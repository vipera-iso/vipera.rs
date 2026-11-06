//! `vipera serve` — the HTTP surface from `contracts/http-chat.md`.
//!
//! One model, one generation at a time: the engine is single-sequence by design, so requests queue on
//! a mutex instead of interleaving into a half-correct batch. Everything else is the contract's shape:
//! OpenAI-compatible request and response bodies, `text/event-stream` for `stream: true`, the frozen
//! error envelope, `GET /v1/models`, `/health`, `/ready` and `/metrics`.
//!
//! No HTTP crate and no JSON crate: a chat endpoint is a request line, headers, one JSON body and one
//! response, and the project keeps its dependency list empty on purpose (`research.md`, R3).
//!
//! What Phase 1 does not do, stated here rather than implied: request bodies must be `Content-Length`
//! framed (chunked bodies are refused with `invalid_request`), `stop` sequences are parsed and ignored,
//! per-key rate limiting (`429`) is not implemented, and a client that disconnects mid-stream still
//! lets the current turn finish before the connection is dropped.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use vipera_backend_cpu::CpuBackend;
use vipera_core::gguf::GgufFile;
use vipera_core::planner::KvCacheType;
use vipera_errors::ViperaError;
use vipera_runtime::{DEFAULT_TOKENS_PER_BLOCK, Engine, EngineConfig, KvCache, KvConfig};
use vipera_sampler::SamplerConfig;
use vipera_tokenizer::{ChatMessage, Tokenizer};

use crate::json::{Json, quote};

/// Bodies larger than this are refused instead of buffered.
const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;
/// Request line plus headers must fit here; anything larger is not a chat request.
const MAX_HEAD_BYTES: usize = 64 * 1024;

pub fn command(args: &[String]) -> Result<u8, ViperaError> {
    let mut path: Option<&str> = None;
    let mut host = "127.0.0.1".to_owned();
    let mut port: u16 = 8080;
    let mut api_key: Option<String> = None;
    let mut model_id: Option<String> = None;
    let mut ctx: Option<u64> = None;
    let mut kv_type = KvCacheType::F16;

    let mut index = 0;
    while index < args.len() {
        match args[index].as_str() {
            "--host" => {
                host = next_value(args, index, "--host")?.to_owned();
                index += 1;
            }
            "--port" => {
                port = next_number(args, index, "--port")?;
                index += 1;
            }
            "--api-key" => {
                api_key = Some(next_value(args, index, "--api-key")?.to_owned());
                index += 1;
            }
            "--model-id" => {
                model_id = Some(next_value(args, index, "--model-id")?.to_owned());
                index += 1;
            }
            "--ctx" => {
                ctx = Some(next_number(args, index, "--ctx")?);
                index += 1;
            }
            "--kv-type" => {
                kv_type = KvCacheType::parse(next_value(args, index, "--kv-type")?)?;
                index += 1;
            }
            flag if flag.starts_with('-') => {
                return Err(ViperaError::usage(format!(
                    "unknown flag \"{flag}\" for serve"
                )));
            }
            other => {
                if path.is_some() {
                    return Err(ViperaError::usage(format!(
                        "unexpected extra argument \"{other}\""
                    )));
                }
                path = Some(other);
            }
        }
        index += 1;
    }

    let path = path.ok_or_else(|| ViperaError::usage("serve needs a model path"))?;
    let model_id = model_id.unwrap_or_else(|| default_model_id(path));

    // A model exposed beyond loopback without a key is an open door; refuse instead of warning.
    let loopback = host == "127.0.0.1" || host == "::1" || host == "localhost";
    if !loopback && api_key.is_none() {
        return Err(ViperaError::usage(
            "binding serve to a non-loopback address needs --api-key",
        ));
    }

    let served = Arc::new(Served::load(path, model_id, api_key, ctx, kv_type)?);
    let listener = TcpListener::bind((host.as_str(), port)).map_err(|error| {
        ViperaError::invalid_argument(format!("cannot bind {host}:{port}: {error}"))
    })?;
    let address = listener
        .local_addr()
        .map_err(|error| ViperaError::internal(format!("bound socket has no address: {error}")))?;
    println!(
        "vipera: serving {} as \"{}\" on http://{address}",
        path, served.model
    );
    std::io::stdout()
        .flush()
        .map_err(|error| ViperaError::internal(format!("cannot flush stdout: {error}")))?;

    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                let served = Arc::clone(&served);
                std::thread::spawn(move || {
                    if let Err(error) = handle(&served, stream) {
                        eprintln!("vipera: connection ended: {error}");
                    }
                });
            }
            Err(error) => eprintln!("vipera: accept failed: {error}"),
        }
    }
    Ok(0)
}

fn default_model_id(path: &str) -> String {
    std::path::Path::new(path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("local")
        .to_owned()
}

fn next_value<'a>(args: &'a [String], index: usize, flag: &str) -> Result<&'a str, ViperaError> {
    args.get(index + 1)
        .map(String::as_str)
        .ok_or_else(|| ViperaError::usage(format!("{flag} needs a value")))
}

fn next_number<T: std::str::FromStr>(
    args: &[String],
    index: usize,
    flag: &str,
) -> Result<T, ViperaError> {
    let raw = next_value(args, index, flag)?;
    raw.parse::<T>()
        .map_err(|_| ViperaError::usage(format!("{flag} needs a number, got \"{raw}\"")))
}

/// The loaded model plus everything the handlers share.
struct Served {
    engine: Mutex<Engine<CpuBackend>>,
    tokenizer: Tokenizer,
    model: String,
    context_length: usize,
    api_key: Option<String>,
    metrics: Mutex<Metrics>,
}

impl Served {
    fn load(
        path: &str,
        model: String,
        api_key: Option<String>,
        ctx: Option<u64>,
        kv_type: KvCacheType,
    ) -> Result<Served, ViperaError> {
        let bytes = std::fs::read(path)?;
        let file = GgufFile::parse(&bytes)?;
        let tokenizer = Tokenizer::from_gguf(&file)?;
        let backend = CpuBackend::from_bytes(bytes, &file)?;
        let config = backend.config();

        let bytes_per_token = config.kv_bytes_per_token_for(kv_type).ok_or_else(|| {
            ViperaError::unsupported(format!(
                "this model cannot store its KV cache as {}",
                kv_type.as_str()
            ))
        })?;
        let limit = file.describe().context_limit.unwrap_or(2048);
        let context_length = ctx.unwrap_or(limit.min(2048));
        let budget = context_length
            .checked_mul(bytes_per_token)
            .ok_or_else(|| ViperaError::invalid_argument("--ctx is too large for this model"))?;
        let kv = KvCache::new(KvConfig::from_bytes(
            budget,
            bytes_per_token,
            DEFAULT_TOKENS_PER_BLOCK,
        )?)?;

        let mut eos = Vec::new();
        for marker in ["<|im_end|>", "<|endoftext|>", "</s>"] {
            if let Some(id) = tokenizer.token_id(marker)
                && !eos.contains(&id)
            {
                eos.push(id);
            }
        }
        let engine = Engine::new(
            kv,
            backend,
            EngineConfig {
                eos,
                sampler: SamplerConfig {
                    temperature: 0.7,
                    ..SamplerConfig::default()
                },
                recent_window: 128,
            },
        );

        Ok(Served {
            engine: Mutex::new(engine),
            tokenizer,
            model,
            context_length: usize::try_from(context_length).unwrap_or(usize::MAX),
            api_key,
            metrics: Mutex::new(Metrics::default()),
        })
    }
}

fn handle(served: &Served, mut stream: TcpStream) -> std::io::Result<()> {
    let Some(request) = read_request(&mut stream)? else {
        return Ok(());
    };
    let started = Instant::now();
    let status = route(served, &mut stream, &request, started)?;
    let endpoint = endpoint_name(&request.path);
    if let Ok(mut metrics) = served.metrics.lock() {
        metrics.record(endpoint, status, started.elapsed().as_secs_f64());
    }
    Ok(())
}

fn route(
    served: &Served,
    stream: &mut TcpStream,
    request: &Request,
    started: Instant,
) -> std::io::Result<u16> {
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/health") => {
            respond(stream, 200, "OK", "application/json", r#"{"status":"ok"}"#)?;
            Ok(200)
        }
        ("GET", "/ready") => {
            let body = format!(r#"{{"status":"ready","model":{}}}"#, quote(&served.model));
            respond(stream, 200, "OK", "application/json", &body)?;
            Ok(200)
        }
        ("GET", "/v1/models") => {
            let body = format!(
                r#"{{"object":"list","data":[{{"id":{},"object":"model","owned_by":"local"}}]}}"#,
                quote(&served.model)
            );
            respond(stream, 200, "OK", "application/json", &body)?;
            Ok(200)
        }
        ("GET", "/metrics") => {
            let blocks = served
                .engine
                .lock()
                .map(|engine| engine.stats().kv_used_blocks)
                .unwrap_or(0);
            let body = served
                .metrics
                .lock()
                .map(|metrics| metrics.render(blocks))
                .unwrap_or_default();
            respond(stream, 200, "OK", "text/plain; version=0.0.4", &body)?;
            Ok(200)
        }
        ("POST", "/v1/chat/completions") => {
            if let Some(key) = &served.api_key
                && !request.authorized(key)
            {
                respond_error(stream, 401, "unauthorized", "missing or wrong API key")?;
                return Ok(401);
            }
            chat(served, stream, request, started)
        }
        ("GET", _) | ("POST", _) | ("PUT", _) | ("DELETE", _) | ("PATCH", _) => {
            respond_error(stream, 404, "not_found", "no such route")?;
            Ok(404)
        }
        _ => {
            respond_error(stream, 405, "invalid_request", "method not allowed")?;
            Ok(405)
        }
    }
}

fn endpoint_name(path: &str) -> &'static str {
    match path {
        "/v1/chat/completions" => "chat_completions",
        "/v1/models" => "models",
        "/metrics" => "metrics",
        "/health" => "health",
        "/ready" => "ready",
        _ => "other",
    }
}

fn chat(
    served: &Served,
    stream: &mut TcpStream,
    request: &Request,
    started: Instant,
) -> std::io::Result<u16> {
    let body = match std::str::from_utf8(&request.body) {
        Ok(text) => text,
        Err(_) => {
            respond_error(stream, 400, "invalid_request", "body is not UTF-8")?;
            return Ok(400);
        }
    };
    let parsed = match Json::parse(body) {
        Ok(value) => value,
        Err(error) => {
            respond_error(stream, 400, "invalid_request", &error)?;
            return Ok(400);
        }
    };
    if let Some(model) = parsed.get("model").and_then(Json::as_str)
        && model != served.model
    {
        respond_error(
            stream,
            404,
            "model_not_found",
            &format!("no model named {model:?}"),
        )?;
        return Ok(404);
    }

    let messages = match messages_of(&parsed) {
        Ok(messages) => messages,
        Err(message) => {
            respond_error(stream, 400, "invalid_request", &message)?;
            return Ok(400);
        }
    };
    let streaming = parsed
        .get("stream")
        .and_then(Json::as_bool)
        .unwrap_or(false);

    let ids = match served.tokenizer.encode_chat(&messages, true) {
        Ok(ids) => ids,
        Err(error) => {
            respond_error(stream, 500, "internal_error", &error.to_string())?;
            return Ok(500);
        }
    };
    if ids.len() >= served.context_length {
        respond_error(
            stream,
            422,
            "context_too_long",
            &format!(
                "prompt is {} tokens but the context holds {}",
                ids.len(),
                served.context_length
            ),
        )?;
        return Ok(422);
    }

    let remaining = (served.context_length - ids.len()) as u32;
    let requested = parsed
        .get("max_tokens")
        .and_then(Json::as_u32)
        .unwrap_or(256)
        .max(1);
    let max_tokens = requested.min(remaining);
    let sampler = sampler_of(&parsed);

    let id = format!("chatcmpl-{}", request_id());
    let created = now_seconds();
    let mut engine = match served.engine.lock() {
        Ok(engine) => engine,
        Err(_) => {
            respond_error(stream, 500, "internal_error", "engine lock was poisoned")?;
            return Ok(500);
        }
    };
    engine.set_sampler(sampler);

    let sequence = match engine.admit(&ids, max_tokens) {
        Ok(sequence) => sequence,
        Err(error) => {
            let (status, code) = map_engine_error(&error);
            respond_error(stream, status, code, &error.to_string())?;
            return Ok(status);
        }
    };

    if streaming {
        stream.set_nodelay(true).ok();
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\n\
             Connection: close\r\n\r\n"
        )?;
        let mut seen: Vec<u32> = Vec::new();
        let mut shown = String::new();
        let mut first = true;
        let mut disconnected = false;
        let generated = engine.generate_with(sequence, &mut |token| {
            if disconnected {
                return;
            }
            seen.push(token);
            let Ok(text) = served.tokenizer.decode(&seen) else {
                return;
            };
            if !text.starts_with(shown.as_str()) {
                return;
            }
            let delta = &text[shown.len()..];
            if delta.is_empty() {
                return;
            }
            let frame = chunk(&id, created, &served.model, Some(delta), None);
            if sse(stream, &frame).is_err() {
                disconnected = true;
                return;
            }
            if first {
                first = false;
                if let Ok(mut metrics) = served.metrics.lock() {
                    metrics.time_to_first_token(started.elapsed().as_secs_f64());
                }
            }
            shown.push_str(delta);
            if let Ok(mut metrics) = served.metrics.lock() {
                metrics.tokens_generated(1);
            }
        });
        let _ = engine.release(sequence);

        let finished = match generated {
            Ok(generation) => generation,
            Err(error) => {
                let _ = sse(
                    stream,
                    &format!(r#"{{"error":{}}}"#, quote(&error.to_string())),
                );
                let _ = sse(stream, "[DONE]");
                return Ok(200);
            }
        };
        if !disconnected {
            // Anything held back because it was not a complete string yet.
            if let Ok(text) = served.tokenizer.decode(&finished.tokens)
                && text.starts_with(shown.as_str())
                && text.len() > shown.len()
            {
                let _ = sse(
                    stream,
                    &chunk(
                        &id,
                        created,
                        &served.model,
                        Some(&text[shown.len()..]),
                        None,
                    ),
                );
            }
            let reason = finish_reason(&finished.finish);
            let final_frame = chunk(&id, created, &served.model, None, Some(reason));
            let _ = sse(stream, &final_frame);
            let _ = sse(stream, "[DONE]");
        }
        let _ = stream.flush();
        return Ok(200);
    }

    let generation = match engine.generate(sequence) {
        Ok(generation) => generation,
        Err(error) => {
            let _ = engine.release(sequence);
            let (status, code) = map_engine_error(&error);
            respond_error(stream, status, code, &error.to_string())?;
            return Ok(status);
        }
    };
    let _ = engine.release(sequence);
    let text = served
        .tokenizer
        .decode(&generation.tokens)
        .unwrap_or_default();
    let prompt_tokens = ids.len();
    let completion_tokens = generation.tokens.len();
    let body = format!(
        r#"{{"id":{},"object":"chat.completion","created":{created},"model":{},"choices":[{{"index":0,"message":{{"role":"assistant","content":{}}},"finish_reason":{}}}],"usage":{{"prompt_tokens":{prompt_tokens},"completion_tokens":{completion_tokens},"total_tokens":{}}}}}"#,
        quote(&id),
        quote(&served.model),
        quote(&text),
        quote(finish_reason(&generation.finish)),
        prompt_tokens + completion_tokens
    );
    respond(stream, 200, "OK", "application/json", &body)?;
    if let Ok(mut metrics) = served.metrics.lock() {
        metrics.tokens_generated(completion_tokens as u64);
    }
    Ok(200)
}

fn messages_of(request: &Json) -> Result<Vec<ChatMessage>, String> {
    let items = request
        .get("messages")
        .and_then(Json::as_array)
        .ok_or_else(|| "messages must be an array".to_owned())?;
    if items.is_empty() {
        return Err("messages must not be empty".to_owned());
    }
    let mut messages = Vec::with_capacity(items.len());
    for item in items {
        let role = item
            .get("role")
            .and_then(Json::as_str)
            .ok_or_else(|| "every message needs a role".to_owned())?;
        let content = match item.get("content") {
            Some(Json::String(text)) => text.clone(),
            Some(Json::Array(parts)) => {
                let mut joined = String::new();
                for part in parts {
                    if let Some(text) = part.get("text").and_then(Json::as_str) {
                        joined.push_str(text);
                    }
                }
                joined
            }
            _ => return Err("every message needs string content".to_owned()),
        };
        messages.push(match role {
            "system" => ChatMessage::system(content),
            "user" => ChatMessage::user(content),
            "assistant" => ChatMessage::assistant(content),
            other => return Err(format!("unsupported role {other:?}")),
        });
    }
    Ok(messages)
}

/// Sampling knobs from the request. Unknown or out-of-range values fall back to the defaults.
fn sampler_of(request: &Json) -> SamplerConfig {
    let mut sampler = SamplerConfig::default();
    if let Some(value) = request.get("temperature").and_then(Json::as_f64) {
        sampler.temperature = value.clamp(0.0, 2.0) as f32;
    }
    if let Some(value) = request.get("top_p").and_then(Json::as_f64) {
        sampler.top_p = (value.clamp(0.0, 1.0)) as f32;
    }
    if let Some(value) = request.get("top_k").and_then(Json::as_u32) {
        sampler.top_k = value as usize;
    }
    if let Some(value) = request.get("min_p").and_then(Json::as_f64) {
        sampler.min_p = (value.clamp(0.0, 1.0)) as f32;
    }
    if let Some(value) = request.get("seed").and_then(Json::as_u32) {
        sampler.seed = u64::from(value);
    }
    if let Some(value) = request.get("repetition_penalty").and_then(Json::as_f64)
        && value > 0.0
    {
        sampler.repetition_penalty = value as f32;
    }
    sampler
}

fn finish_reason(finish: &vipera_runtime::FinishReason) -> &'static str {
    match finish {
        vipera_runtime::FinishReason::EndOfSequence => "stop",
        vipera_runtime::FinishReason::Length => "length",
        vipera_runtime::FinishReason::Cancelled => "cancelled",
        vipera_runtime::FinishReason::Error => "error",
    }
}

fn map_engine_error(error: &ViperaError) -> (u16, &'static str) {
    match error.exit_code() {
        2 => (400, "invalid_request"),
        3 => (422, "context_too_long"),
        _ => (500, "internal_error"),
    }
}

fn chunk(
    id: &str,
    created: u64,
    model: &str,
    content: Option<&str>,
    finish: Option<&str>,
) -> String {
    let delta = match content {
        Some(text) => format!(r#"{{"role":"assistant","content":{}}}"#, quote(text)),
        None => "{}".to_owned(),
    };
    let reason = match finish {
        Some(reason) => quote(reason),
        None => "null".to_owned(),
    };
    format!(
        r#"{{"id":{},"object":"chat.completion.chunk","created":{created},"model":{},"choices":[{{"index":0,"delta":{delta},"finish_reason":{reason}}}]}}"#,
        quote(id),
        quote(model)
    )
}

fn sse(stream: &mut TcpStream, data: &str) -> std::io::Result<()> {
    stream.write_all(format!("data: {data}\n\n").as_bytes())?;
    stream.flush()
}

fn respond(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    content_type: &str,
    body: &str,
) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\
         Connection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()
}

fn respond_error(
    stream: &mut TcpStream,
    status: u16,
    code: &str,
    message: &str,
) -> std::io::Result<u16> {
    let kind = match status {
        401 => "authentication_error",
        404 => "invalid_request_error",
        429 => "rate_limit_error",
        500 | 503 => "server_error",
        _ => "invalid_request_error",
    };
    let body = format!(
        r#"{{"error":{{"message":{},"type":"{kind}","code":{}}}}}"#,
        quote(message),
        quote(code)
    );
    let reason = match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        405 => "Method Not Allowed",
        413 => "Payload Too Large",
        422 => "Unprocessable Entity",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Error",
    };
    respond(stream, status, reason, "application/json", &body)?;
    Ok(status)
}

struct Request {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl Request {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }

    fn authorized(&self, expected: &str) -> bool {
        match self.header("authorization") {
            Some(value) => value
                .strip_prefix("Bearer ")
                .map(str::trim)
                .is_some_and(|key| key == expected),
            None => false,
        }
    }
}

/// Read one request. `Ok(None)` means the peer closed before sending a request line.
fn read_request(stream: &mut TcpStream) -> std::io::Result<Option<Request>> {
    let mut buffer = Vec::with_capacity(1024);
    let mut chunk = [0u8; 8192];
    let head_end = loop {
        if let Some(position) = find(&buffer, b"\r\n\r\n") {
            break position;
        }
        if buffer.len() > MAX_HEAD_BYTES {
            respond_error(
                stream,
                400,
                "invalid_request",
                "request headers are too large",
            )?;
            return Ok(None);
        }
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            return Ok(None);
        }
        buffer.extend_from_slice(&chunk[..read]);
    };

    let head = String::from_utf8_lossy(&buffer[..head_end]).into_owned();
    let mut lines = head.split("\r\n");
    let Some(line) = lines.next() else {
        respond_error(stream, 400, "invalid_request", "malformed request line")?;
        return Ok(None);
    };
    let mut parts = line.split(' ');
    let (Some(method), Some(path)) = (parts.next(), parts.next()) else {
        respond_error(stream, 400, "invalid_request", "malformed request line")?;
        return Ok(None);
    };
    let mut headers = Vec::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.push((name.trim().to_owned(), value.trim().to_owned()));
        }
    }

    let mut request = Request {
        method: method.to_owned(),
        path: path.split('?').next().unwrap_or(path).to_owned(),
        headers,
        body: Vec::new(),
    };

    if request
        .header("transfer-encoding")
        .is_some_and(|value| value.to_ascii_lowercase().contains("chunked"))
    {
        respond_error(
            stream,
            400,
            "invalid_request",
            "chunked request bodies are not supported; send Content-Length",
        )?;
        return Ok(None);
    }
    let length = match request.header("content-length") {
        Some(value) => match value.parse::<usize>() {
            Ok(length) => length,
            Err(_) => {
                respond_error(stream, 400, "invalid_request", "bad Content-Length")?;
                return Ok(None);
            }
        },
        None => 0,
    };
    if length > MAX_BODY_BYTES {
        respond_error(stream, 413, "invalid_request", "request body is too large")?;
        return Ok(None);
    }
    if request
        .header("expect")
        .is_some_and(|value| value.eq_ignore_ascii_case("100-continue"))
    {
        stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
        stream.flush()?;
    }

    let mut body = buffer[head_end + 4..].to_vec();
    while body.len() < length {
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        body.extend_from_slice(&chunk[..read]);
    }
    body.truncate(length);
    request.body = body;
    Ok(Some(request))
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn request_id() -> String {
    // Time in milliseconds plus a per-process counter: unique enough for logs, no dependency.
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis())
        .unwrap_or(0);
    let counter = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    base32(millis << 12 | u128::from(counter & 0xfff))
}

static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn base32(mut value: u128) -> String {
    const ALPHABET: &[u8] = b"0123456789abcdefghjkmnpqrstvwxyz";
    let mut out = Vec::new();
    while value > 0 {
        out.push(ALPHABET[(value % 32) as usize]);
        value /= 32;
    }
    if out.is_empty() {
        out.push(b'0');
    }
    out.reverse();
    String::from_utf8(out).unwrap_or_default()
}

fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

#[derive(Default)]
struct Metrics {
    requests: BTreeMap<(String, u16), u64>,
    tokens: u64,
    duration: Histogram,
    first_token: Histogram,
}

impl Metrics {
    fn record(&mut self, endpoint: &str, status: u16, seconds: f64) {
        *self
            .requests
            .entry((endpoint.to_owned(), status))
            .or_insert(0) += 1;
        self.duration.observe(seconds);
    }

    fn time_to_first_token(&mut self, seconds: f64) {
        self.first_token.observe(seconds);
    }

    fn tokens_generated(&mut self, count: u64) {
        self.tokens += count;
    }

    fn render(&self, kv_blocks: u64) -> String {
        let mut out = String::new();
        for ((endpoint, status), count) in &self.requests {
            out.push_str(&format!(
                "vipera_requests_total{{endpoint=\"{endpoint}\",status=\"{status}\"}} {count}\n"
            ));
        }
        out.push_str(&format!("vipera_tokens_generated_total {}\n", self.tokens));
        self.duration
            .render("vipera_request_duration_seconds", &mut out);
        self.first_token
            .render("vipera_time_to_first_token_seconds", &mut out);
        out.push_str(&format!("vipera_kv_blocks_in_use {kv_blocks}\n"));
        out.push_str("vipera_oom_events_total 0\n");
        out
    }
}

#[derive(Default)]
struct Histogram {
    buckets: [u64; BUCKETS.len()],
    sum: f64,
    count: u64,
}

const BUCKETS: [f64; 10] = [0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0];

impl Histogram {
    fn observe(&mut self, seconds: f64) {
        for (slot, bound) in self.buckets.iter_mut().zip(BUCKETS) {
            if seconds <= bound {
                *slot += 1;
            }
        }
        self.sum += seconds;
        self.count += 1;
    }

    fn render(&self, name: &str, out: &mut String) {
        for (count, bound) in self.buckets.iter().zip(BUCKETS) {
            out.push_str(&format!("{name}_bucket{{le=\"{bound}\"}} {count}\n"));
        }
        out.push_str(&format!(
            "{name}_bucket{{le=\"+Inf\"}} {}\n{name}_sum {}\n{name}_count {}\n",
            self.count, self.sum, self.count
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_roles_the_contract_does_not_define() {
        let request = Json::parse(r#"{"messages":[{"role":"tool","content":"x"}]}"#).unwrap();
        let error = messages_of(&request).expect_err("tool is not supported");
        assert!(error.contains("unsupported role"), "{error}");
    }

    #[test]
    fn accepts_content_parts_as_well_as_strings() {
        let request = Json::parse(
            r#"{"messages":[{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]}"#,
        )
        .unwrap();
        let messages = messages_of(&request).expect("valid");
        assert_eq!(messages.len(), 1);
    }

    #[test]
    fn sampler_fields_are_clamped() {
        let request = Json::parse(
            r#"{"temperature":9,"top_p":2,"top_k":5,"min_p":-1,"seed":7,"repetition_penalty":1.2}"#,
        )
        .unwrap();
        let sampler = sampler_of(&request);
        assert_eq!(sampler.temperature, 2.0);
        assert_eq!(sampler.top_p, 1.0);
        assert_eq!(sampler.top_k, 5);
        assert_eq!(sampler.min_p, 0.0);
        assert_eq!(sampler.seed, 7);
        assert!((sampler.repetition_penalty - 1.2).abs() < 1e-6);
    }

    #[test]
    fn stream_frames_parse_as_json() {
        let frame = chunk("chatcmpl-1", 42, "qwen2", Some("hi \"there\""), None);
        let parsed = Json::parse(&frame).expect("valid frame");
        assert_eq!(
            parsed
                .get("choices")
                .and_then(Json::as_array)
                .and_then(|choices| choices[0].get("delta"))
                .and_then(|delta| delta.get("content"))
                .and_then(Json::as_str),
            Some("hi \"there\"")
        );
        let final_frame = chunk("chatcmpl-1", 42, "qwen2", None, Some("stop"));
        assert_eq!(
            Json::parse(&final_frame)
                .unwrap()
                .get("choices")
                .and_then(Json::as_array)
                .and_then(|choices| choices[0].get("finish_reason"))
                .and_then(Json::as_str),
            Some("stop")
        );
    }

    #[test]
    fn bearer_tokens_are_compared_exactly() {
        let request = Request {
            method: "POST".to_owned(),
            path: "/v1/chat/completions".to_owned(),
            headers: vec![("Authorization".to_owned(), "Bearer secret".to_owned())],
            body: Vec::new(),
        };
        assert!(request.authorized("secret"));
        assert!(!request.authorized("secre"));
        assert!(!request.authorized(""));
    }

    #[test]
    fn histograms_expose_the_prometheus_shape() {
        let mut histogram = Histogram::default();
        histogram.observe(0.2);
        let mut out = String::new();
        histogram.render("vipera_request_duration_seconds", &mut out);
        assert!(out.contains("vipera_request_duration_seconds_bucket{le=\"0.25\"} 1"));
        assert!(out.contains("vipera_request_duration_seconds_bucket{le=\"+Inf\"} 1"));
        assert!(out.contains("vipera_request_duration_seconds_count 1"));
    }

    #[test]
    fn error_bodies_use_the_contract_envelope() {
        let error = map_engine_error(&ViperaError::infeasible("too big"));
        assert_eq!(error, (422, "context_too_long"));
    }
}
