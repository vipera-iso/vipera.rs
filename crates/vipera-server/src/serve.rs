//! `vipera serve` — the HTTP surface from `contracts/http-chat.md`.
//!
//! One model, one stepping thread: the engine is not `Sync` (it owns the weights and the KV cache), so
//! exactly one thread calls [`Engine::step`]. Handlers keep the engine lock only long enough to attach
//! their sampler and queue a turn, then wait on a channel; the scheduler batches whatever is queued
//! into one forward pass, which is what makes four concurrent requests one batch rather than four
//! queueing turns. Everything else is the contract's shape: OpenAI-compatible request and response
//! bodies, `text/event-stream` for `stream: true`, the frozen error envelope, `GET /v1/models`,
//! `/health`, `/ready` and `/metrics`.
//!
//! No HTTP crate and no JSON crate: a chat endpoint is a request line, headers, one JSON body and one
//! response, and the project keeps its dependency list empty on purpose (`research.md`, R3).
//!
//! What Phase 1 does not do, stated here rather than implied: request bodies must be `Content-Length`
//! framed (chunked bodies are refused with `invalid_request`), `stop` sequences are parsed and ignored,
//! and a client that disconnects mid-stream still lets the current turn finish before the
//! connection is dropped. Per-key rate limiting is opt-in via `--rate-limit` (requests per minute
//! per client, `429` + `Retry-After` when exceeded); without the flag there is no limit.

use std::collections::{BTreeMap, HashMap};
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use vipera_backend_cpu::CpuBackend;
use vipera_core::gguf::GgufFile;
use vipera_core::planner::KvCacheType;
use vipera_errors::ViperaError;
use vipera_runtime::{
    DEFAULT_TOKENS_PER_BLOCK, Engine, EngineConfig, Generation, KvCache, KvConfig, StepReport,
};
use vipera_sampler::SamplerConfig;
use vipera_tokenizer::{ChatMessage, Tokenizer};

use crate::json::{Json, quote};

/// Bodies larger than this are refused instead of buffered.
const MAX_BODY_BYTES: usize = 8 * 1024 * 1024;
/// Request line plus headers must fit here; anything larger is not a chat request.
const MAX_HEAD_BYTES: usize = 64 * 1024;

/// OpenAI accepts at most four stop sequences per request.
const MAX_STOP_SEQUENCES: usize = 4;

/// The engine batches at most four turns, so that is the ceiling for `n`.
const MAX_CHOICES: usize = 4;

/// Run `vipera serve`: parse `args`, load the model, and serve until a signal arrives.
///
/// `args` are the flags after the subcommand. Returns the process exit code, which is `0` after a
/// clean shutdown; a bad flag or a model that fails to load comes back as the error instead.
pub fn command(args: &[String]) -> Result<u8, ViperaError> {
    let mut path: Option<&str> = None;
    let mut host = "127.0.0.1".to_owned();
    let mut port: u16 = 8080;
    let mut api_key: Option<String> = None;
    let mut model_id: Option<String> = None;
    let mut ctx: Option<u64> = None;
    let mut kv_type = KvCacheType::F16;
    let mut rate_limit: Option<u32> = None;

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
            "--rate-limit" => {
                let limit = next_number(args, index, "--rate-limit")?;
                if limit == 0 {
                    return Err(ViperaError::usage(
                        "--rate-limit must be at least 1 (omit the flag for no limit)",
                    ));
                }
                rate_limit = Some(limit);
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

    let served = Arc::new(Served::load(
        path,
        model_id,
        api_key.clone(),
        ctx,
        kv_type,
        rate_limit,
    )?);
    let listener = TcpListener::bind((host.as_str(), port)).map_err(|error| {
        ViperaError::invalid_argument(format!("cannot bind {host}:{port}: {error}"))
    })?;
    let address = listener
        .local_addr()
        .map_err(|error| ViperaError::internal(format!("bound socket has no address: {error}")))?;
    // A model exposed beyond loopback without a key is an open door; refuse instead of warning.
    // Decide from the *bound* address, not the requested hostname: a name that
    // resolves to a non-loopback address must not bypass the key requirement.
    if !address.ip().is_loopback() && served.api_key.is_none() {
        return Err(ViperaError::usage(
            "binding serve to a non-loopback address needs --api-key",
        ));
    }
    {
        use std::io::Write as _;
        let mut out = std::io::stdout().lock();
        // Ignore a closed pipe here: the server should keep serving.
        let _ = writeln!(
            out,
            "vipera: serving {} as \"{}\" on http://{address}",
            path, served.model
        );
        let _ = out.flush();
    }

    for incoming in listener.incoming() {
        match incoming {
            Ok(stream) => {
                let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(10)));
                let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(30)));
                let served = Arc::clone(&served);
                match std::thread::Builder::new().spawn(move || {
                    if let Err(error) = handle(&served, stream) {
                        eprintln!("vipera: connection ended: {error}");
                    }
                }) {
                    Ok(_) => {}
                    Err(error) => eprintln!("vipera: cannot spawn connection thread: {error}"),
                }
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

fn resolve_model_path(path: &str) -> Result<String, ViperaError> {
    let candidate = std::path::Path::new(path);
    if !candidate.is_dir() {
        return Ok(path.to_owned());
    }
    let mut found = Vec::new();
    let entries = std::fs::read_dir(candidate).map_err(|error| {
        ViperaError::internal(format!("cannot list model directory \"{path}\": {error}"))
    })?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            ViperaError::internal(format!("cannot read model directory \"{path}\": {error}"))
        })?;
        let entry_path = entry.path();
        if entry_path.is_file()
            && entry_path
                .extension()
                .and_then(|ext| ext.to_str())
                .is_some_and(|ext| ext.eq_ignore_ascii_case("gguf"))
        {
            found.push(entry_path.to_string_lossy().into_owned());
        }
    }
    match found.len() {
        1 => Ok(found.into_iter().next().expect("one gguf")),
        0 => Err(ViperaError::usage(format!(
            "directory \"{path}\" contains no .gguf file; pass a model file directly"
        ))),
        many => Err(ViperaError::usage(format!(
            "directory \"{path}\" contains {many} .gguf files; pass one model file directly"
        ))),
    }
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

/// Token bucket per client, shared across handler threads.
struct RateLimiter {
    per_minute: f64,
    buckets: Mutex<HashMap<String, (f64, Instant)>>,
}

impl RateLimiter {
    fn new(per_minute: u32) -> Self {
        Self {
            per_minute: per_minute as f64,
            buckets: Mutex::new(HashMap::new()),
        }
    }

    fn allow(&self, client: &str) -> Result<(), u64> {
        self.allow_at(client, Instant::now())
    }

    fn allow_at(&self, client: &str, now: Instant) -> Result<(), u64> {
        let mut buckets = match self.buckets.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        let entry = buckets
            .entry(client.to_owned())
            .or_insert((self.per_minute, now));
        let elapsed = now.duration_since(entry.1).as_secs_f64();
        entry.0 = (entry.0 + elapsed * self.per_minute / 60.0).min(self.per_minute);
        entry.1 = now;
        if entry.0 >= 1.0 {
            entry.0 -= 1.0;
            Ok(())
        } else {
            let retry = ((1.0 - entry.0) * 60.0 / self.per_minute).ceil() as u64;
            Err(retry.max(1))
        }
    }
}

/// The loaded model plus everything the handlers share.
struct Served {
    engine: Arc<Mutex<Engine<CpuBackend>>>,
    router: Arc<Router>,
    tokenizer: Tokenizer,
    model: String,
    context_length: usize,
    api_key: Option<String>,
    rate_limit: Option<Arc<RateLimiter>>,
    metrics: Mutex<Metrics>,
}

impl Served {
    fn load(
        path: &str,
        model: String,
        api_key: Option<String>,
        ctx: Option<u64>,
        kv_type: KvCacheType,
        rate_limit: Option<u32>,
    ) -> Result<Served, ViperaError> {
        let bytes = std::fs::read(resolve_model_path(path)?)?;
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
                // The scheduler filters stop tokens out of the batch before it posts them, so it
                // needs its own copy.
                eos: eos.clone(),
                sampler: SamplerConfig {
                    temperature: 0.7,
                    ..SamplerConfig::default()
                },
                recent_window: 128,
                max_batch: 4,
            },
        );

        let engine = Arc::new(Mutex::new(engine));
        let router = Arc::new(Router::new());
        {
            let engine = Arc::clone(&engine);
            let router = Arc::clone(&router);
            std::thread::spawn(move || schedule(engine, router, eos));
        }

        Ok(Served {
            engine,
            router,
            tokenizer,
            model,
            context_length: usize::try_from(context_length).unwrap_or(usize::MAX),
            api_key,
            rate_limit: rate_limit.map(|limit| Arc::new(RateLimiter::new(limit))),
            metrics: Mutex::new(Metrics::default()),
        })
    }
}

/// What a handler receives from the scheduler: one token, the end of its turn, or a failure.
enum Event {
    Token(u32),
    Finished(Box<Generation>),
    Failed(String),
}

/// Hand-off between request handlers and the one thread that steps the engine.
///
/// The engine is not `Sync` (it owns the weights and the KV cache), so exactly one thread touches it.
/// A handler takes the engine lock only to attach its sampler and queue its turn, then waits on its
/// own channel; the scheduler batches whatever is queued into one forward pass and posts each token
/// to the handler that owns it.
struct Router {
    routes: Mutex<HashMap<u64, mpsc::Sender<Event>>>,
    /// Whether a turn was queued since the scheduler last looked.
    pending: Mutex<bool>,
    wake: Condvar,
}

impl Router {
    fn new() -> Router {
        Router {
            routes: Mutex::new(HashMap::new()),
            pending: Mutex::new(false),
            wake: Condvar::new(),
        }
    }

    /// Point `sequence`'s tokens at `sender`. Called before the turn is queued, so a step can never
    /// draw a token for a sequence nobody is listening to.
    fn register(&self, sequence: u64, sender: mpsc::Sender<Event>) {
        if let Ok(mut routes) = self.routes.lock() {
            routes.insert(sequence, sender);
        }
    }

    /// Drop a route: the handler already has its generation, or the client is gone.
    fn unregister(&self, sequence: u64) {
        if let Ok(mut routes) = self.routes.lock() {
            routes.remove(&sequence);
        }
    }

    /// Tell the scheduler that a turn was queued.
    fn wake(&self) {
        if let Ok(mut pending) = self.pending.lock() {
            *pending = true;
        }
        self.wake.notify_one();
    }

    /// Block until a turn is queued. The flag is checked under the lock, so a queue that fills
    /// between stepping and waiting cannot be missed.
    fn wait_for_work(&self) {
        let mut pending = match self.pending.lock() {
            Ok(pending) => pending,
            Err(poisoned) => poisoned.into_inner(),
        };
        while !*pending {
            pending = match self.wake.wait(pending) {
                Ok(pending) => pending,
                Err(poisoned) => poisoned.into_inner(),
            };
        }
        *pending = false;
    }

    /// Post the step's tokens and finished turns to their handlers. A finished turn's route is
    /// removed here, because nothing more will arrive for it.
    fn dispatch(&self, report: StepReport, eos: &[u32]) {
        let Ok(mut routes) = self.routes.lock() else {
            return;
        };
        for (sequence, token) in report.tokens {
            if eos.contains(&token) {
                continue;
            }
            if let Some(sender) = routes.get(&sequence) {
                let _ = sender.send(Event::Token(token));
            }
        }
        for generation in report.finished {
            if let Some(sender) = routes.remove(&generation.sequence_id) {
                let _ = sender.send(Event::Finished(Box::new(generation)));
            }
        }
        for (sequence, message) in report.failed {
            if let Some(sender) = routes.remove(&sequence) {
                let _ = sender.send(Event::Failed(message));
            }
        }
    }

    /// Fail every waiting handler: a step that returned an error left no tokens to hand out.
    fn fail(&self, message: &str) {
        let Ok(mut routes) = self.routes.lock() else {
            return;
        };
        for (_, sender) in routes.drain() {
            let _ = sender.send(Event::Failed(message.to_owned()));
        }
    }
}

/// Step the engine while turns are queued, and post each token to the handler that asked for it.
///
/// One step is one forward pass over every turn the engine can batch (up to `max_batch`), so four
/// concurrent requests share one read of the weights. When nothing is queued the thread sleeps on
/// [`Router::wake`] instead of spinning.
fn schedule(engine: Arc<Mutex<Engine<CpuBackend>>>, router: Arc<Router>, eos: Vec<u32>) {
    loop {
        router.wait_for_work();
        loop {
            let report = {
                // Recover a poisoned lock like Router::wait_for_work does, instead of
                // abandoning the scheduler forever (handlers would block indefinitely).
                let mut engine = match engine.lock() {
                    Ok(engine) => engine,
                    Err(poisoned) => poisoned.into_inner(),
                };
                if !engine.has_work() {
                    break;
                }
                engine.step()
            };
            match report {
                Ok(report) => router.dispatch(report, &eos),
                // The failed pass took its entries with it; the handlers still waiting are told, and
                // the scheduler goes back to sleep until new work arrives.
                Err(error) => {
                    router.fail(&error.to_string());
                    break;
                }
            }
        }
    }
}

fn handle(served: &Served, mut stream: TcpStream) -> std::io::Result<()> {
    let peer = stream
        .peer_addr()
        .map(|addr| addr.ip().to_string())
        .unwrap_or_else(|_| "unknown".to_owned());
    let Some(request) = read_request(&mut stream)? else {
        return Ok(());
    };
    let started = Instant::now();
    let status = route(served, &mut stream, &request, started, &peer)?;
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
    peer: &str,
) -> std::io::Result<u16> {
    // G01: when a key is configured, protect every business route. /health is
    // the only public one (it reports nothing about the model).
    if let Some(key) = &served.api_key
        && !matches!(request.path.as_str(), "/health")
        && !request.authorized(key)
    {
        respond_error(stream, 401, "unauthorized", "missing or wrong API key")?;
        return Ok(401);
    }
    if let Some(limiter) = &served.rate_limit
        && !matches!(request.path.as_str(), "/health")
    {
        let client = request
            .header("authorization")
            .and_then(|value| value.strip_prefix("Bearer "))
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
            .unwrap_or_else(|| peer.to_owned());
        if let Err(retry) = limiter.allow(&client) {
            respond_error_retry(
                stream,
                429,
                "rate_limited",
                "request rate limit exceeded",
                retry,
            )?;
            return Ok(429);
        }
    }
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
        ("POST", "/v1/chat/completions") => chat(served, stream, request, started),
        ("POST", "/v1/completions") => completions(served, stream, request, started),
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
        "/v1/completions" => "completions",
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
    let include_usage = include_usage_of(&parsed);

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
    let stops = match stop_sequences_of(&parsed) {
        Ok(stops) => stops,
        Err(message) => {
            respond_error(stream, 400, "invalid_request", &message)?;
            return Ok(400);
        }
    };
    let wants_logprobs = parsed
        .get("logprobs")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    let top_n = parsed
        .get("top_logprobs")
        .and_then(Json::as_u32)
        .unwrap_or(0) as usize;
    if top_n > 5 {
        respond_error(stream, 400, "invalid_request", "top_logprobs is at most 5")?;
        return Ok(400);
    }
    if top_n > 0 && !wants_logprobs {
        respond_error(
            stream,
            400,
            "invalid_request",
            "top_logprobs needs logprobs: true",
        )?;
        return Ok(400);
    }
    if wants_logprobs && streaming {
        respond_error(
            stream,
            400,
            "invalid_request",
            "logprobs needs stream: false",
        )?;
        return Ok(400);
    }

    let id = format!("chatcmpl-{}", request_id());
    let created = now_seconds();
    let n = match choices_of(&parsed) {
        Ok(n) => n,
        Err(message) => {
            respond_error(stream, 400, "invalid_request", &message)?;
            return Ok(400);
        }
    };
    if streaming && n > 1 {
        respond_error(
            stream,
            400,
            "invalid_request",
            "n greater than 1 needs stream: false",
        )?;
        return Ok(400);
    }
    let mut turns = match attach_turns(served, &ids, max_tokens, sampler, n, wants_logprobs, top_n)
    {
        Ok(turns) => turns,
        Err((status, code, message)) => {
            respond_error(stream, status, code, &message)?;
            return Ok(status);
        }
    };

    if streaming {
        let (sequence, receiver) = turns.remove(0);
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
        let mut stopped = false;
        let mut finished: Option<Generation> = None;
        let mut failure: Option<String> = None;
        for event in receiver {
            let token = match event {
                Event::Token(token) => token,
                Event::Finished(generation) => {
                    finished = Some(*generation);
                    break;
                }
                Event::Failed(message) => {
                    failure = Some(message);
                    break;
                }
            };
            seen.push(token);
            let Ok(text) = served.tokenizer.decode(&seen) else {
                continue;
            };
            if !text.starts_with(shown.as_str()) {
                continue;
            }
            // A stop sequence may arrive across token boundaries, so search the text decoded so far
            // and emit only what precedes it. The held-back tail is what keeps a stop from leaking
            // out one token at a time.
            let stop_at = stops.iter().filter_map(|stop| text.find(stop)).min();
            let cut = match stop_at {
                Some(at) => {
                    stopped = true;
                    at
                }
                None => emit_boundary(&text, shown.len(), &stops, false),
            };
            if cut > shown.len() {
                let delta = &text[shown.len()..cut];
                let frame = chunk(&id, created, &served.model, Some(delta), None);
                if sse(stream, &frame).is_err() {
                    // The client is gone: stop the turn instead of generating into a dead socket.
                    disconnected = true;
                    break;
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
            }
            if stopped {
                break;
            }
        }
        if (disconnected || stopped)
            && let Ok(mut engine) = served.engine.lock()
        {
            let _ = engine.cancel(sequence);
        }
        served.router.unregister(sequence);
        if let Ok(mut engine) = served.engine.lock() {
            let _ = engine.release(sequence);
        }
        if disconnected {
            let _ = stream.flush();
            return Ok(200);
        }
        if let Some(message) = failure {
            let _ = sse(stream, &format!(r#"{{"error":{}}}"#, quote(&message)));
            let _ = sse(stream, "[DONE]");
            let _ = stream.flush();
            return Ok(200);
        }
        if stopped {
            let frame = chunk(&id, created, &served.model, None, Some("stop"));
            let _ = sse(stream, &frame);
            if include_usage {
                let usage = usage_chunk(
                    &id,
                    created,
                    &served.model,
                    "chat.completion.chunk",
                    ids.len(),
                    seen.len(),
                );
                let _ = sse(stream, &usage);
            }
            let _ = sse(stream, "[DONE]");
            let _ = stream.flush();
            return Ok(200);
        }
        let Some(finished) = finished else {
            let _ = stream.flush();
            return Ok(200);
        };
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
        if include_usage {
            let usage = usage_chunk(
                &id,
                created,
                &served.model,
                "chat.completion.chunk",
                ids.len(),
                finished.tokens.len(),
            );
            let _ = sse(stream, &usage);
        }
        let _ = sse(stream, "[DONE]");
        let _ = stream.flush();
        return Ok(200);
    }

    let prompt_tokens = ids.len();
    let mut completion_tokens = 0usize;
    let mut index = 0usize;
    let mut choices = String::new();
    while !turns.is_empty() {
        let (sequence, receiver) = turns.remove(0);
        let mut finished: Option<Generation> = None;
        let mut failure: Option<String> = None;
        for event in receiver {
            match event {
                // A non-streaming caller only needs the generation; the tokens are in it.
                Event::Token(_) => {}
                Event::Finished(generation) => {
                    finished = Some(*generation);
                    break;
                }
                Event::Failed(message) => {
                    failure = Some(message);
                    break;
                }
            }
        }
        release_sequence(served, sequence);
        if let Some(message) = failure {
            for (sequence, _) in turns.drain(..) {
                release_sequence(served, sequence);
            }
            respond_error(stream, 500, "internal_error", &message)?;
            return Ok(500);
        }
        let Some(generation) = finished else {
            for (sequence, _) in turns.drain(..) {
                release_sequence(served, sequence);
            }
            respond_error(
                stream,
                500,
                "internal_error",
                "the turn ended without a generation",
            )?;
            return Ok(500);
        };
        let text = served
            .tokenizer
            .decode(&generation.tokens)
            .unwrap_or_default();
        let (text, stopped) = without_stop(&text, &stops);
        let reason = if stopped {
            "stop"
        } else {
            finish_reason(&generation.finish)
        };
        completion_tokens += generation.tokens.len();
        let choice_logprobs = if wants_logprobs {
            logprobs_json(
                &served.tokenizer,
                &generation.tokens,
                &generation.logprobs,
                &generation.top_logprobs,
                text,
            )
        } else {
            "null".to_owned()
        };
        if !choices.is_empty() {
            choices.push(',');
        }
        choices.push_str(&format!(
            r#"{{"index":{},"message":{{"role":"assistant","content":{}}},"logprobs":{choice_logprobs},"finish_reason":{}}}"#,
            index,
            quote(text),
            quote(reason)
        ));
        index += 1;
    }
    let body = format!(
        r#"{{"id":{},"object":"chat.completion","created":{created},"model":{},"choices":[{choices}],"usage":{{"prompt_tokens":{prompt_tokens},"completion_tokens":{completion_tokens},"total_tokens":{}}}}}"#,
        quote(&id),
        quote(&served.model),
        prompt_tokens + completion_tokens
    );
    respond(stream, 200, "OK", "application/json", &body)?;
    if let Ok(mut metrics) = served.metrics.lock() {
        metrics.tokens_generated(completion_tokens as u64);
    }
    Ok(200)
}

/// `POST /v1/completions`: the legacy text endpoint, with the same sampling, stop and streaming
/// behaviour as chat.
fn completions(
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
    let Some(prompt) = parsed.get("prompt").and_then(Json::as_str) else {
        respond_error(stream, 400, "invalid_request", "prompt must be a string")?;
        return Ok(400);
    };
    let ids = match served.tokenizer.encode(prompt) {
        Ok(ids) => ids,
        Err(error) => {
            respond_error(stream, 500, "internal_error", &error.to_string())?;
            return Ok(500);
        }
    };
    if ids.is_empty() {
        respond_error(stream, 400, "invalid_request", "prompt must not be empty")?;
        return Ok(400);
    }
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
    let stops = match stop_sequences_of(&parsed) {
        Ok(stops) => stops,
        Err(message) => {
            respond_error(stream, 400, "invalid_request", &message)?;
            return Ok(400);
        }
    };
    let streaming = parsed
        .get("stream")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    let include_usage = include_usage_of(&parsed);
    let wants_logprobs = parsed
        .get("logprobs")
        .and_then(Json::as_bool)
        .unwrap_or(false);
    let top_n = parsed
        .get("top_logprobs")
        .and_then(Json::as_u32)
        .unwrap_or(0) as usize;
    if top_n > 5 {
        respond_error(stream, 400, "invalid_request", "top_logprobs is at most 5")?;
        return Ok(400);
    }
    if top_n > 0 && !wants_logprobs {
        respond_error(
            stream,
            400,
            "invalid_request",
            "top_logprobs needs logprobs: true",
        )?;
        return Ok(400);
    }
    if wants_logprobs && streaming {
        respond_error(
            stream,
            400,
            "invalid_request",
            "logprobs needs stream: false",
        )?;
        return Ok(400);
    }

    let id = format!("cmpl-{}", request_id());
    let created = now_seconds();
    let n = match choices_of(&parsed) {
        Ok(n) => n,
        Err(message) => {
            respond_error(stream, 400, "invalid_request", &message)?;
            return Ok(400);
        }
    };
    if streaming && n > 1 {
        respond_error(
            stream,
            400,
            "invalid_request",
            "n greater than 1 needs stream: false",
        )?;
        return Ok(400);
    }
    let mut turns = match attach_turns(served, &ids, max_tokens, sampler, n, wants_logprobs, top_n)
    {
        Ok(turns) => turns,
        Err((status, code, message)) => {
            respond_error(stream, status, code, &message)?;
            return Ok(status);
        }
    };

    if streaming {
        let (sequence, receiver) = turns.remove(0);
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
        let mut stopped = false;
        let mut finished: Option<Generation> = None;
        let mut failure: Option<String> = None;
        for event in receiver {
            let token = match event {
                Event::Token(token) => token,
                Event::Finished(generation) => {
                    finished = Some(*generation);
                    break;
                }
                Event::Failed(message) => {
                    failure = Some(message);
                    break;
                }
            };
            seen.push(token);
            let Ok(text) = served.tokenizer.decode(&seen) else {
                continue;
            };
            if !text.starts_with(shown.as_str()) {
                continue;
            }
            let stop_at = stops.iter().filter_map(|stop| text.find(stop)).min();
            let cut = match stop_at {
                Some(at) => {
                    stopped = true;
                    at
                }
                None => emit_boundary(&text, shown.len(), &stops, false),
            };
            if cut > shown.len() {
                let delta = &text[shown.len()..cut];
                let frame = completion_chunk(&id, created, &served.model, Some(delta), None);
                if sse(stream, &frame).is_err() {
                    disconnected = true;
                    break;
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
            }
            if stopped {
                break;
            }
        }
        if (disconnected || stopped)
            && let Ok(mut engine) = served.engine.lock()
        {
            let _ = engine.cancel(sequence);
        }
        served.router.unregister(sequence);
        if let Ok(mut engine) = served.engine.lock() {
            let _ = engine.release(sequence);
        }
        if disconnected {
            let _ = stream.flush();
            return Ok(200);
        }
        if let Some(message) = failure {
            let _ = sse(stream, &format!(r#"{{"error":{}}}"#, quote(&message)));
            let _ = sse(stream, "[DONE]");
            let _ = stream.flush();
            return Ok(200);
        }
        if stopped {
            let frame = completion_chunk(&id, created, &served.model, None, Some("stop"));
            let _ = sse(stream, &frame);
            if include_usage {
                let usage = usage_chunk(
                    &id,
                    created,
                    &served.model,
                    "text_completion",
                    ids.len(),
                    seen.len(),
                );
                let _ = sse(stream, &usage);
            }
            let _ = sse(stream, "[DONE]");
            let _ = stream.flush();
            return Ok(200);
        }
        let Some(finished) = finished else {
            let _ = stream.flush();
            return Ok(200);
        };
        if let Ok(text) = served.tokenizer.decode(&finished.tokens)
            && text.starts_with(shown.as_str())
            && text.len() > shown.len()
        {
            let _ = sse(
                stream,
                &completion_chunk(
                    &id,
                    created,
                    &served.model,
                    Some(&text[shown.len()..]),
                    None,
                ),
            );
        }
        let reason = finish_reason(&finished.finish);
        let frame = completion_chunk(&id, created, &served.model, None, Some(reason));
        let _ = sse(stream, &frame);
        if include_usage {
            let usage = usage_chunk(
                &id,
                created,
                &served.model,
                "text_completion",
                ids.len(),
                finished.tokens.len(),
            );
            let _ = sse(stream, &usage);
        }
        let _ = sse(stream, "[DONE]");
        let _ = stream.flush();
        return Ok(200);
    }

    let prompt_tokens = ids.len();
    let mut completion_tokens = 0usize;
    let mut index = 0usize;
    let mut choices = String::new();
    while !turns.is_empty() {
        let (sequence, receiver) = turns.remove(0);
        let mut finished: Option<Generation> = None;
        let mut failure: Option<String> = None;
        for event in receiver {
            match event {
                Event::Token(_) => {}
                Event::Finished(generation) => {
                    finished = Some(*generation);
                    break;
                }
                Event::Failed(message) => {
                    failure = Some(message);
                    break;
                }
            }
        }
        release_sequence(served, sequence);
        if let Some(message) = failure {
            for (sequence, _) in turns.drain(..) {
                release_sequence(served, sequence);
            }
            respond_error(stream, 500, "internal_error", &message)?;
            return Ok(500);
        }
        let Some(generation) = finished else {
            for (sequence, _) in turns.drain(..) {
                release_sequence(served, sequence);
            }
            respond_error(
                stream,
                500,
                "internal_error",
                "the turn ended without a generation",
            )?;
            return Ok(500);
        };
        let text = served
            .tokenizer
            .decode(&generation.tokens)
            .unwrap_or_default();
        let (text, stopped) = without_stop(&text, &stops);
        let reason = if stopped {
            "stop"
        } else {
            finish_reason(&generation.finish)
        };
        completion_tokens += generation.tokens.len();
        let choice_logprobs = if wants_logprobs {
            logprobs_json(
                &served.tokenizer,
                &generation.tokens,
                &generation.logprobs,
                &generation.top_logprobs,
                text,
            )
        } else {
            "null".to_owned()
        };
        if !choices.is_empty() {
            choices.push(',');
        }
        choices.push_str(&format!(
            r#"{{"text":{},"index":{},"logprobs":{choice_logprobs},"finish_reason":{}}}"#,
            quote(text),
            index,
            quote(reason)
        ));
        index += 1;
    }
    let body = format!(
        r#"{{"id":{},"object":"text_completion","created":{created},"model":{},"choices":[{choices}],"usage":{{"prompt_tokens":{prompt_tokens},"completion_tokens":{completion_tokens},"total_tokens":{}}}}}"#,
        quote(&id),
        quote(&served.model),
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
    if let Some(value) = request.get("top_p").and_then(Json::as_f64)
        && value > 0.0
    {
        sampler.top_p = (value.clamp(f32::MIN_POSITIVE as f64, 1.0)) as f32;
    }
    if let Some(value) = request.get("top_k").and_then(Json::as_u32) {
        sampler.top_k = value as usize;
    }
    if let Some(value) = request.get("min_p").and_then(Json::as_f64) {
        sampler.min_p = (value.clamp(0.0, 0.999_999)) as f32;
    }
    if let Some(value) = request.get("seed").and_then(Json::as_u32) {
        sampler.seed = u64::from(value);
    }
    if let Some(value) = request.get("repetition_penalty").and_then(Json::as_f64)
        && value > 0.0
        && value.is_finite()
    {
        sampler.repetition_penalty = value as f32;
    }
    sampler
}

/// Stop sequences from a request: one string or an array of up to four, as OpenAI defines them.
fn stop_sequences_of(request: &Json) -> Result<Vec<String>, String> {
    let Some(value) = request.get("stop") else {
        return Ok(Vec::new());
    };
    let mut stops = Vec::new();
    match value {
        Json::String(text) => {
            if !text.is_empty() {
                stops.push(text.clone());
            }
        }
        Json::Array(items) => {
            if items.len() > MAX_STOP_SEQUENCES {
                return Err(format!("stop accepts at most {MAX_STOP_SEQUENCES} strings"));
            }
            for item in items {
                match item {
                    Json::String(text) if !text.is_empty() => stops.push(text.clone()),
                    _ => return Err("stop must be a string or an array of strings".to_owned()),
                }
            }
        }
        _ => return Err("stop must be a string or an array of strings".to_owned()),
    }
    Ok(stops)
}

/// Generated text up to the earliest stop sequence, and whether one was found.
fn without_stop<'a>(text: &'a str, stops: &[String]) -> (&'a str, bool) {
    match stops.iter().filter_map(|stop| text.find(stop)).min() {
        Some(at) => (&text[..at], true),
        None => (text, false),
    }
}

/// How much of `text` may be emitted without cutting a stop sequence in half.
///
/// Until the turn finishes, the last `longest stop - 1` bytes are held back: a stop sequence can
/// arrive across token boundaries, so text that looks safe now may become its prefix next token.
fn emit_boundary(text: &str, shown: usize, stops: &[String], finished: bool) -> usize {
    if finished {
        return text.len();
    }
    let hold = stops
        .iter()
        .map(String::len)
        .max()
        .unwrap_or(0)
        .saturating_sub(1);
    let mut cut = text.len().saturating_sub(hold).max(shown);
    while cut > shown && !text.is_char_boundary(cut) {
        cut -= 1;
    }
    cut
}

/// Attach a turn to the engine and return the receiver its tokens arrive on.
/// A queued turn: its sequence id and the channel its tokens arrive on.
type QueuedTurn = (u64, mpsc::Receiver<Event>);

/// Release one turn's blocks and drop its route.
fn release_sequence(served: &Served, sequence: u64) {
    served.router.unregister(sequence);
    if let Ok(mut engine) = served.engine.lock() {
        let _ = engine.release(sequence);
    }
}

/// Attach `n` turns for one request, releasing the ones already queued if a later admit fails.
fn attach_turns(
    served: &Served,
    ids: &[u32],
    max_tokens: u32,
    sampler: SamplerConfig,
    n: usize,
    logprobs: bool,
    top_n: usize,
) -> Result<Vec<QueuedTurn>, (u16, &'static str, String)> {
    let mut turns = Vec::with_capacity(n);
    for _ in 0..n {
        match attach_turn(served, ids, max_tokens, sampler, logprobs, top_n) {
            Ok(turn) => turns.push(turn),
            Err(error) => {
                for (sequence, _) in turns.drain(..) {
                    release_sequence(served, sequence);
                }
                return Err(error);
            }
        }
    }
    Ok(turns)
}

fn attach_turn(
    served: &Served,
    ids: &[u32],
    max_tokens: u32,
    sampler: SamplerConfig,
    logprobs: bool,
    top_n: usize,
) -> Result<QueuedTurn, (u16, &'static str, String)> {
    let (sender, receiver) = mpsc::channel();
    let sequence = {
        // The lock is held only long enough to attach this request's sampler and queue its turn. The
        // scheduler does the work, so other requests can join the same forward pass instead of
        // queueing behind this one.
        let mut engine = served
            .engine
            .lock()
            .map_err(|_| (500, "internal_error", "engine lock was poisoned".to_owned()))?;
        engine.set_sampler(sampler);
        let sequence = engine.admit(ids, max_tokens).map_err(|error| {
            let (status, code) = map_engine_error(&error);
            (status, code, error.to_string())
        })?;
        engine.set_logprobs(sequence, logprobs).map_err(|error| {
            let (status, code) = map_engine_error(&error);
            (status, code, error.to_string())
        })?;
        engine.set_top_logprobs(sequence, top_n).map_err(|error| {
            let (status, code) = map_engine_error(&error);
            (status, code, error.to_string())
        })?;
        // The route goes in before the turn is queued: a step that starts the moment after
        // `enqueue` must already have somewhere to send the first token.
        served.router.register(sequence, sender);
        if let Err(error) = engine.enqueue(sequence) {
            served.router.unregister(sequence);
            let _ = engine.release(sequence);
            let (status, code) = map_engine_error(&error);
            return Err((status, code, error.to_string()));
        }
        sequence
    };
    served.router.wake();
    Ok((sequence, receiver))
}

/// Whether the caller asked for the final usage chunk on a stream.
fn include_usage_of(request: &Json) -> bool {
    request
        .get("stream_options")
        .and_then(|options| options.get("include_usage"))
        .and_then(Json::as_bool)
        .unwrap_or(false)
}

/// `n` from the request, clamped to what one batch can hold.
fn choices_of(request: &Json) -> Result<usize, String> {
    let n = request.get("n").and_then(Json::as_u32).unwrap_or(1);
    if n == 0 {
        return Err("n must be at least 1".to_owned());
    }
    if n as usize > MAX_CHOICES {
        return Err(format!("n is at most {MAX_CHOICES} on this server"));
    }
    Ok(n as usize)
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

/// OpenAI's `logprobs.content`: one entry per generated token, decoding each token as the delta
/// from the text shown so far. Entries stop before any text past `visible`, so a stop sequence is
/// not reported as a generated token.
fn logprobs_json(
    tokenizer: &Tokenizer,
    tokens: &[u32],
    logprobs: &[f32],
    top: &[Vec<(u32, f32)>],
    visible: &str,
) -> String {
    let mut entries = String::new();
    let mut shown = String::new();
    for (index, _) in tokens.iter().enumerate() {
        let Ok(text) = tokenizer.decode(&tokens[..=index]) else {
            continue;
        };
        if text.len() > visible.len() || !text.starts_with(shown.as_str()) {
            break;
        }
        let delta = &text[shown.len()..];
        let logprob = logprobs.get(index).copied().unwrap_or(f32::NAN);
        let logprob = if logprob.is_finite() {
            format!("{logprob}")
        } else {
            "null".to_owned()
        };
        let mut alternatives = String::new();
        for (id, probability) in top.get(index).map(Vec::as_slice).unwrap_or(&[]) {
            let Ok(token_text) = tokenizer.decode(&[*id]) else {
                continue;
            };
            let probability = if probability.is_finite() {
                format!("{probability}")
            } else {
                "null".to_owned()
            };
            if !alternatives.is_empty() {
                alternatives.push(',');
            }
            alternatives.push_str(&format!(
                r#"{{"token":{},"logprob":{probability},"bytes":null}}"#,
                quote(&token_text)
            ));
        }
        if !entries.is_empty() {
            entries.push(',');
        }
        entries.push_str(&format!(
            r#"{{"token":{},"logprob":{logprob},"bytes":null,"top_logprobs":[{alternatives}]}}"#,
            quote(delta)
        ));
        shown = text;
    }
    format!(r#"{{"content":[{entries}]}}"#)
}

/// The extra chunk OpenAI clients expect when `stream_options.include_usage` is set: no choices,
/// just the token counts for the whole stream.
fn usage_chunk(
    id: &str,
    created: u64,
    model: &str,
    object: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
) -> String {
    format!(
        r#"{{"id":{},"object":{},"created":{created},"model":{},"choices":[],"usage":{{"prompt_tokens":{prompt_tokens},"completion_tokens":{completion_tokens},"total_tokens":{}}}}}"#,
        quote(id),
        quote(object),
        quote(model),
        prompt_tokens + completion_tokens
    )
}

fn completion_chunk(
    id: &str,
    created: u64,
    model: &str,
    content: Option<&str>,
    finish: Option<&str>,
) -> String {
    let text = match content {
        Some(text) => quote(text),
        None => quote(""),
    };
    let reason = match finish {
        Some(reason) => quote(reason),
        None => "null".to_owned(),
    };
    format!(
        r#"{{"id":{},"object":"text_completion","created":{created},"model":{},"choices":[{{"text":{text},"index":0,"logprobs":null,"finish_reason":{reason}}}]}}"#,
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

fn respond_error_retry(
    stream: &mut TcpStream,
    status: u16,
    code: &str,
    message: &str,
    retry_after: u64,
) -> std::io::Result<u16> {
    let kind = "rate_limit_error";
    let body = format!(
        r#"{{"error":{{"message":{},"type":"{kind}","code":{}}}}}"#,
        quote(message),
        quote(code)
    );
    let head = format!(
        "HTTP/1.1 {status} Too Many Requests\r\nContent-Type: application/json\r\nContent-Length: {}\r\nRetry-After: {retry_after}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()?;
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
    fn rate_limiter_allows_burst_then_refills() {
        let limiter = RateLimiter::new(60);
        let start = Instant::now();
        for _ in 0..60 {
            assert!(limiter.allow_at("alice", start).is_ok());
        }
        let retry = limiter
            .allow_at("alice", start)
            .expect_err("61st must be limited");
        assert!(retry >= 1);
        assert!(
            limiter.allow_at("bob", start).is_ok(),
            "other client unaffected"
        );
        assert!(
            limiter
                .allow_at("alice", start + std::time::Duration::from_secs(61))
                .is_ok()
        );
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
    fn n_defaults_to_one_and_is_capped_at_the_batch_size() {
        assert_eq!(choices_of(&Json::parse("{}").unwrap()), Ok(1));
        assert_eq!(choices_of(&Json::parse(r#"{"n":4}"#).unwrap()), Ok(4));
        assert!(choices_of(&Json::parse(r#"{"n":0}"#).unwrap()).is_err());
        assert!(choices_of(&Json::parse(r#"{"n":5}"#).unwrap()).is_err());
    }

    #[test]
    fn stop_accepts_a_string_or_an_array() {
        let request = Json::parse(r#"{"stop":"\n\n"}"#).unwrap();
        assert_eq!(stop_sequences_of(&request).unwrap(), vec!["\n\n"]);
        let request = Json::parse(r#"{"stop":["a","b"]}"#).unwrap();
        assert_eq!(stop_sequences_of(&request).unwrap(), vec!["a", "b"]);
        let request = Json::parse("{}").unwrap();
        assert!(stop_sequences_of(&request).unwrap().is_empty());
    }

    #[test]
    fn stop_rejects_more_than_four_sequences_or_a_bad_type() {
        let request = Json::parse(r#"{"stop":["a","b","c","d","e"]}"#).unwrap();
        assert!(stop_sequences_of(&request).is_err());
        let request = Json::parse(r#"{"stop":7}"#).unwrap();
        assert!(stop_sequences_of(&request).is_err());
    }

    #[test]
    fn stop_cuts_at_the_earliest_match() {
        let stops = vec!["STOP".to_owned(), "TOP".to_owned()];
        assert_eq!(without_stop("abcSTOPdef", &stops), ("abc", true));
        assert_eq!(without_stop("abcdef", &stops), ("abcdef", false));
    }

    #[test]
    fn streaming_holds_back_a_partial_stop() {
        let stops = vec!["world".to_owned()];
        // Four trailing bytes could still become "world"; only "hello" is safe.
        assert_eq!(emit_boundary("hello wor", 0, &stops, false), 5);
        // A finished turn emits everything, because no stop occurred.
        assert_eq!(emit_boundary("hello wor", 0, &stops, true), 9);
        // The hold-back never splits a UTF-8 character.
        assert_eq!(emit_boundary("aé", 0, &["xyz".to_owned()], false), 1);
    }

    #[test]
    fn include_usage_requires_a_true_flag() {
        let parse = |body: &str| include_usage_of(&Json::parse(body).unwrap());
        assert!(parse(r#"{"stream_options":{"include_usage":true}}"#));
        assert!(!parse(r#"{"stream_options":{"include_usage":false}}"#));
        assert!(!parse(r#"{"stream_options":{}}"#));
        assert!(!parse("{}"));
        assert!(!parse(r#"{"stream_options":{"include_usage":"yes"}}"#));
    }

    #[test]
    fn usage_chunk_has_no_choices_and_counts_tokens() {
        let frame = usage_chunk("chatcmpl-1", 42, "qwen2", "chat.completion.chunk", 5, 3);
        let parsed = Json::parse(&frame).expect("valid frame");
        assert_eq!(
            parsed
                .get("choices")
                .and_then(Json::as_array)
                .map(<[Json]>::len),
            Some(0)
        );
        let usage = parsed.get("usage").expect("usage present");
        assert_eq!(usage.get("prompt_tokens").and_then(Json::as_u32), Some(5));
        assert_eq!(
            usage.get("completion_tokens").and_then(Json::as_u32),
            Some(3)
        );
        assert_eq!(usage.get("total_tokens").and_then(Json::as_u32), Some(8));
    }

    #[test]
    fn completion_frames_parse_as_json() {
        let frame = completion_chunk("cmpl-1", 42, "qwen2", Some("hi"), None);
        let parsed = Json::parse(&frame).expect("valid frame");
        assert_eq!(
            parsed.get("object").and_then(Json::as_str),
            Some("text_completion")
        );
        assert_eq!(
            parsed
                .get("choices")
                .and_then(Json::as_array)
                .and_then(|choices| choices[0].get("text"))
                .and_then(Json::as_str),
            Some("hi")
        );
        let final_frame = completion_chunk("cmpl-1", 42, "qwen2", None, Some("stop"));
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
    fn error_bodies_use_the_contract_envelope() {
        let error = map_engine_error(&ViperaError::infeasible("too big"));
        assert_eq!(error, (422, "context_too_long"));
    }
}
