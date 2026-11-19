//! The completion endpoints: `POST /v1/chat/completions` and `POST /v1/completions`, plus the
//! request-shaping between a JSON body and the engine — messages, sampling knobs, stop sequences,
//! logprobs, and the hand-off of a queued turn to the scheduler.

use std::sync::mpsc;
use std::time::Instant;

use vipera_errors::ViperaError;
use vipera_runtime::Generation;
use vipera_sampler::{MAX_LOGIT_BIAS, SamplerConfig};
use vipera_tokenizer::{ChatMessage, Tokenizer};

use crate::http::{
    CORS_HEADERS, Request, Transport, now_seconds, request_id, respond, respond_error, sse,
};
use crate::json::{Json, quote};
use crate::serve::{Event, MAX_CHOICES, MAX_STOP_SEQUENCES, Served};

pub(crate) fn chat(
    served: &Served,
    stream: &mut dyn Transport,
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
        // A conversation this build cannot render (no user turn, a template it does not know) is
        // the caller's mistake, so the status follows the error's kind rather than being 500 just
        // because the failure surfaced inside the engine.
        Err(error) => {
            let (status, code) = map_engine_error(&error);
            respond_error(stream, status, code, &error.to_string())?;
            return Ok(status);
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
    let sampler = match sampler_of(&parsed) {
        Ok(sampler) => sampler,
        Err(message) => {
            respond_error(stream, 400, "invalid_request", &message)?;
            return Ok(400);
        }
    };
    let stops = match stop_sequences_of(&parsed) {
        Ok(stops) => stops,
        Err(message) => {
            respond_error(stream, 400, "invalid_request", &message)?;
            return Ok(400);
        }
    };
    let json_mode = match json_mode_of(&parsed) {
        Ok(mode) => mode,
        Err(message) => {
            respond_error(stream, 400, "invalid_request", &message)?;
            return Ok(400);
        }
    };
    if json_mode && streaming {
        respond_error(
            stream,
            400,
            "invalid_request",
            "response_format json_object needs stream: false",
        )?;
        return Ok(400);
    }
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
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n{CORS_HEADERS}\
             Cache-Control: no-cache\r\nConnection: close\r\n\r\n"
        )?;
        let mut seen: Vec<u32> = Vec::new();
        let mut shown = String::new();
        let mut first = true;
        let mut disconnected = false;
        let mut stopped = false;
        let mut finished: Option<Generation> = None;
        let mut failure: Option<String> = None;
        for event in receiver {
            let (token, logprob, top) = match event {
                Event::Token {
                    token,
                    logprob,
                    top_logprobs,
                } => (token, logprob, top_logprobs),
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
                let entry = logprob
                    .map(|logprob| stream_logprob_json(&served.tokenizer, delta, logprob, &top));
                let frame = chunk(
                    &id,
                    created,
                    &served.model,
                    Some(delta),
                    None,
                    entry.as_deref(),
                );
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
            let frame = chunk(&id, created, &served.model, None, Some("stop"), None);
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
                    None,
                ),
            );
        }
        let reason = finish_reason(&finished.finish);
        let final_frame = chunk(&id, created, &served.model, None, Some(reason), None);
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
                Event::Token { .. } => {}
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
        if json_mode && !is_json_object(text) {
            for (sequence, _) in turns.drain(..) {
                release_sequence(served, sequence);
            }
            respond_error(
                stream,
                422,
                "invalid_json_output",
                "model output is not a valid JSON object",
            )?;
            return Ok(422);
        }
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
pub(crate) fn completions(
    served: &Served,
    stream: &mut dyn Transport,
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
        // Text the vocabulary cannot encode is the caller's input, not a server fault.
        Err(error) => {
            let (status, code) = map_engine_error(&error);
            respond_error(stream, status, code, &error.to_string())?;
            return Ok(status);
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
    let sampler = match sampler_of(&parsed) {
        Ok(sampler) => sampler,
        Err(message) => {
            respond_error(stream, 400, "invalid_request", &message)?;
            return Ok(400);
        }
    };
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
    let json_mode = match json_mode_of(&parsed) {
        Ok(mode) => mode,
        Err(message) => {
            respond_error(stream, 400, "invalid_request", &message)?;
            return Ok(400);
        }
    };
    if json_mode && streaming {
        respond_error(
            stream,
            400,
            "invalid_request",
            "response_format json_object needs stream: false",
        )?;
        return Ok(400);
    }
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
            let (token, _, _) = match event {
                Event::Token {
                    token,
                    logprob,
                    top_logprobs,
                } => (token, logprob, top_logprobs),
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
                Event::Token { .. } => {}
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
        if json_mode && !is_json_object(text) {
            for (sequence, _) in turns.drain(..) {
                release_sequence(served, sequence);
            }
            respond_error(
                stream,
                422,
                "invalid_json_output",
                "model output is not a valid JSON object",
            )?;
            return Ok(422);
        }
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

/// Sampling knobs from the request. Unknown or out-of-range values fall back to the defaults;
/// malformed `logit_bias` is an error because silently dropping a ban changes what is generated.
/// Whether the request asks for JSON-only output.
///
/// Only `text` and `json_object` are understood; anything else is rejected rather than silently
/// treated as plain text.
fn json_mode_of(request: &Json) -> Result<bool, String> {
    let Some(format) = request.get("response_format") else {
        return Ok(false);
    };
    let kind = format.get("type").and_then(Json::as_str).unwrap_or("text");
    match kind {
        "text" => Ok(false),
        "json_object" => Ok(true),
        other => Err(format!("response_format type {other:?} is not supported")),
    }
}

/// A JSON-mode reply must be one complete object, not just any JSON value.
fn is_json_object(text: &str) -> bool {
    matches!(Json::parse(text), Ok(Json::Object(_)))
}

fn sampler_of(request: &Json) -> Result<SamplerConfig, String> {
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
    for (key, target) in [
        ("presence_penalty", &mut sampler.presence_penalty),
        ("frequency_penalty", &mut sampler.frequency_penalty),
    ] {
        if let Some(value) = request.get(key).and_then(Json::as_f64)
            && value.is_finite()
        {
            *target = value.clamp(-2.0, 2.0) as f32;
        }
    }
    if let Some(value) = request.get("logit_bias") {
        let Json::Object(entries) = value else {
            return Err("logit_bias must be an object mapping token ids to numbers".to_owned());
        };
        if entries.len() > MAX_LOGIT_BIAS {
            return Err(format!(
                "logit_bias accepts at most {MAX_LOGIT_BIAS} entries"
            ));
        }
        for (slot, (key, value)) in entries.iter().enumerate() {
            let id = key
                .parse::<u32>()
                .map_err(|_| format!("logit_bias key {key:?} is not a token id"))?;
            let bias = value
                .as_f64()
                .ok_or_else(|| format!("logit_bias for {key:?} must be a number"))?;
            if !bias.is_finite() {
                return Err(format!("logit_bias for {key:?} must be finite"));
            }
            sampler.logit_bias[slot] = Some((id, bias.clamp(-100.0, 100.0) as f32));
        }
    }
    Ok(sampler)
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
    for choice in 0..n {
        match attach_turn(
            served,
            ids,
            max_tokens,
            seed_for_choice(sampler, choice),
            logprobs,
            top_n,
        ) {
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

/// Give each of the `n` choices its own seed.
///
/// Every turn restarts its sampler from the same configuration, so sharing the seed would make
/// every choice draw the same tokens. The first choice keeps the requested seed, which keeps a
/// single-choice request reproducible.
fn seed_for_choice(mut sampler: SamplerConfig, choice: usize) -> SamplerConfig {
    sampler.seed = sampler.seed.wrapping_add(choice as u64);
    sampler
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
    let _admitting = served.router.admitting();
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

/// Keeps the router's admit count honest on every path out of [`attach_turn`].
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

pub(crate) fn map_engine_error(error: &ViperaError) -> (u16, &'static str) {
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
    logprob: Option<&str>,
) -> String {
    let delta = match content {
        Some(text) => format!(r#"{{"role":"assistant","content":{}}}"#, quote(text)),
        None => "{}".to_owned(),
    };
    let reason = match finish {
        Some(reason) => quote(reason),
        None => "null".to_owned(),
    };
    let logprobs = match logprob {
        Some(entry) => format!(r#"{{"content":[{entry}]}}"#),
        None => "null".to_owned(),
    };
    format!(
        r#"{{"id":{},"object":"chat.completion.chunk","created":{created},"model":{},"choices":[{{"index":0,"delta":{delta},"logprobs":{logprobs},"finish_reason":{reason}}}]}}"#,
        quote(id),
        quote(model)
    )
}

/// One streaming token's logprob entry, shaped like an entry of the non-streaming array.
fn stream_logprob_json(
    tokenizer: &Tokenizer,
    delta: &str,
    logprob: f32,
    top: &[(u32, f32)],
) -> String {
    let mut alternatives = String::new();
    for (id, probability) in top {
        let Ok(token_text) = tokenizer.decode(&[*id]) else {
            continue;
        };
        if !alternatives.is_empty() {
            alternatives.push(',');
        }
        alternatives.push_str(&format!(
            r#"{{"token":{},"logprob":{probability},"bytes":null}}"#,
            quote(&token_text)
        ));
    }
    format!(
        r#"{{"token":{},"logprob":{logprob},"bytes":null,"top_logprobs":[{alternatives}]}}"#,
        quote(delta)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_conversation_the_template_cannot_render_is_the_callers_error() {
        // The same mapping the chat handler applies to `encode_chat`, so a system-only request is
        // a 400 instead of a 500 that monitoring counts as the server's fault.
        for error in [
            ViperaError::invalid_argument("a chat request needs a user message"),
            ViperaError::unsupported("chat template is not one this build renders"),
        ] {
            assert_eq!(map_engine_error(&error), (400, "invalid_request"));
        }
    }

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
            r#"{"temperature":9,"top_p":2,"top_k":5,"min_p":-1,"seed":7,"repetition_penalty":1.2,"presence_penalty":5,"frequency_penalty":-5,"logit_bias":{"42":200,"7":-5}}"#,
        )
        .unwrap();
        let sampler = sampler_of(&request).expect("valid request");
        assert_eq!(sampler.logit_bias[0], Some((42, 100.0)));
        assert_eq!(sampler.logit_bias[1], Some((7, -5.0)));
        assert_eq!(sampler.logit_bias[2], None);
        assert_eq!(sampler.presence_penalty, 2.0);
        assert_eq!(sampler.frequency_penalty, -2.0);
        assert_eq!(sampler.temperature, 2.0);
        assert_eq!(sampler.top_p, 1.0);
        assert_eq!(sampler.top_k, 5);
        assert_eq!(sampler.min_p, 0.0);
        assert_eq!(sampler.seed, 7);
        assert!((sampler.repetition_penalty - 1.2).abs() < 1e-6);
    }

    #[test]
    fn json_mode_understands_text_and_object() {
        let object = Json::parse(r#"{"response_format":{"type":"json_object"}}"#).unwrap();
        assert!(json_mode_of(&object).unwrap());
        let text = Json::parse(r#"{"response_format":{"type":"text"}}"#).unwrap();
        assert!(!json_mode_of(&text).unwrap());
        assert!(!json_mode_of(&Json::parse("{}").unwrap()).unwrap());
        let schema = Json::parse(r#"{"response_format":{"type":"json_schema"}}"#).unwrap();
        assert!(json_mode_of(&schema).is_err());
    }

    #[test]
    fn json_validation_requires_one_complete_object() {
        assert!(is_json_object(r#"{"a":1}"#));
        assert!(!is_json_object("[1]"));
        assert!(!is_json_object(r#"{"a":1"#));
        assert!(!is_json_object("Hi"));
    }

    #[test]
    fn malformed_logit_bias_is_rejected() {
        for body in [
            r#"{"logit_bias":[1,2]}"#,
            r#"{"logit_bias":{"x":1}}"#,
            r#"{"logit_bias":{"7":"high"}}"#,
        ] {
            let request = Json::parse(body).unwrap();
            assert!(sampler_of(&request).is_err(), "accepted {body}");
        }
    }

    #[test]
    fn choices_get_distinct_seeds() {
        let base = SamplerConfig {
            seed: 7,
            ..SamplerConfig::default()
        };
        assert_eq!(seed_for_choice(base, 0).seed, 7);
        assert_eq!(seed_for_choice(base, 3).seed, 10);
        let last = SamplerConfig {
            seed: u64::MAX,
            ..base
        };
        assert_eq!(seed_for_choice(last, 1).seed, 0);
    }

    #[test]
    fn stream_frames_parse_as_json() {
        let frame = chunk("chatcmpl-1", 42, "qwen2", Some("hi \"there\""), None, None);
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
        let final_frame = chunk("chatcmpl-1", 42, "qwen2", None, Some("stop"), None);
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
}
