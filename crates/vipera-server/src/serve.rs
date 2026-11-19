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
//! The surface is split by job: [`crate::http`] reads a request and writes a response, [`crate::tls`]
//! builds and reloads the TLS configuration, [`crate::handler`] turns a JSON body into a turn and a
//! turn into frames, and this module owns the process — the flags, the model load, the scheduler, the
//! connection loop, auth and metrics.
//!
//! What Phase 1 does not do, stated here rather than implied: request bodies must be `Content-Length`
//! framed (chunked bodies are refused with `invalid_request`), and a client that disconnects
//! mid-stream still lets the current turn finish before the connection is dropped. Requests carry a per-key (or, without a key, per peer) budget of 600 a
//! minute by default (`429` + `Retry-After` when exceeded); `--rate-limit N` changes the budget and
//! `--rate-limit 0` turns it off. Without a key every `authorization` value is caller-chosen, so it
//! cannot name a client. At most [`MAX_CONNECTIONS`] connections are served at once; past that the
//! server answers `503` instead of spawning another thread.

use std::collections::{BTreeMap, HashMap};
use std::io::Write as _;
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use vipera_backend_cpu::CpuBackend;
use vipera_core::gguf::GgufFile;
use vipera_core::planner::KvCacheType;
use vipera_core::window::attention_window;
use vipera_core::{PlacementPlan, PlanRequest, SystemProbe, plan};
use vipera_errors::ViperaError;
use vipera_runtime::{
    DEFAULT_TOKENS_PER_BLOCK, Engine, EngineConfig, Generation, KvCache, KvConfig, StepReport,
};
use vipera_sampler::SamplerConfig;
use vipera_tokenizer::Tokenizer;
use vipera_traits::HardwareProbe;

use crate::handler::{chat, completions};
use crate::http::{
    Incoming, Request, Transport, read_request, respond, respond_error, respond_error_retry,
    respond_preflight,
};
use crate::json::quote;
use crate::tls::TlsSource;

pub use crate::tls::tls_config;

/// Connections that may be open at once. Each one owns a thread and up to [`MAX_BODY_BYTES`], so this
/// is a memory bound rather than a concurrency target.
const MAX_CONNECTIONS: usize = 64;

/// Bytes kept free beyond the model file when deciding whether it can be read into memory at all.
const LOAD_RESERVE_BYTES: u64 = 256 * 1024 * 1024;

/// Clients the rate limiter tracks at once. Beyond this the map is pruned, then its oldest bucket is
/// dropped: an unbounded map is a memory leak with a network trigger.
const MAX_BUCKETS: usize = 4096;

/// OpenAI accepts at most four stop sequences per request.
pub(crate) const MAX_STOP_SEQUENCES: usize = 4;

/// The engine batches at most four turns, so that is the ceiling for `n`.
pub(crate) const MAX_CHOICES: usize = 4;

/// Requests per minute per key (or, without a key, per peer address) unless `--rate-limit` says
/// otherwise. One conversation streams well under this; the budget is there so a runaway client
/// cannot pin the machine, and `--rate-limit 0` turns it off for a trusted box.
const DEFAULT_RATE_LIMIT: u32 = 600;

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
    let mut rate_limit: Option<u32> = Some(DEFAULT_RATE_LIMIT);
    let mut tls_cert: Option<&str> = None;
    let mut tls_key: Option<&str> = None;
    #[cfg(feature = "acme")]
    let mut acme_domain: Option<String> = None;
    #[cfg(feature = "acme")]
    let mut acme_contact: Option<String> = None;
    #[cfg(feature = "acme")]
    let mut acme_directory = "https://acme-v02.api.letsencrypt.org/directory".to_owned();
    #[cfg(feature = "acme")]
    let mut acme_http_address = "0.0.0.0:80".to_owned();

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
            "--tls-cert" => {
                tls_cert = Some(next_value(args, index, "--tls-cert")?);
                index += 1;
            }
            "--tls-key" => {
                tls_key = Some(next_value(args, index, "--tls-key")?);
                index += 1;
            }
            "--rate-limit" => {
                // Zero is the documented way to say "no budget"; the default is `DEFAULT_RATE_LIMIT`.
                rate_limit = budget_from_flag(next_number(args, index, "--rate-limit")?);
                index += 1;
            }
            #[cfg(feature = "acme")]
            "--acme-domain" => {
                acme_domain = Some(next_value(args, index, "--acme-domain")?.to_owned());
                index += 1;
            }
            #[cfg(feature = "acme")]
            "--acme-contact" => {
                acme_contact = Some(next_value(args, index, "--acme-contact")?.to_owned());
                index += 1;
            }
            #[cfg(feature = "acme")]
            "--acme-directory" => {
                acme_directory = next_value(args, index, "--acme-directory")?.to_owned();
                index += 1;
            }
            #[cfg(feature = "acme")]
            "--acme-http-address" => {
                acme_http_address = next_value(args, index, "--acme-http-address")?.to_owned();
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

    // `--acme-domain` turns the `--tls-cert`/`--tls-key` pair into files this process writes and
    // then keeps current; the challenge listener has to come up before TLS can, and on a first
    // issuance there is nothing to serve until the CA has answered.
    #[cfg(feature = "acme")]
    if let Some(domain) = acme_domain {
        let (Some(cert), Some(key)) = (tls_cert, tls_key) else {
            return Err(ViperaError::usage(
                "--acme-domain needs --tls-cert and --tls-key: they name the files the certificate \
                 is written to",
            ));
        };
        crate::acme::spawn(
            crate::acme::Options {
                domain,
                contact: acme_contact,
                directory: acme_directory,
                http_address: acme_http_address,
                // `vipera serve` talks to a CA the platform already trusts; the test suite is the
                // only caller that names another root.
                root_cert: None,
            },
            std::path::Path::new(cert),
            std::path::Path::new(key),
        )?;
        if !std::path::Path::new(cert).is_file() || !std::path::Path::new(key).is_file() {
            crate::acme::wait_for_certificate(
                std::path::Path::new(cert),
                std::path::Path::new(key),
                std::time::Duration::from_secs(120),
            )?;
        }
    }

    let tls = match (tls_cert, tls_key) {
        (None, None) => None,
        (Some(_), None) => {
            return Err(ViperaError::usage("--tls-cert needs --tls-key"));
        }
        (None, Some(_)) => {
            return Err(ViperaError::usage("--tls-key needs --tls-cert"));
        }
        (Some(cert), Some(key)) => {
            let source = TlsSource::new(cert, key).map_err(|error| {
                ViperaError::invalid_argument(format!("TLS is not usable: {error}"))
            })?;
            Some(Arc::new(source))
        }
    };

    let served = Arc::new(Served::load(
        path,
        model_id,
        api_key.clone(),
        ctx,
        kv_type,
        rate_limit,
        tls,
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
        let scheme = if served.tls.is_some() {
            "https"
        } else {
            "http"
        };
        let _ = writeln!(
            out,
            "vipera: serving {} as \"{}\" on {scheme}://{address}",
            path, served.model
        );
        let _ = out.flush();
    }

    let connections = Arc::new(ConnectionBudget::new(MAX_CONNECTIONS));
    for incoming in listener.incoming() {
        match incoming {
            Ok(mut stream) => {
                let Some(slot) = connections.try_acquire() else {
                    // Refuse instead of spawning a thread the box cannot back with memory.
                    let _ = respond_error(
                        &mut stream,
                        503,
                        "overloaded",
                        "too many concurrent connections",
                    );
                    continue;
                };
                // A renewed certificate is a file write, not a restart: pick it up here, where
                // the socket is still ours and no request is in flight.
                if let Some(tls) = &served.tls
                    && tls.refresh()
                {
                    eprintln!("vipera: reloaded the TLS certificate");
                }
                // Small replies and streamed tokens: Nagle would hold them back.
                let _ = stream.set_nodelay(true);
                let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(10)));
                let _ = stream.set_write_timeout(Some(std::time::Duration::from_secs(30)));
                let served = Arc::clone(&served);
                match std::thread::Builder::new().spawn(move || {
                    // The slot lives as long as the connection's thread and frees itself on exit.
                    let _slot = slot;
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

/// The budget a `--rate-limit` value asks for: zero turns the budget off, anything else sets it.
fn budget_from_flag(limit: u32) -> Option<u32> {
    (limit > 0).then_some(limit)
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

/// Refuse a model file that cannot be held in RAM, before the read allocates it.
///
/// The planner can only answer after the header is parsed, and by then the bytes are already
/// resident. This check runs on the file length alone, so an oversized model is a usage error
/// instead of an OOM kill.
fn check_file_fits(file_bytes: u64, ram_total_bytes: u64) -> Result<(), ViperaError> {
    if file_bytes.saturating_add(LOAD_RESERVE_BYTES) > ram_total_bytes {
        return Err(ViperaError::infeasible(format!(
            "the model file is {file_bytes} bytes and this machine has {ram_total_bytes} bytes of RAM; \
             it cannot be read into memory"
        )));
    }
    Ok(())
}

/// Refuse a placement the planner calls infeasible, quoting the reasons it gave.
fn check_placement(placement: &PlacementPlan) -> Result<(), ViperaError> {
    if placement.feasible {
        return Ok(());
    }
    let reasons: Vec<&str> = placement
        .violations
        .iter()
        .map(|violation| violation.message.as_str())
        .collect();
    Err(ViperaError::infeasible(reasons.join("; ")))
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

/// What a rate limit counts against.
///
/// With a configured key, the key is the identity: it is the only value authentication accepts. With
/// no key, the caller chooses the `authorization` value freely, so the peer address is the identity —
/// keying on an unauthenticated bearer would hand out a new bucket per request and grow the map
/// without bound.
fn rate_limit_client(api_key: Option<&str>, request: &Request, peer: &str) -> String {
    if api_key.is_some()
        && let Some(key) = request
            .header("authorization")
            .and_then(|value| value.strip_prefix("Bearer "))
    {
        let key = key.trim();
        if !key.is_empty() {
            return key.to_owned();
        }
    }
    peer.to_owned()
}

/// Counts open connections and refuses past its limit.
///
/// Every accepted socket owns a thread and can buffer [`MAX_BODY_BYTES`] of request, so an unbounded
/// accept loop is an unbounded allocation with a network trigger.
pub(crate) struct ConnectionBudget {
    limit: usize,
    active: AtomicUsize,
}

impl ConnectionBudget {
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            limit,
            active: AtomicUsize::new(0),
        }
    }

    /// Take a slot, or `None` when every slot is busy. The slot returns its count when dropped.
    pub(crate) fn try_acquire(self: &Arc<Self>) -> Option<ConnectionSlot> {
        let mut active = self.active.load(Ordering::Relaxed);
        loop {
            if active >= self.limit {
                return None;
            }
            match self.active.compare_exchange_weak(
                active,
                active + 1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => {
                    return Some(ConnectionSlot {
                        budget: Arc::clone(self),
                    });
                }
                Err(current) => active = current,
            }
        }
    }
}

/// One connection's slot in a [`ConnectionBudget`].
pub(crate) struct ConnectionSlot {
    budget: Arc<ConnectionBudget>,
}

impl Drop for ConnectionSlot {
    fn drop(&mut self) {
        self.budget.active.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Token bucket per client, shared across handler threads.
pub(crate) struct RateLimiter {
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
        if !buckets.contains_key(client) && buckets.len() >= MAX_BUCKETS {
            // Drop buckets that have already refilled; if the map is still full every client is
            // active, so evict the least recently seen one.
            buckets.retain(|_, (_, last)| {
                now.saturating_duration_since(*last) < Duration::from_secs(60)
            });
            if buckets.len() >= MAX_BUCKETS
                && let Some(oldest) = buckets
                    .iter()
                    .min_by_key(|(_, (_, last))| *last)
                    .map(|(key, _)| key.clone())
            {
                buckets.remove(&oldest);
            }
        }
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
pub(crate) struct Served {
    pub(crate) engine: Arc<Mutex<Engine<CpuBackend>>>,
    pub(crate) router: Arc<Router>,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) model: String,
    pub(crate) context_length: usize,
    pub(crate) api_key: Option<String>,
    pub(crate) rate_limit: Option<Arc<RateLimiter>>,
    pub(crate) metrics: Mutex<Metrics>,
    /// `None` serves plain HTTP. `Some` completes a TLS handshake before reading the request, with
    /// the certificate and key `--tls-cert` / `--tls-key` named, re-read when they change.
    tls: Option<Arc<TlsSource>>,
}

impl Served {
    fn load(
        path: &str,
        model: String,
        api_key: Option<String>,
        ctx: Option<u64>,
        kv_type: KvCacheType,
        rate_limit: Option<u32>,
        tls: Option<Arc<TlsSource>>,
    ) -> Result<Served, ViperaError> {
        let path = resolve_model_path(path)?;
        // This build has no device backend, so plan the placement this process will actually run:
        // the same profile `verify --cpu` uses. Planning after the read would be too late for a file
        // the machine cannot hold, so the length is checked first.
        let mut hardware = SystemProbe::new().profile();
        hardware.devices.clear();
        let file_bytes = std::fs::metadata(&path)?.len();
        check_file_fits(file_bytes, hardware.ram_total_bytes)?;

        let bytes = std::fs::read(&path)?;
        let file = GgufFile::parse(&bytes)?;
        let limit = file.describe().context_limit.unwrap_or(2048);
        let context_length = ctx.unwrap_or(limit.min(2048));
        // The window makes the shapes past it unreachable, so the plan and the cache both pay for
        // the sinks and the window instead of the whole context. One ring per sequence, because up
        // to `max_batch` sequences are live at once.
        let window = attention_window();
        let sequences = 4;
        let placement = plan(
            &file,
            &hardware,
            &PlanRequest {
                ctx: context_length,
                kv_type,
                batch_size: if window.is_some() { sequences } else { 1 },
                window,
            },
        )?;
        check_placement(&placement)?;

        let tokenizer = Tokenizer::from_gguf(&file)?;
        let backend = CpuBackend::from_bytes(bytes, &file)?;
        let config = backend.config();

        let bytes_per_token = config.kv_bytes_per_token_for(kv_type).ok_or_else(|| {
            ViperaError::unsupported(format!(
                "this model cannot store its KV cache as {}",
                kv_type.as_str()
            ))
        })?;
        let kv = KvCache::new(KvConfig::for_run(
            bytes_per_token,
            DEFAULT_TOKENS_PER_BLOCK,
            context_length,
            window.map(|window| (window.tokens, window.sinks)),
            sequences,
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
                // A turn only speculates when it is the only one decoding; with company the batch
                // keeps the weight read shared.
                speculative: true,
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
            tls,
        })
    }
}

/// What a handler receives from the scheduler: one token, the end of its turn, or a failure.
pub(crate) enum Event {
    Token {
        token: u32,
        logprob: Option<f32>,
        top_logprobs: Vec<(u32, f32)>,
    },
    Finished(Box<Generation>),
    Failed(String),
}

/// Hand-off between request handlers and the one thread that steps the engine.
///
/// The engine is not `Sync` (it owns the weights and the KV cache), so exactly one thread touches it.
/// A handler takes the engine lock only to attach its sampler and queue its turn, then waits on its
/// own channel; the scheduler batches whatever is queued into one forward pass and posts each token
/// to the handler that owns it.
pub(crate) struct Router {
    routes: Mutex<HashMap<u64, mpsc::Sender<Event>>>,
    /// Whether a turn was queued since the scheduler last looked.
    pending: Mutex<bool>,
    wake: Condvar,
    /// Handlers waiting for the engine lock to queue a new turn. The scheduler yields the engine
    /// while this is above zero: `std::sync::Mutex` is unfair, and a scheduler that reacquires it
    /// at once can keep a new request out for a whole prompt.
    pub(crate) admitting: AtomicUsize,
}

impl Router {
    fn new() -> Router {
        Router {
            routes: Mutex::new(HashMap::new()),
            pending: Mutex::new(false),
            wake: Condvar::new(),
            admitting: AtomicUsize::new(0),
        }
    }

    /// Note that a handler is about to queue a turn, and undo it when done.
    pub(crate) fn admitting(&self) -> AdmitGuard<'_> {
        self.admitting.fetch_add(1, Ordering::SeqCst);
        AdmitGuard { router: self }
    }

    /// Whether a handler is waiting to queue a turn right now.
    fn someone_is_admitting(&self) -> bool {
        self.admitting.load(Ordering::SeqCst) > 0
    }

    /// Point `sequence`'s tokens at `sender`. Called before the turn is queued, so a step can never
    /// draw a token for a sequence nobody is listening to.
    pub(crate) fn register(&self, sequence: u64, sender: mpsc::Sender<Event>) {
        if let Ok(mut routes) = self.routes.lock() {
            routes.insert(sequence, sender);
        }
    }

    /// Drop a route: the handler already has its generation, or the client is gone.
    pub(crate) fn unregister(&self, sequence: u64) {
        if let Ok(mut routes) = self.routes.lock() {
            routes.remove(&sequence);
        }
    }

    /// Tell the scheduler that a turn was queued.
    pub(crate) fn wake(&self) {
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
        for step in report.tokens {
            if eos.contains(&step.token) {
                continue;
            }
            if let Some(sender) = routes.get(&step.sequence) {
                let _ = sender.send(Event::Token {
                    token: step.token,
                    logprob: step.logprob,
                    top_logprobs: step.top_logprobs,
                });
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

/// Keeps the router's "a handler is waiting to queue" count honest while one does.
pub(crate) struct AdmitGuard<'a> {
    router: &'a Router,
}

impl Drop for AdmitGuard<'_> {
    fn drop(&mut self) {
        self.router.admitting.fetch_sub(1, Ordering::SeqCst);
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
            // The lock is free again, but a handler that wants to queue a turn may be waiting on it.
            // Pause for a moment so it can attach: a prefill chunk takes hundreds of milliseconds,
            // and a starving admit would wait out the whole prompt instead of one chunk.
            if router.someone_is_admitting() {
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
        }
    }
}

fn handle(served: &Served, mut stream: TcpStream) -> std::io::Result<()> {
    let peer = stream
        .peer_addr()
        .map(|addr| addr.ip().to_string())
        .unwrap_or_else(|_| "unknown".to_owned());
    if let Some(source) = &served.tls {
        let connection = rustls::ServerConnection::new(source.config())
            .map_err(|error| std::io::Error::other(format!("TLS: {error}")))?;
        let mut tls = rustls::StreamOwned::new(connection, stream);
        let result = serve_connection(served, &mut tls, &peer);
        // `Connection: close` is the contract here, and over TLS that means close_notify before
        // the socket goes: a client that reads to the end treats a bare EOF as a truncated
        // response, not as the end of one. curl tolerates it; rustls' own client does not.
        tls.conn.send_close_notify();
        let _ = tls.flush();
        return result;
    }
    serve_connection(served, &mut stream, &peer)
}

fn serve_connection(
    served: &Served,
    stream: &mut dyn Transport,
    peer: &str,
) -> std::io::Result<()> {
    let started = Instant::now();
    let (method, path, status) = match read_request(stream)? {
        Incoming::Closed => return Ok(()),
        // A refusal is a response too. Without this arm a body that was too large, or a header
        // block that never ended, would be answered and then leave no trace in the log or on
        // `/metrics`, which is exactly what an operator needs when the server is being flooded.
        Incoming::Rejected { status } => (String::new(), String::new(), status),
        Incoming::Request(request) => {
            let status = route(served, stream, &request, started, peer)?;
            (request.method, request.path, status)
        }
    };
    let endpoint = endpoint_name(&path);
    if let Ok(mut metrics) = served.metrics.lock() {
        metrics.record(endpoint, status, started.elapsed().as_secs_f64());
    }
    // One line per request, in the same JSON-event shape `vipera generate` writes, so a log
    // collector reads both the same way. `VIPERA_ACCESS_LOG=0` silences it.
    if access_log_enabled() {
        eprintln!(
            "{{\"event\":\"request\",\"peer\":{},\"method\":{},\"path\":{},\"status\":{status},\"micros\":{}}}",
            quote(peer),
            quote(&method),
            quote(&path),
            started.elapsed().as_micros()
        );
    }
    Ok(())
}

/// Whether each request writes one line to stderr. On by default; `0`, `false`, `off` or `no`
/// silences it for a host that would rather have nothing between it and its log files.
fn access_log_enabled() -> bool {
    access_log_enabled_for(std::env::var("VIPERA_ACCESS_LOG").ok().as_deref())
}

/// [`access_log_enabled`] against a given value, so the rule can be tested without the process
/// environment.
fn access_log_enabled_for(value: Option<&str>) -> bool {
    value.is_none_or(|value| {
        let value = value.trim().to_ascii_lowercase();
        !matches!(value.as_str(), "0" | "false" | "off" | "no")
    })
}

fn route(
    served: &Served,
    stream: &mut dyn Transport,
    request: &Request,
    started: Instant,
    peer: &str,
) -> std::io::Result<u16> {
    // A preflight carries no credentials, so it has to be answered before authentication. It also
    // must not consume the rate limit: one user action would otherwise spend two requests.
    if request.method == "OPTIONS" {
        respond_preflight(stream)?;
        return Ok(204);
    }
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
        let client = rate_limit_client(served.api_key.as_deref(), request, peer);
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

#[derive(Default)]
pub(crate) struct Metrics {
    requests: BTreeMap<(String, u16), u64>,
    tokens: u64,
    duration: Histogram,
    first_token: Histogram,
}

impl Metrics {
    pub(crate) fn record(&mut self, endpoint: &str, status: u16, seconds: f64) {
        *self
            .requests
            .entry((endpoint.to_owned(), status))
            .or_insert(0) += 1;
        self.duration.observe(seconds);
    }

    pub(crate) fn time_to_first_token(&mut self, seconds: f64) {
        self.first_token.observe(seconds);
    }

    pub(crate) fn tokens_generated(&mut self, count: u64) {
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

    fn request_with_bearer(value: Option<&str>) -> Request {
        Request {
            method: "POST".to_owned(),
            path: "/v1/chat/completions".to_owned(),
            headers: value
                .map(|value| vec![("authorization".to_owned(), value.to_owned())])
                .unwrap_or_default(),
            body: Vec::new(),
        }
    }

    #[test]
    fn an_unauthenticated_bearer_cannot_choose_its_rate_limit_bucket() {
        // No --api-key: every request is the same peer, and a bearer string it made up must not
        // buy a fresh bucket (which is both a bypass and unbounded memory).
        let request = request_with_bearer(Some("Bearer make-believe"));
        assert_eq!(rate_limit_client(None, &request, "127.0.0.1"), "127.0.0.1");
    }

    #[test]
    fn an_authenticated_key_is_the_rate_limit_identity() {
        let request = request_with_bearer(Some("Bearer sk-local"));
        assert_eq!(
            rate_limit_client(Some("sk-local"), &request, "127.0.0.1"),
            "sk-local"
        );
    }

    #[test]
    fn the_rate_limiter_keeps_its_bucket_map_bounded() {
        let limiter = RateLimiter::new(60);
        let start = Instant::now();
        for index in 0..(MAX_BUCKETS + 16) {
            let client = format!("peer-{index}");
            assert!(limiter.allow_at(&client, start).is_ok());
        }
        let buckets = limiter.buckets.lock().expect("bucket map").len();
        assert!(
            buckets <= MAX_BUCKETS,
            "{buckets} buckets is over the {MAX_BUCKETS} cap"
        );
        assert!(
            limiter.allow_at("the-newest-client", start).is_ok(),
            "a new client still gets its own bucket"
        );
    }

    #[test]
    fn a_file_larger_than_ram_is_refused_before_it_is_read() {
        let error = check_file_fits(15 << 30, 15 << 30).expect_err("15 GB does not fit in 15 GB");
        assert!(error.to_string().contains("RAM"), "{error}");
        assert!(check_file_fits(1 << 30, 15 << 30).is_ok());
    }

    #[test]
    fn an_infeasible_placement_is_refused_with_the_planner_reason() {
        let mut hardware = SystemProbe::new().profile();
        hardware.devices.clear();
        hardware.ram_total_bytes = 1 << 30;
        let shape = vipera_core::ModelShape {
            n_layer: 28,
            n_embd: 1536,
            n_head: 12,
            n_head_kv: 2,
            max_context: Some(4096),
            weights_bytes: 4 << 30,
            non_layer_bytes: 0,
        };
        let placement = vipera_core::plan_shape("test-model", &hardware, &shape, &plan_request())
            .expect("plan");
        assert!(!placement.feasible, "4 GB of weights cannot fit in 1 GB");
        let error = check_placement(&placement).expect_err("infeasible placement must be refused");
        assert!(error.to_string().contains("host memory"), "{error}");
    }

    fn plan_request() -> PlanRequest {
        PlanRequest {
            ctx: 2048,
            kv_type: KvCacheType::F16,
            batch_size: 1,
            window: None,
        }
    }

    #[test]
    fn a_connection_budget_refuses_once_it_is_full_and_frees_on_drop() {
        let budget = Arc::new(ConnectionBudget::new(2));
        let first = budget.try_acquire().expect("first connection");
        let second = budget.try_acquire().expect("second connection");
        assert!(budget.try_acquire().is_none(), "the third must be refused");
        drop(first);
        assert!(
            budget.try_acquire().is_some(),
            "a closed connection frees its slot"
        );
        drop(second);
    }

    #[test]
    fn the_access_log_is_on_unless_it_is_turned_off() {
        assert!(access_log_enabled_for(None), "unset means on");
        for off in ["0", "false", "OFF", "no", " 0 "] {
            assert!(!access_log_enabled_for(Some(off)), "{off:?} turns it off");
        }
        for on in ["1", "true", "yes", ""] {
            assert!(access_log_enabled_for(Some(on)), "{on:?} leaves it on");
        }
    }

    #[test]
    fn the_default_budget_is_six_hundred_requests_a_minute() {
        assert_eq!(DEFAULT_RATE_LIMIT, 600);
    }

    #[test]
    fn a_zero_flag_turns_the_budget_off_and_a_positive_one_sets_it() {
        assert_eq!(
            budget_from_flag(0),
            None,
            "zero is the documented off switch"
        );
        assert_eq!(budget_from_flag(5), Some(5));
        assert_eq!(budget_from_flag(600), Some(600));
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
    fn histograms_expose_the_prometheus_shape() {
        let mut histogram = Histogram::default();
        histogram.observe(0.2);
        let mut out = String::new();
        histogram.render("vipera_request_duration_seconds", &mut out);
        assert!(out.contains("vipera_request_duration_seconds_bucket{le=\"0.25\"} 1"));
        assert!(out.contains("vipera_request_duration_seconds_bucket{le=\"+Inf\"} 1"));
        assert!(out.contains("vipera_request_duration_seconds_count 1"));
    }
}
