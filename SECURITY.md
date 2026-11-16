# Security policy

## Reporting a vulnerability

Report privately through GitHub's *Report a vulnerability* (Security → Advisories) on this
repository. If that is unavailable, open an issue that says only that you have a security report
and how to reach you — no details in the issue itself.

Expect an acknowledgement within a week. This is a small project maintained in the open; there is
no bounty, and there is no private branch process beyond the advisory.

## What is in scope

`vipera serve` is the only component designed to face a network, and it carries nearly all of the
risk:

- request parsing and the JSON envelope for `/v1/chat/completions`, `/v1/completions` and the
  management endpoints
- bearer authentication, the opt-in per-key rate limiter, and the connection budget
- TLS termination (`--tls-cert`, `--tls-key`) and the ordering of shutdown
- anything that lets a request read memory or files outside the model it was given

The engine and the CLI are in scope for memory-safety defects. The backend contains `unsafe` in
exactly one file, `crates/vipera-backend-cpu/src/simd.rs`, and every block there carries a `//
SAFETY:` comment naming its invariant; a case where one of those invariants does not hold, or where
a read or write escapes its slice, is a vulnerability.

## Out of scope

- Model output that is wrong, offensive, or produced by a model file the attacker controls.
- Denial of service by requesting a large generation: rate limiting is opt-in by design, memory is
  planned before load, and a request that cannot fit is refused rather than served.
- Anything that requires local control of the machine, the model file, or the build.

## Supported versions

The project does not cut releases. `main` is the only supported version and the only place fixes
land; a report against an older commit is welcome, but the fix will be written against `main`.
