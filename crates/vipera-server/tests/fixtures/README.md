# Throwaway TLS material for `tests/tls.rs`

Generated for the test, not for anything else. The key is committed on purpose: a test that
generates its own certificate would need a certificate-generation dependency, and this pair has no
value to anyone — it is a self-signed authority (`ca.crt`, `ca.key`) plus a leaf it signed for
`localhost` and `127.0.0.1` (`server.crt`, `server.key`), valid ten years from 2026-11-16.

- `ca.crt` — the trust anchor the test client loads.
- `chain.crt` — `server.crt` followed by `ca.crt`, the shape `--tls-cert` expects.
- `server.key` — PKCS#8 key for the leaf.

Regenerate with, from this directory:

```bash
openssl req -x509 -newkey rsa:2048 -nodes -days 3650 -keyout ca.key -out ca.crt \
  -subj "/CN=vipera test CA" \
  -addext "basicConstraints=critical,CA:TRUE" -addext "keyUsage=critical,keyCertSign,cRLSign"
openssl req -newkey rsa:2048 -nodes -keyout server.key -out server.csr -subj "/CN=localhost"
printf "subjectAltName=DNS:localhost,IP:127.0.0.1\nbasicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n" > ext.cnf
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -days 3650 \
  -out server.crt -extfile ext.cnf
cat server.crt ca.crt > chain.crt && rm server.csr ca.srl ext.cnf
```

A single self-signed certificate is not enough: webpki rejects a CA used directly as a server
certificate with `CaUsedAsEndEntity`, which is why this is an authority plus a leaf.
