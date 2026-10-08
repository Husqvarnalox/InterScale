# Security

InferScale is experimental software.

- No authentication, per-client rate limiting or TLS termination. Put it behind a reverse proxy
  if it must be reachable beyond localhost; it binds to `127.0.0.1` by default.
- Downloading and running a model executes code and weights from the model source.
  `trust_remote_code` is never enabled, but only load models you trust.
- It is not designed for hostile or multi-tenant environments. Any client can occupy the queue
  and batch slots.
- Prompts and generated text are not logged.

To report a vulnerability, use GitHub's private vulnerability reporting for this repository
(Security → Report a vulnerability) instead of a public issue.
