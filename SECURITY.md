# Security

InferScale is an educational/experimental project. It has **no authentication, no rate
limiting per client, and no TLS**; do not expose it to untrusted networks. By default it binds
to `127.0.0.1`.

Models loaded from Hugging Face run with the privileges of the server process; only load
models you trust. `trust_remote_code` is never enabled.

To report a vulnerability, please open a private security advisory on the GitHub repository
(Security → Report a vulnerability) rather than a public issue.
