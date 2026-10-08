"""``inferscale`` command line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from inferscale import __version__
from inferscale.config import ConfigError, ServerConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inferscale", description="InferScale: a small educational LLM inference server."
    )
    parser.add_argument("--version", action="version", version=f"inferscale {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser(
        "serve",
        help="start the HTTP server",
        description="Start the server. Precedence: CLI flags > INFERSCALE_* env vars > defaults.",
    )
    serve.add_argument("--host", help="bind address (default 127.0.0.1)")
    serve.add_argument("--port", type=int, help="bind port (default 8000)")
    serve.add_argument(
        "--model",
        help="Hugging Face id/path, or 'builtin:tiny' for an offline random tiny model "
        "(default Qwen/Qwen2.5-0.5B-Instruct)",
    )
    serve.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"])
    serve.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"])
    serve.add_argument("--max-batch-size", type=int, help="max sequences per batch (default 8)")
    serve.add_argument("--max-queue-size", type=int, help="max waiting requests (default 64)")
    serve.add_argument("--max-model-len", type=int, help="max prompt+completion tokens (2048)")
    serve.add_argument("--log-level", choices=["debug", "info", "warning", "error"])
    return parser


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from inferscale.api.app import create_app
    from inferscale.logging import configure_logging
    from inferscale.model.loader import ModelLoadError, load_runner

    cli_values = {
        k: getattr(args, k)
        for k in (
            "host",
            "port",
            "model",
            "device",
            "dtype",
            "max_batch_size",
            "max_queue_size",
            "max_model_len",
            "log_level",
        )
    }
    try:
        config = ServerConfig.resolve(cli_values)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    configure_logging(config.log_level)
    try:
        runner = load_runner(config)  # fail fast with a clean message, before binding the port
    except (ModelLoadError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    app = create_app(config, runner)
    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "serve":
        return _serve(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
