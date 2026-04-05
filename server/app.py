"""OpenEnv-compatible server entrypoint."""

from __future__ import annotations

import argparse

import uvicorn

from api.server import app


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if args.host == "0.0.0.0" and args.port == 7860:
        main()
    else:
        main(host=args.host, port=args.port)
