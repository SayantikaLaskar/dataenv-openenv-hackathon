"""OpenEnv validator-compatible server entry point."""

from __future__ import annotations

import uvicorn

from dataenv.server import app


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Run the ASGI server directly."""

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
