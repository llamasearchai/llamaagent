#!/usr/bin/env python3
"""
FastAPI Server Entry Point for LlamaAgent.

Author: Nik Jois <nikjois@llamasearch.ai>

This module provides the main entry point for running the FastAPI server.
"""

import uvicorn


def main() -> None:
    """Main entry point for the FastAPI server."""
    uvicorn.run(
        "llamaagent.api:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
