"""Launch the MAS demo UI: python -m demo_ui [--host HOST] [--port PORT]."""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    try:
        import gradio
    except ModuleNotFoundError as error:
        if error.name != "gradio":
            raise
        parser.exit(1, "The demo UI requires gradio. Add it to your environment, then run this command again.\n")

    from demo_ui.app import build_app

    build_app().queue().launch(
        server_name=args.host, server_port=args.port,
        share=False, inbrowser=False, ssr_mode=False,
    )


if __name__ == "__main__":
    main()
