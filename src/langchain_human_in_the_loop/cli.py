from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from langchain_human_in_the_loop.tool import HumanInTheLoop

PROJECT_ID_ENV = "CODEVF_PROJECT_ID"
MAX_CREDIT_ENV = "CODEVF_MAX_CREDITS"
API_KEY_ENV = "CODEVF_API_KEY"
BASE_URL_ENV = "CODEVF_BASE_URL"


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.project_id is None:
        args.project_id = _read_int_env(PROJECT_ID_ENV, parser)
    if args.project_id is None:
        parser.error(
            f"Missing project_id configuration. Set --project-id or {PROJECT_ID_ENV}."
        )
    if args.max_credit is None:
        args.max_credit = _read_int_env(MAX_CREDIT_ENV, parser)
    if args.max_credit is None:
        parser.error(
            f"Missing max_credit configuration. Set --max-credit or {MAX_CREDIT_ENV}."
        )
    if args.max_credit <= 0:
        parser.error("--max-credit must be greater than 0.")
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be greater than 0.")

    hitl = HumanInTheLoop(
        project_id=args.project_id,
        max_credits=args.max_credit,
        mode=args.mode,
        poll_interval=args.poll_interval,
        timeout=args.timeout,
        tag_id=args.tag_id,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    try:
        result = hitl.invoke(args.prompt)
    except TimeoutError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="invoke",
        description="Invoke a CodeVF human-in-the-loop task and wait for completion.",
    )
    parser.add_argument("prompt", help="Prompt to send to CodeVF.")
    parser.add_argument(
        "--project-id",
        type=int,
        default=None,
        help=f"CodeVF project ID. Defaults to {PROJECT_ID_ENV}.",
    )
    parser.add_argument(
        "--max-credit",
        "--max-credits",
        dest="max_credit",
        type=int,
        default=None,
        help=f"Max credits for the task. Defaults to {MAX_CREDIT_ENV}.",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=_parse_timeout_value,
        default=None,
        help=(
            "Invoke timeout in seconds. Defaults to (2 * max_credit) + 300. "
            "Use -1 for infinite wait."
        ),
    )
    parser.add_argument(
        "--tag-id",
        type=int,
        default=None,
        help="Optional expertise tag ID from GET /tags.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=2.0,
        help="Polling interval in seconds while waiting for completion.",
    )
    parser.add_argument(
        "--mode",
        default="standard",
        help="CodeVF service mode (for example: standard, fast).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv(API_KEY_ENV),
        help=f"CodeVF API key. Defaults to {API_KEY_ENV}.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv(BASE_URL_ENV),
        help=f"CodeVF API base URL. Defaults to {BASE_URL_ENV}.",
    )
    return parser


def _parse_timeout_value(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "timeout must be a number of seconds or -1 for infinite wait."
        ) from exc

    if value == -1:
        return value
    if value <= 0:
        raise argparse.ArgumentTypeError(
            "timeout must be greater than 0, or -1 for infinite wait."
        )
    return value


def _read_int_env(name: str, parser: argparse.ArgumentParser) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        parser.error(f"{name} must be an integer if set.")
        return None


if __name__ == "__main__":
    raise SystemExit(main())
