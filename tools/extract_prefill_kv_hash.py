# SPDX-License-Identifier: Apache-2.0
"""Extract compact KV hashes by full request ID or its UUID substring."""

from __future__ import annotations

# Standard
import argparse
from pathlib import Path
import re
import socket

_TAG = re.compile(r"\[KVH\]\s+p\d+\s+q\d+\b")


def extract(
    paths: list[Path],
    request_ids: list[str],
    save_request_ids: list[str] | None = None,
) -> int:
    """Print matching request blocks, stripping verbose worker log prefixes.

    Returns the number of matching records. Files are read twice as streams;
    source tensor data and unrelated requests are never loaded into memory.
    """
    count = 0
    print("HOST:", socket.gethostname())
    for path in paths:
        selected = {}
        with path.open(errors="replace") as source:
            for line in source:
                match = _TAG.search(line)
                if not match or " req=" not in line:
                    continue
                stages = set()
                if any(request in line for request in request_ids):
                    stages.update(("P", "D") if save_request_ids else ("S", "P", "D"))
                if any(request in line for request in save_request_ids or ()):
                    stages.add("S")
                if stages:
                    selected[match.group()] = (stages, line[match.start() :].rstrip())
        if not selected:
            continue
        printed = set()
        file_printed = False
        with path.open(errors="replace") as source:
            for line in source:
                match = _TAG.search(line)
                if match and match.group() in selected:
                    stages, header = selected[match.group()]
                    body = line[match.end() :].strip()
                    if not body or body.split()[0] not in stages:
                        continue
                    if not file_printed:
                        print("FILE:", path.name)
                        file_printed = True
                    if match.group() not in printed:
                        print(header)
                        printed.add(match.group())
                    print(line[match.start() :].rstrip())
                    count += 1
    return count


def main() -> int:
    """Extract only requested KVH blocks from one or more server log files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-id", action="append", default=[])
    parser.add_argument(
        "--save-request-id",
        action="append",
        default=[],
        help="Warmup request: print S only; targets print P/D only.",
    )
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    if not args.request_id and not args.save_request_id:
        parser.error("supply --request-id or --save-request-id")
    if not extract(args.logs, args.request_id, args.save_request_id):
        print("No matching KVH records; check request ID and hash switch.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
