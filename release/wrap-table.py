#!/usr/bin/env python3
"""Wrap a markdown table in <!--hidden-start--> / <!--hidden-end--> markers.

Usage:
    python release/wrap-table.py --readme path/to/README.md --header "Scenario | Prev"

The first table whose header line contains --header is wrapped. Idempotent: a table
already wrapped is left untouched. Useful before running a regen_*_perf_table.py
script (see samples/*/Python/).
"""
import argparse
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--readme', required=True)
    ap.add_argument('--header', required=True)
    args = ap.parse_args()
    p = Path(args.readme)
    text = p.read_text(encoding='utf-8')
    if '<!--hidden-start-->' in text:
        print('already wrapped; nothing to do')
        return
    lines = text.splitlines(keepends=True)
    start = None
    for i, ln in enumerate(lines):
        if args.header in ln:
            start = i
            break
    if start is None:
        raise SystemExit(f'header {args.header!r} not found')
    # walk down collecting table rows (lines starting with |)
    end = start
    while end < len(lines) and lines[end].lstrip().startswith('|'):
        end += 1
    block = ''.join(lines[start:end])
    wrapped = '<!--hidden-start-->\n' + block + '<!--hidden-end-->\n'
    new_text = ''.join(lines[:start]) + wrapped + ''.join(lines[end:])
    p.write_text(new_text, encoding='utf-8')
    print(f'wrapped lines {start+1}..{end} in {p}')


if __name__ == '__main__':
    main()
PY