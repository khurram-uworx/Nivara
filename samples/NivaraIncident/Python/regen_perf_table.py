#!/usr/bin/env python3
"""Regenerate a README benchmark results table from a captured JSON baseline.

Usage (from repository root):
    python samples/NivaraInference/Python/regen_perf_table.py \
        --json release\\vX.Y.Z\\nivara-perf-vX.Y.Z.json \
        --readme tests/Nivara.PerformanceTests/README.md

The target table must be bracketed by HTML comments:

    <!--hidden-start-->
    | Scenario | Prev | Current | Δ% | B/op | gen0/op |
    |---|---|---|---|---|---|
    | ... rows ...
    <!--hidden-end-->

The script keeps the header/separator lines, parses every data row between the markers,
looks the scenario name up in the JSON's "results" array (matching on a
backslash-normalized key so Markdown escapes do not break the lookup), shifts the old
"Current" value into "Prev", computes the fresh delta as (Current - Prev) / Prev, and
rewrites only the rows between the markers. Results present in the JSON but absent from
the table are appended with Prev = "—". Recording/machine lines are left for the human
to update deliberately.
"""
import argparse
import json
import re
from pathlib import Path


def normalize(name):
    return re.sub(r'\\([_*])', r'\1', name.strip('`').strip())


def shift_table(readme_text, json_data):
    start_token = '<!--hidden-start-->'
    end_token = '<!--hidden-end-->'
    start = readme_text.index(start_token) + len(start_token)
    end = readme_text.index(end_token)
    old_block = readme_text[start:end]
    old_lines = old_block.splitlines()
    header_lines = [ln for ln in old_lines
                    if 'Scenario | Prev' in ln or ln.strip().startswith('|---')]
    old_rows = [ln for ln in old_lines
                if ln.strip().startswith('|')
                and 'Scenario | Prev' not in ln
                and not ln.strip().startswith('|---')]
    new_rows = []
    used = set()
    by = {normalize(r['name']): r for r in json_data['results']}
    for ln in old_rows:
        fields = [f.strip() for f in ln.strip().strip('|').split('|')]
        if len(fields) < 6:
            new_rows.append(ln)
            continue
        name_cell = fields[0]
        old_curr = fields[2]
        r = by.get(normalize(name_cell))
        if not r:
            new_rows.append(ln)
            continue
        used.add(normalize(r['name']))
        try:
            old_curr_num = float(old_curr.replace(',', ''))
        except ValueError:
            old_curr_num = None
        new_prev = old_curr if old_curr_num is not None else fields[1]
        ops_per_sec = r['opsPerSec']
        new_curr = f"{ops_per_sec:.1f}" if abs(ops_per_sec) < 1 else f"{round(ops_per_sec):,}"
        if old_curr_num is not None and old_curr_num != 0:
            delta = (ops_per_sec - old_curr_num) / old_curr_num * 100
            delta_str = f"+{delta:.1f}%" if delta >= 0 else f"-{abs(delta):.1f}%"
        else:
            delta_str = '—'
        bop = f"{round(r['bytesPerOp']):,}"
        gen0 = f"{r['gen0PerOp']:.2f}"
        new_rows.append(f"| {name_cell} | {new_prev} | {new_curr} | {delta_str} | {bop} | {gen0} |")
    for r in json_data['results']:
        if normalize(r['name']) in used:
            continue
        ops_per_sec = r['opsPerSec']
        bop = f"{round(r['bytesPerOp']):,}"
        gen0 = f"{r['gen0PerOp']:.2f}"
        new_curr = f"{ops_per_sec:.1f}" if abs(ops_per_sec) < 1 else f"{round(ops_per_sec):,}"
        new_rows.append(f"| {r['name']} | — | {new_curr} | — | {bop} | {gen0} |")
    body = header_lines + new_rows
    new_block = '\n' + '\n'.join(body) + '\n'
    return readme_text[:start] + new_block + readme_text[end:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--readme', required=True)
    args = ap.parse_args()
    json_data = json.loads(Path(args.json).read_text(encoding='utf-8'))
    readme_path = Path(args.readme)
    text = readme_path.read_text(encoding='utf-8')
    Path(str(readme_path) + '.bak').write_text(text, encoding='utf-8')
    text = shift_table(text, json_data)
    readme_path.write_text(text, encoding='utf-8')
    print(f'Regenerated {args.readme}; backup at {args.readme}.bak')


if __name__ == '__main__':
    main()
