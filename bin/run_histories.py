#!/usr/bin/env python3
"""Portable BM25 history replay using the study's unchanged implementation."""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from ragmetatest.supplemental_histories import (
    FAULTS, STRATEGIES, ViolationSignature, execute_history,
    generate_continuous_history, initial_documents, shrink_signature,
)
from ragmetatest.workload import confirmation_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=int, default=1)
    parser.add_argument('--seed', type=int, default=901)
    parser.add_argument('--reduce', action='store_true', help='Also shrink each detected history; takes longer')
    parser.add_argument('--output', type=Path, default=ROOT/'verification-output/histories.json')
    args = parser.parse_args()
    if not 1 <= args.cases <= 100:
        parser.error('--cases must be in 1..100')
    if args.output.exists():
        parser.error('Output already exists; choose a new filename to preserve the earlier run')
    rows = []
    for case in confirmation_cases()[:args.cases]:
        initial = initial_documents(case)
        for strategy in STRATEGIES:
            events = generate_continuous_history(case, strategy, args.seed)
            for fault in ('clean', *FAULTS):
                result = execute_history(initial, events, fault)
                row = {'case_id': case.case_id, 'strategy': strategy, 'seed': args.seed,
                       'fault': fault, 'trace': [asdict(e) for e in events], 'execution': result}
                if args.reduce and result['detected']:
                    first = next(c for c in result['checkpoints'] if c['signatures'])
                    signature = ViolationSignature(**first['signatures'][0])
                    row['reduction'] = shrink_signature(initial, events, fault, signature)
                    if not row['reduction']['signature_preserved'] or not row['reduction']['one_minimal']:
                        raise RuntimeError('Witness validation failed')
                if fault == 'clean' and result['detected']:
                    raise RuntimeError('Unexpected clean-history flag')
                rows.append(row)
    result = {'run_type': 'fresh portable BM25 execution; not a replacement for the frozen study', 'rows': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'valid': True, 'histories': len(rows), 'events_per_history': 32,
                      'seeded': sum(r['fault'] != 'clean' for r in rows),
                      'flagged': sum(r['execution']['detected'] for r in rows),
                      'reduced': sum('reduction' in r for r in rows), 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
