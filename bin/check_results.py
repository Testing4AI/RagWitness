#!/usr/bin/env python3
"""Recount headline results from frozen per-checkpoint and per-reduction records."""
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads((ROOT / path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def boolean(value):
    require(type(value) is bool, "Missing/unknown verdict is not a Boolean result")
    return value


def main():
    rows = load("evidence/matched/history-verdicts.json")
    require(len(rows) == 50 and len({r['id'] for r in rows}) == 50, "Matched inventory")
    names = {"RAGWitness": "state_flag", "Query evidence": "evidence_flag",
             "RAG-MT / DeepSeek": "antalchemy-deepseek-v4-pro", "RAG-MT / Kimi": "antalchemy-kimi-k3"}
    expected = [(39, 16, 0), (39, 16, 0), (28, 3, 6), (24, 3, 5)]
    comparison = {}
    for (name, field), target in zip(names.items(), expected):
        recalculated = []
        for row in rows:
            checkpoints = row["checkpoints"]
            require([c['position'] for c in checkpoints] == list(range(1, 33)), "Checkpoint inventory")
            flags = []
            for c in checkpoints:
                if field == "state_flag":
                    flag = bool(c["state_signatures"])
                    require(flag == boolean(c[field]), "State signature/count mismatch")
                elif field == "evidence_flag":
                    probes = c["probes"]
                    require(len(probes) == 4, "Probe inventory")
                    # Reconstruct the evidence relation, not merely its summary flag.
                    values = [p['actual_evidence'] != p['expected_evidence'] for p in probes]
                    require(values == [boolean(p[field]) for p in probes], "Evidence projection mismatch")
                    flag = any(values)
                    require(flag == boolean(c[field]), "Evidence aggregation mismatch")
                else:
                    values = [boolean(p['answers'][field]['alert']) for p in c['probes']]
                    require(len(values) == 4, "Answer probe inventory")
                    flag = any(values)
                    require(flag == boolean(c['answers'][field]), "Answer checkpoint aggregation")
                flags.append(flag)
            require(any(flags) == boolean(row[field]), "History aggregation")
            require(flags[-1] == boolean(row['terminal_' + field]), "Endpoint aggregation")
            require(boolean(row['seeded']) == (row['fault'] != 'clean'), "Condition labels")
            recalculated.append((row['seeded'], any(flags), flags[-1]))
        seeded = [r for r in recalculated if r[0]]
        clean = [r for r in recalculated if not r[0]]
        actual = (sum(r[1] for r in seeded), sum(r[2] for r in seeded), sum(r[1] for r in clean))
        require(len(seeded) == 40 and len(clean) == 10 and actual == target, name + " counts")
        comparison[name] = dict(zip(('checkpoint_detected', 'endpoint_detected', 'clean_alerts'), actual))

    dense = {}
    for backend in ('chroma-contriever', 'qdrant-contriever'):
        cohort = [json.loads(p.read_text()) for p in sorted((ROOT/'evidence/crossbackend/rows').glob(backend+'*.json'))]
        require(len(cohort) == 50, "Dense inventory")
        seeded = [r for r in cohort if r['fault'] != 'clean']
        clean = [r for r in cohort if r['fault'] == 'clean']
        for r in cohort:
            require(len(r['checkpoints']) == 32, "Dense checkpoints")
            require(r['detected'] == any(c['signatures'] for c in r['checkpoints']), "Dense ledger")
            require(r['terminal_detected'] == bool(r['checkpoints'][-1]['signatures']), "Dense endpoint")
        values = (sum(boolean(r['detected']) for r in seeded), sum(boolean(r['terminal_detected']) for r in seeded))
        require(len(seeded) == 40 and len(clean) == 10 and values == (39, 16), "Dense counts")
        require(not any(boolean(r['detected']) for r in clean), "Dense clean control")
        dense[backend] = {'checkpoint': values[0], 'endpoint': values[1], 'seeded': 40, 'clean': 10}

    continuous = load('evidence/continuous/results.json')['rows']
    require(len(continuous) == 420, 'Continuous inventory')
    histories = {}
    for strategy, expected_values in [('relation_guided', (180, 60)), ('operation_random', (178, 128))]:
        seeded = [r for r in continuous if r['strategy'] == strategy and r['fault'] != 'clean']
        clean = [r for r in continuous if r['strategy'] == strategy and r['fault'] == 'clean']
        values = (sum(boolean(r['detected']) for r in seeded), sum(boolean(r['terminal_detected']) for r in seeded))
        require(len(seeded) == 180 and len(clean) == 30 and values == expected_values, 'Continuous counts')
        require(not any(boolean(r['detected']) for r in clean), 'Continuous clean flags')
        histories[strategy] = {'checkpoint': values[0], 'endpoint': values[1]}

    with (ROOT/'evidence/reduction/paired-reductions.csv').open(newline='') as f:
        reductions = list(csv.DictReader(f))
    require(len(reductions) == 716, 'Reduction inventory')
    reduction_summary = {}
    paired = Counter(r['history_id'] for r in reductions)
    require(len(paired) == 358 and set(paired.values()) == {2}, 'Paired history inventory')
    for arm, expected_preserved in [('full', 358), ('no_signature', 54)]:
        group = [r for r in reductions if r['arm'] == arm]
        require(len(group) == 358, 'Arm inventory')
        require(all(r['strict_valid'] == r['strict_failure_reproduced'] == r['one_minimal_under_arm'] == 'True' for r in group), 'Witness validity')
        require(all(r['strict_target_preserved'] in ('True', 'False') for r in group), 'Unknown preservation result')
        preserved = sum(r['strict_target_preserved'] == 'True' for r in group)
        lengths = [int(r['minimized_events']) for r in group]
        require(preserved == expected_preserved and median(lengths) == 2 and min(lengths) == 1 and max(lengths) == 4, 'Reduction counts')
        reduction_summary[arm] = {'preserved': preserved, 'observed_failures': 358, 'median_events': median(lengths), 'range': [min(lengths), max(lengths)]}

    historical = load('evidence/historical/validation.json')
    require(historical['historical_clean_controls'] == {'n': 24, 'flagged': 0}, 'Historical controls')
    require(len(historical['historical_arms']) == 8, 'Historical inventory')
    arms = {}
    for r in historical['historical_arms']:
        expected_detected = 3 if r['variant'] == 'parent' or r['case'] == 'llama-l03' else 0
        require(r['n'] == 3 and r['detected'] == expected_detected, 'Historical outcome')
        arms[r['case'] + '/' + r['variant']] = f"{r['detected']}/{r['n']}"
    print(json.dumps({'valid': True, 'verification_type': 'saved-record recomputation, not new system executions',
                      'matched_comparison': comparison, 'crossbackend': dense, 'continuous': histories,
                      'reduction': reduction_summary, 'historical': arms, 'historical_clean_flags': '0/24'}, indent=2))


if __name__ == '__main__':
    main()
