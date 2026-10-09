"""Read-only matched-trial audit of downloaded native tau conversation records.

Run from the repository root with --base and --opd pointing to directories
created BEFORE `modal volume get ... trial0 DIRECTORY/`.
This probes selection coverage, not the causal correctness of any action.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from configs.tau_bench_eval.harness import paired_delta


def read_records(root):
    records, receipts = {}, []
    for path in sorted(root.rglob('*.json')):
        raw = path.read_bytes()
        row = json.loads(raw)
        key = (row['task_id'], row['trial'])
        if key in records:
            raise ValueError(f'Duplicate task/trial {key}')
        if row['domain'] != 'tau2_airline' or row['trial'] != 0:
            raise ValueError('This preselected probe expects tau2_airline trial0 only')
        if not isinstance(row['success'], bool) or row['success'] != (row['reward'] == 1.0):
            raise ValueError('Inconsistent success label')
        records[key] = row
        receipts.append({'file': str(path.relative_to(root)), 'sha256': hashlib.sha256(raw).hexdigest()})
    if len(records) != 50:
        raise ValueError(f'Expected all 50 Airline tasks, got {len(records)}')
    return records, receipts


def features(row):
    msgs = (row.get('result') or {}).get('messages', [])
    tools = [m for m in msgs if m['role'] == 'tool' and m.get('requestor') == 'assistant']
    errors = [m for m in tools if m.get('error') is True]
    assistants = [m for m in msgs if m['role'] == 'assistant']
    post_tool = [m for i, m in enumerate(msgs) if m['role'] == 'assistant' and i and msgs[i - 1]['role'] == 'tool']
    calls = [c for m in assistants for c in (m.get('tool_calls') or [])]
    signatures = [json.dumps([c['name'], c['arguments']], sort_keys=True) for c in calls]
    # A count diagnostic: repeated arguments can be legitimate, so no error label.
    repeated = sum(n - 1 for n in Counter(signatures).values() if n > 1)
    info = (row.get('result') or {}).get('reward_info') or {}
    return {
        'task_id': row['task_id'], 'trial': row['trial'], 'seed': row['seed'],
        'success': row['success'], 'termination': row['termination'],
        'tool_errors': len(errors), 'tool_calls': len(calls),
        'assistant_messages': len(assistants), 'assistant_messages_after_tool': len(post_tool),
        'repeated_exact_tool_calls': repeated,
        'completion_tokens': sum(c.get('completion_tokens') or 0 for c in row['agent_calls']),
        'truncated_turns': sum(c.get('finish_reason') == 'length' for c in row['agent_calls']),
        'tool_names': dict(Counter(c['name'] for c in calls)),
        'db_match': (info.get('db_check') or {}).get('db_match'),
    }


def aggregate(rows):
    failed = [r for r in rows if not r['success']]
    errors = [r for r in rows if r['tool_errors']]
    assistant_n = sum(r['assistant_messages'] for r in rows)
    return {
        'tasks': len(rows), 'successes': sum(r['success'] for r in rows),
        'success_percent': 100 * sum(r['success'] for r in rows) / len(rows),
        'failed_tasks': len(failed),
        'failed_with_tool_error': sum(bool(r['tool_errors']) for r in failed),
        'failed_without_tool_error': sum(not r['tool_errors'] for r in failed),
        'all_tasks_with_tool_error': len(errors),
        'successful_despite_tool_error': sum(r['success'] for r in errors),
        'natural_user_stop_failures': sum(r['termination'] == 'user_stop' for r in failed),
        'terminations': dict(Counter(r['termination'] for r in rows)),
        'assistant_messages': assistant_n,
        'assistant_messages_after_tool': sum(r['assistant_messages_after_tool'] for r in rows),
        'after_tool_fraction': sum(r['assistant_messages_after_tool'] for r in rows) / assistant_n,
        'completion_tokens': sum(r['completion_tokens'] for r in rows),
        'truncated_turns': sum(r['truncated_turns'] for r in rows),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--opd', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    base, base_receipts = read_records(args.base)
    opd, opd_receipts = read_records(args.opd)
    if set(base) != set(opd):
        raise ValueError('Task/trial sets differ')
    if any(base[k]['seed'] != opd[k]['seed'] for k in base):
        raise ValueError('Matched trials have different simulator seeds')
    bf, of = [features(base[k]) for k in sorted(base)], [features(opd[k]) for k in sorted(opd)]
    b = {k[0]: float(v['success']) for k, v in base.items()}
    o = {k[0]: float(v['success']) for k, v in opd.items()}
    flips = Counter(('both_right' if base[k]['success'] else 'opd_gain') if opd[k]['success']
                    else ('opd_loss' if base[k]['success'] else 'both_wrong') for k in base)
    output = {
        'source': {'modal_environment': 'alex-dev-2', 'volume': 'lightning-weave-tau-eval',
                   'run': 'tau-b712b7c60edc-full', 'domain': 'tau2_airline', 'trial': 0},
        'limitations': ['One matched trial, historical LoopTool student, not mentor latest students.',
                       'Paired bootstrap resamples 50 tasks; does not estimate training-seed variance.',
                       'A visible tool error is not necessarily a semantic mistake; absence is not correctness.',
                       'Tool error coverage alone is not coverage of the mentor union with WRITE states.',
                       'No new generation, tools, reward execution, or GPU training.'],
        'base': aggregate(bf), 'opd': aggregate(of),
        'paired_delta_pp': paired_delta(b, o), 'paired_flips': dict(flips),
        'per_task': {'base': bf, 'opd': of},
        'source_receipts': {'base': base_receipts, 'opd': opd_receipts},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: output[k] for k in ('base', 'opd', 'paired_delta_pp', 'paired_flips')}, indent=2))


if __name__ == '__main__':
    main()
