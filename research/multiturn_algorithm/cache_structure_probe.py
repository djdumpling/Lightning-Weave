#!/usr/bin/env python3
"""Read-only analysis of the historical eight-response projection cache.

This measures sampling/group structure, not correctness. The donor weights are
from the historical efficiency experiment, not new agentic-accuracy donors.
Canonical tool groups are a diagnostic relaxation only: raw serialization and
private reasoning can remain in future policy context even when calls match.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pyarrow.parquet as pq


def action_key(decision: str) -> tuple[str, str]:
    obj = json.loads(decision)
    visible = obj['visible'] or ''
    blocks = list(re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', visible, re.S))
    # Public prose is never paraphrase-grouped. Preserve call order and finish.
    rest = re.sub(r'<tool_call>\s*.*?\s*</tool_call>', '', visible, flags=re.S)
    if blocks and not rest.strip() and obj['finish_reason'] != 'length':
        try:
            calls = [json.loads(m.group(1)) for m in blocks]
            if not all(isinstance(c,dict) and isinstance(c.get('name'),str) and isinstance(c.get('arguments'),dict) for c in calls):
                raise ValueError('not an ordinary tool call')
            return 'pure_tool', json.dumps({'calls': calls, 'finish_reason': obj['finish_reason']},sort_keys=True,separators=(',',':'))
        except (ValueError,TypeError):
            return 'invalid_tool_serialization', decision
    return ('truncated' if obj['finish_reason']=='length' else 'public_text_or_mixed'), decision


def summarize(groups):
    rows = [r for rs in groups.values() for r in rs]
    exact_n, canonical_n, singleton = Counter(), Counter(), 0
    all_same=all_different=0
    mass_tvs=[]
    for rs in groups.values():
        ec=Counter(r['decision'] for r in rs)
        cc=Counter(action_key(r['decision'])[1] for r in rs)
        exact_n[len(ec)]+=1;canonical_n[len(cc)]+=1
        singleton+=sum(n for n in ec.values() if n==1)
        all_same+=len(ec)==1;all_different+=len(ec)==len(rs)
        n=len(rs)
        mass=defaultdict(lambda:[0.,0.,0.])
        for r in rs:
            m=mass[r['decision']]
            m[0]+=1/n;m[1]+=r['weight_ordinary']/n;m[2]+=r['weight_projected']/n
        # Historical projection must preserve every empirical decision marginal.
        assert abs(sum(r['weight_projected'] for r in rs)-n)<1e-8
        assert all(abs(m[0]-m[2])<1e-8 for m in mass.values())
        mass_tvs.append(.5*sum(abs(m[0]-m[1]) for m in mass.values()))
    result={
        'states':len(groups),'responses':len(rows),
        'responses_per_state':dict(sorted(Counter(map(len,groups.values())).items())),
        'exact_decisions_per_state':dict(sorted(exact_n.items())),
        'canonical_tool_decisions_per_state':dict(sorted(canonical_n.items())),
        'all_same_exact_states':all_same,'all_different_exact_states':all_different,
        'singleton_response_fraction':singleton/len(rows),
        'projected_weight_exactly_one_fraction':float(np.mean([abs(r['weight_projected']-1)<1e-12 for r in rows])),
        'mean_abs_projected_weight_change':float(np.mean([abs(r['weight_projected']-1) for r in rows])),
        'mean_abs_ordinary_weight_change':float(np.mean([abs(r['weight_ordinary']-1) for r in rows])),
        'mean_ordinary_action_mass_total_variation':float(np.mean(mass_tvs)),
        'response_types':dict(Counter(action_key(r['decision'])[0] for r in rows)),
    }
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',type=Path,default=Path('/private/tmp/multiturn-algorithm-cache'));ap.add_argument('--output',type=Path,default=Path(__file__).with_name('artifacts')/'cache_structure.json');args=ap.parse_args()
    wp=args.cache/'weights.parquet';pp=args.cache/'prompts.parquet'
    ws=pq.read_table(wp).to_pylist();ps=pq.read_table(pp).to_pylist()
    prompts={p['prompt_id']:p for p in ps};assert len(prompts)==len(ps)
    groups=defaultdict(list)
    for r in ws:groups[r['prompt_id']].append(r)
    assert set(groups)==set(prompts)
    posttool={k:v for k,v in groups.items() if '<tool_response>' in prompts[k]['prompt'].rsplit('<|im_start|>user',1)[-1]}
    other={k:v for k,v in groups.items() if k not in posttool}
    result={
        'description':'Historical DECS efficiency projection cache; structure only, no correctness labels.',
        'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [wp,pp]},
        'metadata':{k.decode():v.decode() for k,v in pq.read_schema(wp).metadata.items()},
        'all':summarize(groups),'after_tool':summarize(posttool),'other':summarize(other),
        'caveats':['No inference about action correctness or new accuracy donors.','Byte-exact output groups are not sufficient for causal next-state equivalence if private reasoning persists.','Canonical JSON grouping is only diagnostic; public prose is never merged.','All group estimates use just eight samples per state.'],
    }
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
