#!/usr/bin/env python3
"""Check usage joins for impossible same-user-turn request message counts. CPU only."""
import argparse,json,sys
from pathlib import Path
from collections import defaultdict
p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--new-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.repo))
from evaluation.bfcl_efficiency import load_run,MULTI_TURN
out={};totalruns=0
for tree in sorted(a.new_root.iterdir()):
 for d in sorted(tree.iterdir()):
  if not (d/'aggregate.json').exists():continue
  totalruns+=1;run=load_run(d,require_usage=False);bad=[];covered=0;total=0
  for c in MULTI_TURN:
   for e in run[c]:
    total+=1
    if 'requests' not in e:continue
    covered+=1;groups=defaultdict(list)
    for req in e['requests']:groups[tuple(req['users'])].append(req)
    for rs in groups.values():
     ns=[r['messages'] for r in rs]
     if any(y<=x for x,y in zip(ns,ns[1:])):
      bad.append({'id':e['id'],'messages':ns,'completion_tokens':[r['completion_tokens'] for r in rs]})
  out[f'{tree.name}/{d.name}']={'multi_turn_entries':total,'reconciled':covered,'impossible_assigned_turns':bad}
r={'definition':'A real single sequential conversation strictly increases input message count within one user-turn prefix. Usage join uses chronological request time; a non-increasing count flags mixed attempts or mixed entries despite token-count reconciliation. This check detects some errors and cannot prove all other joins correct.','runs':out,'runs_checked':totalruns,'runs_with_impossible_lineage':sum(bool(v['impossible_assigned_turns']) for v in out.values()),'impossible_assigned_turns':sum(len(v['impossible_assigned_turns']) for v in out.values())}
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='runs'}));print(json.dumps({k:v for k,v in out.items() if v['impossible_assigned_turns']},indent=2))
