#!/usr/bin/env python3
"""Independent CPU-only audit of downloaded BFCL trees. No cloud jobs or repository edits.

Example:
  python results_audit_recompute.py --raw-root PATH/TO/night --output audit.json
RAW_ROOT holds runs/ and runs_old/. --new-root and --old-root override these.
CIs condition on the evaluated checkpoints and decoding replicates; they do not
estimate variation across newly trained checkpoints. --draws 0 skips intervals.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import numpy as np

SIMPLE = ('simple_python','simple_java','simple_javascript')
OTHER = ('parallel','multiple','parallel_multiple','irrelevance')
LIVE = ('live_simple','live_multiple','live_parallel','live_parallel_multiple','live_irrelevance','live_relevance')
MULTI = ('multi_turn_base','multi_turn_miss_func','multi_turn_miss_param','multi_turn_long_context')
CATS = SIMPLE + OTHER + LIVE + MULTI
NEW = ('bfcl-v3-22b3e917da76-full','bfcl-v3-a9dc3055c864-full','bfcl-v3-3905dd869e16-full')
OLD = ('bfcl-v3-3e6e955a00df-full','bfcl-v3-8b2ad2de30cd-full','bfcl-v3-4743b22b979b-full')
METRICS = ('accuracy','multi_turn_accuracy','single_turn_accuracy','total_tokens','multi_turn_tokens','single_turn_tokens')
J='ae.joint.'

def weights(counts):
    non={c:1/15 for c in SIMPLE} | {c:1/5 for c in OTHER}
    live={c:counts[c]/sum(counts[t] for t in LIVE) for c in LIVE}
    multi={c:1/4 for c in MULTI}
    single={c:v/2 for g in (non,live) for c,v in g.items()}
    overall={c:v/3 for g in (non,live,multi) for c,v in g.items()}
    return {'overall':overall,'non_live':non,'live':live,'multi_turn':multi,'single_turn':single}

def generated_turns(row):
    gen=row['generation']
    return gen if isinstance(gen,list) and gen and all(isinstance(x,list) for x in gen) else [[gen]]

def bad_think(row):
    texts=[s for turn in generated_turns(row) for s in turn if isinstance(s,str)]
    return any('<think>' in s and '</think>' not in s[s.find('<think>'):] for s in texts)

def fail_turn(row,error):
    kind=error['error_type']
    if kind=='multi_turn:instance_state_mismatch':return len(error['execution_result'])-1
    if kind in ('multi_turn:empty_turn_model_response','multi_turn:execution_response_mismatch'):
        match=re.search(r'for turn (\d+)',error['error_message'])
        if not match:raise ValueError(error['error_message'])
        return int(match.group(1))
    if kind=='multi_turn:force_terminated':return len(generated_turns(row))-1
    raise ValueError(f'Unrecognized failure {kind}')

def load_run(path,issues):
    data={}; multi={}; category_stats={}; errors=Counter(); malformed=[]
    for c in CATS:
        rows={};line_count=0
        with (path/f'bfcl_v3.{c}'/'output.jsonl').open() as f:
            for l in f:
                if not l.strip():continue
                e=json.loads(l);line_count+=1
                if e['id'] in rows:issues.append([str(path),'duplicate',e['id']])
                if e.get('num_generated_tokens_list') is not None and sum(e['num_generated_tokens_list'])!=e['num_generated_tokens']:
                    issues.append([str(path),'token_sum_mismatch',e['id']])
                rows[e['id']]=e
                if e.get('error'):errors[str(e['error'])]+=1
                if bad_think(e):malformed.append({'id':e['id'],'category':c,'correct':bool(e['is_correct'])})
        ids=sorted(rows)
        a=np.array([[float(bool(rows[x]['is_correct'])),float(rows[x]['num_generated_tokens'])] for x in ids])
        data[c]=(ids,a)
        head=json.loads((path/'scores'/f'BFCL_v4_{c}_score.json').open().readline())
        head_accuracy=float(head['accuracy'])
        if abs(head_accuracy-a[:,0].mean())>1e-10:issues.append([str(path),c,'category_accuracy',head_accuracy,a[:,0].mean()])
        category_stats[c]={'entries':len(ids),'lines':line_count,'accuracy_pp':float(100*a[:,0].mean()),'generated_total':int(a[:,1].sum()),'overflow':sum(rows[x].get('error')=='_ran_out_of_context_' for x in ids)}
        if c in MULTI:
            score_lines=(path/'scores'/f'BFCL_v4_{c}_score.json').read_text().splitlines()[1:]
            failures={v['id']:v for v in map(json.loads,score_lines) if v}
            for x,e in rows.items():
                failed=failures.get(x)
                if (failed is None)!=bool(e['is_correct']):issues.append([str(path),x,'score_correctness_mismatch'])
                turn=fail_turn(e,failed['error']) if failed else None
                multi[x]={'category':c,'correct':bool(e['is_correct']),'overflow':e.get('error')=='_ran_out_of_context_',
                          'failure_turn':turn,'step_tokens':e.get('num_generated_tokens_list',[]),'first_turn_steps':len(generated_turns(e)[0]),
                          'first_turn_tokens':sum(e.get('num_generated_tokens_list',[])[:len(generated_turns(e)[0])]),
                          'malformed_think':bad_think(e),'error_type':failed['error']['error_type'] if failed else None}
    count={c:len(data[c][0]) for c in CATS};w=weights(count)
    acc={g:float(100*sum(v*data[c][1][:,0].mean() for c,v in ws.items())) for g,ws in w.items()}
    aggregate=json.loads((path/'aggregate.json').read_text())
    official=100*float(aggregate['overall_accuracy']['accuracy'])
    if abs(official-acc['overall'])>1e-9:issues.append([str(path),'aggregate_accuracy',official,acc['overall']])
    raw=int(sum(data[c][1][:,1].sum() for c in CATS));n=sum(count.values())
    summary={'accuracy_pp':acc,'official_accuracy_pp':official,'entries':n,'generated_total':raw,
             'raw_tokens_per_entry':raw/n,'leaderboard_weighted_tokens_per_entry':float(sum(v*data[c][1][:,1].mean() for c,v in w['overall'].items())),
             'multi_turn_generated_total':int(sum(data[c][1][:,1].sum() for c in MULTI)),
             'single_turn_generated_total':int(sum(data[c][1][:,1].sum() for c in CATS if c not in MULTI)),
             'error_counts':dict(errors),'malformed_think_entries':malformed,'categories':category_stats}
    return {'arrays':data,'multi':multi,'summary':summary}

def values(run,idx=None):
    a=run['arrays']; w=weights({c:len(a[c][0]) for c in CATS})
    means={c:(v[:,0] if idx is None else v[idx[c],0]).mean(axis=-1) for c,(_,v) in a.items()}
    sums={c:(v[:,1] if idx is None else v[idx[c],1]).sum(axis=-1) for c,(_,v) in a.items()}
    out=[sum(v*means[c] for c,v in w[g].items()) for g in ('overall','multi_turn','single_turn')]
    out += [sum(sums[c] for c in cats) for cats in (CATS,MULTI,tuple(c for c in CATS if c not in MULTI))]
    return np.stack(out,axis=-1)

def changes(a,b):
    return np.concatenate([100*(a[...,:3]-b[...,:3]),100*(a[...,3:]/b[...,3:]-1)],axis=-1)

def estimate(point,boot=None):
    return {m:{'delta':float(point[i]),**({'ci95':np.percentile(boot[:,i],[2.5,97.5]).tolist()} if boot is not None else {})} for i,m in enumerate(METRICS)}

def first_failure(runs,pairs,ids,weighted):
    labels=('0','1','2','3+','overflow');arr=np.zeros((len(ids),len(labels)));point_per_category={}
    for arm,ref in pairs:
        for i,e in enumerate(ids):
            for k,sign in ((arm,1),(ref,-1)):
                r=runs[k]['multi'][e]
                if r['correct']:continue
                label='overflow' if r['overflow'] else str(min(r['failure_turn'],3))
                if label=='3':label='3+'
                arr[i,labels.index(label)]+=sign/len(pairs)
    if weighted:
        bycat={c:[i for i,e in enumerate(ids) if runs[pairs[0][0]]['multi'][e]['category']==c] for c in MULTI}
        avg=np.mean([arr[ix].mean(0) for ix in bycat.values()],axis=0)
    else:avg=arr.mean(0)
    transition=np.zeros((len(ids),4))
    for arm,ref in pairs:
        for i,e in enumerate(ids):
            a,r=runs[arm]['multi'][e],runs[ref]['multi'][e]
            afirst=not a['correct'] and not a['overflow'] and a['failure_turn']==0
            rfirst=not r['correct'] and not r['overflow'] and r['failure_turn']==0
            transition[i] += np.array([
                r['correct'] and afirst,
                a['correct'] and rfirst,
                not r['correct'] and not rfirst and afirst,
                not a['correct'] and not afirst and rfirst,
            ],float)/len(pairs)
    tx=np.mean([transition[ix].mean(0) for ix in bycat.values()],axis=0) if weighted else transition.mean(0)
    names=('newly_lost_firstturn','newly_gained_firstturn','already_failed_other_to_firstturn','already_failed_firstturn_to_other')
    return {'entries':len(ids),'weighted_categories':weighted,'net_failure_increase_pp':float(100*avg.sum()),
            'by_earliest_scored_failure':{l:float(100*avg[j]) for j,l in enumerate(labels)},
            'firstturn_transition_decomposition_pp':{n:float(100*tx[j]) for j,n in enumerate(names)},
            'firstturn_net_newly_lost_pp':float(100*(tx[0]-tx[1])),
            'firstturn_net_existing_failure_reclassification_pp':float(100*(tx[2]-tx[3]))}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-root',type=Path,required=True);p.add_argument('--new-root',type=Path);p.add_argument('--old-root',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--draws',type=int,default=2000);p.add_argument('--bootstrap-seed',type=int,default=0)
    p.add_argument('--cluster-multiturn',action='store_true',help='Resample the same numeric task suffix across all four multi-turn category variants; verify identical suffix sets first')
    args=p.parse_args(); roots={'new':args.new_root or args.raw_root/'runs','old':args.old_root or args.raw_root/'runs_old'}
    issues=[];runs={}
    for proto,trees in (('new',NEW),('old',OLD)):
        for seed,tree in enumerate(trees):
            for path in sorted((roots[proto]/tree).iterdir()):
                if (path/'aggregate.json').exists():runs[(proto,str(seed),path.name)]=load_run(path,issues)
    first=next(iter(runs.values()))['arrays']
    for k,r in runs.items():
        for c in CATS:
            if r['arrays'][c][0]!=first[c][0]:issues.append([str(k),c,'entry_ids_mismatch'])
    rng=np.random.default_rng(args.bootstrap_seed)
    idx={c:rng.integers(0,len(first[c][0]),(args.draws,len(first[c][0]))) for c in CATS} if args.draws else None
    if args.cluster_multiturn and idx is not None:
        suffixes={c:[int(x.rsplit('_',1)[1]) for x in first[c][0]] for c in MULTI}
        expected=set(range(200))
        if any(set(v)!=expected or len(v)!=200 for v in suffixes.values()):
            raise ValueError('Task-suffix clustering requires each multi-turn category to contain suffixes 0..199 exactly once')
        shared=rng.integers(0,200,(args.draws,200))
        for c in MULTI:
            position={suffix:i for i,suffix in enumerate(suffixes[c])}
            idx[c]=np.array([position[x] for x in shared.reshape(-1)]).reshape(args.draws,200)
    pts={k:values(r) for k,r in runs.items()};boots={};core={};core_vectors={}
    for proto in ('new','old'):
        pairs=[((proto,s,J+'acc-legacy+decs'+suffix),(proto,s,J+'acc-legacy'+suffix)) for suffix in ('','.s5678') for s in ('0','1','2')]
        pointrows=np.array([changes(pts[a],pts[b]) for a,b in pairs]);point=pointrows.mean(0)
        bt=None
        if idx is not None:
            for k in {r for pair in pairs for r in pair}:boots[k]=values(runs[k],idx)
            bt=np.mean([changes(boots[a],boots[b]) for a,b in pairs],axis=0)
        core[proto]={'pairs':[list(map(list,pair)) for pair in pairs],'pooled':estimate(point,bt),'per_training_seed':{
            '1234':estimate(pointrows[:3].mean(0)),'5678':estimate(pointrows[3:].mean(0))},'per_pair':[estimate(x) for x in pointrows]}
        core_vectors[proto]=(point,bt)
        allids=sorted(runs[pairs[0][0]]['multi'])
        core_tags=[(proto,s,J+tag+suffix) for tag in ('acc-legacy','acc-legacy+decs','acc-legacy+decs-protect-turn-starts','acc-legacy+decs-gate-random-multi-turn') for suffix in ('','.s5678') for s in ('0','1','2')]
        available=[k for k in core_tags if k in runs]
        clean=[e for e in allids if not any(runs[k]['multi'][e]['overflow'] for k in available)]
        core[proto]['first_failure_decomposition']={
            'all800':first_failure(runs,pairs,allids,True),
            'common_nonoverflow_unweighted':first_failure(runs,pairs,clean,False),
            'common_nonoverflow_category_weighted':first_failure(runs,pairs,clean,True),
            'common_nonoverflow_selection_runs':len(available)}
        core[proto]['per_category_delta_pp']={c:float(np.mean([100*(runs[a]['arrays'][c][1][:,0].mean()-runs[b]['arrays'][c][1][:,0].mean()) for a,b in pairs])) for c in CATS}
    n,nb=core_vectors['new'];o,ob=core_vectors['old'];core['difference_in_differences']=estimate(n-o,nb-ob if idx is not None else None)
    out={'raw_roots':{k:str(v) for k,v in roots.items()},'definitions':{'accuracy':'Independent BFCL v3 category weights; points','token_changes':'Mean of six within-pair ratios of raw summed generated tokens, percent','bootstrap':'Shared category-stratified entry resampling, checkpoints and decode replicates fixed','malformed_think':'Final generation contains literal <think> with no subsequent </think> within that visible string','first_failure':'Earliest turn reported by official BFCL score; state mismatch inferred from execution_result length minus one; force termination from final generated turn','first_failure_note':'Descriptive decomposition; a shift from a later failure to an earlier one contributes to early bin without changing total task failure'},'runs_checked':len(runs),'draws':args.draws,'cluster_multiturn':args.cluster_multiturn,'discrepancies':issues,'core':core,'runs':{'/'.join(k):v['summary'] for k,v in runs.items()}}
    args.output.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'runs_checked':len(runs),'discrepancies':issues,'core':core,'output':str(args.output)},indent=2))
if __name__=='__main__':main()
