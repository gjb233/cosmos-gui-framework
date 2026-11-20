from pathlib import Path
import json, collections, hashlib

BASE=Path('/data/mai/assets/training-ready')
TRAJ=Path('/data/mai/trajectories/20261111')
LENGTH=Path('/data/mai/benchmarks/gb256-20261112/input-lengths-existing.jsonl')
OUT=Path('/data/mai/assets/training-ready-strict-10k-20261113')
def rows(p):
    with p.open() as f:
        for line in f:
            if line.strip(): yield json.loads(line)
def write(p, data):
    with p.open('w') as f:
        for r in data: f.write(json.dumps(r,ensure_ascii=False)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

bindings={}
for p in sorted(TRAJ.glob('*/sample-bindings.jsonl')):
    for b in rows(p):
        assert b['sample_id'] not in bindings
        bindings[b['sample_id']]=b
lengths={}
for r in rows(LENGTH):
    key=(r['id'],r['level']); value=r['text']+r['visual']
    assert key not in lengths or lengths[key]==value
    lengths[key]=value
train=list(rows(BASE/'train-manifest.jsonl'))
assert len({r['id'] for r in train})==len(train)
kept=[]; excluded=[]; counts=collections.defaultdict(collections.Counter)
for r in train:
    b=bindings.get(r['id']); reasons=[]; c=counts[r['source']];c['original']+=1
    if not b: reasons.append('missing_original_binding')
    else:
        assert b['source']==r['source']
        if not b.get('trajectory_complete'): reasons.append('trajectory_not_strictly_complete')
        if not b.get('history_complete'): reasons.append('history_incomplete')
        if b.get('binding_status')!='original_step_identified': reasons.append('original_step_not_identified')
        issues=b.get('trajectory_issues',{})
        for key in ('missing_step_ids','conflicting_step_ids','unverified_image_step_ids'):
            if issues.get(key): reasons.append(key)
        if issues.get('boundary_known') is False: reasons.append('unknown_boundary')
    missing=[v for v in r['views'] if (r['id'],v) not in lengths]
    assert not missing,(r['id'],missing)
    maximum=max(lengths[r['id'],v] for v in r['views'])
    if reasons: c['excluded_uncertain']+=1
    elif maximum>10000:
        reasons.append('input_prefix_over_10000');c['excluded_length_after_quality']+=1
    if reasons:
        excluded.append(dict(id=r['id'],source=r['source'],reasons=reasons,max_input_tokens=maximum))
    else:
        kept.append(r);c['retained']+=1
assert kept
OUT.mkdir(exist_ok=False)
write(OUT/'train-manifest.jsonl',kept)
write(OUT/'excluded-samples.jsonl',excluded)
ids={r['id'] for r in kept}; indexed={r['id']:r for r in kept}
schedule_stats={}
for phase in (1,2):
    name=f'schedule-phase{phase}.jsonl'; original=list(rows(BASE/name)); selected=[]
    for r in original:
        if r['id'] in ids:
            assert r['level'] in indexed[r['id']]['views']
            assert lengths[r['id'],r['level']]<=10000
            item=dict(r);item['draw']=len(selected);selected.append(item)
    write(OUT/name,selected)
    assert list(rows(OUT/name))==selected
    schedule_stats[name]=dict(original=len(original),retained=len(selected),sha256=sha(OUT/name))
assert len(list(rows(OUT/'train-manifest.jsonl')))==len(ids)
assert len(kept)+len(excluded)==len(train)
report=dict(original_samples=len(train),retained_samples=len(kept),excluded_samples=len(excluded),
    excluded_uncertain=sum(c['excluded_uncertain'] for c in counts.values()),
    excluded_length_after_quality=sum(c['excluded_length_after_quality'] for c in counts.values()),
    by_source=dict(counts),schedules=schedule_stats,output_directory=str(OUT),
    policy=dict(strict_trajectory=True,terminal_length_conflicts_excluded=True,max_input_tokens=10000,
        token_scope='text + visual prefix, excluding answer and FM tokens',all_views_required=True,
        no_truncation=True,schedule_order_preserved=True,draw_indices_renumbered=True),
    inputs=dict(manifest_sha256=sha(BASE/'train-manifest.jsonl'),lengths_sha256=sha(LENGTH)),
    manifest_sha256=sha(OUT/'train-manifest.jsonl'),validation_passed=True)
(OUT/'filter-report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
