"""CPU-only inventory, evidence join and deterministic mixed-SFT schedule."""
import argparse, hashlib, json, random
from collections import Counter, defaultdict
from pathlib import Path

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def lines(p):
    with Path(p).open(encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)

def write_lines(p, rows):
    with Path(p).open('w', encoding='utf-8') as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False)+'\n')

def inventory(root, output):
    counts=read(root/'mixture-counts.json'); pairs=defaultdict(dict)
    for entry in counts['entries']:
        rows=read(root/'snapshot'/entry['file'])
        assert len(rows)==entry['samples'], entry['file']
        seen=Counter()
        for row in rows:
            conv=row['conversations']
            assert conv and conv[-1]['from'] in ('gpt','assistant')
            prefix=conv[:-1]
            key=digest([entry['source'], row['image'], prefix])
            # Repeated identical prompts are not silently discarded.
            occurrence=seen[key]; seen[key]+=1
            uid=key+':'+str(occurrence)
            item=pairs[uid]
            item.update(id=uid, source=entry['source'], image=row['image'], prefix=prefix)
            assert entry['variant'] not in item
            item[entry['variant']]=conv[-1]['value']
            item.setdefault('provenance',{})[entry['variant']]=entry['file']
    rows=list(pairs.values())
    report={'paired_records':sum('reasoning' in r and 'direct' in r for r in rows),
            'unique_prompt_occurrences':len(rows), 'source_counts':dict(Counter(r['source'] for r in rows)),
            'unpaired':sum(not ('reasoning' in r and 'direct' in r) for r in rows),
            'instruction_level':'unknown_until_source_join', 'training_ready':False}
    write_lines(output/'inventory.jsonl', rows)
    (output/'inventory-report.json').write_text(json.dumps(report,indent=2))
    return report

def validate_evidence(row, e, heldout, image_root):
    """Evidence must be constructed from original dataset annotations, not filenames."""
    assert e['id']==row['id'] and e['source']==row['source']
    assert e.get('approved_for_training') is True, 'original evidence is not approved'
    assert e.get('split_verified') is True and e.get('current_image_identity_verified') is True
    assert e['split']=='train' and e['episode_id']
    assert [e['source'], e['episode_id']] not in heldout['episodes']
    assert e['annotation_sha256'] and e['annotation_path']
    assert len(e['annotation_sha256'])==64
    annotation=Path(e['annotation_path'])
    assert annotation.is_file() and hashlib.sha256(annotation.read_bytes()).hexdigest()==e['annotation_sha256']
    assert type(e['step']) is int and e['step']>=0
    images=[e['current_image']]
    assert e['instruction_level'] in ('low','high')
    assert e['instruction'] and e['instruction'] in (e.get('low_instruction'),e.get('goal'))
    assert e['instruction']==e['low_instruction' if e['instruction_level']=='low' else 'goal']
    for h in e['history']:
        assert h['episode_id']==e['episode_id'] and h['step']<e['step']
    assert e['history']==sorted(e['history'],key=lambda h:h['step'])
    next_image=e.get('next_image')
    if next_image:
        assert not e['terminal'] and e['next_step']==e['step']+1
        assert e['next_episode_id']==e['episode_id']
        assert e['action_alignment_verified'] is True
        images.append(next_image)
    for img in images:
        p=Path(img).resolve()
        assert p.is_relative_to(image_root.resolve()) and p.is_file()
        sha=hashlib.sha256(p.read_bytes()).hexdigest()
        assert sha not in heldout['image_sha256']
    return bool(next_image)

def build(config, output, evidence_path, heldout_path):
    cfg=read(config); heldout=read(heldout_path)
    assert set(heldout['benchmarks']) >= {'androidcontrol','gui-odyssey','mm-mind2web-v2'}
    evidence={}
    for e in lines(evidence_path):
        assert e['id'] not in evidence; evidence[e['id']]=e
    accepted=[]; blocked=Counter(); root=Path(cfg['image_root'])
    for row in lines(output/'inventory.jsonl'):
        if row['id'] not in evidence: blocked['missing_original_annotation_join']+=1; continue
        e=evidence[row['id']]
        try: transition=validate_evidence(row,e,heldout,root)
        except (AssertionError,KeyError,OSError) as exc:
            raise ValueError('Invalid evidence: '+row['id']) from exc
        # High/Low targets need separate validation. A Low-specific rationale
        # cannot simply become a High target when the prompt is swapped.
        views={}
        for level,field in [('low','low_instruction'),('high','goal')]:
            targets=e.get('targets',{}).get(level)
            if e.get(field) and targets and targets.get('instruction_consistency_verified') is True:
                action=targets.get('action')
                assert isinstance(action,dict), 'missing audited canonical action'
                from mixed_prompt import target
                direct=target(action)
                assert targets.get('direct')==direct, 'direct target differs from canonical action'
                views[level]={'instruction':e[field], 'action':action, 'direct':direct,
                              'reasoning':targets.get('reasoning')}
        if not views: blocked['missing_level_specific_target_audit']+=1; continue
        accepted.append({'id':row['id'],'source':row['source'],'episode_id':e['episode_id'],
                         'step':e['step'],'current_image':e['current_image'],
                         'history':[h['action'] for h in e['history'][-cfg['history_steps']:]], 'views':views,
                         'next_image':e.get('next_image'),'fm_mask':int(transition),
                         'terminal':e['terminal'],'approved_for_training':True,
                         'action_alignment_verified':bool(e.get('action_alignment_verified')), 'evidence_sha256':digest(e)})
    write_lines(output/'train-manifest.jsonl',accepted)
    by_source=Counter(r['source'] for r in accepted)
    report={'accepted':len(accepted),'sources':dict(by_source),'blocked':dict(blocked),
            'transition_count':sum(r['fm_mask'] for r in accepted),
            'manifest_sha256':hashlib.sha256((output/'train-manifest.jsonl').read_bytes()).hexdigest(),
            'all_sources_present':all(by_source.get(s) for s in cfg['sources']),
            'training_ready':False}
    (output/'build-report.json').write_text(json.dumps(report,indent=2))
    return report

def schedule(config, output, phase, draws):
    cfg=read(config); rng=random.Random(cfg['seed']+phase); groups=defaultdict(list)
    for row in lines(output/'train-manifest.jsonl'): groups[row['source']].append(row)
    assert all(groups.get(s) for s in cfg['sources']), 'Missing source: do not silently renormalize'
    sources=list(cfg['sources']); weights=[cfg['sources'][s]['count'] for s in sources]
    records=[]
    for i in range(draws):
        source=rng.choices(sources,weights=weights,k=1)[0]; row=rng.choice(groups[source])
        level=rng.choice(list(row['views'])) # dual-level rows: 1:1; one-level rows unchanged
        view=row['views'][level]
        variants=[k for k in ('direct','reasoning') if view.get(k)]
        if cfg['sampling'].get('answer_style')=='direct_only': variants=[v for v in variants if v=='direct']
        elif phase==2: variants=[v for v in variants if v=='reasoning']
        assert variants, (row['id'],level,'missing target for phase')
        variant=rng.choice(variants)
        records.append({'draw':i,'id':row['id'],'source':source,'level':level,'variant':variant,
                        'fm_mask':row['fm_mask']})
    write_lines(output/f'schedule-phase{phase}.jsonl',records)
    result={'draws':draws,'seed':cfg['seed']+phase,'source':dict(Counter(r['source'] for r in records)),
            'level':dict(Counter(r['level'] for r in records)),
            'variant':dict(Counter(r['variant'] for r in records))}
    (output/f'schedule-phase{phase}-report.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['inventory','build','schedule'])
    p.add_argument('--config',default='mixture.json');p.add_argument('--output',default='artifacts')
    p.add_argument('--evidence');p.add_argument('--heldout');p.add_argument('--phase',type=int,choices=[1,2],default=1)
    p.add_argument('--draws',type=int,default=81449);a=p.parse_args();o=Path(a.output);o.mkdir(parents=True,exist_ok=True)
    if a.mode=='inventory':result=inventory(Path(read(a.config)['data_root']),o)
    elif a.mode=='build':result=build(a.config,o,a.evidence,a.heldout)
    else: result=schedule(a.config,o,a.phase,a.draws)
    print(json.dumps(result,ensure_ascii=False,indent=2))
