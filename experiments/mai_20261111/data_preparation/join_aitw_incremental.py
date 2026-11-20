import pathlib,json,collections,re,hashlib,fcntl
P=pathlib.Path(__file__).resolve().parent
lock=(P/'aitw-incremental-binding.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
idx=collections.defaultdict(dict);shards=0
for done in (P/'originals/aitw-selected').glob('*/complete.json'):
 meta=json.loads(done.read_text())
 if not meta.get('wanted_hash'):continue
 f=done.parent/'records.jsonl'
 if not f.exists():continue
 shards+=1
 for r in map(json.loads,f.open()):
  semantic=hashlib.sha256(json.dumps(dict(features=r['features'],image_sha256=r['image_sha256']),sort_keys=True,ensure_ascii=False).encode()).hexdigest()
  key=(r['episode_id'],r['step'])
  ref={k:r[k] for k in ['raw_record_path','raw_record_sha256','gcs_name','gcs_generation','record_index']}
  if semantic in idx[key]:idx[key][semantic]['equivalent_raw_provenance'].append(ref)
  else:
   r['equivalent_raw_provenance']=[ref];idx[key][semantic]=r
spl=json.loads((P/'originals/aitw-standard-split.json').read_text());split={str(ep):k for k,eps in spl.items() for ep in eps}
counts=collections.Counter();sources=collections.defaultdict(collections.Counter)
with (P/'artifacts/aitw-official-incremental-bindings.jsonl').open('w') as out:
 for r in map(json.loads,(P/'artifacts/inventory.jsonl').open()):
  if r['source']!='aitw':continue
  name=r['image'].rsplit('.',1)[0];kind=name.split('_',1)[0]
  if name.startswith('single_'):
   ep,step=name[len('single_'):].rsplit('_',1)
  else:
   m=re.fullmatch(r'(general|install|googleapps|google_apps|webshopping|web_shopping)_(\d+)_(\d+)',name)
   if not m:counts['filename_unparsed']+=1;continue
   ep,step=m[2],m[3]
  step=int(step);hits=list(idx.get((ep,step),{}).values())
  row=dict(id=r['id'],source='aitw',image=r['image'],original_episode_id=ep,step=step,official_parent_split=split.get(ep.split('.')[0],'unknown'),approved_for_training=False,fm_approved=False)
  status='unique_original_record' if len(hits)==1 else 'not_extracted_yet' if not hits else 'conflicting_original_records'
  row['status']=status
  if len(hits)==1:
   n=hits[0];row['original_record']=n
   row['next_original_records']=list(idx.get((ep,step+1),{}).values())
   row['previous_original_records']=[list(idx[(ep,i)].values())[0] for i in range(max(0,step-4),step) if len(idx.get((ep,i),{}))==1]
  counts[status]+=1;sources[kind][status]+=1
  out.write(json.dumps(row,ensure_ascii=False)+'\n')
report=dict(completed_shards_indexed=shards,unique_raw_step_keys=len(idx),counts=counts,by_subset=sources,training_ready=False,note='Incremental exact full episode+step binding; unextracted is pending, not bad. Original image/action/Low/next admission remains separate.')
(P/'artifacts/aitw-official-incremental-binding-report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
