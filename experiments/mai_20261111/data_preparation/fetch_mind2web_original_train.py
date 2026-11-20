"""Pinned Mind2Web original train download and compact trajectory index."""
import pathlib,json,hashlib,requests,concurrent.futures,fcntl
P=pathlib.Path(__file__).resolve().parent;D=P/'originals/mind2web-train';D.mkdir(exist_ok=True)
meta=json.loads((P/'originals/osunlp-Mind2Web-tree.json').read_text())
files=[x for x in meta['files'] if x.get('type')=='file' and x['path'].startswith('data/train/') and x['path'].endswith('.json')]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def work(item):
 dst=D/pathlib.Path(item['path']).name
 digest=item.get('lfs',{}).get('oid')
 if not(dst.exists() and dst.stat().st_size==item['size'] and (not digest or sha(dst)==digest)):
  url='https://huggingface.co/datasets/'+meta['repo']+'/resolve/'+meta['revision']+'/'+item['path']
  tmp=dst.with_suffix('.partial')
  with requests.get(url,stream=True,timeout=(30,180)) as r,tmp.open('wb') as out:
   r.raise_for_status()
   for chunk in r.iter_content(8*1024*1024):
    if chunk:out.write(chunk)
  assert tmp.stat().st_size==item['size']
  assert not digest or sha(tmp)==digest
  tmp.replace(dst)
 digest=sha(dst);records=json.loads(dst.read_bytes());count=0
 with dst.with_suffix('.index.jsonl').open('w') as out:
  for r in records:
   steps=r['actions']
   for i,a in enumerate(steps):
    row=dict(episode_id=r['annotation_id'],step=i,action_uid=a['action_uid'],goal=r['confirmed_task'],operation=a['operation'],
     positive_candidates=a['pos_candidates'],action_repr=r['action_reprs'][i],next_action_uid=steps[i+1]['action_uid'] if i+1<len(steps) else None,
     previous_action_uids=[v['action_uid'] for v in steps[max(0,i-4):i]],split='train',original_file=str(dst),original_file_sha256=digest,revision=meta['revision'])
    out.write(json.dumps(row,ensure_ascii=False)+'\n');count+=1
 return dict(file=str(dst),sha256=digest,episodes=len(records),steps=count)
with (P/'mind2web-original-download.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 reports=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for r in pool.map(work,files):
   reports.append(r);(D/'progress.json').write_text(json.dumps(dict(completed=len(reports),total=len(files),files=reports),indent=2));print(json.dumps(r),flush=True)
 (D/'complete.json').write_text(json.dumps(dict(complete=True,revision=meta['revision'],files=reports),indent=2))
