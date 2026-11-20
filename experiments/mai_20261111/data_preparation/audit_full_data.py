
import json,hashlib,io,fcntl,time,collections,concurrent.futures
from pathlib import Path
from PIL import Image
p=Path(__file__).resolve().parent
out=p/'artifacts'
def check(r):
 try:
  data=Path(r['image']).read_bytes()
  with Image.open(io.BytesIO(data)) as im:
   im.load();w,h=im.size;mode=im.mode
  return dict(r,sha256=hashlib.sha256(data).hexdigest(),width=w,height=h,mode=mode,decoded=True)
 except Exception as e:return dict(r,decoded=False,error=repr(e))
def action(s):
 start=s.rfind('<answer>')
 if start<0:raise ValueError('missing answer')
 text=s[start+8:].strip();a,end=json.JSONDecoder().raw_decode(text)
 if text[end:].strip()!='</answer>':raise ValueError('invalid envelope')
 for k in ['action_type','action_target','value','point_2d']:assert k in a,k
 xy=a['point_2d']
 assert xy is None or xy=='None' or (isinstance(xy,list) and len(xy)==2 and all(isinstance(v,(float,int)) and 0<=v<=1000 for v in xy)),xy
 return a
with (p/'full-data-audit.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 counts=collections.Counter();bad=collections.Counter();types=collections.defaultdict(collections.Counter)
 with (out/'action-pair-audit.jsonl').open('w') as f:
  for line in (out/'inventory.jsonl').open():
   r=json.loads(line);counts[r['source']]+=1
   try:
    a=action(r['direct']);b=action(r['reasoning'])
    equal=all(a[k]==b[k] for k in ['action_type','value','point_2d'])
    types[r['source']][a['action_type']]+=1
    result=dict(id=r['id'],source=r['source'],parsed=True,action_consistent=equal,action=a)
    if not equal:bad['action_mismatch']+=1
   except Exception as e:
    result=dict(id=r['id'],source=r['source'],parsed=False,error=repr(e));bad['parse_error']+=1
   f.write(json.dumps(result,ensure_ascii=False)+chr(10))
 (out/'action-pair-report.json').write_text(json.dumps(dict(counts=counts,issues=bad,action_types=types,training_ready=False),indent=2))
 done=set()
 path=out/'image-decode-audit.jsonl'
 if path.exists():
  for line in path.open():
   try:done.add(json.loads(line)['id'])
   except ValueError:raise RuntimeError('incomplete audit line')
 rows=(json.loads(l) for l in (out/'image-resolution.jsonl').open())
 rows=(r for r in rows if r['id'] not in done)
 with path.open('a') as f,concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
  # bounded batches keep memory bounded on Python3.11
  import itertools
  n=len(done)
  while True:
   batch=list(itertools.islice(rows,128))
   if not batch:break
   for r in pool.map(check,batch):f.write(json.dumps(r)+chr(10));n+=1
   f.flush()
   (out/'full-data-audit-progress.json').write_text(json.dumps(dict(images_checked=n,total=81449,complete=False,time=time.time())))
 summary=collections.Counter();sources=collections.defaultdict(collections.Counter)
 for line in path.open():
  r=json.loads(line);key='decoded' if r['decoded'] else 'failed';summary[key]+=1;sources[r['source']][key]+=1
 (out/'image-decode-report.json').write_text(json.dumps(dict(summary=summary,sources=sources,training_ready=False),indent=2))
 (out/'full-data-audit-progress.json').write_text(json.dumps(dict(images_checked=n,total=81449,complete=True,time=time.time())))
 print(json.dumps(dict(action_issues=bad,images=summary)),flush=True)
