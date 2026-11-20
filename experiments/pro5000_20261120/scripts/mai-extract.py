import json,time,tarfile,subprocess,fcntl
from pathlib import Path
root=Path('/data/mai/assets/gui-libra-81k-sft-20261028-v1')
lock=open(root/'extract.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
plan=json.loads((root/'download-plan.json').read_text());groups={}
for item in plan['files']:
 name=item['path']
 if name.startswith('data/images/'):groups.setdefault(name.split('.part-')[0],[]).append(name)
out=root/'extracted';out.mkdir(exist_ok=True)
while True:
 s=json.loads((root/'download-status.json').read_text());verified={r['path'] for r in s['verified']}
 complete=0
 for base,names in sorted(groups.items()):
  marker=root/(Path(base).name+'.extracted.json')
  if marker.exists():complete+=1;continue
  if not all(n in verified for n in names):continue
  print('EXTRACT',base,flush=True)
  parts=[root/'snapshot'/n for n in sorted(names)]
  proc=subprocess.Popen(['cat',*[str(p) for p in parts]],stdout=subprocess.PIPE)
  with tarfile.open(fileobj=proc.stdout,mode='r|gz') as archive:archive.extractall(out,filter='data')
  assert proc.wait()==0
  marker.write_text(json.dumps({'parts':[str(p) for p in parts],'completed':time.time()}))
  print('EXTRACTED',base,flush=True);complete+=1
 if complete==len(groups):break
 if s['stage']=='failed':raise SystemExit('Download failed; unfinished archives not extracted')
 time.sleep(20)
(root/'extraction-status.json').write_text(json.dumps({'stage':'complete','archives':len(groups),'completed':time.time()}))
print('ALL_IMAGES_EXTRACTED',flush=True)
