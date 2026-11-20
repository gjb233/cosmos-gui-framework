import json,pathlib,urllib.request,urllib.parse,re,datetime,subprocess,fcntl
P=pathlib.Path('/rscratch/wzzheng/experiments/mixed-sft-preparation-20261028-v1')
O=pathlib.Path('/rscratch/wzzheng/experiments/guiodyssey-high-a27-20261028-v1')
out=P/'originals/odyssey-train'
lock=(out/'tree-repair.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
rev=json.load(open(O/'dataset-revision.json'))
url='https://huggingface.co/api/datasets/'+rev['repo']+'/tree/'+rev['sha']+'/annotations?limit=1000'
entries=[];seen=set()
while url:
 if url in seen: raise RuntimeError('pagination loop')
 seen.add(url)
 with urllib.request.urlopen(url,timeout=60) as r:
  entries.extend(json.load(r));links=r.headers.get('Link','')
 match=re.search(r'<([^>]+)>; rel="next"',links)
 url=match.group(1) if match else None
 if url and urllib.parse.urlparse(url).netloc!='huggingface.co':raise RuntimeError('unexpected pagination host')
 print('pages',len(seen),'entries',len(entries),flush=True)
tree=P/'originals/odyssey-annotations-full-tree.json'
tree.write_text(json.dumps(entries))
source=(P/'fetch_odyssey_train_annotations.py').read_text().replace("O/'dataset-files.json'","P/'originals/odyssey-annotations-full-tree.json'")
for name in ('download-results.jsonl','progress.json','complete.json'):
 source=source.replace("'"+name+"'","'retry-full-tree-"+name+"'")
retry=P/'fetch_odyssey_train_annotations_full_tree.py';retry.write_text(source)
subprocess.run(['python3',str(retry)],check=True)
