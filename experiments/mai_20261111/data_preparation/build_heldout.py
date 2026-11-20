
from pathlib import Path
import json,hashlib,collections
p=Path(__file__).resolve().parent;root=p.parent
episodes=set();hashes=set();sources=[];missing=[];cache={};counts=collections.Counter()
def image(path,expected=None):
 q=Path(path)
 if str(q) not in cache:
  if not q.is_file():missing.append(str(q));return
  cache[str(q)]=hashlib.sha256(q.read_bytes()).hexdigest()
 h=cache[str(q)]
 if expected and h!=expected:raise ValueError('heldout image SHA mismatch '+str(q))
 hashes.add(h)
files=sorted(set(root.glob('*/manifest.jsonl'))|set(root.glob('*/manifest-low.jsonl')))
for f in files:
 rows=[]
 try:
  with f.open() as stream:
   first=json.loads(next(stream));rows=[first]+[json.loads(l) for l in stream if l.strip()]
 except (StopIteration,ValueError):continue
 source=None
 if 'annotation_id' in first and 'images' in first:source='mind2web'
 elif 'episode_id' in first and 'image_sha256' in first and 'image' in first:source='android_control'
 elif 'episode' in first and 'target' in first and 'guiodyssey' in str(f):source='gui-odyssey'
 if not source:continue
 sources.append(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),rows=len(rows),source=source))
 for r in rows:
  counts[source]+=1
  if source=='mind2web':
   episodes.add((source,str(r['annotation_id'])))
   for im in r['images']:image(im['path'],im.get('sha256'))
  elif source=='android_control':
   episodes.add((source,str(r['episode_id'])));image(r['image'],r.get('image_sha256'))
  else:
   episodes.add((source,str(r['episode'])))
   candidates=[f.parent/'data/test-screenshots'/r['image'],f.parent/'data/screenshots'/r['image'],Path('/rscratch/wzzheng/data/mai/GUIOdyssey/screenshots')/r['image']]
   found=next((q for q in candidates if q.is_file()),None)
   if found:image(found)
   else:missing.append(str(candidates[0]))
 report=dict(benchmarks=['androidcontrol','gui-odyssey','mm-mind2web-v2'],episodes=sorted(episodes),image_sha256=sorted(hashes),provenance=sources,missing_images=sorted(set(missing)),complete=not missing,scope='union of discovered local benchmark manifests; original official split exclusion additionally required')
 (p/'heldout-candidate.json').write_text(json.dumps(report,indent=2))
 print(source,len(rows),len(episodes),len(hashes),len(set(missing)),flush=True)
if not missing and set(counts)=={'android_control','gui-odyssey','mind2web'}:
 (p/'heldout.json').write_text(json.dumps(report,indent=2))
(p/'artifacts/heldout-build-report.json').write_text(json.dumps(dict(source_rows=counts,episodes=len(episodes),images=len(hashes),manifests=len(sources),missing_images=len(set(missing)),complete=not missing),indent=2))
