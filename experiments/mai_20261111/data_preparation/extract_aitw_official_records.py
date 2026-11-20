"""Pinned GCS AITW scan; preserve selected raw records and screenshots, CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['TF_NUM_INTRAOP_THREADS']='1';os.environ['TF_NUM_INTEROP_THREADS']='1'
import sys,pathlib,json,urllib.request,urllib.parse,gzip,struct,hashlib,base64,collections,concurrent.futures,fcntl,time,threading
sys.path.insert(0,'/rscratch/wzzheng/data/androidcontrol/python')
from tensorflow.core.example.example_pb2 import Example
P=pathlib.Path(__file__).resolve().parent;O=P/'originals/aitw-selected';O.mkdir(exist_ok=True)
lock=(P/'aitw-original-scan.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
wanted={r['episode_id'] for r in map(json.loads,(P/'artifacts/aitw-official-split-identifier-audit.jsonl').open()) if r.get('episode_id')}
# Single hindsight episode IDs include the relabelled instruction, not only the numeric parent.
for r in map(json.loads,(P/'artifacts/inventory.jsonl').open()):
 if r['source']=='aitw' and r['image'].startswith('single_'):
  wanted.add(pathlib.Path(r['image']).stem[len('single_'):].rsplit('_',1)[0])
wanted_hash=hashlib.sha256(json.dumps(sorted(wanted)).encode()).hexdigest()
items=[x for x in json.loads((P/'originals/aitw-gcs-full-index.json').read_text()) if '-of-' in x['name']]
order={'single':0,'general':1,'install':2,'web_shopping':3,'google_apps':4}
items.sort(key=lambda x:(order.get(x['name'].split('/')[1],9),x['name']))
def atom(path,obj):
 tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False));tmp.replace(path)
def worker(x):
 name=pathlib.Path(x['name']).name;directory=O/name;directory.mkdir(exist_ok=True)
 done=directory/'complete.json'
 if done.exists():
  previous=json.loads(done.read_text())
  if previous.get('wanted_hash')==wanted_hash:return previous
  done.rename(directory/('complete-previous-'+str(time.time_ns())+'.json'))
 dest=directory/'source.tfrecord.gz';partial=directory/'download.partial'
 try:
  if not dest.exists():
   u='https://storage.googleapis.com/gresearch/'+urllib.parse.quote(x['name'])+'?generation='+x['generation']
   for attempt in range(3):
    try:
     md=hashlib.md5();size=0
     with urllib.request.urlopen(u,timeout=120) as response,partial.open('wb') as f:
      while True:
       b=response.read(4*1024*1024)
       if not b:break
       f.write(b);md.update(b);size+=len(b)
     assert size==int(x['size']) and base64.b64encode(md.digest()).decode()==x['md5Hash'],'GCS size/MD5 mismatch'
     partial.replace(dest);break
    except Exception:
     if attempt==2:raise
     time.sleep(5*(attempt+1))
  count=0;selected=0;eps=set()
  with gzip.open(dest,'rb') as f,(directory/'records.partial').open('w') as out:
   while True:
    header=f.read(8)
    if not header:break
    n=struct.unpack('<Q',header)[0];assert len(f.read(4))==4
    b=f.read(n);assert len(b)==n and len(f.read(4))==4
    ex=Example.FromString(b).features.feature;count+=1
    ep=ex['episode_id'].bytes_list.value[0].decode()
    if ep not in wanted:continue
    step=int(ex['step_id'].int64_list.value[0]);sha=hashlib.sha256(b).hexdigest()
    root=O/'episodes'/hashlib.sha256(ep.encode()).hexdigest();root.mkdir(parents=True,exist_ok=True)
    recordpath=root/(str(step)+'-'+sha+'.pb');recordpath.write_bytes(b)
    picture=bytes(ex['image/encoded'].bytes_list.value[0]);imsha=hashlib.sha256(picture).hexdigest()
    imagepath=root/(str(step)+'-'+imsha+'.img');imagepath.write_bytes(picture)
    fields={}
    for key,value in ex.items():
     if key=='image/encoded':continue
     kind=value.WhichOneof('kind')
     # Protobuf permits an unset Feature; preserve it as an empty value list.
     vals=list(getattr(value,kind).value) if kind is not None else []
     fields[key]=[v.decode('utf-8',errors='replace') for v in vals] if kind=='bytes_list' else vals
    row=dict(episode_id=ep,step=step,features=fields,raw_record_path=str(recordpath),raw_record_sha256=sha,image_path=str(imagepath),image_sha256=imsha,gcs_name=x['name'],gcs_generation=x['generation'],gcs_md5=x['md5Hash'],record_index=count-1,approved_for_training=False)
    out.write(json.dumps(row,ensure_ascii=False)+'\n');selected+=1;eps.add(ep)
  (directory/'records.partial').replace(directory/'records.jsonl')
  result=dict(shard=x['name'],generation=x['generation'],size=int(x['size']),records_scanned=count,selected_steps=selected,episodes=sorted(eps),status='complete',md5_verified=True,wanted_hash=wanted_hash)
  atom(done,result)
  # Only remove this worker's verified download after retaining selected original bytes and metadata.
  dest.unlink()
  return result
 except Exception as e:
  result=dict(shard=x['name'],status='error',error=repr(e));atom(directory/'error.json',result);return result
finished=[];unique=set()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 futures=[pool.submit(worker,x) for x in items]
 for fu in concurrent.futures.as_completed(futures):
  r=fu.result();finished.append(r);unique.update(r.get('episodes',[]))
  summary=dict(completed_shards=sum(x['status']=='complete' for x in finished),failed_shards=sum(x['status']=='error' for x in finished),total_shards=len(items),selected_steps=sum(x.get('selected_steps',0) for x in finished),found_episodes=len(unique),wanted_episodes=len(wanted),training_ready=False)
  atom(O/'progress.json',summary)
  if len(finished)%20==0:print(json.dumps(summary),flush=True)
atom(O/'complete.json',summary)
