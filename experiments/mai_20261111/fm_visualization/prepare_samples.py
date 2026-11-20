import os,sys,json,gzip,struct,hashlib,math
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
sys.path.insert(0,'/rscratch/wzzheng/data/androidcontrol/python')
from tensorflow.core.example.example_pb2 import Example
from PIL import Image
import numpy as np
o=Path(__file__).resolve().parent;p=o.parent
b=p/'evaluation-mai-70284-two-benchmarks-a23-20261110'
rows=[json.loads(l) for l in (b/'joint-300/grounding/manifest.jsonl').open()]
candidates=[r for r in rows if r['level']=='high' and r['history_provenance']['original_shard'].endswith('00001-of-00020')][:4]
wanted={r['episode_id']:r for r in candidates}
train=[json.loads(l) for l in (p/'mai-ui-70283-resume150-fsdp7-micro4-gb56-a22-fullmem/artifacts/train-manifest.jsonl').open()]
episodes={str(r.get('episode_id',r.get('episode',''))) for r in train if r['source']=='android_control'}
assert not episodes.intersection(map(str,wanted))
items=[]
with gzip.open(candidates[0]['history_provenance']['original_shard'],'rb') as f:
 while wanted:
  h=f.read(8)
  if not h:break
  n=struct.unpack('<Q',h)[0];f.read(4);raw=f.read(n);f.read(4)
  ex=Example.FromString(raw).features.feature
  ep=int(ex['episode_id'].int64_list.value[0])
  if ep not in wanted:continue
  r=wanted.pop(ep);s=r['step'];actions=[json.loads(x) for x in ex['actions'].bytes_list.value];shots=ex['screenshots'].bytes_list.value
  assert s+1<len(actions)
  assert hashlib.sha256(raw).hexdigest()==r['history_provenance']['record_sha256']
  d=o/'samples'/str(ep);d.mkdir(parents=True,exist_ok=True)
  for t in [s,s+1,s+2]:(d/f'screen-{t}.png').write_bytes(shots[t])
  with Image.open(d/f'screen-{s}.png') as im:arr=np.array(im.convert('RGB'));w,h=im.size
  with Image.open(r['image']) as im:assert np.array_equal(arr,np.array(im.convert('RGB')))
  scale=min(1,640/max(h,w));hw=[max(32,math.ceil(h*scale/32)*32),max(32,math.ceil(w*scale/32)*32)]
  row=dict(id=r['id'],current_image=str(d/f'screen-{s}.png'),next_image=str(d/f'screen-{s+1}.png'),goal=r['goal'],views={'high':{'instruction':r['goal']}},history_mobile_use=r['history_mobile_use'])
  cap=dict(level='high',goal=r['goal'],instruction=r['goal'],history=r['history_mobile_use'],slot_weight=1.,fm_mask=1,record_id=r['id'])
  items.append(dict(row=row,hw=hw,seed=42,caption=json.dumps(cap),current_rgb_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),action=actions[s],next_action=actions[s+1],after_next_image=str(d/f'screen-{s+2}.png'),provenance=r['history_provenance']))
assert len(items)==4
items.sort(key=lambda x:[r['id'] for r in candidates].index(x['row']['id']))
(o/'selection.json').write_text(json.dumps(items,indent=2))
print('SELECTED',[(i['row']['id'],i['action'],i['next_action']) for i in items],flush=True)
