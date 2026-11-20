"""Native H1 packaging, including nonterminal records without a next frame.
Only approved records may be used for training. probe=True is CPU structural QA.
"""
import json,math,os
_SEEN_SOURCES=set()
from pathlib import Path
from mixed_prompt import render_policy,target
def package(record,level,*,probe=False):
 import torch,numpy as np
 from PIL import Image
 from cosmos_framework.gui_mot import joint_dataset as jd
 if not probe and record.get('admission_basis') != 'user_authorized_published_labels':raise ValueError('unapproved training record')
 if level not in record['views']:raise ValueError('unavailable instruction level')
 view=record['views'][level]
 fm=int(record['fm_mask'])
 if fm not in (0,1):raise ValueError('invalid FM mask')
 if fm and (not record.get('next_image') or record.get('transition_verified') is not True or record.get('terminal')):raise ValueError('unverified transition')
 if not fm and record.get('next_image'):raise ValueError('masked row must not carry a future image')
 with Image.open(record['current_image']) as im:
  im=im.convert('RGB');prefix=torch.from_numpy(np.array(im,copy=True)).permute(2,0,1)
 h,w=prefix.shape[-2:]
 scale=min(1.0,640/max(h,w))
 hw=(max(32,math.ceil(h*scale/32)*32),max(32,math.ceil(w*scale/32)*32))
 current=jd.read_screen(record['current_image'],hw)
 future=jd.read_screen(record['next_image'],hw) if fm else None
 # Disable legacy mobile-only rendering and action codec. Inject explicit
 # native text fields after video-only packing.
 sample=jd.make_sample(current,view['instruction'],future=future,action=None,
                       native_ar_text=False,video_only=True,prefix_image=prefix)
 policy=dict(current_image=record['current_image'],instruction=view['instruction'],
             history=record['history_mobile_use'],level=level,goal=record['goal'])
 messages=render_policy(policy)
 sample['gui_system_prompt']=messages[0]['content']
 sample['ai_caption']=messages[1]['content'][0]['text']
 sample['gui_action_text']=target(view['action'])
 sample['gui_current_screens']=[prefix]
 sample['gui_fm_mask']=torch.tensor(float(fm))
 sample['gui_record_id']=record['id']
 if not fm:assert torch.count_nonzero(sample['video'][:,1:])==0
 if fm and record['source'] not in _SEEN_SOURCES:
  _SEEN_SOURCES.add(record['source'])
  root=Path(os.environ['GUI_ADAPTER_EXPORT_ROOT']);root.mkdir(parents=True,exist_ok=True)
  with (root/('source-packaging-'+str(os.getpid())+'.jsonl')).open('a') as f:
   f.write(json.dumps(dict(source=record['source'],id=record['id'],fm_mask=fm,current_image=record['current_image'],next_image=record['next_image'],video_shape=list(sample['video'].shape),prefix_shape=list(prefix.shape)))+'\n')

 return sample
