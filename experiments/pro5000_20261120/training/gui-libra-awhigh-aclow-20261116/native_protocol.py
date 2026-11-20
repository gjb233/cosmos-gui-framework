import json,math,re
from pathlib import Path
from PIL import Image
SPEC=json.loads((Path(__file__).parent/'templates.json').read_text())
SYSTEM=SPEC['low_system']
SYSTEMS={SPEC['low_system'],SPEC['high_system']}
LOW_FIELDS=('action_type','action_target','value','point_2d')
HIGH_FIELDS=('action_description','action_type','value','point_2d')
def validate(a):
 assert set(a) in (set(LOW_FIELDS),set(HIGH_FIELDS)),a
 assert isinstance(a['action_type'],str)
 p=a['point_2d'];assert isinstance(p,list) and len(p)==2
 assert p==[-100,-100] or all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1000 for x in p),p
 return a
def target(a,level='low',reasoning=None):
 assert level in ('high','low')
 # Direct-only supervision: original reasoning is not emitted.
 fields=HIGH_FIELDS if level=='high' else LOW_FIELDS
 out={k:a[k] for k in fields}
 if out['point_2d'] in (None,'None'):out['point_2d']=[-100,-100]
 validate(out)
 return '<answer>\n'+json.dumps(out,ensure_ascii=False)+'\n</answer>'
def parse(text):
 m=re.fullmatch(r'<answer>\s*(.*?)\s*</answer>',text,re.S)
 assert m,text
 a=validate(json.loads(m[1]))
 return dict(action='terminate' if a['action_type']=='Terminate' else a['action_type'],fields=a)
def render_policy(policy):
 im=policy['current_image']
 if isinstance(im,str):
  with Image.open(im) as opened:size=opened.size
 elif im is not None:size=im.size
 else:size=policy['image_size']
 level=policy['level'];assert level in ('high','low')
 assert all(isinstance(a,str) for a in policy['history'])
 history=''.join(('\nStep '+str(i+1)+(' Action: ' if level=='high' else '\n Action: ')+a+'\n') for i,a in enumerate(policy['history']))
 instruction=policy['instruction'] if level=='low' else policy['goal']
 text=SPEC['question'].format(f'(original image size {size[0]}x{size[1]})',instruction,history)+'\n'+SPEC[level+'_format']
 return [{'role':'system','content':SPEC[level+'_system']},{'role':'user','content':[{'type':'text','text':text},{'type':'image','image':im}]}]
