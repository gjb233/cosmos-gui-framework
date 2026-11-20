import json,math,re
from pathlib import Path
from PIL import Image
SPEC=json.loads(Path('/data/mai/code/f2f8996/experiments/mai_20261111/native_template/official-template-variants.json').read_text())
SYSTEM=SPEC['system']+'\nAdditional dataset actions: Select, KeyboardPress, NavigateHome, Swipe, Answer. Preserve the recorded action type and value. Coordinates are normalized to 0..1000. Return only <answer> JSON; no thinking tags.'
FIELDS=('action_type','action_target','value','point_2d')
def validate(a):
 assert set(a)==set(FIELDS),a
 assert isinstance(a['action_type'],str)
 p=a['point_2d']
 if isinstance(p,list):assert len(p)==2 and all(type(v) in (int,float) and math.isfinite(v) and (-100<=v<=1000) for v in p),p
 else:assert p in (None,'None'),p
 return a
def target(a):
 a=validate({k:a[k] for k in FIELDS})
 return '<answer>\n'+json.dumps(a,ensure_ascii=False)+'\n</answer>'
def parse(text):
 m=re.fullmatch(r'<answer>\s*(.*?)\s*</answer>',text,re.S);assert m,text
 a=validate(json.loads(m[1]));return dict(action='terminate' if a['action_type']=='Terminate' else a['action_type'],fields=a)
def history_action(obj):
 a=obj['arguments'];t=a['action'];v=a.get('text','None');p=a.get('coordinate')
 mapping=dict(click='Click',long_press='LongPress',type='Write',swipe='Scroll',gesture_swipe='Swipe',open='OpenAPP',wait='Wait',terminate='Terminate',answer='Answer',select='Select',drag='Drag')
 if t=='system_button':kind=dict(back='NavigateBack',home='NavigateHome',enter='KeyboardPress',menu='KeyboardPress')[a['button']];v=a['button']
 else:kind=mapping[t]
 if t in ('swipe','gesture_swipe'):v=a['direction']
 if t=='terminate':v=a['status']
 if t=='wait':v=str(a.get('seconds',2))+'s'
 return dict(action_type=kind,action_target='None',value=v,point_2d=[round(x*1000/999) for x in p] if p else 'None')
def render_policy(policy):
 im=policy['current_image']
 if isinstance(im,str):
  with Image.open(im) as opened:size=opened.size
 elif im is not None:size=im.size
 else:size=policy['image_size']
 history=''.join('\nStep '+str(i+1)+'\n Action: '+json.dumps(history_action(a),ensure_ascii=False)+'\n' for i,a in enumerate(policy['history']))
 instruction=policy['instruction'] if policy['level']=='low' else policy['goal']
 text=SPEC['question'].format(f'(original image size {size[0]}x{size[1]})',instruction,history)+'\n'+SPEC['direct']+'\nReturn only the answer block, without a thinking process.'
 # Keep content positions compatible with the audited loader; image precedes text in rendered prompt via backend.
 return [{'role':'system','content':SYSTEM},{'role':'user','content':[{'type':'text','text':text},{'type':'image','image':im}]}]
