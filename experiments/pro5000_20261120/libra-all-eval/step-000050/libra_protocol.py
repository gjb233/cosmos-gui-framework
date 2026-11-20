"""Use the frozen GUI-Libra training renderer for benchmark inference."""
from pathlib import Path
import importlib.util,json,re,base64,hashlib,io,math
from PIL import Image
W=Path('/data/mai/training/gui-libra-zero2-mlp-strict10k-20261114')
spec=importlib.util.spec_from_file_location('frozen_libra_training_protocol',W/'native_protocol.py')
training=importlib.util.module_from_spec(spec);spec.loader.exec_module(training)
SYSTEM=training.SYSTEM
def parse(text):
 text=re.sub(r'^<thinking>.*?</thinking>\s*','',text.strip(),count=1,flags=re.S)
 a=training.parse(text)['fields']
 p=a['point_2d']
 if isinstance(p,list):assert all(0<=x<=1000 for x in p) or p==[-100,-100]
 return a
def convert(text):
 try:return parse(text)
 except (ValueError,AssertionError,TypeError,KeyError):return None
def messages(im,goal,history,instruction='',level='high'):
 p=dict(current_image=im,goal=goal,history=history,instruction=instruction,level=level)
 result=training.render_policy(p);text=result[1]['content'][0]
 buf=io.BytesIO();im.save(buf,format='PNG')
 result[1]['content']=[dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode())),text]
 return result
def checked_image(path,sha):
 raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
 return Image.open(io.BytesIO(raw)).convert('RGB')
def ground_messages(row):
 return messages(checked_image(row['image'],row['image_sha256']),row['goal'],row['history_mobile_use'],row.get('step_instruction',''),row['level'].lower())
_history=None
def m2_messages(row,block):
 global _history
 if _history is None:_history=json.loads(Path('/data/mai/evaluation/assets-20261113/mind2web/history-mobile-use.json').read_text())
 im=row['images'][block]
 return messages(checked_image(im['path'],im['sha256']),row['task'],_history[row['id']])
def android_action(a,width,height):
 t=a['action_type'].lower();v=a['value'];p=a['point_2d'];kw={}
 if isinstance(p,list) and p!=[-100,-100]:kw=dict(x=round(p[0]*(width-1)/1000),y=round(p[1]*(height-1)/1000))
 if t in ('click','longpress','long_press'):
  assert kw,'point required';return dict(action_type='click' if t=='click' else 'long_press',**kw),False
 if t in ('write','type_text'):return dict(action_type='input_text',text=str(v)),False
 if t in ('openapp','open_app'):return dict(action_type='open_app',app_name=str(v)),False
 if t in ('scroll','swipe'):
  direction=str(v).lower();assert direction in ('up','down','left','right')
  if t=='swipe':direction=dict(up='down',down='up',left='right',right='left')[direction]
  return dict(action_type='scroll',direction=direction),False
 if t in ('navigateback','navigatehome'):return dict(action_type={'navigateback':'navigate_back','navigatehome':'navigate_home'}[t]),False
 if t=='keyboardpress':
  assert str(v).lower() in ('enter','return');return dict(action_type='keyboard_enter'),False
 if t=='wait':return dict(action_type='wait'),False
 if t=='answer':return dict(action_type='answer',text=str(v)),True
 if t=='terminate':return dict(action_type='status',goal_status='infeasible' if any(x in str(v).lower() for x in ('fail','infeasible')) else 'complete'),True
 raise ValueError('unsupported action '+t)
