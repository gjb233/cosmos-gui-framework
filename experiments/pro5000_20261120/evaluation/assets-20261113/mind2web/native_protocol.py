"""Shared MAI no-thinking protocol for training and future evaluation."""
import ast,json,math
from pathlib import Path
UP=Path(__file__).resolve().parent/'prompt.py'
tree=ast.parse(UP.read_text())
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MAI_MOBILE_SYS_PROMPT_NO_THINKING' for t in n.targets))
BASE_SYSTEM=ast.literal_eval(node.value.func.value).strip()
EXTENSION='''

## Dataset action extensions (shared by training and evaluation)
{"action": "select", "text": "option value", "coordinate": [x, y]} # Select the named option; coordinate is optional.
{"action": "gesture_swipe", "direction": "up or down or left or right", "coordinate": [x, y]} # Finger-motion direction, distinct from viewport scroll direction; coordinate is optional.
The native swipe action denotes viewport scrolling, following the AndroidControl evaluation mapping.
type and system_button may optionally include coordinate to identify the target.
wait may optionally include seconds to preserve the recorded duration.
click may optionally include label to preserve a supplied semantic label.
All coordinates are [x,y] normalized to 0..999 on the corresponding screenshot.
Return only <tool_call> with mobile_use JSON; do not emit thinking tags.
'''
SYSTEM=BASE_SYSTEM+EXTENSION
ACTIONS={'click','long_press','type','swipe','open','drag','system_button','wait','terminate','answer','select','gesture_swipe'}
def validate(obj):
 assert set(obj)=={'name','arguments'} and obj['name']=='mobile_use'
 a=obj['arguments'];assert isinstance(a,dict) and a['action'] in ACTIONS
 for k in ['coordinate','start_coordinate','end_coordinate']:
  if k in a:assert isinstance(a[k],list) and len(a[k])==2 and all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=999 for x in a[k]),a
 if a['action'] in ['click','long_press']:assert 'coordinate' in a
 if a['action'] in ['type','open','answer','select']:assert isinstance(a.get('text'),str)
 if a['action'] in ['swipe','gesture_swipe']:assert a.get('direction') in ['up','down','left','right']
 if a['action']=='system_button':assert a.get('button') in ['back','home','menu','enter']
 if a['action']=='terminate':assert a.get('status') in ['success','fail']
 return obj
def published(action):
 typ=action['action_type'];v=action['value'];p=action['point_2d']
 args={}
 if isinstance(p,list):
  assert len(p)==2 and all(type(x) in (int,float) and 0<=x<=1000 for x in p)
  args['coordinate']=[round(x*999/1000) for x in p]
 if typ in ['Click','LongPress']:
  args['action']={'Click':'click','LongPress':'long_press'}[typ]
  if typ=='Click' and v not in (None,'None',''):args['label']=str(v)
 elif typ in ['Write','OpenAPP','Answer','Select']:
  args['action']={'Write':'type','OpenAPP':'open','Answer':'answer','Select':'select'}[typ];args['text']=v
 elif typ in ['Scroll','Swipe']:
  args['action']='swipe' if typ=='Scroll' else 'gesture_swipe';args['direction']=v
 elif typ in ['NavigateHome','NavigateBack','KeyboardPress']:
  args['action']='system_button';args['button']={'NavigateHome':'home','NavigateBack':'back','KeyboardPress':str(v).lower()}[typ]
 elif typ=='Wait':
  args['action']='wait';args['seconds']=float(str(v).removesuffix('s'))
 elif typ=='Terminate':
  args['action']='terminate';args['status']='fail' if v in ['fail','failed'] else 'success'
 else:raise ValueError(typ)
 return validate({'name':'mobile_use','arguments':{'action':args.pop('action'),**args}})
def parse(text):
 assert text.startswith('<tool_call>\n') and text.endswith('\n</tool_call>')
 return validate(json.loads(text[len('<tool_call>\n'):-len('\n</tool_call>')]))
def target(action):
 return '<tool_call>\n'+json.dumps(validate(action),ensure_ascii=False)+'\n</tool_call>'
def ac_text(row):
 text='The user query: '+row['goal']
 if row['level'].lower()=='low':text+='\n\nCurrent step query: '+row['step_instruction']
 text+='\n\nTask progress (You have done the following operation on the current device):'
 for i,a in enumerate(row['history_mobile_use'],1):
  text+='\nStep '+str(i)+': '+json.dumps(validate(a),ensure_ascii=False)+';'
 return text
def render_policy(policy):
 assert policy['level'] in ['low','high']
 text=ac_text(dict(goal=policy['goal'],level=policy['level'],step_instruction=policy['instruction'],history_mobile_use=policy['history']))
 return [{'role':'system','content':SYSTEM},{'role':'user','content':[{'type':'text','text':text},{'type':'image','image':policy['current_image']}]}]
