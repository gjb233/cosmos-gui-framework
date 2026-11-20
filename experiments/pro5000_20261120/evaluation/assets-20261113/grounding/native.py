from pathlib import Path
import sys,json,re,math,base64,hashlib,io
from PIL import Image
from qwen_vl_utils import smart_resize
UP=Path(__file__).resolve().parent
sys.path.insert(0,str(UP/'src'))
from prompt import MAI_MOBILE_SYS_PROMPT_NO_THINKING
NAV=MAI_MOBILE_SYS_PROMPT_NO_THINKING
GROUND="""You are a GUI grounding agent.
## Task
Given a screenshot and the user's grounding instruction. Your task is to accurately locate a UI element based on the user's instructions.
## Output Format
Return only the coordinate JSON within <answer></answer> XML tags, without a thinking process:
<answer>
{"coordinate": [x,y]}
</answer>
## Input instruction
"""
def image(path,sha,ground=False):
 raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
 im=Image.open(io.BytesIO(raw)).convert('RGB')
 if ground:
  h,w=smart_resize(im.height,im.width,factor=32,min_pixels=1024,max_pixels=6553600)
  im=im.resize((w,h))
 buf=io.BytesIO();im.save(buf,format='PNG')
 return {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()}}
def nav(goal,hist,im):
 # Offline datasets provide descriptions, not recorded native agent responses.
 # Preserve available history as user context; never fabricate assistant tool calls.
 text=goal
 if hist:text+='\n\nPrevious actions:\n'+'\n'.join(str(x) for x in hist[-4:])
 return [{'role':'system','content':[{'type':'text','text':NAV}]},{'role':'user','content':[{'type':'text','text':text}]},{'role':'user','content':[im]}]
def ac_text(row):
 text='The user query: '+row['goal']
 if row['level'].lower()=='low':
  text+='\n\nCurrent step query: '+row['step_instruction']
 text+='\n\nTask progress (You have done the following operation on the current device):'
 for i,a in enumerate(row['history_mobile_use'],1):
  text+='\nStep '+str(i)+': '+json.dumps(a,ensure_ascii=False)+';'
 return text
def ground_messages(row):
 assert row['dataset'].startswith('androidcontrol')
 return [{'role':'system','content':[{'type':'text','text':NAV}]},
         {'role':'user','content':[{'type':'text','text':ac_text(row)},image(row['image'],row['image_sha256'])]}]
def m2_messages(row,block):
 im=row['images'][block]
 return nav(row['task'],row['previous_actions_descriptions'],image(im['path'],im['sha256']))
def convert(text):
 try:
  m=re.search(r'<tool_call>\s*(.*?)\s*</tool_call>',text,re.S)
  if not m:return None
  obj=json.loads(m[1]);assert obj['name']=='mobile_use'
  a=obj['arguments'];act=a['action'];assert isinstance(act,str)
  p=a.get('coordinate')
  if p is not None:
   assert len(p) in (2,4) and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=999 for v in p)
   if len(p)==4:p=[(p[0]+p[2])/2,(p[1]+p[3])/2]
   p=[v*1000/999 for v in p]
  value=a.get('text','')
  mapped={'type':'type_text','open':'open_app','swipe':'scroll','terminate':'status'}.get(act,act)
  if act=='swipe':value=a.get('direction','')
  if act=='terminate':value='successful' if a.get('status')=='success' else 'failed'
  if act=='system_button':mapped={'back':'navigate_back','home':'navigate_home','enter':'enter','menu':'menu'}.get(a.get('button'),'unknown')
  return {'action_type':mapped,'value':value,'point_2d':p if p is not None else [-100,-100]}
 except Exception:return None
def ground_point(text):
 try:
  m=re.search(r'<answer>\s*(.*?)\s*</answer>',text,re.S)
  p=json.loads(m[1])['coordinate']
  assert len(p)==2 and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1000 for v in p)
  return p
 except Exception:return None

from evaluation_protocol import messages as shared_messages,scored_action
def ground_messages(row):
 return shared_messages(row,image(row['image'],row['image_sha256']))
def convert(text):
 try:return scored_action(text)
 except Exception:return None
