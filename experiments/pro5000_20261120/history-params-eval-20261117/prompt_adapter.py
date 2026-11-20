from pathlib import Path
import json,re,ast,base64,io,hashlib,os
from PIL import Image
SPEC=json.loads(Path('/data/mai/code/f2f8996/experiments/mai_20261111/native_template/official-template-variants.json').read_text())
FORMAT=SPEC['reasoning_qwen3']
EXTRA='\naction_type: Select, action_target: Dropdown element, value: Option to select, point_2d: [x, y]\naction_type: Answer, action_target: None, value: Answer text, point_2d: None\naction_type: KeyboardPress, action_target: None, value: Key name, point_2d: None\naction_type: NavigateHome, action_target: None, value: None, point_2d: None\n'
def convert(text):
 try:
  m=re.search(r'<answer>\s*(.*?)\s*</answer>',text,re.S);a=json.loads(m[1]);assert all(k in a for k in ['action_type','action_target','value','point_2d']);return a
 except Exception:return None
src=Path('/data/mai/evaluation/history-params-20261117/assets/mind2web/upstream/guilibra_plan.py').read_text();tree=ast.parse(src)
WEB_SYSTEM=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='system_prompt' for t in n.targets))
WEB_FORMAT=next(n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str) and '<thinking>' in n.value and 'action_target' in n.value)
def m2_messages(row,block):
 im=row['images'][block];raw=Path(im['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==im['sha256'];img=Image.open(io.BytesIO(raw)).convert('RGB');buf=io.BytesIO();img.save(buf,format='JPEG')
 hist=row.get('previous_actions_descriptions',row.get('previous_actions',[])) or []
 history=''.join(f'\nStep {i+1}\n Action: {a}\n' for i,a in enumerate(hist))
 variant=os.environ['PROMPT_ARM'];system=WEB_SYSTEM if variant=='official' else SPEC['system']+EXTRA
 fmt=WEB_FORMAT if variant=='official' else FORMAT
 text=SPEC['question'].format(f'(original image size {img.width}x{img.height})',row['task'],history)+'\n'+fmt
 return [{'role':'system','content':[{'type':'text','text':system}]},{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode(),'detail':'high'}},{'type':'text','text':text}]}]
def wrap_aw(llm,root):
 original=llm.predict_mm_messages;counter=[0]
 def predict(messages):
  counter[0]+=1
  messages[0]['content']=[{'type':'text','text':SPEC['system']+EXTRA}]
  parts=messages[1]['content'];text=next(x for x in parts if x['type']=='text');text['text']=text['text'].split('The response should be structured')[0].rstrip()+'\n'+FORMAT
  text['text']=re.sub(r'\nStep (\d+) Action:',r'\nStep \1\n Action:',text['text'])
  saved=[{'role':m['role'],'content':[dict(type='image_url',image_url={'url':'[image omitted]'}) if x['type']=='image_url' else x for x in m['content']]} for m in messages]
  output,safe,response=original(messages)
  (root/f'request-{counter[0]:05d}.json').write_text(json.dumps(dict(messages=saved,response=output),ensure_ascii=False))
  a=convert(output)
  if a:
   a['action_description']=f"{a['action_type']}: {a['action_target']}; value={a['value']}"
   output=re.sub(r'<answer>.*?</answer>',lambda m:'<answer>'+json.dumps(a,ensure_ascii=False)+'</answer>',output,flags=re.S)
  return output,safe,response
 llm.predict_mm_messages=predict;return llm

AW_SOURCE=Path('/data/mai/benchmarks/gui-libra-official-repro/android_world/agents/seeact_v_guilibra.py').read_text()
AW_TREE=ast.parse(AW_SOURCE)
def constant(name):
 return resolve(next(n.value for n in AW_TREE.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))
def resolve(node):
 if isinstance(node,ast.Name):return constant(node.id)
 if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add):return resolve(node.left)+resolve(node.right)
 return ast.literal_eval(node)
AW_SYSTEM=constant('system_prompt')
GUIDANCE=constant('GUIDANCE')
AW_FORMAT=next(n.value for n in ast.walk(AW_TREE) if isinstance(n,ast.Constant) and isinstance(n.value,str) and '<thinking>' in n.value and 'action_description' in n.value and 'point_2d' in n.value)
def convert(text):
 try:
  m=re.search(r'<answer>\s*(.*?)\s*</answer>',text,re.S);a=json.loads(m[1]);assert all(k in a for k in ['action_type','value','point_2d']);return a
 except Exception:return None
def build(image_path,sha,goal,history,bench):
 raw=Path(image_path).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
 img=Image.open(io.BytesIO(raw)).convert('RGB');buf=io.BytesIO();fmt='JPEG' if bench=='mind2web' else 'PNG';img.save(buf,format=fmt)
 arm=os.environ['PROMPT_ARM']
 h=''.join(f'\nStep {i+1}\n Action: {a}\n' for i,a in enumerate(history))
 if arm=='official':system=WEB_SYSTEM if bench=='mind2web' else SPEC['system'];output=WEB_FORMAT if bench=='mind2web' else FORMAT
 else:
  system=AW_SYSTEM+'\naction_type: Select, action_target: Dropdown element, value: Option to select, point_2d: [x,y]\naction_type: LongPress, action_target: Element description, value: None, point_2d: [x,y]\n'
  output=AW_FORMAT;h=''.join(f'\nStep {i+1} Action: {a}\n' for i,a in enumerate(history))
 text=SPEC['question'].format(f'(original image size {img.width}x{img.height})',goal,h)
 if arm=='aw-full':text+=GUIDANCE
 text+='\n'+output
 return [{'role':'system','content':[{'type':'text','text':system}]},{'role':'user','content':[{'type':'image_url','image_url':{'url':f'data:image/{fmt.lower()};base64,'+base64.b64encode(buf.getvalue()).decode(),'detail':'high'}},{'type':'text','text':text}]}]
def ground_messages(row):return build(row['image'],row['image_sha256'],row['goal'],row['previous_actions'],'grounding')
def m2_messages(row,block):
 im=row['images'][block];return build(im['path'],im['sha256'],row['task'],row.get('previous_actions_descriptions',row.get('previous_actions',[])) or [],'mind2web')

TRAIN_SPEC=json.loads(Path('/data/mai/training/gui-libra-awhigh-aclow-20261116/templates.json').read_text())
_old_build=build
def direct_build(image_path,sha,instruction,history,bench,level):
 messages=_old_build(image_path,sha,instruction,history,bench)
 messages[0]['content']=[{'type':'text','text':TRAIN_SPEC[level+'_system']}]
 with Image.open(image_path) as im:size=im.size
 h=''.join(('\nStep '+str(i+1)+(' Action: ' if level=='high' else '\n Action: ')+str(a)+'\n') for i,a in enumerate(history))
 text=TRAIN_SPEC['question'].format(f'(original image size {size[0]}x{size[1]})',instruction,h)+'\n'+TRAIN_SPEC[level+'_format']
 assert '<thinking>' not in text
 next(x for x in messages[1]['content'] if x['type']=='text')['text']=text
 return messages
def ground_messages(row):
 level='low' if row['dataset']=='androidcontrol-low' else 'high'
 return direct_build(row['image'],row['image_sha256'],row['step_instruction'] if level=='low' else row['goal'],row['previous_actions'],'grounding',level)
def m2_messages(row,block):
 im=row['images'][block]
 return direct_build(im['path'],im['sha256'],row['task'],row.get('previous_actions_descriptions',row.get('previous_actions',[])) or [],'mind2web','high')

PARAM_HISTORY=json.loads((Path(__file__).parent/'histories.json').read_text())
def ground_messages(row):
 level='low' if row['dataset']=='androidcontrol-low' else 'high'
 return direct_build(row['image'],row['image_sha256'],row['step_instruction'] if level=='low' else row['goal'],PARAM_HISTORY[row['id']],'grounding',level)
def m2_messages(row,block):
 im=row['images'][block]
 return direct_build(im['path'],im['sha256'],row['task'],PARAM_HISTORY[row['id']],'mind2web','high')
