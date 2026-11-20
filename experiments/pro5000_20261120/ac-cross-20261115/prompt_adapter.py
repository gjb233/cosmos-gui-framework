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
src=Path('/data/mai/evaluation/assets-20261113/mind2web/upstream/guilibra_plan.py').read_text();tree=ast.parse(src)
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
