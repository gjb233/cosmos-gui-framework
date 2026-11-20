
from pathlib import Path
import json,re,hashlib,io,base64
from PIL import Image
SYSTEMS=json.loads((Path(__file__).parent/'systems.json').read_text())
def convert(text):
 try:
  m=re.search(r'<answer>\s*(.*?)\s*</answer>',text,re.S);a=json.loads(m[1])
  assert all(k in a for k in ['action_type','value','point_2d'])
  return a
 except Exception:return None
def build(path,sha,instruction,history,source):
 raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
 img=Image.open(io.BytesIO(raw)).convert('RGB');buf=io.BytesIO();fmt='JPEG' if source=='mind2web' else 'PNG';img.save(buf,format=fmt)
 hist='\n'.join(f'Step {i+1} Action: {a}' for i,a in enumerate(history))
 text=f'Please generate the next move according to the UI screenshot (origin image size {img.width}x{img.height}), instruction and previous actions.\n\nInstruction: {instruction}\n\nInteraction History:\n'+hist
 return [{'role':'system','content':[{'type':'text','text':SYSTEMS[source]}]},{'role':'user','content':[{'type':'image_url','image_url':{'url':f'data:image/{fmt.lower()};base64,'+base64.b64encode(buf.getvalue()).decode(),'detail':'high'}},{'type':'text','text':text}]}]
def ground_messages(row):
 return build(row['image'],row['image_sha256'],row['step_instruction'] if row['dataset']=='androidcontrol-low' else row['goal'],row['previous_actions'],'android_control')
def m2_messages(row,block):
 im=row['images'][block];return build(im['path'],im['sha256'],row['task'],row.get('previous_actions_descriptions',row.get('previous_actions',[])) or [],'mind2web')
