from pathlib import Path
import json,base64,re,math
R=Path(__file__).resolve().parent
SPEC=json.loads((R/'official-template-variants.json').read_text())
def messages(row):
 h=''.join(f'\nStep {i+1}\n Action: {txt}\n' for i,txt in enumerate(row['previous_actions']))
 q=SPEC['question'].format(f"(original image size {row['screenshot_width']}x{row['screenshot_height']})",row['goal'],h)+'\n'+SPEC['direct']
 return [{'role':'system','content':[{'type':'text','text':SPEC['system']}]},{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(Path(row['image']).read_bytes()).decode(),'detail':'high'}},{'type':'text','text':q}]}]
def plan_ans(row,text):
 # Match the original guilibra_plan.py extraction exactly.
 def field(key):
  m=re.search(r'"'+key+r'"\s*:\s*"([^"]+)"' if key=='action_type' else r'"'+key+r'"\s*:\s*"([^"]*)"',text,re.DOTALL)
  return m.group(1).strip().strip(',').strip('"') if m else ''
 plan=dict(episode_id=row['episode_id'],step=row['step'],instruction=row['goal'],action_type=field('action_type'),value=field('value'),response=text)
 point=[-100,-100];error=None
 try:
  m=re.search(r'"point_2d"\s*:\s*(\[[^\]]*\])',text,re.DOTALL)
  if m:
   point=json.loads(m.group(1));assert len(point)==2 and all(type(x) in (int,float) and math.isfinite(x) for x in point)
  else:error='missing point_2d; sentinel (-100,-100)'
 except Exception as e:point=[-100,-100];error=repr(e)
 x=point[0]*row['screenshot_width']/1000;y=point[1]*row['screenshot_height']/1000
 ans=dict(episode_id=row['episode_id'],step=row['step'],output=f'({x}, {y})',scale=1)
 return plan,ans,error
