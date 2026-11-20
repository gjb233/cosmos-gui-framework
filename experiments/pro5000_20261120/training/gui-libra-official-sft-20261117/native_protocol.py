import json,re,math
from pathlib import Path
SYSTEMS=set(json.loads((Path(__file__).parent/'official-systems.json').read_text()))
SYSTEM=sorted(SYSTEMS)[0]
HIGH_FIELDS=LOW_FIELDS=('action_description','action_type','action_target','value','point_2d')
SPEC={}
def parse(text):
 m=re.fullmatch(r'<answer>\s*(.*?)\s*</answer>',text.strip(),re.S)
 assert m,text
 a=json.loads(m[1]);assert set(a)==set(HIGH_FIELDS)
 p=a['point_2d'];assert p is None or p=='None' or (isinstance(p,list) and len(p)==2 and all(isinstance(x,(int,float)) and math.isfinite(x) for x in p))
 return dict(action=a['action_type'],fields=a)
def target(a,level='low',reasoning=None):
 text='<answer>\n'+json.dumps({k:a[k] for k in HIGH_FIELDS},ensure_ascii=False,indent=4)+'\n</answer>'
 parse(text);return text
def render_policy(policy):
 assert policy['official_system'] in SYSTEMS
 return [{'role':'system','content':policy['official_system']},{'role':'user','content':[{'type':'text','text':policy['official_user']},{'type':'image','image':policy['current_image']}]}]
