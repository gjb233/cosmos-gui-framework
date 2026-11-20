"""Evaluation adapter sharing the frozen training renderer and action parser.
Call messages(row, encoded_image_content), and send enable_thinking=False.
Existing historical evaluation results are never rewritten.
"""
from native_protocol import SYSTEM,ac_text,parse
ENABLE_THINKING=False
def messages(row,image_content):
 return [{'role':'system','content':SYSTEM},
         {'role':'user','content':[{'type':'text','text':ac_text(row)},image_content]}]
def scored_action(text):
 a=parse(text)['arguments'];t=a['action']
 mapping={'click':'Click','long_press':'LongPress','type':'Write','swipe':'Scroll',
          'gesture_swipe':'Swipe','open':'OpenAPP','wait':'Wait','terminate':'Terminate',
          'answer':'Answer','select':'Select'}
 value=a.get('text','None')
 if t in ['swipe','gesture_swipe']:value=a['direction']
 if t=='click':value=a.get('label','None')
 if t=='wait':value=str(a.get('seconds',2))+'s'
 if t=='terminate':value=a['status']
 if t=='system_button':
  t={'back':'NavigateBack','home':'NavigateHome','enter':'KeyboardPress','menu':'KeyboardPress'}[a['button']]
  value=a['button'].title() if t=='KeyboardPress' else 'None'
 else:t=mapping[t]
 p=a.get('coordinate')
 return {'action_type':t,'value':value,'point_2d':[x*1000/999 for x in p] if p is not None else 'None'}
