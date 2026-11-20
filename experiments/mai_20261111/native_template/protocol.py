from cosmos_framework.gui_mot.mixed_supervision import tokenize_grouped_targets as _original_tokenize_grouped_targets
import json,hashlib
from pathlib import Path
def answer_fields(text):
 from cosmos_framework.gui_mot.mixed_supervision import answer_fields as impl
 return impl(text)
R=Path(__file__).resolve().parent
SPEC=json.loads((R/'official-template-variants.json').read_text())
SYSTEM=SPEC['system'];PROMPT_SHA=hashlib.sha256(SYSTEM.encode()).hexdigest()
def history_action(a):
 if isinstance(a,str):return a
 k=a['type']
 if k in ('click','long_press'):return f"{'Click' if k=='click' else 'Long press'} at {[round(a['x']*1000),round(a['y']*1000)]}"
 if k=='scroll':return 'Scroll '+a['direction']
 if k=='input_text':return 'Write '+a['text']
 if k=='open_app':return 'Open app '+a['app_name']
 return {'navigate_home':'Go to the home screen','navigate_back':'Go back to the previous screen','wait':'Wait','navigate_enter':'Press Enter','terminate':'Terminate'}[k]
def context(goal,history,step_instruction):
 assert isinstance(step_instruction,str) and step_instruction.strip()
 return json.dumps({'step_instruction':step_instruction,'history':history},ensure_ascii=False,separators=(',',':'))
def query(c,size):
 h=''.join(f'\nStep {i+1}\n Action: {history_action(a)}\n' for i,a in enumerate(c['history']))
 return SPEC['question'].format(f'(original image size {size[0]}x{size[1]})',c['step_instruction'],h)+'\n'+SPEC['direct']
def messages(caption,image):
 c=json.loads(caption)
 return [{'role':'system','content':[{'type':'text','text':SYSTEM}]},{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':query(c,image.size)}]}]
def prefix(processor,caption,image):
 text=processor.apply_chat_template(messages(caption,image),tokenize=False,add_generation_prompt=True)
 out=dict(processor(text=[text],images=[image],return_tensors='pt'))
 assert out['input_ids'].shape[-1]+1024<=32768
 return out
def assert_decoded_prefix(processor,inp,caption):
 c=json.loads(caption);d=processor.tokenizer.decode(inp['input_ids'][0])
 assert d.count('<|im_start|>user')==1 and d.count('<|im_start|>assistant')==1
 assert d.count('<|vision_start|>')==1 and d.count('<|vision_end|>')==1
 tail=d.split('<|im_start|>user',1)[1];assert tail.index('<|vision_end|>')<tail.index('Please generate the next move')
 assert 'Instruction: '+c['step_instruction']+'\n\nInteraction History:' in tail
 assert SPEC['direct'] in d and d.rstrip().endswith('<|im_start|>assistant')
 return d
def target(a):
 k=a['type'];kind={'click':'Click','long_press':'LongPress','scroll':'Scroll','input_text':'Write','open_app':'OpenApp','navigate_back':'NavigateBack','navigate_home':'Home','wait':'Wait','navigate_enter':'Enter','terminate':'Terminate'}[k]
 point=[round(a['x']*1000),round(a['y']*1000)] if k in ('click','long_press') else None
 value=a['text'] if k=='input_text' else a['app_name'] if k=='open_app' else a['direction'] if k=='scroll' else a.get('status','success') if k=='terminate' else 'None'
 return '<answer>\n'+json.dumps({'action_type':kind,'action_target':'None','value':value,'point_2d':point},ensure_ascii=False,indent=2)+'\n</answer>'
def grouped(tokenizer,texts,return_masks=False):
 ids,weights=_original_tokenize_grouped_targets(tokenizer,texts,return_masks=True)
 for t,i,w in zip(texts,ids,weights):
  assert tokenizer.decode(i[:-1])==t and abs(sum(w)-1.25)<1e-6
 return (ids,weights) if return_masks else ids

# Run-local structural parser for terminal masking; evaluation keeps the fixed scorer.
def parse(text):
 body=text.split('<answer>',1)[1].split('</answer>',1)[0]
 v=json.loads(body);assert set(v)=={'action_type','action_target','value','point_2d'}
 return dict(action='terminate' if v['action_type']=='Terminate' else v['action_type'], fields=v)
