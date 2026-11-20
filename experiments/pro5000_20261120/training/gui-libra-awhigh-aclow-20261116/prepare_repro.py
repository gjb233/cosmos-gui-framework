from pathlib import Path
import json, re, sys, shutil, collections, hashlib, ast

OLD = Path('/data/mai/training/gui-libra-zero2-mlp-strict10k-20261114')
W = Path('/data/mai/training/gui-libra-awhigh-aclow-20261116')
assert not (W/'train-manifest.jsonl').exists() and not (W/'run/active.json').exists(), 'Do not overwrite a prepared or launched training run'
W.mkdir(exist_ok=True)
(W/'run').mkdir(exist_ok=True)
for p in OLD.iterdir():
    if p.suffix in ('.py', '.toml') or p.name in ('config.json','training-plan.json'):
        (W/p.name).write_text(p.read_text().replace(str(OLD),str(W)))
sys.path.insert(0,'/data/mai/aw-cross-20261116')
import prompt_adapter as pa
templates = dict(question=pa.SPEC['question'], high_system=pa.AW_SYSTEM, low_system=pa.SPEC['system'], high_format=pa.AW_FORMAT, low_format=pa.FORMAT)
# These dataset actions are needed in addition to the benchmark's vocabulary.
definitions={
 'OpenAPP':'action_type: OpenAPP, action_target: None, value: App name, point_2d: [-100,-100]; equivalent to OpenApp in dataset spelling',
 'Select':'action_type: Select, action_target: Dropdown element, value: Option to select, point_2d: [x,y]',
 'LongPress':'action_type: LongPress, action_target: Element description, value: None, point_2d: [x,y]',
 'Swipe':'action_type: Swipe, action_target: Optional position or None, value: up | down | left | right, point_2d: [x,y] or [-100,-100]',
 'Answer':'action_type: Answer, action_target: None, value: Answer text, point_2d: [-100,-100]',
 'KeyboardPress':'action_type: KeyboardPress, action_target: None, value: Key name, point_2d: [-100,-100]',
 'NavigateHome':'action_type: NavigateHome, action_target: None, value: None, point_2d: [-100,-100]',
}
rows=[json.loads(l) for l in (OLD/'train-manifest.jsonl').open()]
types={v['action']['action_type'] for r in rows for v in r['views'].values()}
for level in ('high','low'):
    key=level+'_system'
    for action in sorted(types):
        if 'action_type: '+action+',' not in templates[key]:
            assert action in definitions, action
            templates[key]+='\n'+definitions[action]+'\n'
(W/'templates.json').write_text(json.dumps(templates,ensure_ascii=False,indent=2))
(W/'native_protocol.py').write_text('''import json,math,re
from pathlib import Path
from PIL import Image
SPEC=json.loads((Path(__file__).parent/'templates.json').read_text())
SYSTEM=SPEC['low_system']
SYSTEMS={SPEC['low_system'],SPEC['high_system']}
LOW_FIELDS=('action_type','action_target','value','point_2d')
HIGH_FIELDS=('action_description','action_type','value','point_2d')
def validate(a):
 assert set(a) in (set(LOW_FIELDS),set(HIGH_FIELDS)),a
 assert isinstance(a['action_type'],str)
 p=a['point_2d'];assert isinstance(p,list) and len(p)==2
 assert p==[-100,-100] or all(type(x) in (int,float) and math.isfinite(x) and 0<=x<=1000 for x in p),p
 return a
def target(a,level='low',reasoning=None):
 assert level in ('high','low') and isinstance(reasoning,str) and reasoning.strip()
 assert not any(x in reasoning for x in ('<thinking>','</thinking>','<answer>','</answer>'))
 fields=HIGH_FIELDS if level=='high' else LOW_FIELDS
 out={k:a[k] for k in fields}
 if out['point_2d'] in (None,'None'):out['point_2d']=[-100,-100]
 validate(out)
 return '<thinking>'+reasoning+'</thinking>\\n<answer>\\n'+json.dumps(out,ensure_ascii=False)+'\\n</answer>'
def parse(text):
 m=re.fullmatch(r'<thinking>(.*?)</thinking>\\s*<answer>\\s*(.*?)\\s*</answer>',text,re.S)
 assert m and m[1].strip(),text
 a=validate(json.loads(m[2]))
 return dict(action='terminate' if a['action_type']=='Terminate' else a['action_type'],fields=a)
def render_policy(policy):
 im=policy['current_image']
 if isinstance(im,str):
  with Image.open(im) as opened:size=opened.size
 elif im is not None:size=im.size
 else:size=policy['image_size']
 level=policy['level'];assert level in ('high','low')
 assert all(isinstance(a,str) for a in policy['history'])
 history=''.join(('\\nStep '+str(i+1)+(' Action: ' if level=='high' else '\\n Action: ')+a+'\\n') for i,a in enumerate(policy['history']))
 instruction=policy['instruction'] if level=='low' else policy['goal']
 text=SPEC['question'].format(f'(original image size {size[0]}x{size[1]})',instruction,history)+'\\n'+SPEC[level+'_format']
 return [{'role':'system','content':SPEC[level+'_system']},{'role':'user','content':[{'type':'text','text':text},{'type':'image','image':im}]}]
''')
def edit(name,old,new):
    p=W/name;s=p.read_text();assert old in s,(name,old);p.write_text(s.replace(old,new))
edit('mixed_native_sample.py', "history=record['history_mobile_use']", "history=record['history']")
edit('mixed_native_sample.py', "target(view['action'])", "target(view['action'],level,view['reasoning'])")
edit('training_backend.py', "=='direct_only'", "=='original_reasoning'")
edit('training_backend.py', "d['variant']=='direct'", "d['variant']=='reasoning'")
edit('training_backend.py', "assert ('<thinking>' not in expected)", "assert '<thinking>' in expected")
edit('training_backend.py', "history=r['history_mobile_use']", "history=r['history']")
edit('patch_training.py', "assert all(s==protocol.SYSTEM for s in d['gui_system_prompt'])", "from native_protocol import SYSTEMS\n        assert all(s in SYSTEMS for s in d['gui_system_prompt'])")
edit('patch_training.py', "for t in d['gui_action_text']:assert set(protocol.answer_fields(t)['fields'])=={'action_type','action_target','value','point_2d'}", "for t,c,s in zip(d['gui_action_text'],d['ai_caption'],d['gui_system_prompt'],strict=True):\n            import json\n            from native_protocol import SPEC,HIGH_FIELDS,LOW_FIELDS\n            level=json.loads(c)['level']\n            assert s==SPEC[level+'_system']\n            assert set(protocol.answer_fields(t)['fields'])==set(HIGH_FIELDS if level=='high' else LOW_FIELDS)")
sys.path.insert(0,str(W))
from native_protocol import target,render_policy,parse
from transformers import AutoTokenizer
from PIL import Image
tok=AutoTokenizer.from_pretrained('/data/mai/benchmarks/gui-libra-base/ready-model',local_files_only=True)
root=Path('/data/mai/assets/gui-libra-81k-sft-20261028-v1/snapshot')
indices={}
for src in sorted({r['source'] for r in rows}):
    r=next(r for r in rows if r['source']==src)
    path=root/r['published_annotation_provenance']['reasoning']
    idx=collections.defaultdict(list)
    for i,d in enumerate(json.loads(path.read_text())):
        ans=next((c['value'] for c in d['conversations'] if c['from'] in ('gpt','assistant')),'')
        user=next((c['value'] for c in d['conversations'] if c['from'] in ('human','user')),'')
        try:
            a=json.loads(re.search(r'<answer>(.*?)</answer>',ans,re.S)[1])
            reasoning=re.search(r'<thinking>(.*?)</thinking>',ans,re.S)[1].strip()
            instruction=re.search(r'Instruction: (.*?)\n\s*Interaction History:',user,re.S)[1].strip()
            assert reasoning
            idx[json.dumps(a,sort_keys=True,ensure_ascii=False)].append(dict(index=i,image=d['image'],reasoning=reasoning,instruction=instruction))
        except (AssertionError,TypeError,ValueError):pass
    indices[src]=(idx,str(path),hashlib.sha256(path.read_bytes()).hexdigest())
keep=[];excluded=[];max_prefix=0;max_target=0;examples={};counts=collections.Counter()
for num,r in enumerate(rows):
    idx,path,sha=indices[r['source']]
    with Image.open(r['current_image']) as im:w,h=im.size
    good=True
    for level,v in r['views'].items():
        hits=[x for x in idx[json.dumps(v['action'],sort_keys=True,ensure_ascii=False)] if r['current_image'].endswith('/'+x['image'])]
        # Match image AND every published action field, including its description.
        if len(hits)!=1:
            excluded.append(dict(id=r['id'],level=level,reason='nonunique_original_reasoning',matches=len(hits)));good=False;break
        hit=hits[0]
        normalize=lambda x:' '.join(x.split())
        if normalize(hit['instruction']) not in (normalize(r['goal']),normalize(v['instruction'])):
            excluded.append(dict(id=r['id'],level=level,reason='reasoning_instruction_mismatch'));good=False;break
        v['reasoning']=hit['reasoning'];v['reasoning_provenance']=dict(path=path,index=hit['index'],sha256=sha,instruction=hit['instruction'])
        text=target(v['action'],level,v['reasoning']);parse(text)
        policy=dict(current_image=None,image_size=(w,h),instruction=v['instruction'],history=r['history'],level=level,goal=r['goal'])
        messages=render_policy(policy);messages[1]['content'].reverse()
        n=len(tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))+round(w/32)*round(h/32)-1
        nt=len(tok(text,add_special_tokens=False)['input_ids'])+1
        if n>10000 or nt>4096:
            excluded.append(dict(id=r['id'],level=level,reason='length',prefix_tokens=n,target_tokens=nt));good=False;break
        max_prefix=max(max_prefix,n);max_target=max(max_target,nt)
        if (r['source'],level) not in examples:
            examples[(r['source'],level)]=dict(id=r['id'],source=r['source'],level=level,messages=messages,target=text,prefix_tokens=n,target_tokens=nt)
    if good:keep.append(r);counts[r['source']]+=1
    if num%10000==0:print('ADMISSION',num,len(keep),len(excluded),flush=True)
allowed={r['id'] for r in keep}
manifest=W/'train-manifest.jsonl'
manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in keep))
phase_counts=[]
for i in (1,2):
    phase=[json.loads(l) for l in (OLD/f'phase{i}.jsonl').open()]
    phase=[d for d in phase if d['id'] in allowed]
    for j,d in enumerate(phase):d.update(draw=j,variant='reasoning')
    (W/f'phase{i}.jsonl').write_text(''.join(json.dumps(d)+'\n' for d in phase));phase_counts.append(len(phase))
steps=sum(phase_counts)//256;warm=max(1,round(steps*.01));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((W/'training-plan.json').read_text())
plan.update(optimizer_steps=steps,warmup_steps=warm,original_draws=sum(phase_counts),used_draws=steps*256,dropped_final_incomplete_batch=sum(phase_counts)%256,samples=len(keep),manifest=str(manifest),manifest_sha256=sha(manifest),phase_sha256=[sha(W/f'phase{i}.jsonl') for i in (1,2)],protocol='High: AndroidWorld core reasoning/action_description; Low: AC reasoning/action_target; original published reasoning only',template_sha256=sha(W/'templates.json'))
(W/'training-plan.json').write_text(json.dumps(plan,indent=2))
edit('training_backend.py','draws[:185600]',f'draws[:{steps*256}]')
edit('native.toml','max_iter = 725',f'max_iter = {steps}')
edit('native.toml','cycle_lengths = [725]',f'cycle_lengths = [{steps}]')
edit('native.toml','warm_up_steps = [7]',f'warm_up_steps = [{warm}]')
cfg=json.loads((W/'config.json').read_text());cfg['sampling']['answer_style']='original_reasoning'
for i,n in enumerate(phase_counts,1):cfg['sampling'][f'phase{i}'].update(draws=n,direct_only=False,reasoning_only=True,reasoning_probability_when_both=1)
cfg['train']['max_iter']=steps;cfg['sources']={s:dict(count=n,weight=n/len(keep)) for s,n in counts.items()}
cfg['data_root']=str(root);cfg['image_root']='/data/mai/assets/gui-libra-81k-sft-20261028-v1/extracted'
cfg['history_steps']='published prior action descriptions; full retained history'
cfg['templates']=dict(high='AndroidWorld core, without GUIDANCE; additional dataset action vocabulary',low='AC Low qwen3 reasoning branch; additional dataset action vocabulary',reasoning='original image/action/instruction-matched published reasoning; no synthesized text',coordinates='0..1000; unavailable [-100,-100]')
(W/'config.json').write_text(json.dumps(cfg,indent=2))
(W/'prompt-examples.json').write_text(json.dumps(list(examples.values()),ensure_ascii=False,indent=2))
report=dict(retained=len(keep),excluded=excluded,source_counts=dict(counts),max_prefix=max_prefix,max_target=max_target,optimizer_steps=steps,templates=cfg['templates'])
(W/'token-admission.json').write_text(json.dumps(report,indent=2))
launch=Path('/data/mai/launch_libra_fulltrain.py').read_text().replace(str(OLD),str(W))
(W/'launch.py').write_text(launch)
for p in W.glob('*.py'):ast.parse(p.read_text())
print('PREPARED',json.dumps({k:v for k,v in report.items() if k!='excluded'}), 'excluded',len(excluded),flush=True)
