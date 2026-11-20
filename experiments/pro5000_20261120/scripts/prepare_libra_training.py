from pathlib import Path
import shutil,json,ast,hashlib
old=Path('/data/mai/training/mai-ui-zero2-mlp-strict10k-20261113');new=Path('/data/mai/training/gui-libra-zero2-mlp-strict10k-20261114')
new.mkdir(exist_ok=False)
for p in old.iterdir():
 if p.is_file() and p.suffix in ('.py','.json','.jsonl','.toml') and not p.name.startswith(('pause','check_training','zero2_hook.before')):shutil.copy2(p,new/p.name)
shutil.copy2('/data/mai/libra_native_protocol.py',new/'native_protocol.py')
oldmodel='/data/mai/benchmarks/gb256-20261112/assets/mai-base/model';model='/data/mai/benchmarks/gui-libra-base/model'
for name in ['patch_training.py','training_backend.py']:
 p=new/name;s=p.read_text().replace(oldmodel,model)
 s=s.replace("protocol.answer_fields(t)['name']=='mobile_use'","set(protocol.answer_fields(t)['fields'])=={'action_type','action_target','value','point_2d'}")
 if name=='training_backend.py':
  s=s.replace("messages=render_policy(policy)\n  w,h=image.size","messages=render_policy(policy)\n  messages[1]['content'].reverse()\n  w,h=image.size")
  s=s.replace("enable_thinking=False", "enable_thinking=False")
  s=s.replace("current_image=None,level=c['level']","current_image=None,image_size=c['image_size'],level=c['level']")
  s=s.replace("assert ('Current step query:' in expected)==(c['level']=='low')","assert ('<thinking>' not in expected)")
  s=s.replace("history=c['history'],goal=c['goal']", "history=c['history'],goal=c['goal']")
  s=s.replace("history=r['history_mobile_use'],fm_mask", "image_size=list(reversed(sample['gui_current_screens'][0].shape[-2:])),history=r['history_mobile_use'],fm_mask")
 ast.parse(s);p.write_text(s)
p=new/'mixed_native_sample.py';s=p.read_text().replace("target(view['native_action'])","target(view['action'])");p.write_text(s)
cfg=json.loads((new/'config.json').read_text());cfg['train']['models']=['GUI-Libra-8B'];cfg['protocol']='GUI-Libra native direct answer, full verified action history';(new/'config.json').write_text(json.dumps(cfg,indent=2))
plan=json.loads((new/'training-plan.json').read_text());plan['initial_ar']='GUI-Libra/GUI-Libra-8B original';plan['initial_ar_revision']='15ee285bc1e279eee1b7ea5e953b7dd823e6dbd3';plan['protocol']=cfg['protocol'];(new/'training-plan.json').write_text(json.dumps(plan,indent=2))
(new/'run/audit').mkdir(parents=True)
launcher=Path('/data/mai/launch_mai_fulltrain.py').read_text().replace(str(old),str(new))
Path('/data/mai/launch_libra_fulltrain.py').write_text(launcher)
check=(old/'check_training.py').read_text().replace(str(old),str(new));(new/'check_training.py').write_text(check)
for p in new.glob('*.py'):ast.parse(p.read_text())
print('PREPARED',new)
