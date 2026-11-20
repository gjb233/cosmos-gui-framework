from pathlib import Path
import json,time,subprocess,sys,zipfile,hashlib
import numpy as np
from PIL import Image
o=Path(__file__).resolve().parent
while not (o/'complete.json').exists():
 if any((o/f'exit-{s}.json').exists() and json.loads((o/f'exit-{s}.json').read_text())['exit_code'] for s in [200,300]):
  raise RuntimeError('inference failed')
 time.sleep(10)
items=json.loads((o/'selection.json').read_text())
audit=[]
for s in [100,200,300]:
 assert json.loads((o/f'exit-{s}.json').read_text())['exit_code']==0
 assert json.loads((o/f'step-{s:06d}/loading-audit.json').read_text())['restored_parameters']==584
 for x in items:
  d=o/f'step-{s:06d}'/x['row']['id'].replace(':','-')
  m=json.loads((d/'metadata.json').read_text())
  assert m['step']==s and not m['future_target_supplied'] and not m['current_gold_action_supplied']
  a=Image.open(d/'frame-4.png').convert('RGB')
  pred=np.array(a,dtype=float)/255
  cur=np.array(Image.open(x['row']['current_image']).convert('RGB').resize(a.size,Image.Resampling.BICUBIC),dtype=float)/255
  nex=np.array(Image.open(x['row']['next_image']).convert('RGB').resize(a.size,Image.Resampling.BICUBIC),dtype=float)/255
  audit.append(dict(step=s,id=x['row']['id'],mae_to_current=float(abs(pred-cur).mean()),mae_to_true_next=float(abs(pred-nex).mean()),unchanged_frame_baseline_mae_to_next=float(abs(cur-nex).mean()),image_sha256=hashlib.sha256((d/'frame-4.png').read_bytes()).hexdigest(),**{k:m[k] for k in ['seed','sampling_steps','guidance','seconds','latent_sha256']}))
for x in items:
 a=[r for r in audit if r['id']==x['row']['id']]
 assert len({r['latent_sha256'] for r in a})==3
(o/'comparison-audit.json').write_text(json.dumps(audit,indent=2))
subprocess.run([sys.executable,str(o/'build_gallery.py')],check=True)
with zipfile.ZipFile(o/'fm-test-comparisons.zip','w',zipfile.ZIP_DEFLATED) as z:
 for f in [o/'index.html',o/'selection.json',o/'split-audit.json',o/'comparison-audit.json',*sorted(o.glob('comparison-*.jpg'))]:
  z.write(f,str(f.relative_to(o)))
 for f in sorted((o/'samples').rglob('*.png')):z.write(f,str(f.relative_to(o)))
 for s in [100,200,300]:
  for f in (o/f'step-{s:06d}').rglob('*'):
   if f.is_file() and (f.name in ['frame-0.png','frame-4.png','metadata.json','loading-audit.json']):z.write(f,str(f.relative_to(o)))
(o/'gallery-ready.json').write_text(json.dumps(dict(samples=4,checkpoints=[100,200,300],predictions=12,verified=True)))
print('FINALIZED',flush=True)
