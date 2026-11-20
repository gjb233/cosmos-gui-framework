from pathlib import Path
import json,html,tarfile
from PIL import Image,ImageDraw
r=Path('/data/mai/evaluation/step200-suite-20261118/mai400-fm-visual')
assert json.loads((r/'complete.json').read_text())['samples']==8
rows={x['id']:x for x in json.loads((r/'samples.json').read_text())}
reports=[]
for rank in range(8):
 for kind in ['original-backbone-load','history-fm-prefix']:
  a=json.loads((r/'audit'/f'{kind}-rank{rank}.json').read_text());assert a['passed']
 m=json.loads((r/f'rank-{rank}.json').read_text());assert m['checkpoint']==400 and m['future_pixels_zeroed']
 m['conditioning']='current screenshot + instruction and history';m['action_labels_role']='separate AR branch only, not FM conditioning'
 (r/f'rank-{rank}.json').write_text(json.dumps(m,ensure_ascii=False,indent=2));reports.append(m)
 row=rows[m['record_id']]
 paths=[row['current_image'],row['next_image'],r/f'rank-{rank}-step-000000.png',r/f'rank-{rank}-step-000400.png']
 images=[]
 for path in paths:
  im=Image.open(path).convert('RGB');im.thumbnail((312,680));images.append(im)
 height=max(x.height for x in images)+70
 canvas=Image.new('RGB',(1280,height),'white');d=ImageDraw.Draw(canvas)
 for i,(im,label) in enumerate(zip(images,['Current screenshot','Ground-truth next screenshot','MAI-UI initial prediction','MAI-UI step 400 prediction'])):
  canvas.paste(im,(i*320+(320-im.width)//2,55));d.text((i*320+5,25),label,fill='black')
 canvas.save(r/m['panel'])
page='<meta charset="utf-8"><title>MAI-UI FM: initial vs step 400</title><style>body{max-width:1400px;margin:32px auto;padding:0 20px;font:16px Arial;background:#eef1f5;color:#18202b}section{background:white;padding:16px;border-radius:10px;margin:22px 0}img{width:100%;height:auto}p{line-height:1.65}</style><h1>MAI-UI FM: initial vs step 400</h1><p>8 sources, same 8 training transitions as the GUI-Libra step-200 gallery. Each pair uses the same noise seed and 35 denoising steps. MAI-UI uses its own original training template and base weights.</p><p>Columns: current screenshot / real next screenshot / initial prediction / step-400 prediction. FM conditioning is current screenshot + instruction and history. Action labels belong only to the separate AR branch. Future input pixels are zeroed. These are training diagnostics, not held-out accuracy or an autonomous rollout.</p>'
thumbs=[]
for m in reports:
 page+='<section><h2>'+html.escape(m['source'])+'</h2><img src="'+m['panel']+'"></section>'
 im=Image.open(r/m['panel']);thumbs.append(im.resize((640,im.height//2)))
heights=[max(thumbs[i].height,thumbs[i+1].height)+28 for i in range(0,8,2)]
overview=Image.new('RGB',(1280,sum(heights)),'#eef1f5');d=ImageDraw.Draw(overview);y=0
for pair in range(4):
 for k in range(2):
  i=pair*2+k;x=k*640;d.text((x+5,y+5),reports[i]['source'],fill='black');overview.paste(thumbs[i],(x,y+25))
 y+=heights[pair]
overview.save(r/'overview.png');(r/'index.html').write_text(page)
report=dict(passed=True,model='MAI-UI-8B',checkpoint=400,samples=8,same_samples_as_gui_libra_200=True,steps=[0,400],backbone_loaded_all_ranks=True,trainable_checkpoint_loaded_exactly=True,trainable_dtype='float32',future_pixels_zeroed=True,conditioning='current screenshot + instruction and history; action labels only in separate AR branch',scope='8 training transitions, not held-out accuracy')
(r/'verification.json').write_text(json.dumps(report,indent=2))
with tarfile.open(r/'gallery.tar.gz','w:gz') as tar:
 for name in ['index.html','overview.png','verification.json']+[m['panel'] for m in reports]:tar.add(r/name,arcname=name)
print(json.dumps(report))
