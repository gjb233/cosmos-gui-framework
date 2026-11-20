from pathlib import Path
import json,html,base64,io
from PIL import Image,ImageDraw,ImageFont
o=Path(__file__).resolve().parent
items=json.loads((o/'selection.json').read_text())
b=o.parent/'evaluation-mai-70284-two-benchmarks-a23-20261110'
predictions={}
for step in [100,200,300]:
 predictions[step]={r['id']:r for r in map(json.loads,(b/f'joint-{step}/grounding/predictions.jsonl').open())}
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
cards=[]
for i,item in enumerate(items,1):
 row=item['row'];w,h=320,712
 canvas=Image.new('RGB',(w*6, h+74),'white');draw=ImageDraw.Draw(canvas)
 paths=[row['current_image']]+[str(o/f'step-{s:06d}'/row['id'].replace(':','-')/'frame-4.png') for s in [100,200,300]]+[row['next_image'],item['after_next_image']]
 labels=['Current + action t','FM step 100','FM step 200','FM step 300','True next + action t+1','True after next']
 for j,(path,label) in enumerate(zip(paths,labels)):
  im=Image.open(path).convert('RGB');origw,origh=Image.open(row['current_image']).size
  im=im.resize((w,h),Image.Resampling.LANCZOS)
  dr=ImageDraw.Draw(im)
  action=item['action'] if j==0 else item['next_action'] if j<5 else {}
  if action.get('action_type')=='click':
   x=action['x']/origw*w;y=action['y']/origh*h
   color='#e52631' if j==0 else '#0075ed'
   dr.ellipse((x-12,y-12,x+12,y+12),outline='white',width=6)
   dr.ellipse((x-12,y-12,x+12,y+12),outline=color,width=3)
   dr.line((x-18,y,x+18,y),fill=color,width=2);dr.line((x,y-18,x,y+18),fill=color,width=2)
  canvas.paste(im,(j*w,74));draw.text((j*w+8,42),label,font=font,fill='black')
 draw.text((8,8),row['id']+' | Same high-level goal, history and seed 42 | 20 FM sampling steps',font=font,fill='black')
 file=o/f'comparison-{i}.jpg';canvas.save(file,quality=95)
 raw=io.BytesIO();canvas.save(raw,format='JPEG',quality=93)
 actiontext=html.escape(json.dumps(item['action'],ensure_ascii=False))
 nexttext=html.escape(json.dumps(item['next_action'],ensure_ascii=False))
 ars=''.join('<tr><td>'+str(s)+'</td><td><code>'+html.escape(predictions[s][row['id']]['response']+'\n换算原图坐标: '+predictions[s][row['id']].get('ans',{}).get('output','未知'))+'</code></td></tr>' for s in [100,200,300])
 cards.append(f'<section><h2>样例 {i} · {row["id"]}</h2><p>{html.escape(row["goal"])}</p><img src="data:image/jpeg;base64,{base64.b64encode(raw.getvalue()).decode()}"><p>当前动作（红色）<code>{actiontext}</code></p><p>下一步动作（蓝色）<code>{nexttext}</code></p><h3>三个检查点的当前 GUI 动作预测（独立 AR 评测）</h3><p>工具输出坐标为 0–999 归一化坐标，下方同时给出原图像素坐标；红/蓝标记使用真实动作的原图坐标。AR 动作没有额外输入本次 FM 采样。</p><table><tr><th>检查点</th><th>AR 动作输出</th></tr>{ars}</table></section>')
(o/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>FM 测试样例 100 / 200 / 300</title><style>body{font:16px system-ui;margin:32px;background:#f4f5f7;color:#18212c}section{background:white;padding:24px;margin:28px 0;border-radius:12px}img{width:100%;height:auto}code{white-space:pre-wrap}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}p{line-height:1.6}</style><h1>FM 未来帧预测与 GUI 动作对照</h1><p>AndroidControl 官方测试集四组连续轨迹。输入使用高层目标、当前原图及已执行动作历史。FM 输入不含当前正确动作、下一步动作或未来截图。三个检查点均用 seed 42、20 步采样、guidance 1，FM 长边 640 对齐 32。红圈标当前动作，蓝圈标下一步动作在各预测和真实下一帧上的对应位置。最后一列为执行下一步动作后的真实截图，仅作轨迹参照，并非模型的第二步预测。</p><p>四例取自固定官方分片的前四个高层测试条目，未按预测质量筛选。已按官方 test 划分及训练 transition_provenance 核对，四个 episode 与训练集的 5485 个 AndroidControl episode 无重叠。GUI 动作输出复用相同样例的既有评测结果。这是四例定性观察，不能代替全测试集准确率；第150步后联合训练由双卡迁至七卡，global batch由64变为56。</p>'+''.join(cards))
print('GALLERY_READY',flush=True)
