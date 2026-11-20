from pathlib import Path
import json,statistics
o=Path(__file__).resolve().parent
a=json.loads((o/'fsdp-baseline/active.json').read_text());b=Path(a['out'])/'adapters'
def rows(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def report(root):
 ranks=[{x['step']:x for x in rows(root/f'timing-rank{r}.jsonl')} for r in range(7)]
 seconds=[max(r[s]['seconds'] for r in ranks) for s in (352,353,354)]
 return dict(steps=[352,353,354],seconds=seconds,total_seconds=sum(seconds),mean_seconds=statistics.mean(seconds),median_seconds=statistics.median(seconds),max_seconds=max(seconds),under60=sum(t<60 for t in seconds)/3,samples_per_second=168/sum(seconds),rank_records=[[r[s] for s in (351,352,353,354)] for r in ranks])
base=[x for r in range(7) for x in rows(b/f'draws-rank{r}.jsonl') if 20800<=x['draw']<21024]
pipe=[x for x in rows(o/'stream/draws.jsonl') if 20800<=x['draw']<21024]
for seq in (base,pipe):
 assert len(seq)==224 and {x['draw'] for x in seq}==set(range(20800,21024))
key=lambda x:(x['draw'],x['id'],x['level'])
assert sorted(map(key,base))==sorted(map(key,pipe))
f=report(b);p=report(o)
result=dict(matched_draws=224,warmup_draws=56,timed_draws=168,start_checkpoint=350,global_batch=56,microbatch=4,fsdp=f,pipeline_1f1b=p,speedup=f['total_seconds']/p['total_seconds'],time_reduction_percent=100*(1-p['total_seconds']/f['total_seconds']),timing_scope='synchronized end-to-end optimizer step including input preparation, excluding model startup and checkpoint saving')
if (o/'saves-rank0.jsonl').exists():
 result['pipeline_save354_seconds']=max(next(x['seconds'] for x in rows(o/f'saves-rank{r}.jsonl') if x['step']==354) for r in range(7))
(o/'benchmark-result.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ('fsdp','pipeline_1f1b')},indent=2))
print('FSDP',f['seconds'],'1F1B',p['seconds'])
