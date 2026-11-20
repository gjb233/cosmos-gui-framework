from pathlib import Path
import shutil,json,ast,hashlib
R=Path('/data/mai/benchmarks/gb256-20261112')
W=Path('/data/mai/training/mai-ui-zero2-mlp-strict10k-20261113')
D=Path('/data/mai/assets/training-ready-strict-10k-20261113')
W.mkdir(parents=True,exist_ok=False)
for p in (R/'zero2-micro8-mlp-strict10k-v3').iterdir():
 if p.is_file():shutil.copy2(p,W/p.name)
shutil.copy2('/data/mai/zero2_fulltrain_hook.py',W/'zero2_hook.py')
counts=[sum(1 for _ in (W/f'phase{i}.jsonl').open()) for i in (1,2)]
total=sum(counts)//256
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan=dict(optimizer_steps=total,warmup_steps=max(1,round(total*.01)),global_batch=256,microbatch=8,gradient_accumulation=4,learning_rate=1e-5,
 original_draws=sum(counts),used_draws=total*256,dropped_final_incomplete_batch=sum(counts)%256,
 samples=61887,manifest=str(D/'train-manifest.jsonl'),manifest_sha256=sha(D/'train-manifest.jsonl'),
 phase_sha256=[sha(W/f'phase{i}.jsonl') for i in (1,2)],
 initial_ar='MAI-UI-8B original',initial_fm='Cosmos3-Nano-DCP base',lora_rank=16,lora_alpha=32,
 train_attention_and_mlp_both_branches=True,activation_checkpointing='full',save_every=50,initial_save_step=2)
(W/'training-plan.json').write_text(json.dumps(plan,indent=2))
p=W/'config.json';cfg=json.loads(p.read_text());cfg['train'].update(max_iter=total,checkpoint_every=50);p.write_text(json.dumps(cfg,indent=2))
p=W/'native.toml';s=p.read_text().replace('max_iter = 4',f'max_iter = {total}').replace('save_iter = 10000','save_iter = 50').replace('cycle_lengths = [3746]',f'cycle_lengths = [{total}]').replace('warm_up_steps = [33]',f'warm_up_steps = [{plan["warmup_steps"]}]');p.write_text(s)
p=W/'training_backend.py';s=p.read_text()
assert 'install(records,draws[:1024])' in s
s=s.replace('install(records,draws[:1024])',f'install(records,draws[:{total*256}])')
s=s.replace("def __len__(self):return len(draws)","def __len__(self):return len(draws)-int(os.environ.get('GUI_START_DRAW','0'))")
s=s.replace("draw=draws[min(index,len(draws)-1)]", "index+=int(os.environ.get('GUI_START_DRAW','0'))\n   assert index<len(draws),index\n   draw=draws[index]")
s=s.replace('expected=draws[rank]',"expected=draws[rank+int(os.environ.get('GUI_START_DRAW','0'))]")
ast.parse(s);p.write_text(s)
(W/'run/audit').mkdir(parents=True)
for p in W.glob('*.py'):ast.parse(p.read_text())
print(json.dumps(plan,indent=2))
