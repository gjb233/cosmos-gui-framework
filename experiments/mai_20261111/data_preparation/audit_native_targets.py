import pathlib,json,hashlib
from transformers import AutoTokenizer
from mixed_prompt import target
from mixed_native_targets import tokenize_targets
p=pathlib.Path(__file__).resolve().parent
tok=AutoTokenizer.from_pretrained('/rscratch/wzzheng/code/cosmos-gui/.local/models/GUI-Libra-8B',local_files_only=True)
total=0;maxlen=0;fail=[]
for r in map(json.loads,open(p/'artifacts/action-pair-audit.jsonl')):
 if not r.get('parsed'):continue
 try:
  text=target(r['action']);ids,w=tokenize_targets(tok,[text],True)
  assert json.loads(tok.decode(ids[0][:-1]))=={k:r['action'][k] for k in json.loads(text)}
  assert abs(sum(w[0])-1)<1e-8
  maxlen=max(maxlen,len(ids[0]));total+=1
 except Exception as e:fail.append(dict(id=r['id'],error=str(e)))
report=dict(passed=not fail,total=total,max_target_tokens=maxlen,failures=fail,
 tokenizer_path='/rscratch/wzzheng/code/cosmos-gui/.local/models/GUI-Libra-8B',
 target_module_sha256=hashlib.sha256((p/'mixed_native_targets.py').read_bytes()).hexdigest(),training_ready=False)
(p/'artifacts/native-target-token-audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
