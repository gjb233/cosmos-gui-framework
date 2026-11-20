
from pathlib import Path
import subprocess,json
W=Path('/data/mai/training/gui-libra-awhigh-aclow-ceonly-20261118')
py='/data/mai/envs/mai/bin/python'
subprocess.run([py,'-u',str(W/'validate_processor.py')],check=True)
assert json.loads((W/'processor-validation.json').read_text())['passed']
p=W/'training-plan.json';plan=json.loads(p.read_text());plan.update(ready_to_launch=True,pending_checks=[],runtime_assertion='total == CE; FM gradients and parameter updates == 0');p.write_text(json.dumps(plan,indent=2))
p=subprocess.Popen([py,'-u',str(W/'launch.py')],stdin=subprocess.DEVNULL,stdout=(W/'run/launcher.log').open('a'),stderr=subprocess.STDOUT,start_new_session=True)
print('LAUNCHER',p.pid,flush=True)
