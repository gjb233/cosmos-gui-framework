import subprocess,json,os,sys
from pathlib import Path
r=Path(__file__).resolve().parent
cmd=json.loads((r/'argv.json').read_text())
with (r/'swift-sft.log').open('x') as log:
 job=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
 (r/'active.json').write_text(json.dumps(dict(pid=job.pid,controller=os.getpid())))
 rc=job.wait()
(r/'exit.json').write_text(json.dumps(dict(code=rc)))
sys.exit(rc)
