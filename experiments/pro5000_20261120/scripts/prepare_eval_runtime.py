from pathlib import Path
import json,ast
A=Path('/data/mai/evaluation/assets-20261113')
for bench in ('grounding','mind2web'):
 d=A/bench
 p=d/'native_protocol.py';s=p.read_text();start="UP=Path('/rscratch/wzzheng/experiments/gui-mai-official-alignment-20261006-v1/upstream/MAI-UI/src/prompt.py')"
 assert start in s;s=s.replace(start,"UP=Path(__file__).resolve().parent/'prompt.py'");ast.parse(s);p.write_text(s)
 p=d/'native.py';s=p.read_text();s=s.replace("UP=Path('/rscratch/wzzheng/experiments/gui-mai-official-alignment-20261006-v1/upstream/MAI-UI')","UP=Path(__file__).resolve().parent")
 ast.parse(s);p.write_text(s)
 counts=sum(1 for _ in (d/'manifest.jsonl').open());assert counts==({'grounding':796,'mind2web':3349}[bench])
 print(bench,counts,flush=True)
