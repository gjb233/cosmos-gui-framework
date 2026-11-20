from pathlib import Path
import json,collections
A=Path('/data/mai/evaluation/assets-20261113');T=Path('/data/mai/assets/training-ready-strict-10k-20261113/train-manifest.jsonl')
train=[json.loads(l) for l in T.open()];trainids={r['id'] for r in train};trainsha={r['current_image_sha256'] for r in train}
bindings={}
for source in ('android_control','mind2web'):
 bindings[source]={str(b['episode_id']) for b in map(json.loads,(Path('/data/mai/trajectories/20261111')/source/'sample-bindings.jsonl').open()) if b['sample_id'] in trainids}
report={}
for bench in ('grounding','mind2web'):
 rows=[json.loads(l) for l in (A/bench/'manifest.jsonl').open()];assert len({r['id'] for r in rows})==len(rows)
 images=[dict(path=r['image'],sha256=r['image_sha256']) for r in rows] if bench=='grounding' else [im for r in rows for im in r['images']]
 missing=[im['path'] for im in images if not Path(im['path']).is_file()];assert not missing,missing[:5]
 evaleps={str(r['episode_id']) for r in rows} if bench=='grounding' else {r['annotation_id'] for r in rows}
 source='android_control' if bench=='grounding' else 'mind2web'
 overlap=sorted(evaleps & bindings[source])
 report[bench]=dict(samples=len(rows),image_references=len(images),unique_images=len({im['sha256'] for im in images}),train_image_hash_overlap=len({im['sha256'] for im in images}&trainsha),train_episode_id_overlap=overlap)
 assert not overlap,(bench,overlap[:10])
 assert report[bench]['train_image_hash_overlap']==0,bench
(A/'readiness.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
