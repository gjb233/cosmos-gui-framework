"""Index original GUIAct/CoAT records; do not certify target alignment implicitly."""
import collections
import hashlib
import json
import re
from pathlib import Path

P = Path('/rscratch/wzzheng/experiments/mixed-sft-preparation-20261028-v1')
D = Path('/rscratch/wzzheng/datasets/gui-libra-81k-sft-20261028-v1/extracted')

def canonical(x):
    return json.dumps(x, sort_keys=True, ensure_ascii=False)

def main():
    original = P/'originals/web-multi_train_data.json'
    content = original.read_bytes()
    web = collections.defaultdict(list)
    for index, row in enumerate(json.loads(content)):
        web[row['image_id']].append((index, row))
    web_sha = hashlib.sha256(content).hexdigest()
    images = {r['id']: r for r in map(json.loads, open(P/'artifacts/image-decode-audit.jsonl'))}
    cache = {}
    counts = collections.defaultdict(collections.Counter)
    with (P/'artifacts/original-trajectory-candidates.jsonl').open('w') as out:
        for row in map(json.loads, open(P/'artifacts/inventory.jsonl')):
            source = row['source']
            if source not in ('guiact-web', 'guiact-web-chinese', 'coat-terminal'):
                continue
            result = dict(id=row['id'], source=source, current_image=images[row['id']]['image'],
                          action_alignment_verified=False, training_ready=False)
            if source.startswith('guiact-web'):
                key = Path(row['image']).stem
                matches = web.get(key, [])
                signatures = {canonical(r) for _, r in matches}
                result.update(annotation_path=str(original), annotation_sha256=web_sha,
                              row_indices=[i for i, _ in matches], original_records=[r for _, r in matches])
                if not matches:
                    status = 'missing_original'
                elif len(signatures) != 1:
                    status = 'conflicting_original_records'
                else:
                    status = 'unique_original_content'
                    native = matches[0][1]
                    match = re.fullmatch(r'(uid_record_\d+)_step_(\d+)', native['uid'])
                    if not match:
                        status = 'unrecognized_original_uid'
                    else:
                        episode, step = match.group(1), int(match.group(2))
                        result.update(episode_id=episode, step=step, goal=native['question'],
                                      original_history=native['actions_history'], split_file='train',
                                      terminal=all(a['name']=='answer' for a in native['actions_label']))
                        next_key = episode+'_step_'+str(step+1).zfill(2)
                        successors = web.get(next_key, [])
                        if successors and len({canonical(r) for _, r in successors}) == 1:
                            result['next_original_record'] = successors[0][1]
                            next_path = Path(result['current_image']).with_name(next_key+'.png')
                            result['next_image_candidate'] = str(next_path) if next_path.is_file() else None
            else:
                current = D/'train'/row['image']
                annotation = current.parent/(current.parent.name+'.json')
                if annotation not in cache:
                    raw = annotation.read_bytes()
                    cache[annotation] = (hashlib.sha256(raw).hexdigest(), json.loads(raw))
                sha, records = cache[annotation]
                matches = [r for r in records if r['image_path']==row['image']]
                result.update(annotation_path=str(annotation), annotation_sha256=sha, original_records=matches)
                status = 'unique_original_content' if len(matches)==1 else 'ambiguous_original'
                if len(matches)==1:
                    native = matches[0]
                    result.update(episode_id=str(native['episode_id']), step=int(native['step_id']),
                                  goal=native['instruction'], split_file='train')
                    successors = [r for r in records if str(r['episode_id'])==result['episode_id']
                                  and int(r['step_id'])==result['step']+1]
                    if len(successors)==1:
                        next_path = D/'train'/successors[0]['image_path']
                        result['next_original_record'] = successors[0]
                        result['next_image_candidate'] = str(next_path) if next_path.is_file() else None
            result['status'] = status
            counts[source][status] += 1
            if result.get('next_image_candidate'):
                counts[source]['next_image_candidate_exists'] += 1
            out.write(json.dumps(result, ensure_ascii=False)+'\n')
    report = dict(sources=dict(counts), training_ready=False,
                  note='Original record indexing only. Targets, split exclusions, history and FM alignment remain to be verified.')
    (P/'artifacts/original-trajectory-candidate-report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
