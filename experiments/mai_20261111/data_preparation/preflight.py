"""Fail closed: preparation is not a successful training smoke test."""
import hashlib,json,subprocess
from pathlib import Path

def check():
    reasons=[]
    for f in ['original-evidence.jsonl','heldout.json','artifacts/train-manifest.jsonl',
              'artifacts/schedule-phase1.jsonl','artifacts/schedule-phase2.jsonl',
              'training_backend.py','backend-audit.json']:
        if not Path(f).is_file():reasons.append('missing: '+f)
    if not reasons:
        audit=json.loads(Path('backend-audit.json').read_text())
        for key in ['native_action_roundtrip','low_high_prompt_audit','fm_mask_gradient_audit',
                    'distributed_collectives','finite_forward_backward','heldout_exclusion']:
            if audit.get(key) is not True:reasons.append('unverified: '+key)
        for file,sha in audit.get('file_sha256',{}).items():
            if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=sha:reasons.append('changed: '+file)
        for f in ['training_backend.py','launch_mixed.py','mixed_prompt.py','mixed_native_targets.py','mixed_native_sample.py',
                  'unified-template-debug/patch_training.py','unified-template-debug/protocol.py',
                  'artifacts/train-manifest.jsonl','artifacts/schedule-phase1.jsonl','artifacts/schedule-phase2.jsonl',
                  'original-evidence.jsonl','heldout.json','mixture.json']:
            if f not in audit.get('file_sha256',{}):reasons.append('unbound audit: '+f)
    result={'training_ready':not reasons,'blockers':reasons}
    Path('preflight-result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
    return result
if __name__=='__main__':
    raise SystemExit(0 if check()['training_ready'] else 2)
