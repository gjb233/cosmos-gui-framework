from pathlib import Path
import subprocess,json,time,os,signal,fcntl
W=Path('/data/mai/training/mai-ui-zero2-mlp-strict10k-20261113');T=W/'run'
R=Path('/data/mai/evaluation/vllm-ckpts-through-400-20261113');R.mkdir(parents=True,exist_ok=True)
lock=(R/'controller.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
active=json.loads((T/'active.json').read_text());pid=active['pid']
identity=Path(f'/proc/{pid}/stat').read_text().split()[21]
def status(phase,**extra):
    tmp=R/'controller.tmp';tmp.write_text(json.dumps(dict(phase=phase,at=time.time(),training_pid=pid,**extra)));tmp.replace(R/'controller.json')
try:
    status('waiting_for_checkpoint_400')
    while True:
        assert Path(f'/proc/{pid}').exists(),'training exited before checkpoint400'
        assert Path(f'/proc/{pid}/stat').read_text().split()[21]==identity,'training process identity changed'
        state=json.loads((T/'checkpoints/latest-complete.json').read_text())
        if state['completed_steps']>=400:
            assert state['completed_steps']==400,state
            ck=T/'checkpoints'/state['tag']
            assert len(list(ck.glob('*optim_states.pt')))==8
            assert len(list(ck.glob('rng-rank*.pt')))==8
            assert (ck/'mp_rank_00_model_states.pt').is_file()
            assert (T/'ar-adapters/ar-step-000400.ready.json').is_file()
            cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\x00',' ')
            assert str(W/'train_entry.py') in cmd and os.getpgid(pid)==pid
            (T/'pause-request-400.json').write_text(json.dumps(dict(reason='user_requested_evaluation_at_step_400',checkpoint=state['tag'],pid=pid,signal_sent_at=time.time())))
            os.killpg(pid,signal.SIGTERM)
            status('waiting_for_training_shutdown',checkpoint=state['tag'])
            break
        time.sleep(2)
    deadline=time.time()+300
    while Path(f'/proc/{pid}').exists():
        assert time.time()<deadline,'training shutdown timeout'
        time.sleep(2)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU still occupied'
    status('merging_checkpoints')
    subprocess.run(['/data/mai/envs/mai/bin/python','/data/mai/eval400/merge_eval_models.py'],check=True)
    status('evaluating')
    subprocess.run(['/data/mai/envs/mai/bin/python','/data/mai/eval400/run_vllm_eval.py'],check=True)
    assert json.loads((R/'complete.json').read_text())['completed']
    status('completed_training_remains_paused')
except Exception as e:
    status('failed',error=repr(e));raise
