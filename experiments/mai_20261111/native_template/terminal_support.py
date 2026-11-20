"""Run-local terminal CE support. No real future target exists for terminal rows."""
import json,inspect,textwrap,os
from pathlib import Path
from types import SimpleNamespace

def load_manifest(path):
    import protocol
    rows=[];keys=set()
    for line in Path(path).read_text().splitlines():
        d=json.loads(line);key=(d['episode_id'],d['step']);assert key not in keys;keys.add(key)
        assert d['split']=='train' and d['goal'].strip() and Path(d['current_image']).is_file()
        terminal=d['action']['type']=='terminate'
        assert (d['next_image'] is None)==terminal
        if not terminal:assert Path(d['next_image']).is_file()
        for a in [d['action'],*d['previous_actions']]:protocol.parse(protocol.target(a))
        rows.append(SimpleNamespace(**d))
    return rows

def apply_fm_mask(values,mask):
    import torch
    m=torch.as_tensor(mask,device=values.device,dtype=values.dtype)
    assert values.ndim==m.ndim==1 and values.shape==m.shape
    return values*m

def install():
    import torch,numpy as np,protocol
    import torch.distributed as dist
    from PIL import Image
    from cosmos_framework.gui_mot import joint_dataset as jd,mixed_supervision as ms,native_model as nm
    jd.load_manifest=load_manifest
    oldinit=ms.MixedJointPolicyDataset.__init__
    def init(self,*args,**kwargs):
        oldinit(self,*args,**kwargs)
        if os.environ.get('GUI_RUN_NAME')=='smoke-B':
            terminal=[e for e in self.entries if self.windows[e[0]][0].next_image is None][:2]
            self.entries=terminal+[e for e in self.entries if e not in terminal]
    ms.MixedJointPolicyDataset.__init__=init
    oldget=ms.MixedJointPolicyDataset.__getitem__
    def get(self,index):
        source,reasoning=self.entries[index];assert not reasoning
        row=self.windows[source][0]
        if row.next_image is not None:return oldget(self,index)
        assert row.action['type']=='terminate' and self.video_only and self.native_ar_text
        with Image.open(row.current_image) as im:
            im=im.convert('RGB');prefix=torch.from_numpy(np.array(im,copy=True)).permute(2,0,1)
        import math
        h,w=prefix.shape[-2:];scale=64/max(h,w)
        hw=(max(32,math.ceil(h*scale/32)*32),max(32,math.ceil(w*scale/32)*32))
        current=jd.read_screen(row.current_image,hw)
        # Fixed zeros are packing placeholders, not copied screenshots or supervised transitions.
        sample=jd.make_sample(current,row.instruction,future=None,action=row.action,previous_actions=row.previous_actions,native_ar_text=True,video_only=True,prefix_image=prefix)
        sample['ai_caption']=protocol.context(row.goal,row.previous_actions,row.instruction)
        sample['gui_system_prompt']=protocol.SYSTEM
        assert torch.count_nonzero(sample['video'][:,1:])==0
        return sample
    ms.MixedJointPolicyDataset.__getitem__=get
    oldprepare=nm.GuiJointModel._prepare_gui_prefix
    def prepare(self,batch):
        targets=batch['gui_action_text']
        while isinstance(targets,list) and len(targets)==1 and isinstance(targets[0],list):targets=targets[0]
        self._terminal_fm_mask=[float(protocol.parse(t)['action']!='terminate') for t in targets]
        return oldprepare(self,batch)
    nm.GuiJointModel._prepare_gui_prefix=prepare
    # Modify the original loss body locally, before its global mean and CE addition.
    # Retain all ranks' collectives and graph participation; zero only terminal FM gradients.
    original=nm.GuiJointModel._compute_losses
    # patch_training wraps the original loss; recover the class source method directly.
    tree=__import__('ast').parse(Path(nm.__file__).read_text())
    cls=next(x for x in tree.body if isinstance(x,__import__('ast').ClassDef) and x.name=='GuiJointModel')
    method=next(x for x in cls.body if isinstance(x,__import__('ast').FunctionDef) and x.name=='_compute_losses')
    source=textwrap.dedent(__import__('ast').get_source_segment(Path(nm.__file__).read_text(),method))
    line='vision_loss = global_sample_mean(vision_per_sample, group=group)'
    assert source.count(line)==1
    indent=next(x[:len(x)-len(x.lstrip())] for x in source.splitlines() if line in x)
    source=source.replace(line,'vision_per_sample = apply_fm_mask(vision_per_sample, self._terminal_fm_mask)\n'+indent+line)
    namespace=dict(vars(nm),apply_fm_mask=apply_fm_mask)
    exec(source,namespace);masked=namespace['_compute_losses']
    def losses(self,out_net,*args,**kwargs):
        result=masked(self,out_net,*args,**kwargs)
        mask=self._terminal_fm_mask;assert len(mask)==1
        terminal=mask[0]==0
        if terminal:assert result[1]['flow_matching_loss_vision_per_instance'].eq(0).all()
        assert torch.isfinite(result[0]).all() and torch.isfinite(result[1]['gui_ar_ce']).all()
        if os.environ.get('GUI_RUN_NAME')=='B-ce-fm':
            out=Path(os.environ['GUI_ADAPTER_EXPORT_ROOT']);out.mkdir(parents=True,exist_ok=True)
            record=dict(terminal=terminal,fm_weight=mask[0],fm=float(result[1]['flow_matching_loss_vision_per_instance'][0].detach()),ce=float(result[1]['gui_ar_ce'].detach()),finite=True)
            with (out/f'terminal-loss-rank{dist.get_rank()}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        if os.environ.get('GUI_RUN_NAME')=='smoke-B':
            out=Path(os.environ['GUI_ADAPTER_EXPORT_ROOT']);out.mkdir(parents=True,exist_ok=True)
            record=dict(terminal=terminal,fm_weight=mask[0],fm=float(result[1]['flow_matching_loss_vision_per_instance'][0].detach()),ce=float(result[1]['gui_ar_ce'].detach()),finite=True)
            with (out/f'terminal-loss-rank{dist.get_rank()}.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if terminal and not getattr(self,'_terminal_ce_captured',False):
                self._terminal_ce_captured=True
                torch.save(dict(labels=out_net['gui_ar_labels'].detach().cpu(),weights=out_net['gui_ar_token_weights'].detach().cpu(),logits=out_net['gui_ar_logits'].detach().float().cpu()),out/f'terminal-ce-forward-rank{dist.get_rank()}.pt')
            if dist.get_rank()==0 and not getattr(self,'_ce_debug_captured',False):
                self._ce_debug_captured=True
                torch.save({'prefix':[{k:v.detach().cpu() for k,v in d.items()} for d in self._gui_prefix], 'logits':out_net['gui_ar_logits'].detach().float().cpu(),'labels':out_net['gui_ar_labels'].detach().cpu(),'weights':out_net['gui_ar_token_weights'].detach().cpu()},out/'first-ce-forward.pt')
        return result
    nm.GuiJointModel._compute_losses=losses
