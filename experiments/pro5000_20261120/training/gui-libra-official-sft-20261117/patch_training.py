def install():
    import torch
    from torch.utils.checkpoint import checkpoint
    from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import Qwen3VLTextMLP
    original_mlp = Qwen3VLTextMLP.forward
    def bounded_mlp(self, x):
        if x.shape[-2] <= 1024:
            return original_mlp(self, x)
        def project(part):
            return original_mlp(self, part)
        return torch.cat([project(part)
                          for part in x.split(1024, dim=-2)], dim=-2)
    Qwen3VLTextMLP.forward = bounded_mlp
    from cosmos_framework.model.generator.reasoner.qwen3_vl.qwen3_vl import Qwen3VLTextRMSNorm
    original_norm = Qwen3VLTextRMSNorm.forward
    def bounded_norm(self, hidden_states):
        if hidden_states.numel() <= 1048576:
            return original_norm(self, hidden_states)
        axis = max(range(hidden_states.ndim - 1), key=lambda i: hidden_states.shape[i])
        per_slice = hidden_states.numel() // hidden_states.shape[axis]
        chunk = max(1, 1048576 // per_slice)
        def normalize(part):
            return original_norm(self, part)
        return torch.cat([normalize(part)
                          for part in hidden_states.split(chunk, dim=axis)], dim=axis)
    Qwen3VLTextRMSNorm.forward = bounded_norm
    import types
    from cosmos_framework.model.generator.reasoner.qwen3_vl import qwen3_vl as qmod
    # Keep imported references working by patching the same function object.
    fn=qmod.apply_rotary_pos_emb
    original=types.FunctionType(fn.__code__,fn.__globals__,fn.__name__,fn.__defaults__,fn.__closure__)
    def chunked_rope(q,k,cos,sin,position_ids=None,unsqueeze_dim=1):
        if q.ndim!=4 or cos.ndim!=3 or unsqueeze_dim not in (1,2) or q.numel()<=1048576:
            return _gui_original_rope(q,k,cos,sin,position_ids,unsqueeze_dim)
        axis=2 if unsqueeze_dim==1 else 1
        if cos.shape[1]!=q.shape[axis] or sin.shape[1]!=q.shape[axis]:
            return _gui_original_rope(q,k,cos,sin,position_ids,unsqueeze_dim)
        qr=[];kr=[]
        size=max(1,1048576//(q.numel()//q.shape[axis]))
        for qt,kt,ct,st in zip(q.split(size,axis),k.split(size,axis),cos.split(size,1),sin.split(size,1),strict=True):
            lhs,rhs=_gui_original_rope(qt,kt,ct,st,None,unsqueeze_dim)
            qr.append(lhs);kr.append(rhs)
        return torch.cat(qr,dim=axis),torch.cat(kr,dim=axis)
    fn.__globals__['_gui_original_rope']=original
    # The replacement uses the module's torch global, not an install-scope closure.
    import textwrap,inspect
    ns=dict(fn.__globals__)
    exec(textwrap.dedent(inspect.getsource(chunked_rope)),ns)
    fn.__code__=ns['chunked_rope'].__code__

    import protocol
    from cosmos_framework.gui_mot import mai_weights
    original_load = mai_weights.load_gui_reasoner
    def audited_load(language_model,directory,**kwargs):
        import os,json
        from pathlib import Path
        import torch.distributed as dist
        expected=Path('/data/mai/benchmarks/gui-libra-base/ready-model')
        assert Path(directory).resolve()==expected.resolve()
        assert kwargs.get('copy_generator',False) is False
        count=original_load(language_model,directory,**kwargs)
        assert count>288
        out=Path(os.environ['GUI_ADAPTER_EXPORT_ROOT']);out.mkdir(parents=True,exist_ok=True)
        (out/f'original-backbone-load-rank{dist.get_rank()}.json').write_text(json.dumps(dict(passed=True,model=str(expected),loaded_tensors=count,copy_generator=False,cosmos_source=os.environ['BASE_CHECKPOINT_PATH'])))
        return count
    mai_weights.load_gui_reasoner=audited_load
    from cosmos_framework.gui_mot import mixed_supervision as ms,multimodal as mm,native_model as nm
    from PIL import Image
    base_init=ms.JointPolicyDataset.__init__
    def init_all(self,*args,**kwargs):
        base_init(self,*args,**kwargs)
        assert self.horizon==1
        self.windows=[[row] for row in self.records]
    ms.JointPolicyDataset.__init__=init_all
    from cosmos_framework.gui_mot import joint_dataset as jd
    jd.official_action_text=lambda action,instruction:protocol.target(action)
    jd.official_history_action=lambda action:action
    jd.official_query=lambda row,size:row['step_instruction']
    old=ms.JointPolicyDataset.__getitem__
    def get(self,index):
        source,reasoning=self.entries[index];assert not reasoning
        sample=old(self,source)
        row=self.windows[source][0]
        import torch
        # Reduce only the VAE/FM video; keep the separate original ViT prefix untouched.
        video=sample['video'];h,w=video.shape[-2:]
        import math
        scale=64/max(h,w)
        th=max(32,math.ceil(h*scale/32)*32);tw=max(32,math.ceil(w*scale/32)*32)
        frames=[]
        for j in range(video.shape[1]):
            im=Image.fromarray(video[:,j].permute(1,2,0).numpy())
            import numpy as np
            frames.append(torch.from_numpy(np.array(im.resize((tw,th),Image.Resampling.BICUBIC))).permute(2,0,1))
        sample['video']=torch.stack(frames,dim=1).contiguous()
        sample['image_size']=torch.tensor([th,tw,th,tw])
        sample['padding_mask']=torch.zeros(1,th,tw)
        sample['ai_caption']=protocol.context(row.goal,row.previous_actions,row.instruction);sample['gui_system_prompt']=protocol.SYSTEM
        sample['gui_action_text']=protocol.target(row.action)
        return sample
    ms.MixedJointPolicyDataset.__getitem__=get
    def prepare(batch,**kw):
        d=dict(batch)
        for key in ('gui_action_text','ai_caption','gui_system_prompt'):
            v=d[key]
            while isinstance(v,list) and len(v)==1 and isinstance(v[0],list):v=v[0]
            d[key]=v
        from native_protocol import SYSTEMS
        assert all(s in SYSTEMS for s in d['gui_system_prompt'])
        for t,c,s in zip(d['gui_action_text'],d['ai_caption'],d['gui_system_prompt'],strict=True):
            import json
            from native_protocol import SPEC,HIGH_FIELDS,LOW_FIELDS
            level=json.loads(c)['level']
            assert s==json.loads(c)['official_system']
            assert set(protocol.answer_fields(t)['fields'])==set(HIGH_FIELDS if level=='high' else LOW_FIELDS)
        return d
    ms.prepare_grouped_batch=prepare;ms.tokenize_grouped_targets=protocol.grouped
    def prefix(processor,batch):
        screens=batch['gui_current_screens']
        while isinstance(screens,list) and len(screens)==1 and isinstance(screens[0],list):screens=screens[0]
        if hasattr(screens,'ndim') and screens.ndim==4:screens=list(screens)
        screens=[x.squeeze(0) if hasattr(x,'ndim') and x.ndim==4 and x.shape[0]==1 else x for x in screens]
        return [protocol.prefix(processor,c,Image.fromarray(x.detach().cpu().permute(1,2,0).numpy())) for x,c in zip(screens,batch['ai_caption'],strict=True)]
    mm.prepare_prefix=prefix;nm.prepare_prefix=prefix
    # Select HF vision BEFORE construction/FSDP wrapping, preserving sharding ownership.
    from cosmos_framework.model.generator.mot import unified_mot
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    unified_mot.Qwen3VLVisionModel=Qwen3VLVisionModel
    from cosmos_framework.gui_mot.training_artifacts import NativeTrainingArtifacts
    previous_start=NativeTrainingArtifacts.on_train_start
    def aligned_start(self,model,iteration=0):
        import os,json,torch
        import torch.distributed as dist
        from pathlib import Path
        visual=model.net.language_model.visual
        assert isinstance(visual,Qwen3VLVisionModel)
        assert all(not p.requires_grad for p in visual.parameters())
        # Cosmos meta->to_empty initialization does not recognize the HF rotary class.
        # Recreate its nonpersistent buffer exactly as HF CPU construction does.
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionRotaryEmbedding
        rotary_dim=(visual.config.hidden_size//visual.config.num_heads)//2
        reference=Qwen3VLVisionRotaryEmbedding(rotary_dim).inv_freq
        old_buffer=visual.rotary_pos_emb.inv_freq
        buffer_matched_before=torch.equal(old_buffer.detach().cpu(),reference)
        visual.rotary_pos_emb.register_buffer('inv_freq',reference.to(device=next(visual.parameters()).device),persistent=False)
        assert torch.equal(visual.rotary_pos_emb.inv_freq.cpu(),reference)
        assert visual.rotary_pos_emb.inv_freq.dtype==torch.float32
        if dist.get_rank()==0:
            out=Path(os.environ['GUI_ADAPTER_EXPORT_ROOT']);out.mkdir(parents=True,exist_ok=True)
            (out/'visual-implementation-audit.json').write_text(json.dumps({'class':type(visual).__module__+'.'+type(visual).__name__,'frozen':True,'rotary_dtype':str(visual.rotary_pos_emb.inv_freq.dtype),'installed_before_fsdp':True,'rotary_buffer_matched_before':buffer_matched_before,'rotary_buffer_exact_after':True}))
        return previous_start(self,model,iteration)
    NativeTrainingArtifacts.on_train_start=aligned_start
    old_losses=nm.GuiJointModel._compute_losses
    def losses(self,out_net,*args,**kwargs):
        import os,torch
        import torch.distributed as dist
        from pathlib import Path
        if os.environ.get('GUI_RUN_NAME')=='smoke-B' and not getattr(self,'_ce_debug_captured',False):
            self._ce_debug_captured=True
            if dist.get_rank()==0:
                torch.save({'prefix':[{k:v.detach().cpu() for k,v in d.items()} for d in self._gui_prefix], 'logits':out_net['gui_ar_logits'].detach().float().cpu(),'labels':out_net['gui_ar_labels'].detach().cpu(),'weights':out_net['gui_ar_token_weights'].detach().cpu()},Path(os.environ['GUI_ADAPTER_EXPORT_ROOT'])/'first-ce-forward.pt')
        return old_losses(self,out_net,*args,**kwargs)
    nm.GuiJointModel._compute_losses=losses

    # Verify the history-bearing prefix at the actual FM denoise entry.
    old_prepare=nm.GuiJointModel._prepare_gui_prefix
    def audited_prepare(self,batch):
        result=old_prepare(self,batch)
        import json
        v=batch['ai_caption']
        while isinstance(v,list) and len(v)==1:v=v[0]
        self._history_fm_contexts=[json.loads(x) for x in v];self._history_fm_context=self._history_fm_contexts[0]
        return result
    nm.GuiJointModel._prepare_gui_prefix=audited_prepare
    old_denoise=nm.GuiJointModel.denoise
    def audited_denoise(self,*args,**kwargs):
        import os,json,hashlib
        import torch.distributed as dist
        from pathlib import Path
        if not getattr(self,'_history_fm_verified',False):
            c=self._history_fm_context
            text=protocol.assert_decoded_prefix(self._gui_processor(),self._gui_prefix[0],json.dumps(c))
            p=Path(os.environ['GUI_ADAPTER_EXPORT_ROOT'])/f'history-fm-prefix-rank{dist.get_rank()}.json'
            p.write_text(json.dumps({'passed':True,'at_fm_denoise_entry':True,'history_as_assistant':False,'history_as_step_text':True,'text_then_image':False,'image_then_text':True,'instruction_level':c['level'],'goal_in_input':bool(c.get('goal')),'history_actions':len(c['history']),'prefix_tokens':self._gui_prefix[0]['input_ids'].shape[-1],'prefix_sha256':hashlib.sha256(text.encode()).hexdigest()}))
            self._history_fm_verified=True
        return old_denoise(self,*args,**kwargs)
    nm.GuiJointModel.denoise=audited_denoise
