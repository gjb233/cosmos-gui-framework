"""Training extension; never imported by lightweight data/evaluation modules."""

import torch

from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel

from .backbone import backbone_path
from .hybrid_action import (
    PredictedFutureConditioner,
    PredictedPlanFutureConditioner,
    ZeroInitJointCrossAttention,
    autoregressive_ce,
    hybrid_stage,
    teacher_kl,
    tokenize_action_targets,
)
from .joint_loss import (
    action_flow_loss,
    future_flow_per_sample,
    global_sample_mean,
    inactive_action_flow_loss,
    plan_flow_per_sample,
    plan_x0_per_sample,
)
from .multimodal import capture_current_screens, prepare_prefix


class GuiJointModel(OmniMoTModel):
    def __init__(
        self,
        config,
        action_inactive_loss_weight=0.0,
        hybrid_ar=False,
        hybrid_align_steps=500,
        hybrid_bridge_steps=1000,
        hybrid_ce_weight=1.0,
        hybrid_kd_weight=0.5,
        hybrid_plan_fm_weight=0.5,
        hybrid_plan_x0_weight=0.1,
        mot_joint=False,
        video_only=False,
    ):
        if not 0 <= action_inactive_loss_weight <= 10:
            raise ValueError("Invalid inactive action loss weight")
        # OmniMoTModel.__init__ calls the overridden build_net immediately.
        # Plain attributes must therefore exist before delegating to it.
        object.__setattr__(self, "hybrid_ar", bool(hybrid_ar))
        object.__setattr__(self, "mot_joint", bool(mot_joint))
        object.__setattr__(self, "video_only", bool(video_only))
        if mot_joint and hybrid_ar:
            raise ValueError("MoT joint and legacy AR-bridge modes are exclusive")
        if video_only and (not mot_joint or config.action_gen):
            raise ValueError("Video-only GUI MoT requires mot_joint=True and action_gen=False")
        object.__setattr__(self, "hybrid_align_steps", int(hybrid_align_steps))
        object.__setattr__(self, "hybrid_bridge_steps", int(hybrid_bridge_steps))
        object.__setattr__(self, "hybrid_ce_weight", float(hybrid_ce_weight))
        object.__setattr__(self, "hybrid_kd_weight", float(hybrid_kd_weight))
        object.__setattr__(self, "hybrid_plan_fm_weight", float(hybrid_plan_fm_weight))
        object.__setattr__(self, "hybrid_plan_x0_weight", float(hybrid_plan_x0_weight))
        object.__setattr__(self, "_hybrid_iteration", 0)
        super().__init__(config)
        self.action_inactive_loss_weight = float(action_inactive_loss_weight)

    def build_net(self, *args, **kwargs):
        network = super().build_net(*args, **kwargs)
        if self.hybrid_ar or self.mot_joint:
            hidden = int(network.hidden_size)
            network.gui_joint_ar_bridge = ZeroInitJointCrossAttention(hidden)
            if self.video_only:
                network.gui_future_conditioner = PredictedFutureConditioner(int(self.config.state_ch), hidden)
            else:
                network.gui_plan_future_conditioner = PredictedPlanFutureConditioner(
                    int(self.config.max_action_dim), int(self.config.state_ch), hidden
                )
        return network

    def training_step(self, data_batch, iteration=0):
        self._hybrid_iteration = int(iteration)
        return super().training_step(data_batch, iteration)

    def get_data_and_condition(self, data_batch, *args, **kwargs):
        capture_current_screens(data_batch)
        return super().get_data_and_condition(data_batch, *args, **kwargs)

    def add_lora(self, network, lora_rank, lora_alpha, lora_target_modules):
        network = super().add_lora(network, lora_rank, lora_alpha, lora_target_modules)
        # Native LoRA freezes all other parameters. Fresh GUI heads must learn;
        # re-enable these before FSDP constructs its parameter groups.
        heads = (
            "time_embedder",
            "vae2llm",
            "llm2vae",
            "action2llm",
            "llm2action",
            "action_modality_embed",
            "gui_joint_ar_bridge",
            "gui_plan_future_conditioner",
            "gui_future_conditioner",
        )
        for name, parameter in network.named_parameters():
            if not name.startswith("language_model.") and any(key in name for key in heads):
                parameter.requires_grad_(True)
            if "lora_" in name and parameter.requires_grad:
                # Keep pretrained GUI reasoning fixed during latent alignment
                # and bridge fitting. This live hook lets one optimizer span
                # all three stages without rebuilding FSDP parameter groups.
                if not self.mot_joint:
                    parameter.register_hook(self._hybrid_lora_gradient)
        return network

    def _hybrid_lora_gradient(self, gradient):
        if not self.hybrid_ar:
            return gradient
        stage = hybrid_stage(
            self._hybrid_iteration,
            align_steps=self.hybrid_align_steps,
            bridge_steps=self.hybrid_bridge_steps,
        )
        return gradient if stage.enable_gui_lora else gradient * 0

    def install_attention_dispatch(self, net):
        from .attention import install_gui_attention

        install_gui_attention(net)

    def load_pretrained_model_if_needed(self, *, has_resumable_checkpoint, has_load_path):
        if has_resumable_checkpoint:
            return
        if not has_load_path:
            raise ValueError("Supply a Cosmos base checkpoint; this recipe does not initialize DM from a GUI VLM")
        from .mai_weights import load_gui_reasoner

        # Keep the two pretrained roles separate. GUI-Libra seeds only the
        # understanding/AR pathway; the Cosmos DCP remains the source of the
        # diffusion generation experts. Copying the GUI reasoner into
        # ``*_moe_gen`` here would overwrite the world-model initialization
        # that was restored immediately before this hook.
        load_gui_reasoner(self.net.language_model, backbone_path(), copy_generator=False)
        if self.config.ema.enabled:
            load_gui_reasoner(self.net_ema.language_model, backbone_path(), copy_generator=False)

    def generate_samples_from_batch(self, data_batch, **kwargs):
        if kwargs.get("guidance", 1.5) != 1.0 or kwargs.get("upsample_task") is not None:
            raise ValueError("GUI policy requires guidance=1 and no prompt upsampling")
        return super().generate_samples_from_batch(data_batch, **kwargs)

    def _gui_processor(self):
        if not hasattr(self, "_gui_processor_instance"):
            from transformers import AutoProcessor

            self._gui_processor_instance = AutoProcessor.from_pretrained(backbone_path(), local_files_only=True)
        return self._gui_processor_instance

    def _prepare_gui_prefix(self, batch):
        if self.config.parallelism.context_parallel_shard_degree != 1:
            raise ValueError("GUI prefix currently requires CP=1")
        if self.config.compile.enabled or self.config.compile.use_cuda_graphs:
            raise ValueError("GUI multimodal bridge currently requires compilation and CUDA graphs disabled")
        self._gui_prefix = prepare_prefix(self._gui_processor(), batch)
        if self.hybrid_ar or self.mot_joint:
            texts = batch.get("gui_action_text")
            if texts is None:
                raise ValueError("Hybrid AR batch is missing gui_action_text")
            while isinstance(texts, list) and len(texts) == 1 and isinstance(texts[0], list):
                texts = texts[0]
            if all(isinstance(text, str) and not text for text in texts):
                # Inference deliberately has no clean action target. The joint
                # sampler can run without decoding a future frame or invoking
                # the teacher-forced AR training branch.
                self._gui_action_target_ids = None
                self._gui_action_target_masks = None
            elif any(not isinstance(text, str) or not text for text in texts):
                raise ValueError("Hybrid AR batch mixes training and inference samples")
            else:
                self._gui_action_target_ids, self._gui_action_target_masks = tokenize_action_targets(
                    self._gui_processor().tokenizer, texts, return_masks=True
                )
        return [sample["input_ids"].squeeze(0).tolist() for sample in self._gui_prefix]

    def _load_and_tokenize_text_data(self, data_batch, iteration):
        return self._prepare_gui_prefix(data_batch)

    def _get_inference_text_tokens(self, data_batch, has_negative_prompt, caption_groups=None):
        if has_negative_prompt or caption_groups is not None:
            raise ValueError("GUI initial policy supports one current screen, guidance=1 only")
        tokens = self._prepare_gui_prefix(data_batch)
        return tokens, tokens

    def _can_reuse_inference_text_kv(self, *args, **kwargs):
        # Enable only after cached multimodal-prefix equivalence is validated.
        return False

    def denoise(self, net=None, data_batch_packed=None, memory=None, video_temporal_causal=None):
        if not getattr(self, "_gui_prefix", None):
            raise RuntimeError("Current-screen prefix was not prepared")
        data_batch_packed.gui_multimodal_inputs = self._gui_prefix
        data_batch_packed.gui_special_tokens = self.llm_special_tokens
        if (self.hybrid_ar or self.mot_joint) and self._gui_action_target_ids is not None:
            data_batch_packed.gui_action_target_ids = self._gui_action_target_ids
            data_batch_packed.gui_action_target_masks = self._gui_action_target_masks
            if self.mot_joint:
                data_batch_packed.gui_mot_joint = True
                if self.video_only:
                    data_batch_packed.gui_video_only = True
            else:
                data_batch_packed.gui_hybrid_stage = hybrid_stage(
                    self._hybrid_iteration,
                    align_steps=self.hybrid_align_steps,
                    bridge_steps=self.hybrid_bridge_steps,
                )
        # The parent rebuilds a whitelist-only dictionary and would discard
        # the hybrid AR logits attached inside the FSDP network forward.
        net = net or self.net
        return net(
            packed_seq=data_batch_packed,
            memory=memory,
            video_temporal_causal=video_temporal_causal,
        )

    def _replace_clean_with_noised(self, packed_sequence, gen_data_noised):
        super()._replace_clean_with_noised(packed_sequence, gen_data_noised)
        if self.hybrid_ar or self.mot_joint:
            # Sigma and x_t are inference-available quantities. Attach only
            # those; clean U0/Z0 remain outside the network and cannot leak.
            # MoT uses the same x0 reconstruction to train the conditioner
            # that consumes final FM samples during inference.
            if not self.video_only:
                packed_sequence.gui_sigmas_action = gen_data_noised.sigmas_action
            packed_sequence.gui_sigmas_vision = gen_data_noised.sigmas_vision

    def _compute_losses(
        self,
        out_net,
        data_batch_packed,
        gen_data_noised,
        timesteps,
        is_image_batch,
        **kwargs,
    ):
        rf = self.config.rectified_flow_training_config
        if rf.train_time_weight != "uniform" or rf.independent_action_schedule:
            raise ValueError("Initial GUI objective requires uniform weighting and shared modality sigma")
        if is_image_batch or self.config.sound_gen or self.config.lbl.coeff_gen or self.config.lbl.coeff_und:
            raise ValueError("GUI loss supports dense H1 video/action batches without auxiliary modalities")
        group, _ = self._loss_averaging_group()
        vision_per_sample = future_flow_per_sample(out_net["preds_vision"], gen_data_noised.vt_target_vision)
        vision_loss = global_sample_mean(vision_per_sample, group=group)
        action_loss = inactive_loss = vision_loss.new_zeros(())
        groups = {}
        clean = []
        if self.video_only:
            if data_batch_packed.action is not None or "preds_action" in out_net:
                raise ValueError("Video-only GUI MoT must not pack or predict continuous actions")
        else:
            action = data_batch_packed.action
            if action is None or any(mask.any() for mask in action.condition_mask):
                raise ValueError("GUI actions must all be denoising targets")
            if action.action_valid_mask is not None:
                raise ValueError("Do not expose GUI applicability via action_valid_mask")
            # x0 = eps - (eps-x0): recover labels only HERE, after the joint forward.
            clean = [
                noise - target
                for noise, target in zip(
                    gen_data_noised.epsilon_action,
                    gen_data_noised.vt_target_action,
                    strict=True,
                )
            ]
            if self.hybrid_ar or self.mot_joint:
                plan_per_sample = plan_flow_per_sample(out_net["preds_action"], gen_data_noised.vt_target_action)
                action_loss = global_sample_mean(plan_per_sample, group=group)
            else:
                action_loss, groups = action_flow_loss(
                    out_net["preds_action"], gen_data_noised.vt_target_action, clean, group=group
                )
                inactive_loss = inactive_action_flow_loss(
                    out_net["preds_action"], gen_data_noised.vt_target_action, clean, group=group
                )
        plan_x0_loss = vision_loss.new_zeros(())
        if self.hybrid_ar or self.mot_joint:
            if not self.video_only:
                plan_x0_loss = global_sample_mean(
                    plan_x0_per_sample(
                        gen_data_noised.xt_tokens_action,
                        out_net["preds_action"],
                        gen_data_noised.sigmas_action,
                        clean,
                    ),
                    group=group,
                )
            total = (
                self.hybrid_plan_fm_weight * action_loss
                + rf.loss_scale * vision_loss
                + self.hybrid_plan_x0_weight * plan_x0_loss
            )
        else:
            total = (
                rf.action_loss_weight * (action_loss + self.action_inactive_loss_weight * inactive_loss)
                + rf.loss_scale * vision_loss
            )
        ar_loss = vision_loss.new_zeros(())
        kd_loss = vision_loss.new_zeros(())
        ar_logit_delta = vision_loss.new_zeros(())
        ar_residual_ratio = vision_loss.new_zeros(())
        ar_token_accuracy = vision_loss.new_zeros(())
        ar_base_token_accuracy = vision_loss.new_zeros(())
        ar_base_ce = vision_loss.new_zeros(())
        ar_sequence_exact = vision_loss.new_zeros(())
        if self.hybrid_ar or self.mot_joint:
            stage = (
                None
                if self.mot_joint
                else hybrid_stage(
                    self._hybrid_iteration,
                    align_steps=self.hybrid_align_steps,
                    bridge_steps=self.hybrid_bridge_steps,
                )
            )
            if self.mot_joint or stage.enable_ar_loss:
                ar_mask = out_net["gui_ar_token_mask"]
                ar_loss = autoregressive_ce(out_net["gui_ar_logits"], out_net["gui_ar_labels"], token_mask=ar_mask)
                kd_loss = teacher_kl(out_net["gui_ar_logits"], out_net["gui_ar_base_logits"], token_mask=ar_mask)
                ar_logit_delta = out_net["gui_ar_max_logit_delta"]
                ar_residual_ratio = out_net["gui_ar_residual_norm_ratio"]
                total = total + self.hybrid_ce_weight * ar_loss + self.hybrid_kd_weight * kd_loss
                with torch.no_grad():
                    labels = out_net["gui_ar_labels"]
                    ar_token_accuracy = (
                        (out_net["gui_ar_logits"].argmax(-1) == labels) * ar_mask
                    ).sum() / ar_mask.sum()
                    ar_base_token_accuracy = (
                        (out_net["gui_ar_base_logits"].argmax(-1) == labels) * ar_mask
                    ).sum() / ar_mask.sum()
                    ar_base_ce = autoregressive_ce(out_net["gui_ar_base_logits"], labels, token_mask=ar_mask)
                    ar_sequence_exact = ((out_net["gui_ar_logits"].argmax(-1) == labels) | ~ar_mask).all().float()
        action_metrics = {}
        if not self.video_only:
            with torch.no_grad():
                recovered = [
                    xt.float() - sigma.float() * prediction.float()
                    for xt, sigma, prediction in zip(
                        gen_data_noised.xt_tokens_action,
                        gen_data_noised.sigmas_action,
                        out_net["preds_action"],
                        strict=True,
                    )
                ]
                type_accuracy = (
                    vision_loss.new_zeros(())
                    if (self.hybrid_ar or self.mot_joint)
                    else global_sample_mean(
                        torch.stack(
                            [
                                (prediction[:, :8].argmax(-1) == target[:, :8].argmax(-1)).float().mean()
                                for prediction, target in zip(recovered, clean, strict=True)
                            ]
                        ),
                        group=group,
                    )
                )
                mean_sigma = global_sample_mean(
                    torch.stack([sigma.float().mean() for sigma in gen_data_noised.sigmas_action]), group=group
                )
                _, x0_groups = action_flow_loss(recovered, clean, clean, group=group)
            action_metrics = {
                "gui_action_inactive": inactive_loss,
                "gui_action_denoised_type_accuracy": type_accuracy,
                "gui_action_sigma": mean_sigma,
                **{f"gui_action_x0_{name}_mse": value for name, value in x0_groups.items()},
                "flow_matching_loss_action": action_loss,
                "gui_plan_x0_huber": plan_x0_loss,
                **{f"gui_action_{name}": loss for name, loss in groups.items()},
            }
        return total, {
            "flow_matching_loss_vision": vision_loss,
            "flow_matching_loss_vision_per_instance": vision_per_sample.detach(),
            "gui_ar_ce": ar_loss,
            "gui_ar_kd": kd_loss,
            "gui_ar_max_logit_delta": ar_logit_delta,
            "gui_ar_residual_norm_ratio": ar_residual_ratio,
            "gui_ar_token_accuracy": ar_token_accuracy,
            "gui_ar_base_token_accuracy": ar_base_token_accuracy,
            "gui_ar_base_ce": ar_base_ce,
            "gui_ar_sequence_exact": ar_sequence_exact,
            **action_metrics,
        }
