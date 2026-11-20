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
        assert rf.loss_scale == 1.0 and self.hybrid_ce_weight == 1.0
        if rf.train_time_weight != "uniform" or rf.independent_action_schedule:
            raise ValueError("Initial GUI objective requires uniform weighting and shared modality sigma")
        if is_image_batch or self.config.sound_gen or self.config.lbl.coeff_gen or self.config.lbl.coeff_und:
            raise ValueError("GUI loss supports dense H1 video/action batches without auxiliary modalities")
        group, _ = self._loss_averaging_group()
        vision_per_sample = future_flow_per_sample(out_net["preds_vision"], gen_data_noised.vt_target_vision)
        vision_loss = global_sample_mean(mixed_mask(vision_per_sample, self._mixed_fm_mask)*self._slot_weight, group=group)
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
                + 0.0 * rf.loss_scale * vision_loss
                + self.hybrid_plan_x0_weight * plan_x0_loss
            )
        else:
            total = (
                rf.action_loss_weight * (action_loss + self.action_inactive_loss_weight * inactive_loss)
                + 0.0 * rf.loss_scale * vision_loss
            )
        ar_loss = vision_loss.new_zeros(())
        kd_loss = vision_loss.new_zeros(())
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
                if self.grouped_ce:
                    from .mixed_supervision import grouped_ce

                    ar_loss = grouped_ce(
                        out_net["gui_ar_logits"], out_net["gui_ar_labels"], out_net["gui_ar_token_weights"]
                    )
                else:
                    ar_loss = autoregressive_ce(out_net["gui_ar_logits"], out_net["gui_ar_labels"], token_mask=ar_mask)
                if self.hybrid_kd_weight:
                    kd_loss = teacher_kl(out_net["gui_ar_logits"], out_net["gui_ar_base_logits"], token_mask=ar_mask)
                total = total + self.hybrid_ce_weight * (ar_loss * self._slot_weight) + self.hybrid_kd_weight * kd_loss
        action_metrics = {}
        if not self.video_only:
            action_metrics = {
                "gui_action_inactive": inactive_loss,
                "flow_matching_loss_action": action_loss,
                "gui_plan_x0_huber": plan_x0_loss,
                **{f"gui_action_{name}": loss for name, loss in groups.items()},
            }
        audit_fm_weights(self,total,ar_loss,vision_loss)
        return total, {
            "flow_matching_loss_vision": vision_loss,
            "flow_matching_loss_vision_per_instance": vision_per_sample.detach(),
            "gui_ar_ce": ar_loss,
            **({"gui_ar_kd": kd_loss} if self.hybrid_kd_weight else {}),
            **action_metrics,
        }