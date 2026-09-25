"""Register the GUI experiment before invoking the official training entrypoint."""

import copy
import os

from hydra.core.config_store import ConfigStore

from cosmos_framework.configs.base.experiment.action.posttrain_config.action_policy_droid_nano import (
    action_policy_droid_nano,
)
from cosmos_framework.utils.lazy_config import LazyCall as L

from .joint_dataset import StreamingJointPolicyDataset


def register():
    horizon = int(os.environ.get("GUI_HORIZON", "1"))
    if horizon < 1:
        raise ValueError("GUI_HORIZON must be positive")
    recipe = copy.deepcopy(action_policy_droid_nano)
    for default in recipe.defaults:
        if hasattr(default, "keys") and "override /optimizer" in default:
            default["override /optimizer"] = "adamw"
    recipe.model._target_ = "cosmos_framework.gui_mot.native_model.GuiJointModel"
    recipe.model.action_inactive_loss_weight = 1.0
    recipe.model.hybrid_ar = os.environ.get("GUI_HYBRID_AR") == "1"
    recipe.model.mot_joint = os.environ.get("GUI_MOT_JOINT") == "1"
    recipe.model.hybrid_align_steps = int(os.environ.get("GUI_HYBRID_ALIGN_STEPS", "500"))
    recipe.model.hybrid_bridge_steps = int(os.environ.get("GUI_HYBRID_BRIDGE_STEPS", "1000"))
    recipe.model.hybrid_ce_weight = float(os.environ.get("GUI_HYBRID_CE_WEIGHT", "1.0"))
    recipe.model.hybrid_kd_weight = float(os.environ.get("GUI_HYBRID_KD_WEIGHT", "0.5"))
    recipe.model.hybrid_plan_fm_weight = float(os.environ.get("GUI_HYBRID_PLAN_FM_WEIGHT", "0.5"))
    recipe.model.hybrid_plan_x0_weight = float(os.environ.get("GUI_HYBRID_PLAN_X0_WEIGHT", "0.1"))
    recipe.model.hybrid_future_x0_weight = float(os.environ.get("GUI_HYBRID_FUTURE_X0_WEIGHT", "0.1"))
    cfg = recipe.model.config
    cfg.ema.enabled = False
    cfg.compile.enabled = False
    cfg.compile.use_cuda_graphs = False
    cfg.parallelism.context_parallel_shard_degree = 1
    cfg.parallelism.fsdp_master_dtype = "float32"
    cfg.tokenizer.encode_exact_durations = [1 + 4 * horizon]
    cfg.resolution = "480"
    cfg.diffusion_expert_config.load_weights_from_pretrained = False
    cfg.vlm_config.model_instance.config.include_visual = True
    cfg.vlm_config.model_instance.config.tie_word_embeddings = False
    cfg.parallelism.fsdp_master_dtype = os.environ.get("GUI_FSDP_MASTER_DTYPE", "float32")
    cfg.vlm_config.pretrained_weights.enabled = True
    cfg.vlm_config.pretrained_weights.backbone_path = "${oc.env:GUI_BACKBONE_PATH}"
    cfg.vlm_config.tokenizer.pretrained_model_name = "${oc.env:GUI_BACKBONE_PATH}"
    cfg.rectified_flow_training_config.loss_scale = 1.0
    cfg.rectified_flow_training_config.action_loss_weight = 1.0
    cfg.rectified_flow_training_config.normalize_loss_by_active = True
    # Original AndroidControl screenshots are kept near their native aspect
    # ratio and are assigned to Cosmos' 768 resolution bucket. The upstream
    # action recipe only defines shifts through 720, so extend the schedule
    # with the same high-resolution shift used by the 720 bucket.
    if "768" not in cfg.rectified_flow_training_config.shift:
        cfg.rectified_flow_training_config.shift["768"] = cfg.rectified_flow_training_config.shift["720"]
    cfg.rectified_flow_training_config.train_time_video_distribution = "logitnormal"
    cfg.rectified_flow_training_config.independent_action_schedule = False
    recipe.job.name = f"gui_joint_h{horizon}"
    recipe.optimizer.lr = 1e-5
    # Torch fused AdamW updates the FP32 FSDP master parameters.
    recipe.optimizer.optimizer_type = "AdamW"
    recipe.optimizer.lr_multipliers = {}
    recipe.dataloader_train.dataset_name = "androidcontrol_joint"
    recipe.dataloader_train.max_samples_per_batch = 1
    loader = recipe.dataloader_train.dataloader
    loader.batch_size = 1
    loader.num_workers = 0
    loader.persistent_workers = False
    loader.prefetch_factor = None
    loader.in_order = True
    loader.datasets = {
        "gui": {
            "ratio": 1,
            "dataset": L(StreamingJointPolicyDataset)(
                manifest="${oc.env:GUI_TRAIN_MANIFEST}",
                split="train",
                max_action_dim=64,
                instruction_level="${oc.env:GUI_INSTRUCTION_LEVEL,low}",
                horizon=horizon,
                plan_cache=os.environ.get("GUI_ACTION_PLAN_CACHE") or None,
                normalize_plan=os.environ.get("GUI_NORMALIZE_PLAN", "1") == "1",
            ),
        }
    }
    # All visual weights are replaced by the GUI backbone after DCP load. Action heads are new.
    recipe.checkpoint.keys_to_skip_loading.append("language_model.visual")
    recipe.trainer.callbacks.compile_tokenizer = {"enabled": False}
    if os.environ.get("GUI_OFFICIAL_DCP_EVAL_OUTPUT"):
        from .official_dcp_eval_callback import OfficialAndroidControlDCPEval

        recipe.trainer.callbacks.gui_official_dcp_eval = L(OfficialAndroidControlDCPEval)(
            official_samples="${oc.env:GUI_OFFICIAL_SAMPLES}",
            screenshot_dir="${oc.env:GUI_OFFICIAL_SCREENSHOT_DIR}",
            output="${oc.env:GUI_OFFICIAL_DCP_EVAL_OUTPUT}",
            target_height=int(os.environ.get("GUI_TARGET_HEIGHT", "640")),
            target_width=int(os.environ.get("GUI_TARGET_WIDTH", "384")),
            max_new_tokens=int(os.environ.get("GUI_OFFICIAL_MAX_NEW_TOKENS", "512")),
            sampling_steps=int(os.environ.get("GUI_OFFICIAL_SAMPLING_STEPS", "20")),
            max_samples=int(os.environ.get("GUI_OFFICIAL_MAX_SAMPLES", "0")),
        )
    ConfigStore.instance().store(group="experiment", package="_global_", name="gui_joint_h1", node=recipe)

    ConfigStore.instance().store(
        group="experiment", package="_global_", name="gui_libra_joint_h1", node=copy.deepcopy(recipe)
    )

    mot = copy.deepcopy(recipe)
    mot.model.config.lora_enabled = True
    mot.model.config.lora_rank = 16
    mot.model.config.lora_alpha = 32
    mot.optimizer.lr_multipliers = {}
    mot.checkpoint.keys_to_skip_loading.append("lora_")
    mot.model.mot_joint = True
    mot.model.hybrid_ar = False
    mot.model.video_only = True
    mot.model.config.action_gen = False
    mot.model.hybrid_kd_weight = float(os.environ.get("GUI_HYBRID_KD_WEIGHT", "0.2"))
    mot.model.config.lora_target_modules = (
        "q_proj,k_proj,v_proj,o_proj,q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen"
    )
    mot.dataloader_train.dataloader.datasets["gui"]["dataset"].native_ar_text = True
    mot.dataloader_train.dataloader.datasets["gui"]["dataset"].video_only = True
    mot.dataloader_train.dataloader.datasets["gui"]["dataset"].plan_cache = None
    mot.dataloader_train.dataloader.datasets["gui"]["dataset"].normalize_plan = False
    mot.optimizer.keys_to_select = [
        "lora_",
        "time_embedder",
        "vae2llm",
        "llm2vae",
        "gui_joint_ar_bridge",
        "gui_future_conditioner",
    ]
    if os.environ.get("MOT_CHECKPOINT_PATH"):
        # Base Cosmos checkpoints have no LoRA or semantic-plan head tensors,
        # so fresh training skips them. MoT evaluation/resume must restore all
        # learned U/Z and AR adapters from its own checkpoint.
        learned = {
            "lora_",
            "action2llm",
            "llm2action",
            "action_modality_embed",
            "action_pos_embed",
        }
        mot.checkpoint.keys_to_skip_loading = [key for key in mot.checkpoint.keys_to_skip_loading if key not in learned]
    ConfigStore.instance().store(group="experiment", package="_global_", name="gui_libra_mot_h1", node=mot)
