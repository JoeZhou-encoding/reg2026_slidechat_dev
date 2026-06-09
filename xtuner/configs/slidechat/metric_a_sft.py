# =============================================================================
# REG2026 Metric A — continue-SFT config (single-choice MCQA over CONCH features).
# Adapted from configs/slidechat/stage_2.py. Continue-SFT from the released SlideChat
# stage-2 weights (LongNet+projector+LLM) on our IMG-only MCQA data.
#
# ⚠️ Review every value before pjsub. Run with deepspeed_zero2/zero3 (see runbook).
#
# D-A6 = FULL fine-tune — DECIDED (2026-06-07 meeting + user 2026-06-09): train
#        LongNet+projector+LLM (freeze_llm=False, train_stage='2'). LoRA alt left
#        commented for reference only (NOT used).
# Aligned with SlideChat stage_2.py recipe: lr=2e-5, bs=1, accum=8, cosine,
#        warmup 0.03. Ours: epochs=1 (POC, meeting) vs 3; bf16 (Qwen2.5-native) vs fp16;
#        seed=2026 (repro) vs None; max_length=12288 (10240 patch tok + short MCQA) vs 19600.
# Note: original 'sample_type=wsi' is a DEAD config var (never reaches LLaVADataset;
#        only EvaluateChatHook uses it, which we drop) -> intentionally omitted.
#        per_image_length set to sample_num (NOT None) -> avoids modality_length TypeError.
# =============================================================================
import os
import torch
from mmengine.dataset import DefaultSampler
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

from xtuner.dataset import LLaVADataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.engine.hooks import DatasetInfoHook
from xtuner.engine.runner import TrainLoop
from xtuner.model import LLaVAModel
from xtuner.utils import PROMPT_TEMPLATE

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Cross-cluster root. Genkai default; on TSUBAME the qsub script does
#   export REG2026=/gs/bs/tgh-26IDE/ethanx/reg_data_2026
# so the SAME config drives both clusters (data at $REG2026/data/reg2026, models at $REG2026/models).
# Documented env override (config_registry.md); NOT a hidden default.
REG2026 = os.environ.get('REG2026', '/home/pj24003162/ku40003404/weihao/00/reg_2026')

# Model — base LLM (for arch+tokenizer; weights overwritten by pretrained_pth)
llm_name_or_path = f'{REG2026}/models/Qwen2.5-7B-Instruct'
# SlideChat stage-2 weights extracted to a plain state_dict (extract_stage2_model.py)
pretrained_pth = f'{REG2026}/models/SlideChat_Weight/stage2_model.pth'

# Data
data_path = f'{REG2026}/data/reg2026/metric_a/sft_metric_a_train.json'   # place the json here
image_folder = f'{REG2026}/data/reg2026/train_feat_conch'                # B5: resolves <stem>.h5
image_path_list = None

prompt_template = PROMPT_TEMPLATE.qwen_chat        # S3-verified: masks prompt, no <think>
sample_num = 10240                                 # downsample WSIs with N>10240 (user choice)
max_length = 12288                                 # 10240 patch tokens + short MCQA prompt
per_image_length = sample_num                      # S8: NOT None (avoids modality_length TypeError)

# Scheduler & Optimizer  [D-A6: full fine-tune; LoRA alt commented below]
batch_size = 1                 # per-device (one WSI's ~10k patch tokens -> bs must be 1)
accumulative_counts = 8        # global batch = 1 * 8 * num_gpus  (=32 on 4 GPUs)
dataloader_num_workers = 8
max_epochs = 1                 # POC: 1 epoch
optim_type = AdamW
lr = 2e-5                      # SlideChat stage-2 value; consider 1e-5 for gentler continue-SFT
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1
warmup_ratio = 0.03

# Save
save_steps = 500
save_total_limit = 2

SYSTEM = ''

#######################################################################
#            PART 2  Model & Tokenizer                                #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right')

model = dict(
    type=LLaVAModel,
    freeze_llm=False,             # [D-A6] full LLM fine-tune (train_stage='2')
    pretrained_pth=pretrained_pth,
    train_stage='2',             # trains LongNet + projector + LLM
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16),
    # [D-A6] LoRA alternative (lighter POC) — uncomment to use instead of full:
    # llm_lora=dict(type='peft.LoraConfig', r=64, lora_alpha=16, lora_dropout=0.05,
    #               bias='none', task_type='CAUSAL_LM'),
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
llava_dataset = dict(
    type=LLaVADataset,
    data_path=data_path,
    image_folder=image_folder,
    image_path_list=image_path_list,
    tokenizer=tokenizer,
    sample_num=sample_num,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    per_image_length=per_image_length,
    pad_image_to_square=False)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    pin_memory=True,
    dataset=llava_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16')            # bf16 (Qwen2.5-native; more stable than fp16)

param_scheduler = [
    dict(type=CosineAnnealingLR, eta_min=0.0, by_epoch=True, begin=0,
         end=max_epochs, convert_to_iter_based=True)
]

train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
custom_hooks = [dict(type=DatasetInfoHook, tokenizer=tokenizer)]

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=save_steps,
                    max_keep_ckpts=save_total_limit),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
visualizer = None
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=2026, deterministic=False)   # fixed seed for reproducibility
log_processor = dict(by_epoch=False)
