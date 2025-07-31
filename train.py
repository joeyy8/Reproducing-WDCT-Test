
print("train.py script started")
import os

os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['HF_MODULES_CACHE'] = 'output'
os.environ['TRANSFORMERS_CACHE'] = 'output'

import torch

torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from dpo.utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port
import hydra
import torch.multiprocessing as mp
#from omegaconf import OmegaConf, DictConfig 这是非智谱系列模型
from omegaconf import OmegaConf, DictConfig, ListConfig #这是智谱系列模型使用的

import dpo.trainers as trainers
import wandb
import json
import socket
from typing import Optional, Set
import resource
from peft import LoraConfig, get_peft_model, PeftModel

OmegaConf.register_new_resolver("get_local_run_dir",
                                lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module,
                reference_model: Optional[nn.Module] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    if 'FSDP' in config.trainer:
        init_distributed(rank, world_size, port=config.fsdp_port)

    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and config.wandb.enabled:
        print("W&B is enabled but initialization is disabled. Logging to local file.")
        log_file = os.path.join(config.local_run_dir, 'training_log.txt')
        with open(log_file, 'a') as f:
            f.write(f"Starting training for {config.exp_name}\n")
            f.write(f"Config: {OmegaConf.to_yaml(config)}\n")
    else:
        print("W&B is disabled or rank != 0, proceeding without W&B logging.")

    TrainerClass = getattr(trainers, config.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, reference_model=reference_model,
                           rank=rank, world_size=world_size)

    # 检查是否存在checkpoint
    checkpoint_path = os.path.join(config.local_run_dir, 'LATEST')
    if os.path.exists(os.path.join(checkpoint_path, 'policy.pt')):
        print(f"发现已有checkpoint: {checkpoint_path}，加载模型并保存完整文件，不进行训练")
        # 加载训练状态（确保current_epoch等参数正确）
        if os.path.exists(os.path.join(checkpoint_path, 'trainer_state.pt')):
            state = torch.load(os.path.join(checkpoint_path, 'trainer_state.pt'), map_location=trainer.device)
            trainer.current_epoch = state.get('epoch', 0)
            print(f"已加载checkpoint，当前epoch: {trainer.current_epoch}")
        else:
            print("警告：未找到trainer_state.pt，可能无法正确恢复训练状态")

        # 直接保存模型
        trainer.save()
        print(f"模型已保存至 {checkpoint_path}")
        return

    # 正常训练逻辑（仅当没有checkpoint时执行）
    trainer.train()
    trainer.save()


@hydra.main(version_base=None, config_path="dpo/config", config_name="dpo_config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""
    OmegaConf.resolve(config)
    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")
    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size
    if 'FSDP' in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print('no FSDP port specified; using open port for FSDP:', free_port)
        config.fsdp_port = free_port
    print(OmegaConf.to_yaml(config))
    config_path = os.path.join(config.local_run_dir, 'dpo_config.yaml')
    os.makedirs(os.path.join(config.local_run_dir, 'LATEST'), exist_ok=True)
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)

    model_kwargs = {'device_map': {"": 0}} if config.trainer == 'BasicTrainer' else {}

    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)
    print('building policy')
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path,
        cache_dir=get_local_dir(config.local_dirs),
        low_cpu_mem_usage=True,
        torch_dtype=policy_dtype,
        trust_remote_code=True,
        **model_kwargs
    )
    disable_dropout(policy)

    # Apply LoRA
    # 将 target_modules 处理为 Python 列表，支持字符串或 ListConfig 格式——————————非智谱
    #target_modules = config.model.lora.target_modules.split(',') if isinstance(config.model.lora.target_modules, str) else \
    #    list(config.model.lora.target_modules) if isinstance(config.model.lora.target_modules, OmegaConf.ListConfig) else \
    #    config.model.lora.target_modules

    #智谱系列
    target_modules = config.model.lora.target_modules.split(',') if isinstance(config.model.lora.target_modules,
                                                                               str) else \
        list(config.model.lora.target_modules) if isinstance(config.model.lora.target_modules, ListConfig) else \
            config.model.lora.target_modules

    lora_config = LoraConfig(
        r=config.model.lora.r,
        lora_alpha=config.model.lora.lora_alpha,
        lora_dropout=config.model.lora.dropout,
        target_modules=target_modules,  # 使用转换后的列表
        bias=config.model.lora.bias,
        task_type=config.model.lora.task_type
    )
    policy = get_peft_model(policy, lora_config)

    # Load checkpoint
    checkpoint_path = os.path.join(config.local_run_dir, 'LATEST')
    if os.path.exists(os.path.join(checkpoint_path, 'policy.pt')):
        print(f"发现已有checkpoint: {checkpoint_path}，加载完整模型权重")
        state_dict = torch.load(os.path.join(checkpoint_path, 'policy.pt'), map_location='cpu')
        policy.load_state_dict(state_dict, strict=False)
    elif config.model.archive is not None:
        print(f"加载预训练 LoRA 权重 from {config.model.archive}")
        policy = PeftModel.from_pretrained(policy, config.model.archive)

    for name, param in policy.named_parameters():
        if 'lora' in name or 'query_key_value.adapter' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    policy.print_trainable_parameters()

    if config.loss.name in {'dpo', 'ipo'}:
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path,
            cache_dir=get_local_dir(config.local_dirs),
            low_cpu_mem_usage=True,
            torch_dtype=reference_model_dtype,
            trust_remote_code=True,
            **model_kwargs
        )
        disable_dropout(reference_model)
        # 应用相同的 LoRA 配置到参考模型————————非智谱
        #ref_target_modules = config.model.lora.target_modules.split(',') if isinstance(config.model.lora.target_modules, str) else \
        #    list(config.model.lora.target_modules) if isinstance(config.model.lora.target_modules, OmegaConf.ListConfig) else \
        #    config.model.lora.target_modules

        #智谱
        ref_target_modules = config.model.lora.target_modules.split(',') if isinstance(config.model.lora.target_modules,str) else \
            list(config.model.lora.target_modules) if isinstance(config.model.lora.target_modules, ListConfig) else \
                config.model.lora.target_modules

        ref_lora_config = LoraConfig(
            r=config.model.lora.r,
            lora_alpha=config.model.lora.lora_alpha,
            lora_dropout=config.model.lora.dropout,
            target_modules=ref_target_modules,  # 使用转换后的列表
            bias=config.model.lora.bias,
            task_type=config.model.lora.task_type
        )
        reference_model = get_peft_model(reference_model, ref_lora_config)
        if os.path.exists(os.path.join(checkpoint_path, 'policy.pt')):
            ref_path = config.model.archive
            if os.path.exists(ref_path):
                print(f"加载参考模型 LoRA 权重 from {ref_path}")
                reference_model = PeftModel.from_pretrained(reference_model, ref_path)
                print("参考模型 LoRA 权重已加载")
    else:
        reference_model = None

    if 'FSDP' in config.trainer:
        world_size = torch.cuda.device_count()
        print('starting', world_size, 'processes for FSDP training')
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
        mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, policy, reference_model), join=True)
    else:
        print('starting single-process worker')
        worker_main(0, 1, config, policy, reference_model)


if __name__ == '__main__':
    main()