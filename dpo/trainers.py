import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tensor_parallel as tp
import contextlib

from dpo.preference_datasets import get_batch_iterator
from dpo.utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
from utils.utils import write_file
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple
from peft import get_peft_model_state_dict


def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1 / (2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor,
                     average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch


class BasicTrainer(object):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str,
                 reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir

        # 加载tokenizer
        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            tokenizer_name_or_path,
            cache_dir=get_local_dir(config.local_dirs),
            trust_remote_code=True
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # 数据迭代器配置
        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == 'sft',
        )

        # 模型和参考模型
        self.policy = policy
        self.reference_model = reference_model
        self.rank0 = (rank == 0)
        self.device = torch.device("cuda:0")  # 固定使用cuda:0
        self.policy.to(self.device)  # 仅需执行一次设备迁移

        # 断点续训：加载优化器状态和当前epoch
        self.current_epoch = 0  # 默认从第0轮开始
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=config.lr)  # 初始化优化器
        checkpoint_path = os.path.join(run_dir, 'LATEST')
        if os.path.exists(os.path.join(checkpoint_path, 'trainer_state.pt')):
            # 加载训练状态（优化器+epoch）
            state = torch.load(os.path.join(checkpoint_path, 'trainer_state.pt'), map_location=self.device)
            self.optimizer.load_state_dict(state['optimizer'])
            self.current_epoch = state.get('epoch', 0)
            print(f"已从 checkpoint 恢复，当前 epoch: {self.current_epoch}")

        # 初始化训练迭代器
        self.data_iterator_kwargs = data_iterator_kwargs
        self.batch_size = config.batch_size
        self.n_examples = config.n_examples
        self.train_iterator = get_batch_iterator(
            **data_iterator_kwargs,
            split='train',
            n_epochs=1,
            n_examples=self.n_examples,
            batch_size=self.batch_size,
            silent=not self.rank0,
            cache_dir=get_local_dir(config.local_dirs),
            model_name=config.model.name,
            change_mode=config.change_mode,
            train_mode=config.loss
        )

    def train_epoch(self):
        # 强制切换到训练模式（关键：确保 dropout 等层生效，梯度开启）
        self.policy.train()
        iterator = get_batch_iterator(
            **self.data_iterator_kwargs,
            split='train',
            n_epochs=1,
            n_examples=self.n_examples,
            batch_size=self.batch_size,
            silent=not self.rank0,
            cache_dir=get_local_dir(self.config.local_dirs),
            model_name=self.config.model.name,
            change_mode=self.config.change_mode,
            train_mode=self.config.loss
        )
        for step, batch in enumerate(
                tqdm.tqdm(iterator, desc=f'Training epoch {self.current_epoch}', disable=not self.rank0)):
            # 清零梯度（防止上一步残留梯度干扰）
            self.optimizer.zero_grad()

            # 加载数据到设备
            input_ids = batch['chosen_input_ids'].to(self.device)
            attention_mask = batch['chosen_attention_mask'].to(self.device)
            labels = batch['chosen_labels'].to(self.device)

            # 前向传播计算损失
            outputs = self.policy(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # 检查损失是否有梯度（调试用，可保留）
            if loss.grad_fn is None:
                print(f"Warning: Step {step} loss has no grad_fn! This will cause backward() failure.")

            # 反向传播 + 优化器更新
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()  # 再次清零梯度

            if step % 10 == 0 and self.rank0:
                print(f"Epoch {self.current_epoch} Step {step}, loss: {loss.item():.4f}")

        self.current_epoch += 1

    def train(self):
        """严格按照配置的总轮次训练，避免超额训练"""
        total_epochs = self.config.n_epochs  # 从配置中获取总轮次（例如4）
        start_epoch = self.current_epoch  # 从断点恢复的起始轮次（可能不为0）

        # 计算需要训练的轮次：从起始轮次到总轮次，确保不超过
        for epoch in range(start_epoch, total_epochs):
            # 显示当前轮次（人类可读的1-based格式，例如"3/4"）
            rank0_print(f"Starting epoch {epoch + 1}/{total_epochs}")
            self.train_epoch()  # 训练当前轮次

            # 每轮结束后保存 checkpoint
            self.save()

            # 评估（保持原逻辑）
            if self.config.do_first_eval or (self.config.eval_every > 0 and (epoch + 1) % self.config.eval_every == 0):
                self.evaluate()

        # 训练结束提示
        rank0_print(f"Training completed! Total epochs trained: {total_epochs} (as configured)")

    def evaluate(self):
        """Evaluate the model (placeholder)."""
        # Implement evaluation logic if needed
        rank0_print("Evaluation not implemented yet.")

    def save(self, path: Optional[str] = None):
        import os
        if path is None:
            path = os.path.join(self.run_dir, 'LATEST')
        os.makedirs(path, exist_ok=True)
        if self.rank0:
            # 保存完整模型（包括配置和tokenizer）
            self.policy.save_pretrained(path)  # 保存完整模型（包括LoRA适配器）
            self.tokenizer.save_pretrained(path)  # 保存tokenizer配置

            # 保存训练状态（可选，如果你需要断点续训）
            state = {
                'optimizer': self.optimizer.state_dict(),
                'epoch': self.current_epoch,
            }
            torch.save(state, os.path.join(path, 'trainer_state.pt'))
            print(f'Saved checkpoint to {path}')

    def load(self, path: str):
        # 加载 policy 模型权重
        policy_state_dict = torch.load(os.path.join(path, 'policy.pt'), map_location=self.device)
        self.policy.load_state_dict(policy_state_dict)
        self.policy.to(self.device)

        # 加载训练状态
        state_path = os.path.join(path, 'trainer_state.pt')
        if os.path.exists(state_path):
            state = torch.load(state_path, map_location=self.device)
            self.optimizer.load_state_dict(state['optimizer'])
            self.current_epoch = state.get('epoch', 0)
            if self.rank0:
                print(f"Loaded checkpoint from {path}, resuming at epoch {self.current_epoch}")
        else:
            if self.rank0:
                print(f"No trainer_state.pt found at {state_path}, starting from scratch")