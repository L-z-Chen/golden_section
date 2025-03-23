import os
import math
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import (
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2Tokenizer,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm
import wandb
import numpy as np
import warnings
from torch import nn
from torch.optim.optimizer import Optimizer
from utils import *

distributed_initialized = False


def exists(val):
  return val is not None

# update functions

def softsign_update_fn(p, grad, momentum, lr, wd, beta, eps, correct_bias, step):
  # apply decoupled weight decay
  if wd != 0:
    p.data.mul_(1. - lr * wd)

  # accumulate momentum
  momentum.mul_(beta).add_(grad, alpha=1. - beta)
  if correct_bias:
    bias_correction = 1.0 - beta ** step
    momentum_hat = momentum / bias_correction
  else:
    momentum_hat = momentum
  
  soft_sign = momentum_hat / (momentum_hat.abs() + eps)

  # Replace NaNs with the sign of momentum_hat
  p.add_(soft_sign, alpha=-lr)

class SoftSign(Optimizer):
  def __init__(
    self,
    params,
    lr: float = 1e-3,
    beta: float = 0.9,
    eps: float = 1e-7,
    weight_decay: float = 0.0,
    correct_bias: bool = True,
  ):
    assert lr > 0.
    assert 0. <= beta <= 1.

    defaults = dict(
      lr = lr,
      beta = beta,
      eps=eps,
      weight_decay = weight_decay,
      correct_bias = correct_bias
    )

    super().__init__(params, defaults)

    self.update_fn = softsign_update_fn

  @torch.no_grad()
  def step(self, closure=None):
    loss = None
    if closure is not None:
      loss = closure()

    for group in self.param_groups:
      weight_decay = group['weight_decay']
      for p in filter(lambda p: exists(p.grad), group['params']):
        grad, lr, beta, eps, correct_bias, state = (
          p.grad, group['lr'], group['beta'], group['eps'], group['correct_bias'], self.state[p]
        )

        # initialize state
        if len(state) == 0:
          state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
          state['step'] = 0

        momentum = state['exp_avg']
        state['step'] += 1

        # update parameters
        self.update_fn(
          p,
          grad,
          momentum,
          lr,
          weight_decay,
          beta,
          eps,
          correct_bias,
          state['step']
        )

    return loss



# ====================================
# Utility: Create Optimizer from dict
# ====================================
import inspect
def create_optimizer_from_dict(optimizer_class, model_params, params_dict, x):
    """
    Constructs an optimizer from a dictionary of hyperparameters.

    Args:
        optimizer_class: The optimizer class
        model_params: model.parameters()
        params_dict: dict of all hyperparameters (e.g., {"lr": 1e-3, "weight_decay": 1e-4, ...})

    Returns:
        optimizer: instantiated optimizer
        valid_args: used optimizer args (excluding 'params')
    """
    # Validate args
    sig = inspect.signature(optimizer_class).parameters
    valid_args = {k: v for k, v in params_dict.items() if k in sig}
    keys = list(valid_args.keys())
    filled = {}

    for i, k in enumerate(keys):
        if i < len(x):
            if k in ["lr", "weight_decay", "eps"]:
              filled[k] = 10 ** x[i]
            else:
              filled[k] = x[i]
        else:
            filled[k] = valid_args[k]
    valid_args = filled
    unknown_args = [k for k in params_dict if k not in sig and k != "params"]
    if unknown_args:
        raise ValueError(f"Unknown hyperparameter(s) for {optimizer_class.__name__}: {unknown_args}")

    # Create optimizer
    optimizer = optimizer_class(model_params, **valid_args)
    return optimizer, valid_args

def main():
    trainer = QwenTrainer("qwen", "openwebtext-100k")
    def wrapped_train_fn(x):
        return trainer.train_on(
            params_dict,
            SoftSign,
            x
        )
    params_dict = {
        "lr": [-5., -2.],
        "weight_decay": [-3.2, -.7],
        "eps": [-12.3, -1.],
        "beta": 0.9,
    }
    intervals = [v for v in params_dict.values() if isinstance(v, list)]
    result = golden_search(wrapped_train_fn, intervals, n_calls=10000, random_seed=42)
    if dist.get_rank() == 0:
        print(result.x_opt, result.opt_value)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
