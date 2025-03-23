
import os
import math
import torch
from loguru import logger
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
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

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from typing import Callable, Iterable, Tuple

# functions

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


class MoonDataset(Dataset):
    def __init__(self, dataset_name, dataset, tokenizer, max_length=512):
        self.dataset_name = dataset_name
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.texts = dataset["train"]["text"]
        self.max_length = max_length
        self.tokens = []
        self._tokenize_texts()

    def _tokenize_texts(self):
        if os.path.exists(f"{self.dataset_name}.bin"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.tokens = torch.load(f"{self.dataset_name}.bin")

        else:
            for text in tqdm(self.texts, desc="Tokenizing texts"):
                encoded = self.tokenizer.encode(text, add_special_tokens=True)
                self.tokens.extend(encoded)
            torch.save(self.tokens, f"{self.dataset_name}.bin")

    def __len__(self):
        return len(self.tokens) // self.max_length

    def __getitem__(self, idx):
        start_idx = idx * (self.max_length)
        end_idx = start_idx + (self.max_length)
        token_slice = self.tokens[start_idx:end_idx]
        data = torch.tensor(token_slice, dtype=torch.long)
        return data



def get_model_and_dataloader(model_name, dataset_name, hidden_size):
    name2path = {
        "openwebtext-100k": "Elriggs/openwebtext-100k",
    }
    train_dataset = load_dataset(name2path[dataset_name], trust_remote_code=True)
    if model_name == "qwen":
        tokenizer = Qwen2Tokenizer.from_pretrained(
            "Qwen/Qwen2.5-0.5B", trust_remote_code=True
        )
    else:
        assert 0, f"model {model_name} not supported"
    train_dataset = MoonDataset(dataset_name, train_dataset, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)

    if model_name == "qwen":
        config = Qwen2Config(
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151643,
            hidden_act="silu",
            hidden_size=768,  # Reduced from 1024
            initializer_range=0.02,
            intermediate_size=2048,  # Reduced from 2700
            max_position_embeddings=513,
            max_window_layers=6,  # Adjusted
            model_type="qwen2",
            num_attention_heads=6,  # Adjusted
            num_hidden_layers=6,  # Reduced further
            num_key_value_heads=6,  # Match attention heads
            rms_norm_eps=1e-06,
            rope_theta=1000000.0,
            sliding_window=1024,
            tie_word_embeddings=True,
            torch_dtype="bfloat16",
            use_cache=True,
            use_mrope=False,
            use_sliding_window=False,
            vocab_size=151936,
        )


        model = Qwen2ForCausalLM(config)
    else:
        assert 0, f"model {model_name} not supported"
    return model, train_loader


def add_new_golden_point_2(a,b, where):
  gr = (np.sqrt(5)-1)/2
  if where=='larger':
    c = a + (b-a)*gr
  elif where == 'smaller':
    c = b + (a-b)*gr
  else:
    c = np.where(np.random.normal()>0, a+(b-a)*gr,  b+(a-b)*gr)
  return c

def add_new_golden_point_3(a,b,c):
  gr = (np.sqrt(5)-1)/2
  if (b-a)>(c-b):
    d = a + (b-a)*gr
  else:
    d = c + (b-c)*gr
  return d

class emptyclass(): pass

# multivariate golden section search with greedy coordinate descent
def golden_search(obj, intervals, n_calls,
                  initials = ['random'],
                  interval_tol = 1e-2, random_seed=None):

  np.random.seed(random_seed)

  xx = np.array(intervals, dtype=float)
  xm = np.mean(xx, 1)*0
  improvement = np.zeros_like(xm)+np.inf

  interval_curve = []
  x_curve = []
  loss_curve = []


  # initial=['random', 'larger', 'smaller', ...] deciding which of the two golden points should we initialize to
  if isinstance(initials, list)==False:
    initials = [initials]
  if len(initials)==1:
    initials = [initials[0] for i in range(len(xx))]

  # initial to one of the golden points
  for i in range(len(xx)):
    lower_bound = xx[i][0]; upper_bound = xx[i][1]
    xm[i] = add_new_golden_point_2(lower_bound, upper_bound, initials[i])

  # evaluating the first point
  loss_value = obj(xm)

  for iter in range(int(n_calls)):

    # find the coordinate with the maximum improvement last time.
    i = np.argmax(improvement)
    lower_bound = xx[i][0]; upper_bound = xx[i][1]; current_xi = xm[i]

    # pass if the interval is already very small
    if (upper_bound - lower_bound) <=  interval_tol*(intervals[i][1] - intervals[i][0]):
      continue

    # add new query point
    new_xi = add_new_golden_point_3(lower_bound, current_xi, upper_bound)

    xtmp = np.copy(xm); xtmp[i]=new_xi
    loss_value_tmp = obj(xtmp)

    improvement[i] = np.abs(loss_value_tmp - loss_value)

    if loss_value_tmp <= loss_value:
      xm[i] = new_xi
      loss_value = loss_value_tmp
      if new_xi<=current_xi:
        xx[i][1] = current_xi
      else:
        xx[i][0] = current_xi
    else:
      xm[i] = current_xi
      if new_xi<=current_xi:
        xx[i][0] = new_xi
      else:
        xx[i][1] = new_xi
    print(f'Iter{iter}, var{i}: [{xx[i][0]:.2e}, {xx[i][1]:.2e}], (point:{xm[i]}, loss:{loss_value:.5e}), improve: {improvement[i]}')
    interval_curve.append(np.array(xx))
    x_curve.append(np.array(xm))
    loss_curve.append(loss_value)

  result = emptyclass()
  result.interval_curve = interval_curve
  result.x_curve = x_curve
  result.loss_curve = loss_curve
  result.x_opt = xm
  result.interval = xx
  result.opt_value = loss_value

  return result

def train_qwen(x):


    model, train_loader = get_model_and_dataloader(
        "qwen", "openwebtext-100k", 1024
    )
    n_total_params = sum(p.numel() for _, p in model.named_parameters())

    print(f"Total params_M: {n_total_params / 1_000_000}")

    ##################################################
    ###############Change To Your Opt#################
    ##################################################
    optimizer = SoftSign(
        model.parameters(),
            lr=10**x[0],
            weight_decay=10**x[1],
            eps=10**x[2],
            beta=0.9,
        )
    ##################################################
    ##################################################
    ##################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    ##################################################
    wandb.init(project="Debugging", 
            name=f"softsign_lr={10**x[0]:.2e}_wd={10**x[1]:.2e}_eps={10**x[2]:.2e}_zero_init",
            config={
                "lr": 10**x[0],
                "wd": 10**x[1],
                "eps": 10**x[2],
                "beta": 0.9,
                "n_total_params": n_total_params,
            })
    ##################################################
    epoch = 1
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_loader) * epoch,
        num_cycles=0.5,
    )
    score = 0
    for epoch in range(epoch):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epoch}")

        for step, batch in progress_bar:

            batch = batch.to(device)
            outputs = model(input_ids=batch, labels=batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Update tqdm progress bar
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.6f}"
            })
            
            wandb.log({
                "loss": loss.item(),
                "lr": optimizer.param_groups[0]["lr"],
                "score": score,
            }, step=step)

            score = score * 0.9 + loss.item() * 0.1

            if np.isnan(loss.item()):
                print(f"Early stopping at step {step} with score {score}")
                wandb.finish()
                return 1e10
            if step > 300 and score > 6.2:
                print(f"Early stopping at step {step} with score {score}")
                wandb.finish()
                return score
            
    wandb.finish()
    print(f"Normal stopping at step {step} with score {score}")
    return score




if __name__ == "__main__":

    
    # Run optimization
    # Step1: Define the search space
    intervals = [
        [-5., -2.],  # Learning rate (lr)
        [-3.2, -.7],  # Weight decay (wd)
        [-12.3, -1.],  # eps
    ]
    # Step2: Run optimization
    # Go to @train_qwen to change the optimizer!!

    n_calls = 10000  # Number of evaluations

    result = golden_search(train_qwen, intervals, n_calls, random_seed=42)
    print(result.x_opt, result.opt_value)
  






