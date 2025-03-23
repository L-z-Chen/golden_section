import numpy as np
import os
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
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
        start_idx = idx * self.max_length
        end_idx = start_idx + self.max_length
        token_slice = self.tokens[start_idx:end_idx]
        data = torch.tensor(token_slice, dtype=torch.long)
        return data

def get_model_and_dataset(model_name, dataset_name, hidden_size):
    name2path = {"openwebtext-100k": "Elriggs/openwebtext-100k"}
    train_dataset = load_dataset(name2path[dataset_name], trust_remote_code=True)
    tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
    train_dataset = MoonDataset(dataset_name, train_dataset, tokenizer)
    config = Qwen2Config(
        attention_dropout=0.0,
        bos_token_id=151643,
        eos_token_id=151643,
        hidden_act="silu",
        hidden_size=768,
        initializer_range=0.02,
        intermediate_size=2048,
        max_position_embeddings=513,
        max_window_layers=6,
        model_type="qwen2",
        num_attention_heads=6,
        num_hidden_layers=6,
        num_key_value_heads=6,
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
    return model, train_dataset



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
        start_idx = idx * self.max_length
        end_idx = start_idx + self.max_length
        token_slice = self.tokens[start_idx:end_idx]
        data = torch.tensor(token_slice, dtype=torch.long)
        return data

distributed_initialized = False
def init_distributed():

    global distributed_initialized
    if distributed_initialized:
        return int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])

    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    distributed_initialized = True
    print(f"[Rank {global_rank}] Local Rank: {local_rank}, World Size: {world_size}, Using GPU: {torch.cuda.current_device()} - {torch.cuda.get_device_name(torch.cuda.current_device())}")
    return global_rank, local_rank, world_size

class QwenTrainer:
    def __init__(self, model_name, dataset_name):
        self.rank, self.local_rank, self.world_size = init_distributed()
        self.device = torch.device(f"cuda:{self.local_rank}")
        self.model, self.dataset = self.get_model_and_dataset(model_name, dataset_name)
        self.train_sampler = DistributedSampler(self.dataset)
        self.train_loader = DataLoader(self.dataset, batch_size=16, sampler=self.train_sampler)
        self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.local_rank])

    def get_model_and_dataset(self, model_name, dataset_name):
        name2path = {"openwebtext-100k": "Elriggs/openwebtext-100k"}
        raw_dataset = load_dataset(name2path[dataset_name], trust_remote_code=True)
        tokenizer = Qwen2Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
        dataset = MoonDataset(dataset_name, raw_dataset, tokenizer)
        config = Qwen2Config(
            attention_dropout=0.0,
            bos_token_id=151643,
            eos_token_id=151643,
            hidden_act="silu",
            hidden_size=768,
            initializer_range=0.02,
            intermediate_size=2048,
            max_position_embeddings=513,
            max_window_layers=6,
            model_type="qwen2",
            num_attention_heads=6,
            num_hidden_layers=6,
            num_key_value_heads=6,
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
        return model, dataset


    def train_on(self, params_dict, optimizer_class, x):
        optimizer, optimizer_args = create_optimizer_from_dict(
            optimizer_class=optimizer_class,
            model_params=self.model.parameters(),
            params_dict=params_dict,
            x = x,
        )
        # Log to wandb
        if self.rank == 0:
            print(f"Using optimizer: {optimizer_class.__name__} with args: {optimizer_args}")

            n_total_params = sum(p.numel() for _, p in self.model.named_parameters())
            run_name = f"{optimizer_class.__name__}_" + "_".join([f"{k}={v:.1e}" if isinstance(v, float) else f"{k}={v}" for k, v in optimizer_args.items()])
            wandb.init(
                project="Debugging",
                name=run_name,
                config={**optimizer_args, "n_total_params": n_total_params}
            )



        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=100,
            num_training_steps=len(self.train_loader),
            num_cycles=0.5,
        )
        score = 0
        self.model.train()
        for epoch in range(1):
            self.train_sampler.set_epoch(epoch)
            progress_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), desc=f"Epoch {epoch+1}/1") if self.rank == 0 else enumerate(self.train_loader)
            for step, batch in progress_bar:
                batch = batch.to(self.device)
                outputs = self.model(input_ids=batch, labels=batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                score = score * 0.9 + loss.item() * 0.1
                if self.rank == 0:
                    progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "LR": f"{optimizer.param_groups[0]['lr']:.6f}"})
                    wandb.log({"loss": loss.item(), "lr": optimizer.param_groups[0]["lr"], "score": score}, step=step)
                if np.isnan(loss.item()) or (step > 30 and score > 6.2):
                    if self.rank == 0:
                        wandb.finish()
                    return 1e10 if np.isnan(loss.item()) else score
        if self.rank == 0:
            wandb.finish()
            print(f"Finished with score {score}")
        return score

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



import ast
import os
import importlib.util
def extract_optimizer_info_from_file(file_path: str):
    # Load source code and parse AST
    with open(file_path, "r", encoding="utf-8") as f:
        code = f.read()
    tree = ast.parse(code)

    optimizer_name = None
    hyperparams = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            if any(isinstance(base, ast.Name) and base.id == 'Optimizer' for base in node.bases):
                optimizer_name = node.name

                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                        args = item.args
                        param_names = [arg.arg for arg in args.args[2:]]
                        defaults = args.defaults
                        start = len(param_names) - len(defaults)

                        for i, default in enumerate(defaults):
                            param_name = param_names[start + i]
                            try:
                                value = ast.literal_eval(default)
                            except Exception:
                                value = ast.unparse(default) if hasattr(ast, 'unparse') else str(default)
                            hyperparams[param_name] = value
                break

    # Dynamically import the module and get the class
    module_name = os.path.splitext(os.path.basename(file_path))[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    optimizer_class = getattr(mod, optimizer_name)

    return {
        "optimizer_name": optimizer_name,
        "optimizer_class": optimizer_class,
        "hyperparameters": hyperparams
    }
