import torch.distributed as dist
from utils import *


def main():
    file_path = "opt.py"
    info = extract_optimizer_info_from_file(file_path)
    trainer = QwenTrainer("qwen", "openwebtext-100k")

    def wrapped_train_fn(x):
        return trainer.train_on(
            params_dict,
            info['optimizer_class'],
            x
        )
    params_dict = {k: v for k, v in info["hyperparameters"].items() if isinstance(v, list)}


    intervals = [v for v in params_dict.values() if isinstance(v, list)]
    result = golden_search(wrapped_train_fn, intervals, n_calls=10000, random_seed=42)
    if dist.get_rank() == 0:
        print(result.x_opt, result.opt_value)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
