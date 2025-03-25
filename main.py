# main.py
import argparse
from utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_batch_size', type=int, default=24)
    parser.add_argument('--global_batch_size', type=int, default=192)
    return parser.parse_args()

def main():
    args = parse_args()
    file_path = "opt.py"
    info = extract_optimizer_info_from_file(file_path)
    trainer = QwenTrainer(dataset_name="openwebtext-100k", batch_size=args.local_batch_size)
    # gradient accumulation
    if args.global_batch_size % args.local_batch_size != 0:
        raise ValueError("Global batch size must be divisible by local batch size.")
    trainer.gradient_accumulation_steps = args.global_batch_size // (args.local_batch_size * dist.get_world_size())
    def wrapped_train_fn(x):
        return trainer.train_on(
            params_dict,
            info['optimizer_class'],
            x,
        )
    params_dict = {k: v for k, v in info["hyperparameters"].items() if isinstance(v, list)}


    intervals = [v for v in params_dict.values() if isinstance(v, list)]
    result = golden_search(wrapped_train_fn, intervals, n_calls=10000, random_seed=42)
    if dist.get_rank() == 0:
        print(result.x_opt, result.opt_value)
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
