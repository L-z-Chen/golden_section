export WANDB_API_KEY=<Your Wandb API Key>
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node=1 --master-port=29502 /home/jl77863/muon/main.py \
                                                        --local_batch_size 24 \
                                                        --global_batch_size 48
