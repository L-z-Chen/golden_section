export WANDB_API_KEY=9533feefe54c8861797f495e7872fa9631fe5a77
export CUDA_VISIBLE_DEVICES=2
torchrun --nproc_per_node=1 --master-port=29500 /home/jl77863/muon/main.py \
                                                        --local_batch_size 16 \
                                                        --global_batch_size 16
