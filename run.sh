export CUDA_VISIBLE_DEVICES=0,2,3
torchrun --nproc_per_node=3 /home/jl77863/muon/main.py
