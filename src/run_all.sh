./run_incremental_finetune.py --gpu_start 4
CUDA_VISIBLE_DEVICES=4 ./test_incremental_finetune_wpe.py
./eval_incremental_finetune_wpe.py --gpu 4 --idx 0 --total 1 --BF MVDR &
