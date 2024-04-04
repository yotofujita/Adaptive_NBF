CUDA_VISIBLE_DEVICES=2 ./test_incremental_finetune.py --idx 0 --BF MVDR &
CUDA_VISIBLE_DEVICES=3 ./test_incremental_finetune.py --idx 1 --BF MVDR &
CUDA_VISIBLE_DEVICES=4 ./test_incremental_finetune.py --idx 2 --BF MVDR
