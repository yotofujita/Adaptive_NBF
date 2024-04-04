# CUDA_VISIBLE_DEVICES=0 ./test_finetune_wpe.py  --idx 0 --BF MVDR &
# CUDA_VISIBLE_DEVICES=1 ./test_finetune_wpe.py  --idx 1 --BF MVDR &
# CUDA_VISIBLE_DEVICES=2 ./test_finetune_wpe.py  --idx 2 --BF MVDR &
# CUDA_VISIBLE_DEVICES=3 ./test_finetune_wpe.py  --idx 3 --BF MVDR &
CUDA_VISIBLE_DEVICES=4 ./test_finetune_wpe.py  --idx 4 --BF MVDR &
CUDA_VISIBLE_DEVICES=5 ./test_finetune_wpe.py  --idx 5 --BF MVDR &
CUDA_VISIBLE_DEVICES=7 ./test_finetune_wpe.py  --idx 6 --BF MVDR &
CUDA_VISIBLE_DEVICES=2 ./test_finetune_wpe.py  --idx 7 --BF MVDR &
