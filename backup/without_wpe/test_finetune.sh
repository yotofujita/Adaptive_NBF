# (CUDA_VISIBLE_DEVICES=0 ./test_finetune.py  --idx 0 --BF GEV ; CUDA_VISIBLE_DEVICES=0 ./test_finetune.py  --idx 0 --BF MA_MVDR ; CUDA_VISIBLE_DEVICES=0 ./test_finetune.py  --idx 0 --BF MVDR_SV ;) &
# (CUDA_VISIBLE_DEVICES=1 ./test_finetune.py  --idx 1 --BF GEV ; CUDA_VISIBLE_DEVICES=1 ./test_finetune.py  --idx 1 --BF MA_MVDR ; CUDA_VISIBLE_DEVICES=1 ./test_finetune.py  --idx 1 --BF MVDR_SV ;) &
# (CUDA_VISIBLE_DEVICES=6 ./test_finetune.py  --idx 2 --BF GEV ; CUDA_VISIBLE_DEVICES=6 ./test_finetune.py  --idx 2 --BF MA_MVDR ; CUDA_VISIBLE_DEVICES=6 ./test_finetune.py  --idx 2 --BF MVDR_SV ;) &
# (CUDA_VISIBLE_DEVICES=3 ./test_finetune.py  --idx 3 --BF GEV ; CUDA_VISIBLE_DEVICES=3 ./test_finetune.py  --idx 3 --BF MA_MVDR ; CUDA_VISIBLE_DEVICES=3 ./test_finetune.py  --idx 3 --BF MVDR_SV ;) &
# (CUDA_VISIBLE_DEVICES=4 ./test_finetune.py  --idx 4 --BF GEV ; CUDA_VISIBLE_DEVICES=4 ./test_finetune.py  --idx 4 --BF MA_MVDR ; CUDA_VISIBLE_DEVICES=4 ./test_finetune.py  --idx 4 --BF MVDR_SV ;) &
# (CUDA_VISIBLE_DEVICES=5 ./test_finetune.py  --idx 5 --BF GEV ; CUDA_VISIBLE_DEVICES=5 ./test_finetune.py  --idx 5 --BF MA_MVDR ; CUDA_VISIBLE_DEVICES=5 ./test_finetune.py  --idx 5 --BF MVDR_SV ;) &
# CUDA_VISIBLE_DEVICES=5 ./test_finetune.py  --idx 3 --BF MA_MVDR &
# CUDA_VISIBLE_DEVICES=3 ./test_finetune.py  --idx 4 --BF MA_MVDR &
# CUDA_VISIBLE_DEVICES=4 ./test_finetune.py  --idx 5 --BF MA_MVDR &
CUDA_VISIBLE_DEVICES=4 ./test_finetune.py  --idx 0 --BF MVDR &
CUDA_VISIBLE_DEVICES=5 ./test_finetune.py  --idx 1 --BF MVDR &
# CUDA_VISIBLE_DEVICES=3 ./test_finetune.py  --idx 3 --BF MA_MVDR &
# CUDA_VISIBLE_DEVICES=4 ./test_finetune.py  --idx 4 --BF MA_MVDR &
# CUDA_VISIBLE_DEVICES=5 ./test_finetune.py  --idx 5 --BF MA_MVDR &
