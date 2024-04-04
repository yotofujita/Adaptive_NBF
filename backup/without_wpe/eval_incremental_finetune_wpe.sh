# (./eval_incremental_finetune.py --gpu 1 --idx 1 --total 13 --BF MVDR; ./eval_incremental_finetune.py --gpu 1 --idx 5 --total 13 --BF MVDR; ./eval_incremental_finetune.py --gpu 1 --idx 9  --total 13 --BF MVDR) &
# (./eval_incremental_finetune.py --gpu 2 --idx 2 --total 13 --BF MVDR; ./eval_incremental_finetune.py --gpu 2 --idx 6 --total 13 --BF MVDR; ./eval_incremental_finetune.py --gpu 2 --idx 10 --total 13 --BF MVDR) &
# (./eval_incremental_finetune.py --gpu 3 --idx 3 --total 13 --BF MVDR; ./eval_incremental_finetune.py --gpu 3 --idx 7 --total 13 --BF MVDR; ./eval_incremental_finetune.py --gpu 3 --idx 11 --total 13 --BF MVDR) &
# (./eval_incremental_finetune.py --gpu 4 --idx 4 --total 13 --BF MVDR; ./eval_incremental_finetune.py --gpu 4 --idx 8 --total 13 --BF MVDR; ./eval_incremental_finetune.py --gpu 4 --idx 12 --total 13 --BF MVDR) &

# (./eval_incremental_finetune.py --gpu 5 --idx 1 --total 13 --BF MVDR; ./eval_incremental_finetune.py --gpu 5 --idx 4 --total 13 --BF MVDR; ./eval_incremental_finetune.py --gpu 5 --idx 5 --total 13 --BF MVDR; ./eval_incremental_finetune.py --gpu 5 --idx 9  --total 13 --BF MVDR) &
# (./eval_incremental_finetune.py --gpu 6 --idx 2 --total 13 --BF MVDR; ./eval_incremental_finetune.py --gpu 6 --idx 8 --total 13 --BF MVDR; ./eval_incremental_finetune.py --gpu 6 --idx 6 --total 13 --BF MVDR; ./eval_incremental_finetune.py --gpu 6 --idx 10 --total 13 --BF MVDR) &
# (./eval_incremental_finetune.py --gpu 7 --idx 3 --total 13 --BF MVDR; ./eval_incremental_finetune.py --gpu 7 --idx 7 --total 13 --BF MVDR; ./eval_incremental_finetune.py --gpu 7 --idx 11 --total 13 --BF MVDR; ./eval_incremental_finetune.py --gpu 7 --idx 12 --total 13 --BF MVDR) &

# (./eval_incremental_finetune.py --gpu 4 --idx 5 --total 10 --BF MVDR; ./eval_incremental_finetune.py --gpu 4 --idx 9  --total 10 --BF MVDR) &
# (./eval_incremental_finetune.py --gpu 5 --idx 6 --total 10 --BF MVDR; ./eval_incremental_finetune.py --gpu 5 --idx 10 --total 10 --BF MVDR) &
# (./eval_incremental_finetune.py --gpu 6 --idx 7 --total 10 --BF MVDR; ./eval_incremental_finetune.py --gpu 6 --idx 11 --total 10 --BF MVDR) &
# (./eval_incremental_finetune.py --gpu 7 --idx 8 --total 10 --BF MVDR; ./eval_incremental_finetune.py --gpu 7 --idx 12 --total 10 --BF MVDR) &

# ./eval_incremental_finetune_wpe.py --gpu 4 --idx 0 --total 4 --BF MVDR &
# ./eval_incremental_finetune_wpe.py --gpu 5 --idx 1 --total 4 --BF MVDR &
./eval_incremental_finetune_wpe.py --gpu 6 --idx 2 --total 4 --BF MVDR &
./eval_incremental_finetune_wpe.py --gpu 7 --idx 3 --total 4 --BF MVDR &

# OK in sacs10 09

# ./eval_incremental_finetune.py --gpu 3 --idx 3 --total 13 --BF MVDR &
# ./eval_incremental_finetune.py --gpu 4 --idx 4 --total 13 --BF MVDR &

# ./eval_incremental_finetune.py --gpu 7 --idx 9 --total 13 --BF MVDR &
# ./eval_incremental_finetune.py --gpu 5 --idx 10 --total 13 --BF MVDR &
# ./eval_incremental_finetune.py --gpu 6 --idx 11 --total 13 --BF MVDR &
# ./eval_incremental_finetune.py --gpu 7 --idx 12 --total 13 --BF MVDR &

# for i in `seq 0 7`
# do
#     ./eval_incremental_finetune.py --gpu $i --idx $i --total 8 &
# done