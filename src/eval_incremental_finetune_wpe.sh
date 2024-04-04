
# ./eval_incremental_finetune_wpe.py --gpu 4 --idx 0 --total 4 --BF MVDR &
# ./eval_incremental_finetune_wpe.py --gpu 5 --idx 1 --total 4 --BF MVDR &
#  ./eval_incremental_finetune_wpe.py --gpu 5 --idx 2 --total 4 --BF MVDR &
#  ./eval_incremental_finetune_wpe.py --gpu 7 --idx 3 --total 4 --BF MVDR &

(./eval_incremental_finetune_wpe.py --gpu 4 --idx 0 --total 4 --BF MVDR ;  ./eval_incremental_finetune_wpe.py --gpu 4 --idx 2 --total 4 --BF MVDR) & 
(./eval_incremental_finetune_wpe.py --gpu 5 --idx 1 --total 4 --BF MVDR ;  ./eval_incremental_finetune_wpe.py --gpu 5 --idx 3 --total 4 --BF MVDR) & 
