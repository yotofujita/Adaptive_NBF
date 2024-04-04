./eval_finetune.py --gpu 3 --idx 0 --total 4 --BF MVDR &
./eval_finetune.py --gpu 4 --idx 1 --total 4 --BF MVDR &
./eval_finetune.py --gpu 5 --idx 2 --total 4 --BF MVDR &
./eval_finetune.py --gpu 6 --idx 3 --total 4 --BF MVDR &

# (./eval_finetune.py  --gpu 4 --idx 0 --total 4 --BF MVDR ; ./eval_finetune.py  --gpu 4 --idx 0 --total 4 --BF MA_MVDR ) &
# (./eval_finetune.py  --gpu 5 --idx 1 --total 4 --BF MVDR ; ./eval_finetune.py  --gpu 5 --idx 1 --total 4 --BF MA_MVDR ) &
# (./eval_finetune.py  --gpu 2 --idx 2 --total 4 --BF MVDR ; ./eval_finetune.py  --gpu 2 --idx 2 --total 4 --BF MA_MVDR ) &
# (./eval_finetune.py  --gpu 3 --idx 3 --total 4 --BF MVDR ; ./eval_finetune.py  --gpu 3 --idx 3 --total 4 --BF MA_MVDR ) &
# ./eval_finetune.py --gpu 0 --idx 0 --total 4 &
# ./eval_finetune.py --gpu 2 --idx 1 --total 4 &
# ./eval_finetune.py --gpu 4 --idx 2 --total 4 &
# ./eval_finetune.py --gpu 6 --idx 3 --total 4 &
# for i in `seq 0 7`
# do
#     ./eval_finetune.py --gpu $i --idx $i --total 8 &
# done