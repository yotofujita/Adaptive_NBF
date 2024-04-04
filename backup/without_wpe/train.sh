for i in `seq 0 1`
do
    nohup ./train_CSS.py --gpus 8 --setting $i --threshold -10 --batch_size 16
done