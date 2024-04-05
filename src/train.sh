#!/bin/bash

nohup python src/train_CSS.py --gpus 6 --setting 0 --threshold -10 --batch_size 16 > train_CSS.out