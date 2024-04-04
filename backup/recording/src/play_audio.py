#! /usr/bin/env python3
# coding:utf-8

import os, sys, time
from glob import glob

import argparse

import numpy as np
import rospy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("fname", type=str)
    args = parser.parse_args()

    rospy.init_node('AudioPlayer', anonymous=False)

    rospy.set_param("record_ready", "False")

    with open(args.fname) as f:
        for line in f.readlines():
            param_list = line.replace(" ", "").split(",")
            print(line, param_list)
            if len(param_list) == 5: # target
                ID, gender, known, length, fname = param_list
                save_fname = f"../data/{ID}_{gender}_{known}_{length}.wav"
            elif len(param_list) == 2: # target
                ID, fname = param_list
                save_fname = f"../data/{ID}.wav"
            else:
                ID, length, fname = param_list
                save_fname = f"../data/{ID}_{length}.wav"

            if os.path.isfile(save_fname):
                continue

            rospy.set_param("fname", save_fname)
            rospy.set_param("recording", "True")

            while not rospy.is_shutdown() and rospy.has_param("record_ready") and rospy.get_param("record_ready") == "False":
                print("wait becoming ready to record")
                time.sleep(0.2)
            if rospy.is_shutdown():
                rospy.set_param("recording", "False")
                exit()
            rospy.set_param("recording", "True")

            os.system(f"aplay {fname}")
            print("Finish recording")

            rospy.set_param("recording", "False")

            while (not rospy.is_shutdown()) and rospy.get_param("save_ok") == "False":
                print("wait saving")
                time.sleep(0.2)
            if rospy.is_shutdown():
                rospy.set_param("recording", "False")
                exit()
            
