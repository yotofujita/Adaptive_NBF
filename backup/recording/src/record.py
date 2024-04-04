#! /usr/bin/env python3
# coding:utf-8

import os
import numpy as np
import soundfile as sf
import rospy
import time

from tqdm import tqdm
import argparse
import queue
import socket

import numpy as np
import rospy
from std_msgs.msg import MultiArrayDimension, MultiArrayLayout, Float32MultiArray
# Float32MultiArray = [MultiArrayLayout layout, float32[] data]
# MultiArrayLayout = [MultiArrayDimension[] dims, uint32 data_offset]
# MultiArrayDimension = [string label, uint32 size, uint32 stride]

import soundfile as sf


def recv_nbytes(sock, n_bytes):
    """Receive exactly `n_bytes`.
    If the connection is closed, `None` will be returned.

    Args:
        sock (socket.socket): An socket object.
        n_bytes (int): Amount of data to receive.

    Returns:
        bytes: Received data.
    """
    ret = bytes()
    while len(ret) < n_bytes:
        part = sock.recv(min(4096, n_bytes - len(ret)))
        if part == b"":
            return None  # the connection is closed
        ret += part
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("host", type=str)
    parser.add_argument("--port", type=int, default=50001)
    parser.add_argument("--n_sample", type=int, default=4096)
    args = parser.parse_args()

    rospy.init_node('AudioRecorder', anonymous=False)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((args.host, args.port))
    print(f"Connect {args.host}:{args.port}")

    while not rospy.has_param("recording") or not rospy.has_param("fname"):
        time.sleep(0.1)

    rospy.set_param("recording", "False")

    print("Param recording and fname are ready")
    buffer = None
    buffer_start_idx = 0
    while not rospy.is_shutdown():
        rospy.set_param("record_ready", "True")

        while rospy.get_param("recording") == "True":
            fname = rospy.get_param("fname")
            print("---recording---", fname)
            rospy.set_param("save_ok", "False")

            header = recv_nbytes(sock, 3 * 4)
            if header is None:
                print("\n header is empty \n")
                break
            header = np.frombuffer(header, dtype=np.uint32)
            assert header.shape == (3,)
            n_channel, n_recv_sample, sr = header[0], header[1], header[2]
            assert sr == 48000

            if buffer is None:
                buffer = np.zeros([48000, n_channel], dtype=np.float32)
            if len(buffer) < buffer_start_idx + n_recv_sample:
                buffer = np.append(buffer, np.zeros([48000, n_channel]), axis=0)

            data = recv_nbytes(sock, n_channel * n_recv_sample * 4)
            data = np.frombuffer(data, dtype=np.float32).reshape(-1, n_channel)
            buffer[buffer_start_idx:buffer_start_idx+n_recv_sample] = data
            buffer_start_idx += n_recv_sample

        # if rospy.get_param("recording") == "False":
        if buffer is not None and len(buffer) > 48000:
            sf.write(fname, buffer[:buffer_start_idx], 48000, subtype="PCM_24")
            print("\n---save--- ", fname, "\n")
            buffer = None
            buffer_start_idx = 0

        rospy.set_param("save_ok", "True")
        print("wait recording")
        if rospy.is_shutdown():
            exit()
        time.sleep(0.1)
