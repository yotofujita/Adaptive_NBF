import numpy as np


def DEMAND_mic_pos():
    n_rows, n_cols = (4, 4)
    dist_btw_mics = 0.05    # in meter
    dist_btw_cols = dist_btw_mics
    dist_btw_rows = np.sqrt(dist_btw_mics**2 - (dist_btw_mics / 2)**2)
    mic_x_pos = np.arange(0, n_cols * dist_btw_cols, dist_btw_cols)
    mic_y_pos = np.arange(0, n_rows * dist_btw_rows, dist_btw_rows)
    mic_pos = np.zeros((len(mic_x_pos), len(mic_y_pos), 2))
    mic_pos[:, :, 0] = mic_x_pos
    mic_pos[1, :, 0] += dist_btw_mics / 2
    mic_pos[3, :, 0] += dist_btw_mics / 2
    mic_pos[:, :, 1] = -1 * mic_y_pos[:, None]
    return mic_pos.reshape(-1, 2)


def DEMAND_5mic_trapezoid_arr_type_dict():
    dict_mic_idxs = {
        "1_270":  [1, 0, 4, 5, 2],
        "2_270":  [2, 1, 5, 6, 3],
        "4_30":   [4, 9, 5, 1, 0],
        "4_330":  [4, 8, 9, 5, 1],
        "5_30":   [5, 10, 6, 2, 1],
        "5_90":   [5, 6, 2, 1, 4],
        "5_150":  [5, 2, 1, 4, 9],
        "5_210":  [5, 1, 4, 9, 10],
        "5_270":  [5, 4, 9, 10, 6],
        "5_330":  [5, 9, 10, 6, 2],
        "6_30":   [6, 11, 7, 3, 2],
        "6_90":   [6, 7, 3, 2, 5],
        "6_150":  [6, 3, 2, 5, 10],
        "6_210":  [6, 2, 5, 10, 11],
        "6_270":  [6, 5, 10, 11, 7],
        "6_330":  [6, 10, 11, 7, 3],
        "9_30":   [9, 13, 10, 5, 4],
        "9_90":   [9, 10, 5, 4, 8],
        "9_150":  [9, 5, 4, 8, 12],
        "9_210":  [9, 4, 8, 12, 13],
        "9_270":  [9, 8, 12, 13, 10],
        "9_330":  [9, 12, 13, 10, 5],
        "10_30":  [10, 14, 11, 6, 5],
        "10_90":  [10, 11, 6, 5, 9],
        "10_150": [10, 6, 5, 9, 13],
        "10_210": [10, 5, 9, 13, 14],
        "10_270": [10, 9, 13, 14, 11],
        "10_330": [10, 13, 14, 11, 6],
        "11_150": [11, 7, 6, 10, 14],
        "11_210": [11, 6, 10, 14, 15],
        "13_90":  [13, 14, 10, 9, 12],
        "14_90":  [14, 15, 11, 10, 13],
    }
    return dict_mic_idxs


def DEMAND_7mic_circular_arr_type_dict():
    dict_mic_idxs = {
        "5":   [5, 1, 2, 4, 6, 9, 10],
        "6":   [2, 2, 3, 5, 7, 10, 11],
        "9":   [9, 4, 5, 8, 10, 12, 13],
        "10":  [10, 5, 6, 9, 11, 13, 14]
    }
    return dict_mic_idxs


def DEMAND_5mic_trapezoid_pos(arr_type, center_arr=False):
    dict_mic_idxs = DEMAND_5mic_trapezoid_arr_type_dict()
    mic_idxs = dict_mic_idxs[arr_type]
    all_mic_pos = DEMAND_mic_pos()
    out_mic_pos = all_mic_pos[mic_idxs]
    if center_arr:
        out_mic_pos -= out_mic_pos[0]
    return out_mic_pos


def DEMAND_7mic_circular_pos(arr_type, center_arr=False):
    dict_mic_idxs = DEMAND_7mic_circular_arr_type_dict()
    mic_idxs = dict_mic_idxs[arr_type]
    all_mic_pos = DEMAND_mic_pos()
    out_mic_pos = all_mic_pos[mic_idxs]
    if center_arr:
        out_mic_pos -= out_mic_pos[0]
    return out_mic_pos


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    all_mic_pos = DEMAND_mic_pos()
    import ipdb; ipdb.set_trace()
    assert False
