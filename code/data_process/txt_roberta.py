import pandas as pd
import ast
import torch
import numpy as np
import gzip
import pickle
from torch.nn.utils.rnn import pad_sequence
import random
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

def random_mask(sequences, mask_ratio, seed=42):
    if seed is not None:
        random.seed(seed)

    masked_sequence_delete, masked_sequence_zero_pad, masked_sequence_fill_mask = [], [], []
    for sequence in sequences:
        seq_len = len(sequence)
        num_masked_positions = max(1, int(seq_len * mask_ratio))
        mask = [True] * seq_len
        masked_positions = random.sample(range(seq_len), num_masked_positions)
        for pos in masked_positions:
            mask[pos] = False

        masked_sequence_delete.append([seq_item for i, seq_item in enumerate(sequence) if mask[i]])
        masked_sequence_zero_pad.append([seq_item if mask[i] else 0 for i, seq_item in enumerate(sequence)])
        masked_sequence_fill_mask.append([seq_item if mask[i] else "<mask>" for i, seq_item in enumerate(sequence)])

    return masked_sequence_delete, masked_sequence_zero_pad, masked_sequence_fill_mask

def timestamps2interval(ip_timestamps):
    in_val = []
    for timestamps in ip_timestamps:
        time_diffs = [0] + [(timestamps[i] - timestamps[i-1]) * 1e4 for i in range(1, len(timestamps))]
        processed_inval = [int(min(10000, max(1, x))) for x in time_diffs[1:]]
        processed_inval.insert(0, int(time_diffs[0]))
        in_val.append(processed_inval)
    return in_val

def select_from_df(df, Label2num):
    df.replace({"Label": Label2num}, inplace=True)
    df1 = df[df['Label'].apply(lambda x: isinstance(x, int) if pd.notna(x) else False).astype(bool)]
    return df1

def file_save(file_name, x, y):
    f = open(file_name, "w") # 0
    for k in range(len(x)):
        for i, pkt_len in enumerate(x[k]):
            # if i == 1000:
            #     break
            # f.write(str(pkt_len) + " ")
            if len(x[k]) <= 1000:
                if i < len(x[k]) - 1:
                    f.write(str(pkt_len) + " ")
                else:
                    f.write(str(pkt_len))
                    break
            else:
                if i < 999:
                    f.write(str(pkt_len) + " ")
                else:
                    f.write(str(pkt_len))
                    break
        f.write("\n")
    f.close()


def df2fsnet_mask(path_pkl, mask_ratio, save_txt_name,Label2num, Num2label):
    df = pd.read_pickle(path_pkl)
    if len(Num2label) != 0:
        df = select_from_df(df, Label2num)
        df.replace({"Label": Num2label}, inplace=True)
    pkt_len_, Label = df['ip_lengths'].values.tolist(), df['Label'].values.tolist()
    sequences = []
    for sequence in pkt_len_:
        seq = [s+1600 for s in sequence]
        sequences.append(seq)

    x_train, x_test, y_train, y_test = train_test_split(sequences, Label, test_size=0.2, stratify=Label, random_state=42)
    result_del, result_pad0, result_fmask = random_mask(x_test, mask_ratio = mask_ratio)

    X_train = x_train
    X_test = x_test
    X_mask = result_del
    X_pad = result_pad0

    file_name_train = save_txt_name + "_train_x.txt"
    file_name_test = save_txt_name + "_test_x.txt"
    file_name_mask = save_txt_name + "_mask_" + str(mask_ratio) + "_x.txt"
    file_name_pad = save_txt_name + "_pad_" + str(mask_ratio) + "_x.txt"
    file_name_fmask = save_txt_name + "_fmask_" + str(mask_ratio) + "_x.txt"

    name_list = [file_name_mask, file_name_pad, file_name_fmask]
    X_list = [X_mask, X_pad, result_fmask]
    Y_list = [y_test, y_test, y_test]
    for i in range(len(X_list)):
        print(i)
        file_save(name_list[i], X_list[i], Y_list[i])

def Generate_Pretrain_Lengths(path_pkl, Label2num, Num2label, save_txt_name):
    df = pd.read_pickle(path_pkl)
    if len(Num2label) != 0:
        df = select_from_df(df, Label2num)
        df.replace({"Label": Num2label}, inplace=True)
    pkt_len_, Label = df['ip_lengths'].values.tolist(), df['Label'].values.tolist()
    sequences = []
    for sequence in pkt_len_:
        seq = [s+1600 for s in sequence]
        sequences.append(seq)

    x_train, x_test, y_train, y_test = train_test_split(sequences, Label, test_size=0.2, stratify=Label, random_state=42)
    file_name_all = save_txt_name + "_Pretrain_lengths_all.txt"
    file_name_train = save_txt_name + "_Pretrain_lengths.txt"
    file_save(file_name_all, sequences, 0)
    file_save(file_name_train, x_train, 0)

def Generate_Pretrain_Timestamps(path_pkl, Label2num, Num2label, save_txt_name):
    df = pd.read_pickle(path_pkl)
    if len(Num2label) != 0:
        df = select_from_df(df, Label2num)
        df.replace({"Label": Num2label}, inplace=True)
    pkt_len_, Label = df['ip_timestamps'].values.tolist(), df['Label'].values.tolist()
    sequences = timestamps2interval(pkt_len_)

    x_train, x_test, y_train, y_test = train_test_split(sequences, Label, test_size=0.2, stratify=Label, random_state=42)
    file_name_all = save_txt_name + "_Pretrain_Timestamps_all.txt"
    file_name_train = save_txt_name + "_Pretrain_timestamps.txt"
    file_save(file_name_all, sequences, 0)
    file_save(file_name_train, x_train, 0)

