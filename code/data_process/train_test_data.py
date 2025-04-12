import pandas as pd
import ast
import torch
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import random
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from txt_roberta import Generate_Pretrain_Lengths, Generate_Pretrain_Timestamps


def generate_mask_pad(path_pkl, Label2num,part, mask_ratio, save, save_name):
    df = pd.read_pickle(path_pkl)
    if part:
        data = select_from_df(df, Label2num)
    else:
        df.replace({"Label": Label2num}, inplace=True)
        data = df

    X = data["ip_lengths"]
    y = data["Label"]
 
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_test, X_test_mask, X_test_pad0, x_fmask = dfx2tensor_split(x_test, mask=1, mask_ratio=mask_ratio)
    print("X_mask & pad shape:", X_test_mask.shape, X_test_pad0.shape)

    if save:
        torch.save(X_test_mask,"./" + str(save_name) + "_x_mask" + str(mask_ratio) + ".pt")
        torch.save(X_test_pad0,"./" + str(save_name) + "_x_pad" + str(mask_ratio) + ".pt")
        filename_fmask = "./" + str(save_name) + "_mask_len" + str(mask_ratio) + ".txt"
        mask_file_save(filename_fmask, x_fmask)


def df2tensor_mask(path_pkl, Label2num,part, mask_ratio, save, save_name):
    df = pd.read_pickle(path_pkl)
    print("Initial Num of class: \n", df['Label'].value_counts())
    if part:
        data = select_from_df(df, Label2num)
        print("Selected Num of class: \n", data['Label'].value_counts())
    else:
        df.replace({"Label": Label2num}, inplace=True)
        data = df
        print("Selected Num of class: \n", data["Label"].value_counts())

    X = data["ip_lengths"]
    y = data["Label"]
 
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train = dfx2tensor_split(x_train, mask=0, mask_ratio=mask_ratio)
    X_test, X_test_mask, X_test_pad0, x_fmask = dfx2tensor_split(x_test, mask=1, mask_ratio=mask_ratio)

    y_train = y_train.values.tolist()
    y_test = y_test.values.tolist()
    Y_train = torch.Tensor(y_train)
    Y_test = torch.Tensor(y_test)

    print("X_train shape:", X_train.shape)
    print("y_train shape:", Y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", Y_test.shape)
    print("X_mask & pad shape:", X_test_mask.shape, X_test_pad0.shape)

    if save:
        torch.save(X_train,"./" + str(save_name) + "_x_train.pt")
        torch.save(X_test,"./" + str(save_name) + "_x_test.pt")
        torch.save(Y_train,"./" + str(save_name) + "_y_train.pt")
        torch.save(Y_test,"./" + str(save_name) + "_y_test.pt")
        torch.save(X_test_mask,"./" + str(save_name) + "_x_mask" + str(mask_ratio) + ".pt")
        torch.save(X_test_pad0,"./" + str(save_name) + "_x_pad" + str(mask_ratio) + ".pt")
        filename_fmask = "./" + str(save_name) + "_mask_len" + str(mask_ratio) + ".txt"
        mask_file_save(filename_fmask, x_fmask)

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
        masked_sequence_fill_mask.append([seq_item+1600 if mask[i] else "<mask>" for i, seq_item in enumerate(sequence)])

    return masked_sequence_delete, masked_sequence_zero_pad, masked_sequence_fill_mask

def list2tensor(df_list):
    sequences = [torch.tensor(seq) for seq in df_list]
    max_length = 512
    padded_sequence = [F.pad(sequence, (0, max_length - sequence.size(0))) for sequence in sequences]
    padded_sequences = pad_sequence(padded_sequence, batch_first=True, padding_value=0).unsqueeze(1)
    data_x = padded_sequences.float() / 1600
    print("data_x.shape: ",data_x.shape, '\n data_x.dtype: ', data_x.dtype)
    return data_x


def dfx2tensor_split(df_list, mask, mask_ratio):
    sequences = [torch.tensor(seq) for seq in df_list]
    max_length = 512
    padded_sequence = [F.pad(sequence, (0, max_length - sequence.size(0))) for sequence in sequences]
    data_x = pad_sequence(padded_sequence, batch_first=True, padding_value=0).unsqueeze(1).float() / 1500

    if mask:
        sequences_ = [seq_ for seq_ in df_list]
        # print("sequences_: ", len(sequences_), sequences_[:2])
        result_del, result_pad0, result_fmask = random_mask(sequences_, mask_ratio = mask_ratio)
        result_del, result_pad0 = [torch.tensor(seq) for seq in result_del], [torch.tensor(seq) for seq in result_pad0]
        result_del = [F.pad(sequence, (0, max_length - sequence.size(0))) for sequence in result_del]
        result_pad0 = [F.pad(sequence, (0, max_length - sequence.size(0))) for sequence in result_pad0]
        padded_mask = pad_sequence(result_del, batch_first=True, padding_value=0).unsqueeze(1)
        padded_pad0 = pad_sequence(result_pad0, batch_first=True, padding_value=0).unsqueeze(1)
        data_x_mask, data_x_pad0 = padded_mask.float() / 1600, padded_pad0.float() / 1600
    else: return data_x
    return data_x, data_x_mask, data_x_pad0, result_fmask   # type(result_fmask) --> list


def df2fsnet_mask(path_pkl, mask_ratio, save_txt_name,Label2num, Num2label):
    df = pd.read_csv(path_pkl)
    if len(Num2label) != 0:
        df = select_from_df(df, Label2num)
        df.replace({"Label": Num2label}, inplace=True)
    pkt_len_, Label = df['ip_lengths'].values.tolist(), df['Label'].values.tolist()
    x_train, x_test, y_train, y_test = train_test_split(pkt_len_, Label, test_size=0.2, stratify=Label, random_state=42)
    result_del, result_pad0, result_fmask = random_mask(x_test, mask_ratio = mask_ratio)

    desired_dimension = 512
    X_train = [sublist + [0] * (desired_dimension - len(sublist)) for sublist in x_train]
    X_test = [sublist + [0] * (desired_dimension - len(sublist)) for sublist in x_test]
    X_mask = [sublist + [0] * (desired_dimension - len(sublist)) for sublist in result_del]
    X_pad = [sublist + [0] * (desired_dimension - len(sublist)) for sublist in result_pad0]

    file_name_train = save_txt_name + "_train_x.txt"
    file_name_test = save_txt_name + "_test_x.txt"
    file_name_mask = save_txt_name + "_mask_" + str(mask_ratio) + "_x.txt"
    file_name_pad = save_txt_name + "_pad_" + str(mask_ratio) + "_x.txt"
    

    def file_save(file_name, x, y):
        f = open(file_name, "w") # 0
        for k in range(len(x)):
            f.write(y[k] + ". ") # str(L)
            for i, pkt_len in enumerate(x[k]):
                if i == 512:
                    break
                f.write(str(pkt_len) + " ")
            f.write("\n")
        f.close()
    
    name_list = [file_name_train, file_name_test, file_name_mask, file_name_pad]
    X_list = [X_train, X_test, X_mask, X_pad]
    Y_list = [y_train, y_test, y_test, y_test]

    for i in range(len(X_list)):
        print(i)
        file_save(name_list[i], X_list[i], Y_list[i])

def select_from_df(df, Label2num):
    df.replace({"Label": Label2num}, inplace=True)
    df1 = df[df['Label'].apply(lambda x: isinstance(x, int) if pd.notna(x) else False).astype(bool)]
    return df1

def mask_file_save(file_name, x):
    f = open(file_name, "w") # 0
    for k in range(len(x)):
        for i, pkt_len in enumerate(x[k]):
            if i == 1000:
                break
            f.write(str(pkt_len) + " ")
        f.write("\n")
    f.close()

def Generate_train_test_data(path_pkl, Label2num, save_name):
    mask_ratio_list = [0.2, 0.3]
    df2tensor_mask(path_pkl, Label2num,part=0, mask_ratio=0.1, save=1, save_name=save_name)
    for i in range(len(mask_ratio_list)):
        generate_mask_pad(path_pkl, Label2num,part=0, mask_ratio=mask_ratio_list[i], save=1, save_name=save_name)
    
    Generate_Pretrain_Lengths(path_pkl, Label2num, [], save_txt_name=save_name)
    Generate_Pretrain_Timestamps(path_pkl, Label2num, [], save_txt_name=save_name)

if __name__ == '__main__':

    path_USTC = 'E:/BertFC/rd_data/pkl/USTC.pkl'
    path_DoH = 'E:/BertFC/rd_data/pkl/DoH.pkl'
    path_iot = 'E:/BertFC/rd_data/pkl/IoT2023.pkl'
    path_MalAnd = 'E:/BertFC/rd_data/pkl/MalAnd.pkl'


    USTC_Label2num = {"Weibo":0,"SMB":1,"Virut":2, "Htbot":3, "Neris":4, "Miuref":5, "Nsis":6, "Zeus":7, "Geodo":8, "Shifu":9}

    DoH_Label2num5 = {"BenignDoH_NonDoH-Chrome-AdGuard":0, "BenignDoH_NonDoH-Chrome-Cloudflare":0, "BenignDoH_NonDoH-Chrome-Google":0, "BenignDoH_NonDoH-Chrome-Quad9":0, 
                    "BenignDoH_NonDoH-Firefox-AdGuard":1,"BenignDoH_NonDoH-Firefox-CloudFlare":1, "BenignDoH_NonDoH-Firefox-Google":1, "BenignDoH_NonDoH-Firefox-Quad9":1, 
                    "MaliciousDoH-dns2tcp-Pcaps":2, "MaliciousDoH-dnscat2-Pcaps":3, "MaliciousDoH-iodine-Pcaps":4}
    
    Iot2023_Label2num = {"DoS-HTTP_Flood":0, "DDoS-HTTP_Flood":1, "DoS-TCP_Flood":2, "MITM-ArpSpoofing":3, "DDoS-TCP_Flood":4, "DNS_Spoofing":5}
    
    CICMalAnd_Label2num = {"Benign":0, "Scareware":1, "Adware":2, "Ransomware":3, "SMSMalware":4}

    Generate_train_test_data(path_USTC, USTC_Label2num, "USTC")
