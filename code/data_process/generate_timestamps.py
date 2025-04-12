import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import random
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
# from txt_roberta import Generate_Pretrain_Timestamps, Generate_Pretrain_Lengths
from train_test_data import random_mask, select_from_df, mask_file_save


def generate_mask_pad(path_pkl, Label2num,part, mask_ratio, save, save_name):
    df = pd.read_pickle(path_pkl)
    if part:
        data = select_from_df(df, Label2num)
    else:
        df.replace({"Label": Label2num}, inplace=True)
        data = df

    X = timestamps2interval(data["ip_timestamps"].to_list())
    y = data["Label"]
 
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_test, X_test_mask, X_test_pad0, x_fmask = df_tsp2tensor_split(x_test, mask=1, mask_ratio=mask_ratio)
    print("X_mask & pad shape:", X_test_mask.shape, X_test_pad0.shape)

    if save:
        torch.save(X_test_mask,"./" + str(save_name) + "_Tsp_x_mask" + str(mask_ratio) + ".pt")
        torch.save(X_test_pad0,"./" + str(save_name) + "_Tsp_x_pad" + str(mask_ratio) + ".pt")
        filename_fmask = "./" + str(save_name) + "_Tsp_mask_tsp" + str(mask_ratio) + ".txt"
        mask_file_save(filename_fmask, x_fmask)


def df_tsp2tensor_split(df_list, mask, mask_ratio):
    sequences = [torch.tensor(seq) for seq in df_list]
    max_length = 1000
    padded_sequence = [F.pad(sequence, (0, max_length - sequence.size(0))) for sequence in sequences]
    data_x = pad_sequence(padded_sequence, batch_first=True, padding_value=0).unsqueeze(1).float() / 10000

    if mask:
        sequences_ = [seq_ for seq_ in df_list]
        result_del, result_pad0, result_fmask = random_mask(sequences_, mask_ratio = mask_ratio)
        result_del, result_pad0 = [torch.tensor(seq) for seq in result_del], [torch.tensor(seq) for seq in result_pad0]
        result_del = [F.pad(sequence, (0, max_length - sequence.size(0))) for sequence in result_del]
        result_pad0 = [F.pad(sequence, (0, max_length - sequence.size(0))) for sequence in result_pad0]
        padded_mask = pad_sequence(result_del, batch_first=True, padding_value=0).unsqueeze(1)
        padded_pad0 = pad_sequence(result_pad0, batch_first=True, padding_value=0).unsqueeze(1)
        data_x_mask, data_x_pad0 = padded_mask.float() / 10000, padded_pad0.float() / 10000
    else: return data_x
    
    return data_x, data_x_mask, data_x_pad0, result_fmask   # type(result_fmask) --> list

def timestamps2interval(ip_timestamps):
    in_val = []
    for timestamps in ip_timestamps:
        time_diffs = [0] + [(timestamps[i] - timestamps[i-1]) * 1e4 for i in range(1, len(timestamps))]
        processed_inval = [int(min(10000, max(1, x))) for x in time_diffs[1:]]
        processed_inval.insert(0, int(time_diffs[0]))
        in_val.append(processed_inval)
    return in_val

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

    X = timestamps2interval(data["ip_timestamps"].to_list())
    y = data["Label"]
 
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train = df_tsp2tensor_split(x_train, mask=0, mask_ratio=mask_ratio)
    X_test, X_test_mask, X_test_pad0, x_fmask = df_tsp2tensor_split(x_test, mask=1, mask_ratio=mask_ratio)

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("X_mask & pad shape:", X_test_mask.shape, X_test_pad0.shape)

    if save:
        torch.save(X_train,"./" + str(save_name) + "_Tsp_x_train.pt")
        torch.save(X_test,"./" + str(save_name) + "_Tsp_x_test.pt")
        torch.save(X_test_mask,"./" + str(save_name) + "_Tsp_x_mask" + str(mask_ratio) + ".pt")
        torch.save(X_test_pad0,"./" + str(save_name) + "_Tsp_x_pad" + str(mask_ratio) + ".pt")
        filename_fmask = "./" + str(save_name) + "_Tsp_mask_tsp" + str(mask_ratio) + ".txt"
        mask_file_save(filename_fmask, x_fmask)


if __name__ == '__main__':

    path_USTC = 'E:/BertFC/rd_data/pkl/USTC_copy_6.pkl'
    path_DoH = 'E:/BertFC/rd_data/pkl/DoH_len_tsp.pkl'
    path_iot = 'E:/BertFC/rd_data/pkl/DoS-DDoS-MQTT-IoT.pkl'
    path_MalAnd = 'E:/BertFC/rd_data/pkl/MalDroid.pkl'

    USTC_Label2num = {"Weibo":0,"SMB":1,"Virut":2, "Htbot":3, "Neris":4, "Miuref":5, "Nsis":6, "Zeus":7, "Geodo":8, "Shifu":9}
    USTC_Label2num6 = {"Weibo-1":0,"Weibo-2":0,"Weibo-3":0, "Weibo-4":0, "SMB-1":1, "SMB-2":1, "Htbot":2, "Virut":3, "Miuref":4, "Neris":5,
                       "Nsis-ay":5, "Zeus":5,"Geodo":5,"Shifu":5}
    USTC_Label2num4 = {"Weibo-1":0,"Weibo-2":0,"Weibo-3":0, "Weibo-3":0, "SMB-1":1, "SMB-2":1, "Htbot":2, "Virut":3, "Miuref":3, "Neris":3}

    # Benign --> class 8; Malicious --> 3
    DoH_Label2num11 = {"BenignDoH_NonDoH-Chrome-AdGuard":0, "BenignDoH_NonDoH-Chrome-Cloudflare":1, "BenignDoH_NonDoH-Chrome-Google":2, "BenignDoH_NonDoH-Chrome-Quad9":3, 
                    "BenignDoH_NonDoH-Firefox-AdGuard":4,"BenignDoH_NonDoH-Firefox-CloudFlare":5, "BenignDoH_NonDoH-Firefox-Google":6, "BenignDoH_NonDoH-Firefox-Quad9":7, 
                    "MaliciousDoH-dns2tcp-Pcaps":8, "MaliciousDoH-dnscat2-Pcaps":9, "MaliciousDoH-iodine-Pcaps":10}

    DoH_Label2num4 = {"BenignDoH_NonDoH-Chrome-AdGuard":0, "BenignDoH_NonDoH-Chrome-Cloudflare":0, "BenignDoH_NonDoH-Chrome-Google":0, "BenignDoH_NonDoH-Chrome-Quad9":0, 
                    "BenignDoH_NonDoH-Firefox-AdGuard":0,"BenignDoH_NonDoH-Firefox-CloudFlare":0, "BenignDoH_NonDoH-Firefox-Google":0, "BenignDoH_NonDoH-Firefox-Quad9":0, 
                    "MaliciousDoH-dns2tcp-Pcaps":1, "MaliciousDoH-dnscat2-Pcaps":2, "MaliciousDoH-iodine-Pcaps":3}
    
    Iot_Label2num = {"NormalData":0, "SYN_TCP_Flooding":1, "Delayed_Connect_Flooding":2, "Basic_Connect_Flooding":3,"Invalid_Subscription_Flooding":4, "Connect_Flooding_with_WILL_payload":5}

    MalAnd_Label2num = {"sc":0, "ra":1, "ad":2, "sms":3, "ps":4}


    mask_ratio_list = [0.2, 0.3]
    df2tensor_mask(path_USTC, USTC_Label2num6,part=0, mask_ratio=0.1, save=1, save_name="USTC_6")
    for i in range(len(mask_ratio_list)):
        generate_mask_pad(path_USTC, USTC_Label2num6,part=0, mask_ratio=mask_ratio_list[i], save=1, save_name="USTC_6")
