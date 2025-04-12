import imp
from flowcontainer.extractor import extract
import pandas as pd
from collections import OrderedDict
import os

def extract_seq(path_pcap):
    result = extract(path_pcap,filter='tcp')
    ip_lengths = []
    for key in result:
        value = result[key]
        ip_lens = value.ip_lengths
        ip_lengths.append(ip_lens)

    return ip_lengths

def process_sequences(input_list):
    result_list = []
    for sequence in input_list:
        if len(sequence) >= 20 and all(abs(num) <= 1600 for num in sequence):
            result_list.append(sequence)

    return result_list

def extract_seq_tsp(path_pcap):
    ip_lengths = []
    ip_timestamps = []
    result = extract(path_pcap,filter='tcp')
    for key in result:
        value = result[key]
        ip_lens = value.ip_lengths
        ip_tsp = value.ip_timestamps
        ip_lengths.append(ip_lens)
        ip_timestamps.append(ip_tsp)
    # print("=== Len(ip_lens: & ip_tim)", len(ip_lengths), len(ip_timestamps))

    return ip_lengths, ip_timestamps

def process_sequences_timestamps(input_list, ip_timestamps):
    # print("=== after Len(): ", len(input_list), len(ip_timestamps))
    result_list = []
    result_tsp = []
    for sequence, ip_tsp in list(zip(input_list, ip_timestamps)):
         if len(sequence) >= 20 and all(abs(num) <= 1600 for num in sequence):
            result_list.append(sequence)
            result_tsp.append(ip_tsp)


    return result_list, result_tsp

def all_pkt_length(root_path):
        lengths = pd.DataFrame(columns=['ip_lengths','Label'])
        root_list = os.listdir(root_path)   #[DoH-C, dnscat2, ...]
        print("root_list: ", root_list)
        for p in root_list:
            label_ = p
            p1 = os.path.join(root_path,p)
            print("p1: ", p1)
            p_list = os.listdir(p1)
            # print(p_list)
            for p2 in p_list:           # F:\datasets\CICIDS2017\output\DoS
                path_pcap = os.path.join(p1,p2)
                print(path_pcap)
                ip_lens = extract_seq(path_pcap)
                seq = process_sequences(ip_lens)
                if len(seq) != 0:
                    # for seq_ in seq:
                    #     lengths = lengths.append(pd.DataFrame({'ip_lengths': [seq_],'Label': label_}))
                     lengths_ = pd.DataFrame({'ip_lengths': seq, 'Label': [label_] * len(seq)})
                     print("length_ : ", lengths_.shape[0])
                lengths = lengths.append(lengths_)
                print("num of sample: ",lengths.shape[0])
                   
        return lengths

def all_pkt_length_timestamps(root_path):
        lengths = pd.DataFrame(columns=['ip_lengths', 'ip_timestamps','Label'])
        root_list = os.listdir(root_path)   #[DoH-C, dnscat2, ...]
        print("root_list: ", root_list)
        for p in root_list:
            label_ = p
            p1 = os.path.join(root_path,p)
            print("p1: ", p1)
            p_list = os.listdir(p1)
            # print(p_list)
            for p2 in p_list:
                path_pcap = os.path.join(p1,p2)
                print(path_pcap)
                ip_lens, ip_tsps = extract_seq_tsp(path_pcap)
                seq, tsp = process_sequences_timestamps(ip_lens, ip_tsps)
                if len(seq) != 0:
                     lengths_ = pd.DataFrame({'ip_lengths': seq, 'ip_timestamps': tsp, 'Label': [label_] * len(seq)})
                     print("length_ : ", lengths_.shape[0])
                lengths = lengths.append(lengths_)
                print("num of sample: ",lengths.shape[0])
                   
        return lengths

def USTC_all_pkt_length(root_path): #  Applicable to multiple folders, each containing different categories of pcap files, labeled by the name of each pcap.
        lengths = pd.DataFrame(columns=['ip_lengths','Label'])
        root_list = os.listdir(root_path)   #[Benign, Malware]
        print("root_list: ", root_list)
        for p in root_list:
            p1 = os.path.join(root_path,p)
            print("p1: ", p1)
            p_list = os.listdir(p1)
            # print(p_list)
            for p2 in p_list:
                lengths_ = pd.DataFrame(columns=['ip_lengths','Label'])
                label_ = p2.split('.')[0]
                path_pcap = os.path.join(p1,p2)
                print(path_pcap)
                ip_lens = extract_seq(path_pcap)
                seq = process_sequences(ip_lens)
                if len(seq) != 0:
                     lengths_ = pd.DataFrame({'ip_lengths': seq, 'Label': [label_] * len(seq)})
                     print("length_ : ", lengths_.shape[0])
                lengths = lengths.append(lengths_)
                print("num of sample: ",lengths.shape[0])
        
        return lengths

def USTC_all_pkt_length_timestamps(root_path):
        lengths = pd.DataFrame(columns=['ip_lengths', 'ip_timestamps', 'Label'])
        root_list = os.listdir(root_path)   #[DoH-C, dnscat2, ...]
        print("root_list: ", root_list)
        for p in root_list:
            p1 = os.path.join(root_path,p)
            print("p1: ", p1)
            p_list = os.listdir(p1)
            # print(p_list)
            for p2 in p_list:           # ..\datasets\CICIDS2017\output\DoS
                lengths_ = pd.DataFrame(columns=['ip_lengths','Label'])
                label_ = p2.split('.')[0]
                path_pcap = os.path.join(p1,p2)
                print(path_pcap)
                ip_lens, ip_tsps = extract_seq_tsp(path_pcap)
                seq, tsp = process_sequences_timestamps(ip_lens, ip_tsps)
                if len(seq) != 0:
                     lengths_ = pd.DataFrame({'ip_lengths': seq, 'ip_timestamps': tsp, 'Label': [label_] * len(seq)})
                     print("length_ : ", lengths_.shape[0])
                lengths = lengths.append(lengths_)
                print("num of sample: ",lengths.shape[0])
        
        return lengths

def Andriod_pkt_length_timestamps(root_path):
        lengths = pd.DataFrame(columns=['ip_lengths', 'ip_timestamps', 'Label'])
        root_list = os.listdir(root_path)   #[DoH-C, dnscat2, ...]
        print("root_list: ", root_list)
        for p in root_list:
            p1 = os.path.join(root_path,p)
            print("p1: ", p1)
            p_list = os.listdir(p1)
            label_ = p 
            # print(p_list)
            for p2 in p_list:      # ..\datasets\CICIDS2017\output\DoS
                # label_ = p2.split('_')[0]
                path_pcap = os.path.join(p1,p2)
                print(path_pcap)
                ip_lens, ip_tsps = extract_seq_tsp(path_pcap)
                seq, tsp = process_sequences_timestamps(ip_lens, ip_tsps)
                if len(seq) != 0:
                     lengths_ = pd.DataFrame({'ip_lengths': seq, 'ip_timestamps': tsp, 'Label': [label_] * len(seq)})
                     print("length_ : ", lengths_.shape[0])
                lengths = lengths.append(lengths_)
                print("num of sample: ",lengths.shape[0])
        
        return lengths


if __name__ == '__main__':
    path_root = "..\\datasets\\DoH\\Processed"
    pkt = Andriod_pkt_length_timestamps(path_root)
    pkt.to_pickle('DoH.pkl')
