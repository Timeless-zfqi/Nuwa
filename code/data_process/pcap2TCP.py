import os


def pcap2TCP(path_pcap, path_wireshark, path_savepcap):
    file_list = os.listdir(path_pcap)
    for p in file_list:
        print(p)
        os.system('cd %s' % (path_wireshark))
        os.system('tshark -2 -R "tcp.stream && not tcp.analysis.retransmission" -r %s -w %s -F pcap' % (path_pcap+'/'+str(p), path_savepcap+'/'+p))
        # os.system('tshark -2 -R "not tcp.analysis.retransmission" -r %s -w %s' % (path_savepcap+'/'+p, path_savepcap+'/B_'+p))


# DoHBrw-2020
def Benign_DoH2TCP(path_root, path_wireshark, path_savepcap):
    root_list = os.listdir(path_root) # [Chrom-ad, ...]
    for p in root_list:
        print("root:", p)
        os.system('mkdir %s' % (path_savepcap + '\\' + p)) # 在processed文件中创建子文件
        p1 = os.path.join(path_root,p)
        file_list = os.listdir(p1)
        for pcap in file_list:
            path_pcap = p1 + '\\' + pcap
            save_file = path_savepcap + '\\' + p +'\\' + pcap
            # print("read pacp: ", path_pcap)
            # print("save_pcap: ", save_file)
            os.system('cd %s' % (path_wireshark))
            os.system('tshark -2 -R "tcp.stream && not tcp.analysis.retransmission" -r %s -w %s -F pcap' % (path_pcap, save_file))

    pass

def Mal_DoH2TCP(path_root, path_wireshark, path_savepcap):
    file_list = os.listdir(path_root)
    class_name = path_root.split('\\')[-1]
    print(class_name)
    os.system('mkdir %s' % (path_savepcap + '\\' + class_name)) # 在processed文件中创建子文件
    for p in file_list:
        path_pcap = path_root + '\\' + p
        path_save = path_savepcap + '\\' + class_name + '\\' +p
        # print("path_pcap: ", path_pcap)
        # print("path_save: ", path_save)
        os.system('cd %s' % (path_wireshark))
        os.system('tshark -2 -R "tcp.stream && not tcp.analysis.retransmission" -r %s -w %s -F pcap' % (path_pcap, path_save))

    pass

    pass

def TwoLayer2TCP(path_root, path_wireshark, path_savepcap):
    root_list = os.listdir(path_root) # [Chrom-ad, ...]
    for p in root_list:
        print("root:", p)
        os.system('mkdir %s' % (path_savepcap + '\\' + p)) # 在processed文件中创建Class1文件夹
        p1 = os.path.join(path_root,p)
        file_list = os.listdir(p1)
        print("file_list:", file_list)
        for sub_folder in file_list:
            print("Reading file: ", sub_folder)
            p2 = os.path.join(p1, sub_folder)
            file_list_sub = os.listdir(p2)
            os.system('mkdir %s' % (path_savepcap + '\\' + p + '\\' + sub_folder)) # 在processed\\Class1文件中创建子文件夹
            for pcap in file_list_sub:
                path_pcap = p2 + '\\' + pcap
                save_file = path_savepcap + '\\' + p +'\\' + sub_folder +'\\' + pcap
                # print("read pacp: ", path_pcap)
                # print("save_pcap: ", save_file)
                os.system('cd %s' % (path_wireshark))
                os.system('tshark -2 -R "tcp.stream && not tcp.analysis.retransmission" -r %s -w %s -F pcap' % (path_pcap, save_file))



if __name__ == '__main__':

    path_pcap = "./datasets/DoH"
    path_wireshark = "./wireshark"
    path_savepcap = "./datasets/DoH/processed"

    pcap2TCP(path_pcap, path_wireshark, path_savepcap)



