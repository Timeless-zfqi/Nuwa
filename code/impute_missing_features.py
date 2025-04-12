import pickle
import torch
from transformers import pipeline
from transformers import RobertaTokenizerFast
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

def predict_mask2list(texts, predict_result):
    predict_list = []
    for input_sequence, result in zip(texts, predict_result):
        fill_num = []
        if isinstance(result[0], list):
            for i in range(len(result)):
                fill_num.append(result[i][0]['token_str'])
        else:
            fill_num.append(result[0]['token_str'])
        
        n_pad = []
        for i in range(len(fill_num)):
            if fill_num[i] == '<pad>':
                n_pad = n_pad.append(i+1)
                fill_num[i] = '1600'
            input_sequence = input_sequence.replace('<mask>', fill_num[i], 1)
        # print("fill mask sequence: ", input_sequence, type(input_sequence))
        input_list = list(map(int, input_sequence.split()))
        # print("fill mask list: ", input_list, type(input_list))
        
        predict_list.append(input_list)

    return predict_list, n_pad

def roberta_mask_list2tensor(fmask_list):
    froberta2seq = []
    for fmask in fmask_list:
        fmask = [(num - 1600) for num in fmask] #还原原始序列
        froberta2seq.append(fmask)
    
    sequences = [torch.tensor(seq) for seq in froberta2seq]
    max_length = 512
    padded_sequence = [F.pad(sequence, (0, max_length - sequence.size(0))) for sequence in sequences]
    fmask_tensor = pad_sequence(padded_sequence, batch_first=True, padding_value=0).unsqueeze(1).float() / 1500

    return fmask_tensor

def fill_mask2tensor(mask_name, fmask_name):

    with open(mask_name, 'r') as file:
        texts = [line.rstrip('\n') for line in file]

    print("Starting predicted {mask_name}...")
    pre = predict_mask(texts)
    print("Done...")

    print("Running predicted_mask2list...")
    pre_list, n_pad = predict_mask2list(texts, pre)
    print("Done... and n_pad = ", len(n_pad))
    
    print("Running To Tennsor...")
    fmask_tensor = roberta_mask_list2tensor(pre_list)
    torch.save(fmask_tensor, fmask_name)

def save_mid_list(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_mid_list(filename):
    with open(filename, 'rb') as f:
        load_data = pickle.load(f)
    return load_data


# path
model_dir = './model'
# load tokenizer
deberta_tokenizer = RobertaTokenizerFast(tokenizer_file=model_dir+'/wordlevel.json', max_len=512)
from transformers import DebertaForMaskedLM
model = DebertaForMaskedLM.from_pretrained(model_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

predict_mask = pipeline('fill-mask',
                        model=model,
                        tokenizer=deberta_tokenizer)

mask_path1 = './data/IoT_mask_len0.1.txt'
mask_path2 = './data/IoT_mask_len0.2.txt'
mask_path3 = './data/IoT_mask_len0.3.txt'
mask_path = [mask_path1, mask_path2, mask_path3]

fmask_path1 = './data/IoT_impute_len0.1.pt'
fmask_path2 = './data/IoT_impute_len0.2.pt'
fmask_path3 = './data/IoT_impute_len0.3.pt'
fmask_path = [fmask_path1, fmask_path2, fmask_path3]

for i in range(len(mask_path)):
    fill_mask2tensor(mask_path[i], fmask_path[i])
print("Successful!")