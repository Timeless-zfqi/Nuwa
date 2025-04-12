

# Nüwa: A Network Traffic Side-channel Feature Imputation Framework Based on Pre-training



**The repository of Nüwa, a network feature imputation model on encrypted traffic.**

Nüwa is a framework for learning side-channel contextual relationships from encrypted traffic, which could be effectiveness applied to different encrypted traffic scenarios to **enhance the ML/DL model's performance with pack loss**. Nüwa consists of two stages: (1) pre-training to learn patterns of application side-channel features with unlabeled data, and (2) imputation of missing data within input sequences.

<img src="https://github.com/AnonymousCodeBaseA/AnonymousCodeBaseA-nvwa/blob/main/images/nvwa.png" alt="nvwa" style="zoom:25%;" />

<p align="center">Fig. 1 Overview of Nüwa</p>


# Contents
- [Introduction](#Introduction)
- [Requirements](#Requirements)
- [Dataset and preparation](#Dataset-and-preparation)
- [Using Nüwa](#Using-Nüwa)
- [Reproduce Nüwa](#Reproduce-Nüwa)
- [Acknowledgement](#Acknowledgement) 

# Introduction  
As illustrated in Figure 1, the presence of incomplete features significantly degrades the performance of current traffic analysis models. Our project in this paper introduces a pre-training-based augmentation framework, denoted as Nüwa, which imputes the side-channel features of encrypted network traffic due to packet loss. Nüwa utilizes a Transformer-based structure comprising three modules: Sequence2Embedding for traffic representation, a Traffic Noising based Self-supervised Pre-training Mask strategy (TFM), and a Traffic Side-channel Feature Imputation model (TFI). The framework’s architecture is depicted in Fig. 2.

__Modules of Nüwa include:__

* __Sequence2Embedding Traffic Representation__.
This module serves as a word-level embedding technique for side-channel feature extraction, encoding time series data into a token sequence to facilitate the pre-training of the TFI.
* __TFM: Traffic Noise-based Self-supervised Pre-trained Masking Strategy__.
  This module is designed for use in pre-training strategies, specifically tailored for Masked Language Model (MLM) tasks.
* __TFI: a Traffic Side-Channel Feature Imputation Module.__
  TFI is the model used to impute the missing side-channel features in the input sequences.



<div align="center">
<img src=https://github.com/AnonymousCodeBaseA/AnonymousCodeBaseA-nvwa/blob/main/images/background.png width=50% />
</div>  


<p align="center">Fig. 2 Background</p>



# Requirements

Before using this project, you should configure the following environment.  
1. Requirements
```
python >= 3.8
transformer = 4.30.2
pytorch = 1.12.0
torchvision==0.13.0
torchaudio==0.12.0
```
2. Basic Dependencies
```
scikit-learn
flowcontainer
tokenizers
tqdm
```
3. Others

```shell
Ubuntu 20.04
```



# Dataset and preparation
1.Dataset 

We use the open source [CIC-AndMal-2017](https://www.unb.ca/cic/datasets/andmal2017.html "CIC-AndMal-2017")  dataset, [CIRA-CICDoHBrw-2020](https://www.unb.ca/cic/datasets/dohbrw-2020.html "CIRA-CICDoHBrw-2020") dataset, [CIC-IoT-2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html "CIC-IoT-2023")  IoT devices dataset, and [USTC-TFC](https://github.com/yungshenglu/USTC-TK2016 "USTC-TFC")  PC terminals dataset.

2.Pcap to side-channel features

You need to use split TCP pcaps into flow sessions. Then, using flowcontainer tool to extract time series features with the maximum packet length at 1600 and set the minimum session length to 20.

```shell
# run pcap2TCP.py to filliter TCP flows
# modify: theh path of 'path_pcap', 'path_wireshark' and 'path_save_pcap' to generate TCP flows.
python pcap2TCP.py
# Extract side-channel features from flow sessions
python session2features.py
# Generate train, test and pre-training data
python train_test_data.py
-> tensor(train, test), txt(pre-training)
```



# USing Nüwa

You can now use Nüwa directly through the pre-trained [model](https://drive.google.com/drive/folders/1CtZaJN0-gMPKFv3z8F8GBivZHo4pboSP?usp=sharing) or download via:

After obtaining the pre-trained model, Nüwa could be applied to the missing feature imputation task by TFI at session level. The default path of the TFI model is `model/model.safetensors`, and the `vocab` et .al are all in the `model`. Then you can do inference with the feature imputation model:

```shell
# import the download pre-training model, and run:
python impute_missing_features.py

# You can make up by yourself for the format of imputed output side-channel, such as tensor, list... 
```



# Reproduce Nüwa

## pre-training

If you want to train a new Nüwa in your own scenario, you should firstly run the `train_test_data.py` to generate pre-training data (`file.txt`) by your own network traffic. And then put the generated `file.txt` into the training folder, and create a new folder to save `vocab`, `config`,`model.safetensors`  and `training_args.bin`. Finally, set the `Config` and run the `pre-training.py` for Nüwa pre-training.

```python
# Generate pre-training data

# Create new folder to save pre-traing model

# pre-training
'''
Model_config = Config(
    vocab_size = 3106, # vocab size
    max_position_embeddings = 514,
    num_attention_heads = 12,
    num_hidden_layers = 6,
    type_vocab_size = 0
)
'''
# Run
python pre-training.py

```

In addition, if you want to change the TFM strategy to suit for your network environment, you could modify the ratio of TFM sub-strategy.

```python
# Modified MLM strategies
class DataCollatorForLanguageModeling(DataCollatorMixin):
    #....
    
    mask_strategy = torch.bernoulli(torch.full(labels.shape, 0.50)).bool() & masked_indices
    delete_strategy = torch.bernoulli(torch.full(labels.shape, 0.10)).bool() & masked_indices & ~mask_strategy
    infill_strategy = torch.bernoulli(torch.full(labels.shape, 0.20)).bool() & masked_indices & ~mask_strategy & ~delete_strategy
    permute_strategy = torch.bernoulli(torch.full(labels.shape, 0.10)).bool() & masked_indices & ~mask_strategy & ~delete_strategy & ~infill_strategy
    
    # ...
```



## Side-channel feature imputation

To see an example of how to use Nüwa for the encrypted traffic classification tasks, go to the [Using Nüwa](#Using-Nüwa) and run `python impute_missing_features.py`.



# Acknowledgement
Thanks for these awesome resources that were used during the development of the Nüwa：  
* https://www.unb.ca/cic/datasets/index.html
* https://huggingface.co/
* https://github.com/yungshenglu/USTC-TK2016
* https://timeseriesai.github.io/tsai/
