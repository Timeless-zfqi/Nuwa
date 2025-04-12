import os
import json
import torch
import shutil
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from transformers import DebertaForMaskedLM, RobertaTokenizerFast, DebertaConfig
from transformers import LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

model_dir='./model'
if(os.path.exists(model_dir)):
    shutil.rmtree(model_dir)
os.mkdir(model_dir)

paths = [str(x) for x in Path(".").glob("./*.txt")]
print(paths)
tokenizer = Tokenizer(WordLevel())
trainer = WordLevelTrainer(special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

from tokenizers.pre_tokenizers import Whitespace
tokenizer.pre_tokenizer = Whitespace()

tokenizer.train(files=paths,trainer=trainer)
tokenizer.save(model_dir+'/wordlevel.json')

# create tokenizer
deberta_tokenizer = RobertaTokenizerFast(tokenizer_file=model_dir+'/wordlevel.json', max_len=512)

config = DebertaConfig(
    vocab_size = len(deberta_tokenizer), # vocab size
    max_position_embeddings = 514, # position size
    num_attention_heads = 12, # attention head
    num_hidden_layers = 6,
    type_vocab_size = 0 
)

# init model
deberta_model = DebertaForMaskedLM(config=config)
deberta_model.resize_token_embeddings(len(deberta_tokenizer))

dataset = LineByLineTextDataset(tokenizer = deberta_tokenizer,
                                file_path = './IoT_pretraining_data.txt',
                                block_size=1024)
data_collector = DataCollatorForLanguageModeling(tokenizer=deberta_tokenizer,
                                                 mlm=True,
                                                 mlm_probability=0.15)

trainArgs = TrainingArguments(
    output_dir='./output',
    overwrite_output_dir=True,
    do_train=True,
    save_steps=10000,
    num_train_epochs=3,
    learning_rate=1e-4,
)
#  define trainer
trainer = Trainer(
    model=deberta_model,
    args=trainArgs,
    data_collator=data_collector,
    train_dataset=dataset
)

trainer.train()
trainer.save_model(model_dir)