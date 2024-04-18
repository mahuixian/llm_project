from datasets import load_dataset
from datasets import DatasetDict
from tokenizers import (
    Tokenizer,
    normalizers,
    pre_tokenizers,
    models,
    processors,
    decoders,
    trainers,
)
import json
import os
from pathlib import Path


def dataloader(data_dir):
    data_files = []
    for file in os.listdir(data_dir):
        if file.endswith('.json'):
            data_files.append(os.path.join(data_dir, file))
    datasets = load_dataset('json', data_files=data_files)
    return datasets
        


def batch_iterator(datasets, batch_size=10000):
    for i in range(0, len(datasets), batch_size):
        items = [instruction+' '+input+' '+output for instruction, input, output in zip(datasets[i:i+batch_size]["instruction"], datasets[i:i+batch_size]["input"], datasets[i:i+batch_size]["output"])]
        yield items
    


def train_tokenizer(data_dir, tokenizer_path):
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    normalizer = normalizers.Sequence([normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents(), normalizers.Strip()])
    # # #标点符号，空白，数字
    pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation(), pre_tokenizers.Digits(individual_digits=True)])
    model = models.BPE(unk_token="[UNK]")
    decoder = decoders.BPEDecoder()
    tokenizer.normalizer = normalizer
    tokenizer.pre_tokenizer = pre_tokenizer
    tokenizer.model = model
    tokenizer.decoder = decoder
    
    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "<s>", "</s>", "<t>", "</t>"]
    trainer = trainers.BpeTrainer(special_tokens=special_tokens, vocab_size=54000)
    tokenizer.train_from_iterator(batch_iterator(dataloader(data_dir)), trainer=trainer)
    
    # #字节级 BPE 可能在生成的令牌中包括空白。如果您不希望偏移量包含这些空格，那么必须使用这个 PostProcessor。
    post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],)

    tokenizer.post_processor = post_processor
    tokenizer.save(tokenizer_path)
    

def test_tokenizer(tokenizer_path, input):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(tokenizer.encode(input).tokens)
    

if __name__ == '__main__':
    data_dir = './data'
    
    tokenizer_path = './tokenizer.json'
    train_tokenizer(data_dir, tokenizer_path)
    test_tokenizer(tokenizer_path, '今天天气真好啊！')


            
