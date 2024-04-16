from datasets import load_dataset
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

def batch_iterator(file_path, batch_size=10000):
    with open(file_path, 'r', encoding='utf-8') as f:
        infos = json.load(f)
        for i in range(0, len(infos), batch_size):
            items = [info['instruction'] + " " + info['input'] + " " + info['output'] for info in infos[i:i+batch_size]]
            yield items
            

def train_tokenizer(file_path):
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
    
    special_tokens = ["[CLS]", "[SEP]", "[PAD]", "[MASK]", "<s>", "</s>", "<t>", "</t>"]
    trainer = trainers.BpeTrainer(special_tokens=special_tokens, vocab_size=54000)
    tokenizer.train_from_iterator(batch_iterator(file_path), trainer=trainer)
    
    # #字节级 BPE 可能在生成的令牌中包括空白。如果您不希望偏移量包含这些空格，那么必须使用这个 PostProcessor。
    post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],)

    tokenizer.post_processor = post_processor
    tokenizer.save("./mytokenizer.json")
    

def test_tokenizer(tokenizer_path, input):
    tokenizer = Tokenizer.from_file(tokenizer_path)
    print(tokenizer.encode(input).tokens)
    

if __name__ == '__main__':
    train_tokenizer('all.json')
    test_tokenizer('今天天气真好啊！')


            
