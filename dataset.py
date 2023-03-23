from torch.utils.data import Dataset,DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils import *
import re
import os
from pypinyin import pinyin, lazy_pinyin, Style

def split_line(line, max_length=512):
    chunks = [line[i:i+max_length] for i in range(0, len(line), max_length)]
    return chunks

def translate(input, path):
    output = input.copy()
    input_path=osp.join(path, 'data_label.txt')
    file = open(input_path, 'w',encoding="utf-8")
    for k in range(len(output)):
        i = 0
        while i < len(output[k]):
            if re.search('[\u4e00-\u9fff]', output[k][i]):
                py = str(pinyin(output[k][i], style=0)[0][0])
                output[k] = output[k][:i] + py + output[k][i+1:]
                i += len(py)
            else:
                i += 1
        file.write("%s\n" % output[k])
    file.close()
    return output

def load_data(path, max_length=128):
    input_path=osp.join(path)
    file_list=os.listdir(input_path)
    input=[]
    for file in file_list:
        file_path=osp.join(input_path,file)
        if (os.path.basename(file_path) == "data_label.txt"):
            continue
        with open(file_path,'r',encoding="utf-8") as f:
            lines=f.readlines()
            for line in lines:
                if (len(line) <= max_length):
                    input.append(line.strip().rstrip("\n"))
                else:
                    input.extend(split_line(line.strip().rstrip("\n")))
        f.close()
    return input

class InputDataset(Dataset):
    def __init__(self, path, tokenizer, max_source_chinese_length=128):
        self.test=load_data(path, max_source_chinese_length)
        label_path=osp.join(path, 'data_label.txt')
        if os.path.exists(label_path):
            self.input = []
            with open(label_path, 'r',encoding="utf-8") as file:
                for line in file:
                    self.input.append(line)
        else:
            self.input=translate(self.test, path)
        self.tokenizer=tokenizer
        self.max_target_length = max_source_chinese_length
        self.max_source_length= max_source_chinese_length*6
        self.prefix = "Translate from Pinyin to Chinese: "
        
    def __len__(self,):
        return len(self.test)
    
    def __getitem__(self,item):
        input_sequence = self.input[item].strip()
        output_sequence = self.test[item].strip()
        
        encoding = self.tokenizer(
            input_sequence,
            padding='max_length',
            max_length=self.max_source_length, #max pinyin for a CN char is 6
            truncation=True,
            return_tensors="pt",
        )

        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
        
        target_encoding = self.tokenizer(
            output_sequence,
            padding='max_length',
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        )
        labels = target_encoding.input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids":input_ids.flatten(),
            "attention_mask":attention_mask.flatten(),
            "labels":labels.flatten(),
            "input_sents":input_sequence,
            "output_sents":output_sequence
        }

if __name__=='__main__':
    path='./data'
    data = load_data(path)
    output = translate(data, path)
    print("*******")
    print(len(data))
    print(len(output))
    input_path=osp.join(path, 'data_label.txt')
    line_count = 0

    with open(input_path, 'r',encoding="utf-8") as file:
        for line in file:
            line_count += 1

    print("Number of lines:", line_count)
    # tokenizer = T5Tokenizer.from_pretrained("t5-small")
    # model = T5ForConditionalGeneration.from_pretrained("t5-small")
    # train_dataset=InputDataset(path,tokenizer)
    # train_dataloader = DataLoader(train_dataset,batch_size=6)
    # batch = next(iter(train_dataloader))
    # print(batch)
    # loss = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels']).loss
    # print(loss.item())