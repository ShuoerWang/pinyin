from torch.utils.data import Dataset,DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils import *
import re
from pypinyin import pinyin, lazy_pinyin, Style

def split_line(line, max_length=512):
    chunks = [line[i:i+max_length] for i in range(0, len(line), max_length)]
    return chunks

def translate(input):
    output = input
    for k in range(len(output)):
        i = 0;
        while i < len(output[k]):
            if re.search('[\u4e00-\u9fff]', output[k][i]):
                py = str(pinyin(output[k][i], style=0)[0][0])
                output[k] = output[k][:i] + py + output[k][i+1:]
                i += len(py)
            else:
                i += 1
    return output

def load_data(path, max_length=128):
    input_path=osp.join(path)
    file_list=os.listdir(input_path)
    input=[]
    for file in file_list:
        file_path=osp.join(input_path,file)
        with open(file_path,'r',encoding="utf-8") as f:
            lines=f.readlines()
            for line in lines:
                if (len(line) <= max_length):
                    input.append(line.strip())
                else:
                    input.extend(split_line(line.strip()))
        f.close()
    return input

class InputDataset(Dataset):
    def __init__(self, path, tokenizer, max_source_chinese_length=128):
        self.test=load_data(path, max_source_chinese_length)
        self.input=translate(self.test)
        self.tokenizer=tokenizer
        self.max_target_length = max_source_chinese_length
        self.max_source_length= max_source_chinese_length*6
        
    def __len__(self,):
        return len(self.test)
    
    def __getitem__(self,item):
        input_sequence = self.input[item]
        output_sequence = self.test[item]
        
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
    path='../data'
    # data = load_data(path)
    # print(data)
    # output = translate(data)
    # print(output)
    # print("11111")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    train_dataset=InputDataset(path,tokenizer)
    train_dataloader = DataLoader(train_dataset,batch_size=6)
    batch = next(iter(train_dataloader))
    print(batch)
    loss = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels']).loss
    print(loss.item())