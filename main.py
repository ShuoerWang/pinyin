from torch.optim import AdamW
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
# from transformers import BertTokenizer, T5ForConditionalGeneration, Text2TextGenerationPipeline
import os
from utils import*
from dataset import*
import torch
from transformers import get_scheduler
from tqdm.auto import tqdm


def main():
    path='./data'
    # tokenizer = T5Tokenizer.from_pretrained("t5-base")
    # model = T5ForConditionalGeneration.from_pretrained("t5-base")

    # pretrained_model = "IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese"
    # special_tokens = ["<extra_id_{}>".format(i) for i in range(100)]
    # tokenizer = T5Tokenizer.from_pretrained(
    #     pretrained_model,
    #     do_lower_case=True,
    #     max_length=512,
    #     truncation=True,
    #     additional_special_tokens=special_tokens,
    # )
    # config = T5Config.from_pretrained(pretrained_model)
    # model = T5ForConditionalGeneration.from_pretrained(pretrained_model, config=config)
    # model.resize_token_embeddings(len(tokenizer))

    special_tokens = ["<extra_id_{}>".format(i) for i in range(100)]
    tokenizer = T5Tokenizer.from_pretrained("IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese", additional_special_tokens=special_tokens,)
    model = T5ForConditionalGeneration.from_pretrained("./saved_model_epoch2")
    model.resize_token_embeddings(len(tokenizer))

    train_dataset=InputDataset(path,tokenizer)
    train_dataloader = DataLoader(train_dataset,batch_size=4)

    optimizer = AdamW(model.parameters(), lr=3e-4)
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    model.train()
    mkdir("model")
    for epoch in range(num_epochs):
        model.save_pretrained(f"model/saved_model_epoch{-1}")

        for batch in train_dataloader:
            input_ids=batch['input_ids'].to(device)
            attention_mask=batch['attention_mask'].to(device)
            labels=batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        model.save_pretrained(f"model/saved_model_epoch{epoch}")

if __name__=='__main__':
    main()
    