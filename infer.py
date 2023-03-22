from transformers import BertTokenizer, T5Config, T5Tokenizer, T5ForConditionalGeneration, Text2TextGenerationPipeline
import sys

if __name__=='__main__':
    path = sys.argv[1]
    text = sys.argv[2]
    # tokenizer = BertTokenizer.from_pretrained("uer/t5-small-chinese-cluecorpussmall")
    # model = T5ForConditionalGeneration.from_pretrained(path)
    special_tokens = ["<extra_id_{}>".format(i) for i in range(100)]
    tokenizer = T5Tokenizer.from_pretrained("IDEA-CCNL/Randeng-T5-784M-MultiTask-Chinese", additional_special_tokens=special_tokens,)
    input = tokenizer.encode(text, return_tensors="pt")
    model = T5ForConditionalGeneration.from_pretrained(path)
    model.resize_token_embeddings(len(tokenizer))
    output=model.generate(input)
    print(tokenizer.decode(output[0]))
