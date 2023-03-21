from transformers import BertTokenizer, T5ForConditionalGeneration, Text2TextGenerationPipeline
import sys

if __name__=='__main__':
    text = sys.argv[2]
    path = sys.argv[1]
    tokenizer = BertTokenizer.from_pretrained("uer/t5-small-chinese-cluecorpussmall")
    model = T5ForConditionalGeneration.from_pretrained(path)
    input = tokenizer.encode(text, return_tensors="pt")
    output=model.generate(input)
    print(tokenizer.decode(output[0]))
