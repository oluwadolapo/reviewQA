import json
import pandas as pd
import textwrap

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BartTokenizer, BartForConditionalGeneration


class AnswerabilityModel:
  def __init__(self, output_dir):
    self.tokenizer = BertTokenizer.from_pretrained(output_dir)
    self.model = BertForSequenceClassification.from_pretrained(output_dir)
    
  def encode(self,question_context):
    encoded = self.tokenizer.encode_plus(question_context, max_length = 512, truncation=True)
    return encoded["input_ids"], encoded["attention_mask"]

  def predict(self,question_context):
    input_ids, attention_mask = self.encode(question_context)
    output = self.model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))
    output = torch.softmax(output[0], dim=1)
    output = torch.argmax(output).item()
    return output


class SummarizerModel:
  def __init__(self, output_dir):
    self.model = BartForConditionalGeneration.from_pretrained(output_dir)
    self.tokenizer = BartTokenizer.from_pretrained(output_dir)
    
  def encode(self,question_context):
    encoded = self.tokenizer.encode_plus(question_context, max_length = 512, truncation=True, return_tensors='pt')
    return encoded["input_ids"], encoded["attention_mask"]

  def predict(self,question_context):
    input_ids, attention_mask = self.encode(question_context)
    summary_ids = self.model.generate(input_ids, attention_mask = attention_mask, num_beams=4, max_length=50)
    summary = [self.tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for id in summary_ids]
    return summary


def main():
    # Wrap text to 70 character
    wrapper = textwrap.TextWrapper(width=70)
    df = pd.read_json('/content/drive/MyDrive/Datasets/demo.json', orient='split')
    print()
    answerability = df['is_answerable']
    print("Answerable idx ===> ", [idx for idx, val in enumerate(answerability) if val == 1])
    print("Unanswerable idx ===> ", [idx for idx, val in enumerate(answerability) if val == 0])
    print()
    
    idx = int(input("Enter idx to be tested ===> "))

    question = df["questionText"][idx]
    print()
    print("Question")
    print(wrapper.fill(question))

    reviews = ""
    for rev in df["review_snippets"][idx]:
        reviews += rev + " "
    print()
    print("Reviews")
    print(wrapper.fill(reviews))
    print()
    print("Ref Answers")
    for index, ans in enumerate(df["answers"][idx]):
        print(index+1, ": ", wrapper.fill(ans['answerText']))
        print()
    
    bert_question_context = "[CLS] "+question+" [SEP] "+reviews
    bart_question_context = "<s> "+question+" </s> "+reviews

    bert_model_dir = "/content/drive/My Drive/Experiments/BertQAnswerability/model_save1"
    model = AnswerabilityModel(bert_model_dir)
    answerability = int(model.predict(bert_question_context))

    if answerability == 1:
        bart_model_dir = "/content/drive/My Drive/Experiments/Bart_QA/model_save1"
        model = SummarizerModel(bart_model_dir)
        answer = model.predict(bart_question_context)[0]
        print("Generated answer: ", answer)
    else:
        print("I don't have an answer to your question")

if __name__ == "__main__":
    main()
