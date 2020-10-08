import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import pandas as pd
from rouge import Rouge
import argparse

from summarizer.bart.data import prepare_data

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', dest='model_path', type=str, default=None,
                        help='Directory for bart model')
    parser.add_argument('--data_path', dest='data_path', type=str, default=None,
                        help='Directory for testing data')
    args = parser.parse_args()
    return args

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
    args = get_params()
    model = SummarizerModel(args.model_path)
    df = pd.read_json(args.data_path, orient='split')
    df = df[49500:]
    quest_rev, ref_answers = prepare_data(df)
    pred_answers = []
    for i in range(len(quest_rev)):
        answer = model.predict(quest_rev[i])[0]
        pred_answers.append(answer)

    rouge = Rouge()
    scores = rouge.get_scores(pred_answers, ref_answers, avg=True)

    print(scores)

if __name__ == "__main__":
    main()