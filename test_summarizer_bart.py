import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import pandas as pd
from rouge import Rouge
from rouge_metric import PyRouge
import argparse

#from summarizer.bart.data import prepare_data

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', dest='model_path', type=str, default=None,
                        help='Directory for bart model')
    parser.add_argument('--data_path', dest='data_path', type=str, default=None,
                        help='Directory for testing data')
    args = parser.parse_args()
    return args


def prepare_data(df):
    questions = [q if q.endswith("?") else q+"?" for q in df.question]
    reviews = [r for r in df.passages]
    multiple_answers = [a for a in df.multiple_answers]

    questions = [q.lower() + " </s>" for q in questions]
    reviews = [[p.lower() for p in r] for r in reviews]
    #answers = ["<s> " + a.lower() + " </s>" for a in answers]
    multiple_answers = [["<s> " + a.lower() + " </s>" for a in ans] for ans in multiple_answers]

    quest_rev = []
    for i, q in enumerate(questions):
        for r in reviews[i]:
            q = q + " " + r
        quest_rev.append(q)

    return quest_rev, multiple_answers

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
    #df = df[:4]
    quest_rev, ref_answers = prepare_data(df)
    pred_answers = []
    for i in range(len(quest_rev)):
        answer = model.predict(quest_rev[i])[0]
        pred_answers.append(answer)

    #rouge = Rouge()
    #scores = rouge.get_scores(pred_answers, ref_answers, avg=True)
    rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
                rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)
    scores = rouge.evaluate(pred_answers, ref_answers)

    print(scores)

if __name__ == "__main__":
    main()