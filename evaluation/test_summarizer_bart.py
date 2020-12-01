import torch
from transformers import BartTokenizer, BartForConditionalGeneration
import pandas as pd
import argparse
import json

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', dest='model_path', type=str, default=None,
                        help='Directory for bart model')
    parser.add_argument('--data_path', dest='data_path', type=str, default=None,
                        help='Directory for testing data')
    parser.add_argument('--output_path', dest='output_path', type=str, default='output_bart.json',
                        help='Directory for saving output json file')
    args = parser.parse_args()
    return args


def prepare_data(df):
    questions = [q if q.endswith("?") else q+"?" for q in df.question]
    reviews = [r for r in df.passages]
    #answers = [a for a in df.answers]
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
  def __init__(self, output_dir, device):
    self.model = BartForConditionalGeneration.from_pretrained(output_dir)
    self.model.to(device)
    self.device = device
    self.tokenizer = BartTokenizer.from_pretrained(output_dir)

  def encode(self,question_context):
    encoded = self.tokenizer.encode_plus(question_context, max_length = 512, truncation=True, return_tensors='pt')
    return encoded["input_ids"], encoded["attention_mask"]

  def predict(self,question_context):
    input_ids, attention_mask = self.encode(question_context)
    input_ids = input_ids.to(self.device)
    attention_mask = attention_mask.to(self.device)
    summary_ids = self.model.generate(input_ids, attention_mask = attention_mask, num_beams=4, max_length=50)
    summary = [self.tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=True) for id in summary_ids]
    return summary

def main():
    args = get_params()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print("No GPU available, using CPU instead.")
        device = torch.device("cpu")

    model = SummarizerModel(args.model_path, device)
    df = pd.read_json(args.data_path, orient='split')
    #df = df[:1]
    quest_rev, ref_answers = prepare_data(df)
    print()
    print('Testing the model')
    pred_answers = []
    for i in range(len(quest_rev)):
        answer = model.predict(quest_rev[i])[0]
        pred_answers.append(answer)
    
    questions = [q.lower() if q.endswith("?") else q.lower()+"?" for q in df.question]
    multiple_answers = [[a.lower() for a in ans] for ans in df.multiple_answers]
    print()
    print('Saving the outputs')

    output_json = {'Question':questions,
                    'Pred_answer':pred_answers,
                    'Ref_answers':multiple_answers}

    with open(args.output_path, 'w') as json_file:
        json.dump(output_json, json_file)
    
    print()
    print('Done')

if __name__ == "__main__":
    main()
