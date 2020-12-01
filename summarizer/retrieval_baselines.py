import argparse
import random
from nltk.tokenize import sent_tokenize
import json

import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', dest='model_path', type=str, default=None,
                        help='Directory for bert model')
    parser.add_argument('--data_path', dest='data_path', type=str, default=None,
                        help='Directory for testing data')
    parser.add_argument('--output_path', dest='output_path', type=str, default=None,
                        help='Directory for saving output json file')
    parser.add_argument('--heuristic_type', dest='heuristic_type', type=str, default=None,
                        help='Heuristic baseline to use')
    parser.add_argument('--random_seed', dest='random_seed', type=int, default=42,
                        help='Random Seed')
    parser.add_argument('--test_batch_size', dest='test_batch_size', type=int, default=1,
                        help='Testing batch size')
    args = parser.parse_args()
    return args

def _set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class AnswerabilityModel:
  def __init__(self, output_dir):
    self.tokenizer = BertTokenizer.from_pretrained(output_dir)
    self.model = BertForSequenceClassification.from_pretrained(output_dir)
    
  def encode(self,question_context):
    encoded = self.tokenizer.encode_plus(question_context, max_length = 100, truncation=True)
    return encoded["input_ids"], encoded["attention_mask"]

  def predict(self,question_context):
    input_ids, attention_mask = self.encode(question_context)
    output = self.model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))
    output = torch.softmax(output[0], dim=1)
    pred = torch.argmax(output).item()
    prob = output[0][pred].item()
    return pred, prob


def main():
    args = get_params()
    _set_random_seeds(args.random_seed)
    df = pd.read_json(args.data_path, orient='split')
    #df = df.head(100)

    questions = [q if q.endswith("?") else q+"?" for q in df.question]
    reviews = [r for r in df.passages]
    multiple_answers = [a for a in df.multiple_answers]

    questions = [q.lower() for q in questions]
    reviews = [[p.lower() for p in r] for r in reviews]
    multiple_answers = [[a.lower() for a in ans] for ans in multiple_answers]

    pred_answers = []
    if args.heuristic_type == 'first':
        for i in range(len(questions)):
            review_concat = ''
            for j in reviews[i]:
                review_concat += j + '' 
            sent_list = sent_tokenize(review_concat)
            pred_answers.append(sent_list[0])

    elif args.heuristic_type == 'random':
        for i in range(len(questions)):
            review_concat = ''
            for j in reviews[i]:
                review_concat += j + '' 
            sent_list = sent_tokenize(review_concat)
            ans_id = random.randint(0, len(sent_list)-1)
            pred_answers.append(sent_list[ans_id])

    elif args.heuristic_type == 'bert':
        model = AnswerabilityModel(args.model_path)
        for i in range(len(questions)):
            review_concat = ''
            for j in reviews[i]:
                review_concat += j + '' 
            sent_list = sent_tokenize(review_concat)

            pred_list = []
            for idx, sent in enumerate(sent_list):
                bert_question_context = "[CLS] "+questions[i]+" [SEP] "+sent
                pred, prob = model.predict(bert_question_context)
                pred_dict = {'id':idx, 'pred':pred, 'prob':prob}
                pred_list.append(pred_dict)
            
            best_positive_prob = float('-inf')
            best_positive_idx = None
            best_negative_prob = float('inf')
            best_negative_idx = None
            for pred in pred_list:
                if pred['pred'] == 1:
                    if pred['prob'] > best_positive_prob:
                        best_positive_prob = pred['prob']
                        best_positive_idx = pred['id']
                elif pred['pred'] == 0:
                    if pred['prob'] < best_negative_prob:
                        best_negative_prob = pred['prob']
                        best_negative_idx = pred['id']
            
            if best_positive_idx:
                pred_answers.append(sent_list[best_positive_idx])
            else:
                pred_answers.append(sent_list[best_negative_idx])

    output_json = {'Question':questions,
                    'Pred_answer':pred_answers,
                    'Ref_answers':multiple_answers}

    #df = pd.DataFrame(output_json, columns = ['Question', 'Pred_answer', 'Ref_answers'])
    #df.to_json(args.output_path, orient='split', index = False)

    with open(args.output_path, 'w') as json_file:
        json.dump(output_json, json_file)


if __name__ == "__main__":
    main()
