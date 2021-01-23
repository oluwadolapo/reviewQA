import random
import numpy as np
import pandas as pd
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import BertTokenizer, BertForSequenceClassification
from nltk.tokenize import sent_tokenize
import flask
from flask import Flask, request, render_template
import json

app = Flask(__name__)

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


@app.route('/')
def index():
    return render_template('index.html')

def _set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

@app.route('/get_answer', methods=['POST'])
def get_answer():
    try:
        _set_random_seeds(42)
        q_id = request.json['input_idx']
        q_id = int(q_id)
        
        try:
            df = pd.read_json('/home/oluwadolapo/Datasets/amazonQA_test.json', orient='split')
            question_text = df["question"][q_id]
            reviews = ""
            for rev in df["passages"][q_id]:
                reviews += rev + " "

            question_text = question_text.lower()
            reviews = reviews.lower()

            if not question_text.endswith("?"):
                question_text += "?"
            
            bart_model_dir = "/home/oluwadolapo/Experiments/bartQA/model_save1"
            model = SummarizerModel(bart_model_dir)
            bart_question_context = "<s> "+question_text+" </s> "+reviews
            bart_response = model.predict(bart_question_context)[0]

            sent_list = sent_tokenize(reviews)

            first_response = sent_list[0]
            random_id = random.randint(0, len(sent_list)-1)
            random_response = sent_list[random_id]

            bert_model_dir = "/home/oluwadolapo/Experiments/bertQAnswerability/bert/model_save"
            model = AnswerabilityModel(bert_model_dir)
            pred_list = []
            for idx, sent in enumerate(sent_list):
                bert_question_context = "[CLS] "+question_text+" [SEP] "+sent
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
                bert_response = sent_list[best_positive_idx]
            else:
                bert_response = sent_list[best_negative_idx]


            res = {'question': question_text,
                    'reviews': reviews,
                    'first': first_response,
                    'random': random_response,
                    'bert': bert_response,
                    'bart': bart_response}

            return flask.jsonify(res)

        except KeyError:
            return app.response_class(response=json.dumps("Index out of range. Enter a lower question index"), status=500, mimetype='application/json')
    
    except Exception as error:
        res = str(error)
        return app.response_class(response=json.dumps(res), status=500, mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8002, use_reloader=True)