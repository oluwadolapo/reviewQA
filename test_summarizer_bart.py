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
    pred_answers = []
    for i in range(len(quest_rev)):
        answer = model.predict(quest_rev[i])[0]
        pred_answers.append(answer)

    rouge = Rouge()

    all_rouge_1p = []
    all_rouge_1r = []
    all_rouge_1f = []
    all_rouge_2p = []
    all_rouge_2r = []
    all_rouge_2f = []
    all_rouge_lp = []
    all_rouge_lr = []
    all_rouge_lf = []

    for count in range(len(pred_answers)):
        rouge_1p = []
        rouge_1r = []
        rouge_1f = []
        rouge_2p = []
        rouge_2r = []
        rouge_2f = []
        rouge_lp = []
        rouge_lr = []
        rouge_lf = []
        for ans in ref_answers[count]:
            hyp = []
            ref = []
            hyp.append(pred_answers[count])
            ref.append(ans)
            scores = rouge.get_scores(hyp, ref, avg=True)
            rouge_1p.append(scores['rouge-1']['p'])
            rouge_1r.append(scores['rouge-1']['r'])
            rouge_1f.append(scores['rouge-1']['f'])
            rouge_2p.append(scores['rouge-2']['p'])
            rouge_2r.append(scores['rouge-2']['r'])
            rouge_2f.append(scores['rouge-2']['f'])
            rouge_lp.append(scores['rouge-l']['p'])
            rouge_lr.append(scores['rouge-l']['r'])
            rouge_lf.append(scores['rouge-l']['f'])

        rouge_1p.sort()
        rouge_1r.sort()
        rouge_1f.sort()
        rouge_2p.sort()
        rouge_2r.sort()
        rouge_2f.sort()
        rouge_lp.sort()
        rouge_lr.sort()
        rouge_lf.sort()

        all_rouge_1p.append(rouge_1p[-1])
        all_rouge_1r.append(rouge_1r[-1])
        all_rouge_1f.append(rouge_1f[-1])
        all_rouge_2p.append(rouge_2p[-1])
        all_rouge_2r.append(rouge_2r[-1])
        all_rouge_2f.append(rouge_2f[-1])
        all_rouge_lp.append(rouge_lp[-1])
        all_rouge_lr.append(rouge_lr[-1])
        all_rouge_lf.append(rouge_lf[-1])
    
    print()
    print("rouge_1p:", sum(all_rouge_1p)/len(all_rouge_1p))
    print("rouge_1r:", sum(all_rouge_1r)/len(all_rouge_1r))
    print("rouge_1f:", sum(all_rouge_1f)/len(all_rouge_1f))
    print()
    print("rouge_2p:", sum(all_rouge_2p)/len(all_rouge_2p))
    print("rouge_2r:", sum(all_rouge_2r)/len(all_rouge_2r))
    print("rouge_2f:", sum(all_rouge_2f)/len(all_rouge_2f))
    print()
    print("rouge_lp:", sum(all_rouge_lp)/len(all_rouge_lp))
    print("rouge_lr:", sum(all_rouge_lr)/len(all_rouge_lr))
    print("rouge_lf:", sum(all_rouge_lf)/len(all_rouge_lf))

    #rouge = Rouge()
    #scores = rouge.get_scores(new_pred_answers, new_ref_answers, avg=True)
    #rouge = PyRouge(rouge_n=(1, 2, 4), rouge_l=True, rouge_w=True,
                #rouge_w_weight=1.2, rouge_s=True, rouge_su=True, skip_gap=4)
    #scores = rouge.evaluate(pred_answers, ref_answers)

    #print(scores)

if __name__ == "__main__":
    main()
