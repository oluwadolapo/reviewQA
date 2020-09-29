import torch
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BartTokenizer, BartForConditionalGeneration

print()

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

question = "is bert simple ?"
context = """
We introduce a new language representation model called BERT, which stands for
Bidirectional Encoder Representations from Transformers. Unlike recent language
representation models (Peters et al., 2018a; Radford et al., 2018), BERT is
designed to pretrain deep bidirectional representations from unlabeled text by
jointly conditioning on both left and right context in all layers. As a result,
the pre-trained BERT model can be finetuned with just one additional output
layer to create state-of-the-art models for a wide range of tasks, such as
question answering and language inference, without substantial taskspecific
architecture modifications. BERT is conceptually simple and empirically
powerful. It obtains new state-of-the-art results on eleven natural language
processing tasks, including pushing the GLUE score to 80.5% (7.7% point absolute
improvement), MultiNLI accuracy to 86.7% (4.6% absolute improvement), SQuAD v1.1
question answering Test F1 to 93.2 (1.5 point absolute improvement) and SQuAD
v2.0 Test F1 to 83.1 (5.1 point absolute improvement).
"""
bert_question_context = "[CLS] "+question+" [SEP] "+context
bart_question_context = "<s> "+question+" </s> "+context

bert_model_dir = "/home/oluwadolapo/Experiments/bertQAnswerability/model_save"
model = AnswerabilityModel(bert_model_dir)
answerability = int(model.predict(bert_question_context))

if answerability == 1:
    bart_model_dir = "/home/oluwadolapo/model_save"
    model = SummarizerModel(bart_model_dir)
    answer = model.predict(bart_question_context)[0]
    print(answer)
else:
    print("I don't have an answer to your question")
