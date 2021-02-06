import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers import BartTokenizer, BartForSequenceClassification, BartConfig

class MyBart(BartForSequenceClassification):
    def classifier_head(self, input_size, device):
        self.fc_head = nn.Linear(input_size, 1)
        self.fc_head.to(device)

    def forward(self, input_ids, labels, attention_mask=None, encoder_outputs=None,
            decoder_input_ids=None, decoder_attention_mask=None, decoder_cached_states=None,
            use_cache=False, is_training=False, device = None):

        if is_training:
            decoder_start_token_id = self.config.decoder_start_token_id
            _decoder_input_ids = decoder_input_ids.new_zeros(decoder_input_ids.shape)
            _decoder_input_ids[..., 1:] = decoder_input_ids[..., :-1].clone()
            _decoder_input_ids[..., 0] = decoder_start_token_id
        else:
            _decoder_input_ids = decoder_input_ids.clone()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=_decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            decoder_cached_states=decoder_cached_states,
            use_cache=use_cache,
            return_dict=True
        )
        class_head_size = outputs['last_hidden_state'].size()[2]
        self.classifier_head(class_head_size, device)
        logits = outputs['last_hidden_state'][:,-1,:]
        classifier_output = self.fc_head(logits)
        classifier_output = classifier_output.squeeze()
        classifier_output = classifier_output.unsqueeze(dim=0)
        act_fcn = nn.Sigmoid()
        criterion = nn.BCELoss()
        #import IPython; IPython.embed(); exit(1)
        loss = criterion(act_fcn(classifier_output[0]), labels.float())
        return act_fcn(classifier_output[0]), loss


def model_choice(bart_type, from_scratch, model_path):
    if from_scratch:
        model = MyBart.from_pretrained(bart_type)
        tokenizer = BartTokenizer.from_pretrained(bart_type)
    else:
        model = MyBart.from_pretrained(model_path)
        tokenizer = BartTokenizer.from_pretrained(model_path)
    
    return model, tokenizer