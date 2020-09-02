
import time
import random
import wandb

import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from classifier.bert import config
from classifier.bert.data import data
from classifier.bert.train_eval import train, eval
from utils import format_time

def _set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def optim(args, model, train_dataloader):
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer = AdamW(model.parameters(),
                 lr=args.learning_rate,
                 eps=args.adam_epsilon)

    total_steps = len(train_dataloader) * args.n_epochs
    # Create the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                num_warmup_steps=args.warmup_steps,
                                num_training_steps=total_steps)

    return optimizer, scheduler


def training(args, model, train_dataloader, val_dataloader, tokenizer, device):
    training_stats = []
    total_t0 = time.time()
    min_val_loss = None

    optimizer, scheduler = optim(args, model, train_dataloader)

    
    for epoch in range(1, args.n_epochs+1):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch, args.n_epochs))

        avg_train_loss, training_time = train(args, model, train_dataloader, 
                                                                device, optimizer, scheduler)
        avg_val_loss, avg_val_accuracy, validation_time = eval(model, val_dataloader, device)
        wandb.log({"avg_train_loss": avg_train_loss, "avg_val_loss": avg_val_loss, "avg_val_accuracy": avg_val_accuracy}, step=epoch)


        print("")
        print("   Average training loss: {0:.2f}".format(avg_train_loss))
        print("   Training epoch took: {:}".format(training_time))

        print("   Validation Loss: {0:.2f}".format(avg_val_loss))
        print("   Validation took: {:}".format(validation_time))

        if min_val_loss == None or avg_val_loss < min_val_loss:
            print("Saving model to %s" % args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            
            min_val_loss = avg_val_loss
            wandb.run.summary["best-loss"] = min_val_loss
            wandb.run.summary["best-loss-epoch"] = epoch
            wandb.run.summary.update()

        print("")
        print("Training complete!")
        print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))

def main():
    args = config.get_params()
    wandb.init(config=args, project=args.project_name)

    _set_random_seeds(args.random_seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print("No GPU available, using CPU instead.")
        device = torch.device("cpu")

    if args.from_scratch:
        model = BertForSequenceClassification.from_pretrained(
                args.bert_type,
                num_labels = 2,
                output_attentions = False,
                output_hidden_states = False)
        tokenizer = BertTokenizer.from_pretrained(args.bert_type, do_lower_case=True)
    else:
        model = BertForSequenceClassification.from_pretrained(
                args.model_path,
                num_labels = 2,
                output_attentions = False,
                output_hidden_states = False)
        tokenizer = BertTokenizer.from_pretrained(args.model_path)
    
    model.to(device)
    wandb.watch(model)

    train_dataloader, val_dataloader = data(args, tokenizer)

    training(args, model, train_dataloader, val_dataloader, tokenizer, device)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('-' * 10)
        print('Exiting Early')