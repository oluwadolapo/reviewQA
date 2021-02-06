import wandb

import time
import random

import numpy as np
import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from classifier.bart.model import model_choice
from classifier.bart import config
from utils import format_time
from classifier.bart.train_eval import train, eval
from classifier.bart.data import training_data


def _set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def optim(args, model, train_dataloader):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params':[p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params':[p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    total_steps = len(train_dataloader) * args.num_train_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                num_warmup_steps=args.warmup_steps,
                                num_training_steps=total_steps)
    return optimizer, scheduler

def training(args, model, train_dataloader, val_dataloader, tokenizer, device):
    training_stats = []
    total_t0 = time.time()
    min_val_loss = None

    optimizer, scheduler = optim(args, model, train_dataloader)

    for epoch in range(1, args.num_train_epochs+1):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch, args.num_train_epochs))
 
        avg_train_loss, training_time = train(args, model, train_dataloader, 
                                                                device, optimizer, scheduler)
        avg_val_loss, validation_time = eval(model, val_dataloader, device)
        
        wandb.log({"avg_train_loss": avg_train_loss, "avg_val_loss": avg_val_loss}, step=epoch)
 
        print("")
        print("   Average training loss: {0:.2f}".format(avg_train_loss))
        print("   Average validation loss: {0:.2f}".format(avg_val_loss))
        print("   Training epoch took: {:}".format(training_time))
 
        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            })
    
        if min_val_loss == None or avg_val_loss < min_val_loss:
            print("Saving model to %s" % args.output_dir)
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
 
            # Copy the model files to a directory in your Google Drive.
            #!cp -r ./model_save4/ "/content/drive/My Drive/Experiments/Bart_QA"
 
            min_val_loss = avg_val_loss
 
    # Good practice: save your training arguments together with the trained model
    # torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
 
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

    model_path = args.model_path
    model, tokenizer = model_choice(args.bart_type, args.from_scratch, args.model_path)
    model.to(device)

    wandb.watch(model)

    train_dataloader, val_dataloader = training_data(args, tokenizer)
    training(args, model, train_dataloader, val_dataloader, tokenizer, device)
    print()
    print("End of Training")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('-' * 10)
        print('Exiting Early')