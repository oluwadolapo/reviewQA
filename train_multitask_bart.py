import wandb

import random
import time
import numpy as np
import torch
from transformers import AdamW, get_linear_schedule_with_warmup

from multi_task_bart import config
from utils import format_time
from multi_task_bart.model import model_choice
from multi_task_bart.train_eval import multi_task_train, multi_task_eval
from multi_task_bart.data import summarizer_data, classifier_data


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


def training(args, model, summarizer_train_dataloader, summarizer_val_dataloader,
                classifier_train_dataloader, classifier_val_dataloader, tokenizer, device):
    training_stats = []
    total_t0 = time.time()
    min_val_loss1 = None
    min_val_loss2 = None

    optimizer, scheduler = optim(args, model, summarizer_train_dataloader)

    for epoch in range(1, args.num_train_epochs+1):
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch, args.num_train_epochs))

        
        avg_train_loss1, avg_train_loss2, training_time = multi_task_train(args, model, summarizer_train_dataloader, 
                                          classifier_train_dataloader, device, optimizer, scheduler)
        avg_val_loss1, avg_val_loss2, validation_time = multi_task_eval(model, summarizer_val_dataloader,
                                                         classifier_val_dataloader, device)
        
        wandb.log({"avg_train_loss1": avg_train_loss1, "avg_train_loss2": avg_train_loss2, "avg_val_loss1": avg_val_loss1, "avg_val_loss2": avg_val_loss2}, step=epoch)
 
        print()
        print("   Average training loss for summarizer: {0:.2f}".format(avg_train_loss1))
        print("   Average validation loss for summarizer: {0:.2f}".format(avg_val_loss1))
        print()
        print("   Average training loss for classifier: {0:.2f}".format(avg_train_loss2))
        print("   Average validation loss for classifier: {0:.2f}".format(avg_val_loss2))
        print()
        print("   Training epoch took: {:}".format(training_time))
 
        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch,
                'Training Loss1': avg_train_loss1,
                'Valid. Loss1': avg_val_loss1,
                'Training Loss2': avg_train_loss2,
                'Valid. Loss2': avg_val_loss2,
                'Training Time': training_time,
                'Validation Time': validation_time
            })
    
        if min_val_loss1 == None or avg_val_loss1 < min_val_loss1:
            if min_val_loss2 == None or avg_val_loss2 < min_val_loss2:
                print("Saving model to %s" % args.output_dir)
                # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                # They can then be reloaded using `from_pretrained()`
                model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                model_to_save.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
 
                # Copy the model files to a directory in your Google Drive.
                #!cp -r ./model_save4/ "/content/drive/My Drive/Experiments/Bart_QA"
 
                min_val_loss1 = avg_val_loss1
                min_val_loss2 = avg_val_loss2
 
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

    summarizer_train_dataloader, summarizer_val_dataloader = summarizer_data(args, tokenizer, 'summarization')
    classifier_train_dataloader, classifier_val_dataloader = classifier_data(args, tokenizer, 'classification')

    training(args, model, summarizer_train_dataloader, summarizer_val_dataloader,
                classifier_train_dataloader, classifier_val_dataloader, tokenizer, device)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('-' * 10)
        print('Exiting Early')