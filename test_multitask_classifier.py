import wandb
import numpy as np
import torch
from transformers import BartTokenizer, BartForSequenceClassification, BartConfig

from multi_task_bart.model import model_choice, classifier_h
from utils import flat_accuracy, joint_metrics, confusion
from multi_task_bart import config
from multi_task_bart.data import classifier_testing_data

def eval(model, class_h, test_dataloader, device):
    print("")
    print("Running Evaluation...")
    
    # Tracking variables
    total_test_accuracy = 0
    total_test_precision = 0
    total_test_recall = 0
    total_test_f1 = 0

    total_TN = 0
    total_FP = 0
    total_FN = 0
    total_TP = 0
    # Evaluate data for one epoch
    for batch in test_dataloader:

        with torch.no_grad():
            logits = model(input_ids = batch[0].to(device),
                            attention_mask = batch[1].to(device),
                            decoder_input_ids = batch[2].to(device),
                            decoder_attention_mask = batch[3].to(device),
                            labels = batch[4].to(device),
                            device = device)

            pred, _ = class_h(logits, batch[4].to(device))

        b_labels = batch[4].to(device)
        # Move logits and labels to CPU
        pred = pred.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        #pred_flat = np.argmax(logits, axis=1).flatten()
        pred_flat = pred.flatten()
        pred_flat = np.round_(pred_flat)
        labels_flat = label_ids.flatten()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches
        total_test_accuracy += flat_accuracy(pred_flat, labels_flat)
    
        precision, recall, f1 = joint_metrics(pred_flat, labels_flat)
    
        total_test_precision += precision
        total_test_recall += recall
        total_test_f1 += f1

        TN, FP, FN, TP = confusion(pred_flat, labels_flat)
        total_TN += TN
        total_FP += FP
        total_FN += FN
        total_TP += TP

    # Report the final accuracy for this validation run.
    avg_test_accuracy = total_test_accuracy / len(test_dataloader)
    avg_test_precision = total_test_precision / len(test_dataloader)
    avg_test_recall = total_test_recall / len(test_dataloader)
    avg_test_f1 = total_test_f1 / len(test_dataloader)

    wandb.run.summary["test_accuracy"] = avg_test_accuracy
    wandb.run.summary["test_precision"] = avg_test_precision
    wandb.run.summary["test_recall"] = avg_test_recall
    wandb.run.summary["test_f1"] = avg_test_f1
    wandb.run.summary["TN"] = total_TN
    wandb.run.summary["FP"] = total_FP
    wandb.run.summary["FN"] = total_FN
    wandb.run.summary["TP"] = total_TP
    wandb.run.summary.update()

    print(" Accuracy:  {0:.2f}".format(avg_test_accuracy))
    print(" Precision: {0:.2f}".format(avg_test_precision))
    print(" Recall:    {0:.2f}".format(avg_test_recall))
    print(" F1:        {0:.2f}".format(avg_test_f1))

    print()

    print(" TN:  {0:.2f}".format(total_TN))
    print(" FP:  {0:.2f}".format(total_FP))
    print(" FN:  {0:.2f}".format(total_FN))
    print(" TP:  {0:.2f}".format(total_TP))


def main():
    args = config.get_params()
    wandb.init(config=args, project=args.project_name)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print("No GPU available, using CPU instead.")
        device = torch.device("cpu")

    #model = BartForSequenceClassification.from_pretrained(
    #            args.model_path)
    #tokenizer = BartTokenizer.from_pretrained(args.model_path)
    #model.to(device)
    model_path = args.load_model_path
    model, tokenizer = model_choice(args.bart_type, args.from_scratch, args.load_model_path)
    model.to(device)
    class_h = classifier_h()
    class_h.load_state_dict(torch.load(args.load_classifier_path, map_location=device))
    class_h.to(device)

    wandb.watch(model)
    wandb.watch(class_h)

    test_dataloader = classifier_testing_data(args, tokenizer)

    model.eval()
    eval(model, class_h, test_dataloader, device)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('-' * 10)
        print('Exiting Early')