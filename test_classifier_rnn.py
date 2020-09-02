import wandb

import torch

from classifier.rnn import config
from classifier.rnn.data import test_data
from train_classifier_rnn import rnn_model, _set_random_seeds
from utils import flat_accuracy, joint_metrics, confusion

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate(args, model, test_iterator, device):
    print("")
    print('Running Evaluation...')

    # Tracking variables
    total_test_accuracy = 0
    total_test_precision = 0
    total_test_recall = 0
    total_test_f1 = 0

    total_TN = 0
    total_FP = 0
    total_FN = 0
    total_TP = 0
    
    model.load_state_dict(torch.load(args.load_model_path))
    model = model.to(device)
    model.eval()
    
    for batch in test_iterator:
        text, text_lengths = batch.text
        with torch.no_grad():
            predictions = model(text, text_lengths).squeeze(1)

        # Move predictions and labels to CPU
        predictions = predictions.to(dtype=int).detach().cpu().numpy()
        labels = batch.label.to(dtype=int).detach().cpu().numpy()
    
        # Calculate the metrics for this batch of test sentences, and
        # accumulate it over all batches
        total_test_accuracy += flat_accuracy(predictions, labels)
    
        precision, recall, f1 = joint_metrics(predictions, labels)
    
        total_test_precision += precision
        total_test_recall += recall
        total_test_f1 += f1

        TN, FP, FN, TP = confusion(predictions, labels)
        total_TN += TN
        total_FP += FP
        total_FN += FN
        total_TP += TP
    #import IPython; IPython.embed(); exit(1)
    # Report the final metrics for this validation run.
    avg_test_accuracy = total_test_accuracy / len(test_iterator)
    avg_test_precision = total_test_precision / len(test_iterator)
    avg_test_recall = total_test_recall / len(test_iterator)
    avg_test_f1 = total_test_f1 / len(test_iterator)

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
    _set_random_seeds(args.random_seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))
    else:
        print("No GPU available, using CPU instead.")
        device = torch.device("cpu")

    test_iterator, vocab = test_data(args, device)
    model = rnn_model(args, vocab, 1)
    wandb.watch(model)
    evaluate(args, model, test_iterator, device)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting Early')