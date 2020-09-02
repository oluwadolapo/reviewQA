import datetime
import numpy as np

from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

import torch

def format_time(elapsed):
  '''
  Takes a time in seconds and returns a string hh:mm:ss
  '''
  # Round to the nearest second.
  elapsed_rounded = int(round((elapsed)))

  # Format as hh:mm:ss
  return str(datetime.timedelta(seconds=elapsed_rounded))

# Helper function to calculate the accuracy of our predictions vs labels
def flat_accuracy(pred_flat, labels_flat):
  return np.sum(pred_flat == labels_flat) / len(labels_flat)


def joint_metrics(pred_flat, labels_flat):
  precision, recall, f1, _ = precision_recall_fscore_support(labels_flat, pred_flat, average='macro')
  return precision, recall, f1


def confusion(pred_flat, labels_flat):
  tn, fp, fn, tp = confusion_matrix(labels_flat, pred_flat, labels=[0,1]).ravel()
  return tn, fp, fn, tp