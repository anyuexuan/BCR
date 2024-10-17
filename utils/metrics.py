import numpy as np
import torch
from sklearn.metrics import average_precision_score, hamming_loss, zero_one_loss, label_ranking_loss, f1_score, roc_auc_score, coverage_error


def convert_torch_sparse_to_array(matrix):
    if matrix is not None:
        if 'sparse' in str(matrix):
            matrix = matrix.toarray()
        if type(matrix) == torch.Tensor:
            matrix = matrix.detach().cpu().numpy()
    return matrix

def evaluation(y_true, y_prob, y_pred=None):
    y_true = convert_torch_sparse_to_array(y_true)
    y_prob = convert_torch_sparse_to_array(y_prob)
    y_pred = convert_torch_sparse_to_array(y_pred)
    if y_pred is None:
        if np.min(y_prob) < 0:
            y_pred = np.zeros_like(y_prob)
            y_pred[y_prob > 0] = 1
        else:
            y_pred = np.zeros_like(y_prob)
            y_pred[y_prob > 0.5] = 1
    class_freq = np.sum(y_true, axis=0)
    mAP = average_precision_score(y_true[:, class_freq != 0], y_prob[:, class_freq != 0], average='macro')
    return {'mAP': mAP}