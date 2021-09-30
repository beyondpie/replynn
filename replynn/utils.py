import os
import shutil
import logging
from logging import Logger
from typing import Dict, List, OrderedDict
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize

def get_fnm_prefix(f: str):
    basename = os.path.basename(f)
    if "." in basename:
        prefix, _ = basename.split(".")
    else:
        prefix = basename
    return prefix

def empty_content_under_dir(folder) -> None:
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))
    return None


def setup_logger(name: str, log_file: str, level=logging.DEBUG) -> Logger:
    """Setup as many loggers as we want."""
    formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    logger: Logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


def get_chaintype(v: str, j: str, c: str) -> int:
    s = ""
    if v != "*":
        s = v
    elif c != "*":
        s = c
    elif j != "*":
        s = j
    else:
        return -1

    if s[0:3] == "IGH":
        return 0
    elif s[0:3] == "IGK":
        return 1
    elif s[0:3] == "IGL":
        return 2
    elif s[0:3] == "TRA":
        return 3
    elif s[0:3] == "TRB":
        return 4
    elif s[0:3] == "TRG":
        return 5
    elif s[0:3] == "TRD":
        return 6
    else:
        return -1

def eval_tumor_normal_classifier(out: torch.Tensor, y: torch.Tensor) -> Dict[str, int]:
    class_correct = 2 * [0]
    class_total = 2 * [0]
    y_long = y.to(torch.long)
    with torch.no_grad():
        _, predicted = torch.max(out, dim=1)
        c = (predicted == y_long).squeeze()
        for i in range(c.shape[0]):
            label = y_long[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
    r: Dict[str, float] = {
        "without_tumor_correct": class_correct[0],
        "without_tumor_total": class_total[0],
        "with_tumor_correct": class_correct[1],
        "with_tumor_total": class_total[1],
    }
    return r

def get_binaryclass_auc(
    y_true: torch.Tensor, y_score: torch.Tensor, col_of_t: int = 0
) -> float:
    """Get AUC of ROC curve."""
    y: np.ndarray = y_true.cpu().detach().numpy()
    score = torch.exp(y_score[:, col_of_t]) / torch.sum(torch.exp(y_score), dim=1)
    y_pred: np.ndarray = score.cpu().detach().numpy()
    r = round(roc_auc_score(y_true=y, y_score=y_pred), 3)
    return r



def get_multiclass_ovr_auc(
    y_true: torch.Tensor, y_score: torch.Tensor, y2n: Dict[int, str]
) -> Dict[str, float]:
    """Get one verse others/rest AUC for multiclass model.

    - Input:
      - classes, typically [0, ..., num_class(y_true) - 1] or y2n.keys()
    """
    result: Dict[str, float] = OrderedDict()
    y_true: np.ndarray = y_true.cpu().detach().numpy()
    uclass: List[int] = list(y2n.keys())
    y_true = label_binarize(y=y_true, classes=uclass)
    y_score = torch.exp(y_score) / torch.sum(torch.exp(y_score), dim=1, keepdim=True)
    y_score = y_score.cpu().detach().numpy()
    for c in uclass:
        auc: float = roc_auc_score(y_true=y_true[:, c], y_score=y_score[:, c])
        result[y2n[c]] = auc
    for k, v in result.items():
        print(f"AUC of {k} ovr: {v:.3f}")
    return result

def get_topk_acc(
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    topk: List[int] = [1, 3, 5],
    tag: str = "",
) -> Dict[str, float]:
    """Get the topk accuracy.
    - Input:
      - y_true: [batch_size]
      - y_score: [batch_size, num_class]
      - topk, List/Tuple, (1,3,5) is the default
    NOTE:
    - Ref: https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840/3
    """
    maxk = max(topk)
    batch_size = y_true.size(0)
    res: Dict[str, float] = {}
    if batch_size < 1:
        for k in topk:
            res[f"top{k}{tag}"] = 0.0
        return res
    ## pred: [batch_size, maxk]
    _, pred_indexes = torch.topk(
        input=y_score, k=maxk, dim=1, largest=True, sorted=True
    )
    pred_indexes = pred_indexes.t()
    ## y_true: [batch_size] -> [1, batch_size]  -> [maxk, batch_size] (repeat maxk times)
    correct = pred_indexes.eq(y_true.reshape(1, -1).expand_as(pred_indexes))
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res[f"top{k}{tag}"] = round(correct_k.item() * 100.0 / batch_size, 3)
    return res


def get_topk_acc_given_threshold_of_auc(
    y_true: torch.Tensor,
    y_score: torch.Tensor,
    auc: Dict[str, float],
    tnm2index: Dict[str, int],
    threshold_of_auc: float = 0.7,
    topk: List[int] = [1],
    ignore_nms: List[str] = [],
) -> Dict[str, float]:
    """Get topk acc for only tumor types with auc larger than a threshold.
    Compared with all the tumors.

    - Input:
      - y_true: [batch_size]
      - y_score: [batch_size, nclass]
      - ignore_index: ignore some classes when considering topk
    """
    cols: List[int] = [
        tnm2index[k.replace("test/", "")]
        for k, v in auc.items()
        if (v >= threshold_of_auc) and (k.replace("test/", "") not in ignore_nms)
    ]
    if len(cols) < 1:
        res: Dict[str, float] = {}
        for k in topk:
            res[f"top{k}auc{threshold_of_auc}"] = 0.0
        return res
    rows: List[int] = [i for i, v in enumerate(y_true) if v in cols]
    topk_acc: Dict[str, float] = get_topk_acc(
        y_true=y_true[rows],
        y_score=y_score[rows, :],
        topk=topk,
        tag=f"auc{threshold_of_auc}",
    )
    return topk_acc
