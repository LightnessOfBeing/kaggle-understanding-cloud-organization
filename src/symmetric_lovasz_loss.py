import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.loss import _Loss

try:
    from itertools import ifilterfalse
except ImportError:  # py3k
    from itertools import filterfalse as ifilterfalse


def _lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:  # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
        logits: [B, H, W] Variable, logits at each pixel (between -infinity and +infinity)
        labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        per_image: compute the loss per image instead of per batch
        ignore: void class id
    """
    if per_image:
        loss = mean(
            _lovasz_hinge_flat(
                *_flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore)
            )
            for log, lab in zip(logits, labels)
        )
    else:
        loss = _lovasz_hinge_flat(*_flatten_binary_scores(logits, labels, ignore))
    return loss


def _lovasz_hinge_flat(logits, labels):
    """Binary Lovasz hinge loss
    Args:
        logits: [P] Variable, logits at each prediction (between -iinfinity and +iinfinity)
        labels: [P] Tensor, binary ground truth labels (0 or 1)
        ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * Variable(signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = _lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def _flatten_binary_scores(scores, labels, ignore=None):
    """Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    # print(f"scores = {scores.size()} labels = {labels.size()}")
    # scores = scores.view(-1)
    # labels = labels.view(-1)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    if ignore is None:
        return scores, labels
    valid = labels != ignore
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x


def mean(values, ignore_nan=False, empty=0):
    """Nanmean compatible with generators.
    """
    values = iter(values)
    if ignore_nan:
        values = ifilterfalse(isnan, values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(values, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


# ([2, 4, 320, 640]) B x C x H x W


def symmetric_lovasz(outputs, targets):
    return (_lovasz_hinge(outputs, targets) + _lovasz_hinge(-outputs, 1 - targets)) / 2


class SymmetricLovaszLoss(_Loss):
    def __init__(self, per_image=False, ignore=None):
        super().__init__()
        self.ignore = ignore
        self.per_image = per_image

    def forward(self, logits, target):
        # {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}
        # B x C x H x W

        logits_fish = logits[:, 0, :, :]
        logits_flower = logits[:, 1, :, :]
        logits_gravel = logits[:, 2, :, :]
        logits_sugar = logits[:, 3, :, :]

        target_fish = target[:, 0, :, :]
        target_flower = target[:, 1, :, :]
        target_gravel = target[:, 2, :, :]
        target_sugar = target[:, 3, :, :]

        lovasz_fish = symmetric_lovasz(logits_fish, target_fish)

        lovasz_flower = symmetric_lovasz(logits_flower, target_flower)

        lovasz_gravel = symmetric_lovasz(logits_gravel, target_gravel)

        lovasz_sugar = symmetric_lovasz(logits_sugar, target_sugar)

        return lovasz_fish + lovasz_flower + lovasz_gravel + lovasz_sugar
