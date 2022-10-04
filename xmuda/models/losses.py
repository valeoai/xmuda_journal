import numpy as np
import torch
import torch.nn.functional as F
import logging


def entropy_loss(v):
    """
    Entropy loss.
    Reference: https://github.com/valeoai/ADVENT

    :param v: Input tensor after softmax of size (num_points, num_classes).
    :return: Scalar entropy loss.
    """
    # (num points, num classes)
    if v.dim() == 2:
        v = v.transpose(0, 1)
        v = v.unsqueeze(0)
    # (1, num_classes, num_points)
    assert v.dim() == 3
    n, c, p = v.size()
    return -torch.sum(torch.mul(v, torch.log2(v + 1e-30))) / (n * p * np.log2(c))


def robust_entropy_loss(x, eta=2.0):
    """
    Robust entropy loss.
    Reference: https://github.com/YanchaoYang/FDA

    :param x: Logits before softmax, size: (batch_size, num_classes, number of points).
    :param eta: Hyperparameter of the robust entropy loss.
    :return: Scalar entropy loss.
    """
    if x.dim() != 3:
        raise ValueError(f'Expected 3-dimensional vector, but received {x.dim()}')
    P = F.softmax(x, dim=1)  # [B, C, N]
    logP = F.log_softmax(x, dim=1)  # [B, C, N]
    PlogP = P * logP  # [B, C, N]
    ent = -1.0 * PlogP.sum(dim=1)  # [B, C, N]
    num_classes = x.shape[1]
    ent = ent / torch.log(torch.tensor(num_classes, dtype=torch.float))
    # compute robust entropy
    ent = ent ** 2.0 + 1e-8
    ent = ent ** eta
    return ent.mean()


def logcoral_loss(x_src, x_trg):
    """
    Geodesic loss (log coral loss), reference:
    https://github.com/pmorerio/minimal-entropy-correlation-alignment/blob/master/svhn2mnist/model.py
    :param x_src: source features of size (N, ..., F), where N is the batch size and F is the feature size
    :param x_trg: target features of size (N, ..., F), where N is the batch size and F is the feature size
    :return: geodesic distance between the x_src and x_trg
    """
    # check if the feature size is the same, so that the covariance matrices will have the same dimensions
    assert x_src.shape[-1] == x_trg.shape[-1]
    assert x_src.dim() >= 2
    batch_size = x_src.shape[0]
    if x_src.dim() > 2:
        # reshape from (N1, N2, ..., NM, F) to (N1 * N2 * ... * NM, F)
        x_src = x_src.flatten(end_dim=-2)
        x_trg = x_trg.flatten(end_dim=-2)

    # subtract the mean over the batch
    x_src = x_src - torch.mean(x_src, 0)
    x_trg = x_trg - torch.mean(x_trg, 0)

    # compute covariance
    factor = 1. / (batch_size - 1)

    cov_src = factor * torch.mm(x_src.t(), x_src)
    cov_trg = factor * torch.mm(x_trg.t(), x_trg)

    # dirty workaround to prevent GPU memory error due to MAGMA (used in SVD)
    # this implementation achieves loss of zero without creating a fork in the computation graph
    # if there is a nan or big number in the cov matrix, use where (not if!) to set cov matrix to identity matrix
    condition = (cov_src > 1e30).any() or (cov_trg > 1e30).any() or torch.isnan(cov_src).any() or torch.isnan(cov_trg).any()
    cov_src = torch.where(torch.full_like(cov_src, condition, dtype=torch.uint8), torch.eye(cov_src.shape[0], device=cov_src.device), cov_src)
    cov_trg = torch.where(torch.full_like(cov_trg, condition, dtype=torch.uint8), torch.eye(cov_trg.shape[0], device=cov_trg.device), cov_trg)

    if condition:
        logger = logging.getLogger('xmuda.train')
        logger.info('Big number > 1e30 or nan in covariance matrix, return loss of 0 to prevent error in SVD decomposition.')

    _, e_src, v_src = cov_src.svd()
    _, e_trg, v_trg = cov_trg.svd()

    # nan can occur when taking log of a value near 0 (problem occurs if the cov matrix is of low rank)
    log_cov_src = torch.mm(v_src, torch.mm(torch.diag(torch.log(e_src)), v_src.t()))
    log_cov_trg = torch.mm(v_trg, torch.mm(torch.diag(torch.log(e_trg)), v_trg.t()))

    # Frobenius norm
    return torch.mean((log_cov_src - log_cov_trg) ** 2)
