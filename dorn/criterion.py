import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class berHuLoss(nn.Module):
    def __init__(self):
        super(berHuLoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        huber_c = torch.max(pred - target)
        huber_c = 0.2 * huber_c

        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        diff = diff.abs()

        huber_mask = (diff > huber_c).detach()

        diff2 = diff[huber_mask]
        diff2 = diff2 ** 2

        self.loss = torch.cat((diff, diff2)).mean()

        return self.loss


class ScaleInvariantError(nn.Module):
    """
    Scale invariant error defined in Eigen's paper!
    """

    def __init__(self, lamada=0.5):
        super(ScaleInvariantError, self).__init__()
        self.lamada = lamada
        return

    def forward(self, y_true, y_pred):
        first_log = torch.log(torch.clamp(y_pred, min, max))
        second_log = torch.log(torch.clamp(y_true, min, max))
        d = first_log - second_log
        loss = torch.mean(d * d) - self.lamada * torch.mean(d) * torch.mean(d)
        return loss


class ordLoss(nn.Module):
    """
    Ordinal loss is defined as the average of pixelwise ordinal loss F(h, w, X, O)
    over the entire image domain:
    """

    def __init__(self, device):
        super(ordLoss, self).__init__()
        self.loss = 0.0
        self.device = device

    def forward(self, ord_labels, target):
        """
        :param ord_labels: ordinal labels for each position of Image I.
        :param target:     the ground_truth discreted using SID strategy.
        :return: ordinal loss
        """

        N, C, H, W = ord_labels.size()
        ord_num = C

        self.loss = 0.0

        K = torch.zeros((N, C, H, W), dtype=torch.int).to(self.device)
        for i in range(ord_num):
            K[:, i, :, :] = K[:, i, :, :] + i * torch.ones((N, H, W), dtype=torch.int).to(self.device)

        mask_0 = (K <= target).detach()
        mask_1 = (K > target).detach()

        one = torch.ones(ord_labels[mask_1].size()).to(self.device)

        self.loss += (torch.sum(torch.log(torch.clamp(ord_labels[mask_0], min=1e-8, max=1e8)))
                     + torch.sum(torch.log(torch.clamp(one - ord_labels[mask_1], min=1e-8, max=1e8))))

        N = N * H * W
        self.loss /= (-N)  # negative
        return self.loss
