import torch
from torch import nn
import torch.nn.functional as F

class knnLoss(nn.Module):
    def __init__(self):
        super(knnLoss, self).__init__()

    def loss(self, logits, knn_logits, targets, coeff):
        loss = F.cross_entropy(logits, targets, reduction="mean")

        p = knn_logits / torch.sum(knn_logits, -1, keepdims=True)
        knn_loss = F.nll_loss(torch.clamp(torch.log(p), min=-100),
            targets, reduction="mean")

        loss = loss + torch.mul(loss, knn_loss * coeff)
        return torch.sum(loss) / targets.shape[0]

    def forward(
        self, pred_logits, knn_logits,
        targets, coeff
    ):
        loss = self.loss(
            pred_logits, knn_logits, targets,
            coeff)
        return loss


class knnFocalLikeLoss(nn.Module):
    def __init__(self):
        super(knnFocalLikeLoss, self).__init__()

    def is_single(self):
        return False

    def loss(self, logits, knn_logits, targets, gamma):
        loss = F.cross_entropy(logits, targets, reduction="none")

        num_classes = logits.shape[1]
        targets = self.multi_hot(targets, num_classes)
        # modulator = (1 - p_t) ** gamma
        # below is a numerically stable version
        p = knn_logits / torch.sum(knn_logits, -1, keepdims=True)
        p_t = torch.sum(p * targets, -1)
        # a mask of p == 0
        modulator = torch.exp(gamma * torch.log1p(-1 * p_t))

        loss = loss * modulator
        return torch.sum(loss) / targets.shape[0]

    def forward(
        self, pred_logits, knn_logits,
        targets, coeff
    ):
        loss = self.loss(
            pred_logits, knn_logits, targets, coeff)
        return loss
    
    def multi_hot(self, labels: torch.Tensor, nb_classes: int) -> torch.Tensor:
        labels = labels.unsqueeze(1)  # (batch_size, 1)
        target = torch.zeros(
            labels.size(0), nb_classes, device=labels.device
        ).scatter_(1, labels, 1.)
        return target