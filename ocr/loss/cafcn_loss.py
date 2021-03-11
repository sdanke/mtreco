import torch
import torch.nn as nn


class CAFCNLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(CAFCNLoss, self).__init__()
        self.alpha = alpha
        self.pred_crit = torch.nn.functional.cross_entropy
        self.attn_loss = nn.BCELoss()

    def cal_weights(self, targets):
        batch, h, w = targets.shape
        n = batch * h * w
        t = targets.view(batch, -1).to(float)
        n_neg = (t == 0).to(float).sum(dim=1)
        ratio = n_neg / (n - n_neg)
        weights = torch.where(
            t == 0, torch.ones_like(t), torch.ones_like(t) * ratio.unsqueeze(1)
        )
        return weights.view(batch, h, w)

    def forward(self, preds, targets):
        hm, a2, a3, a4, a5 = preds
        tar_hm, tar_a2, tar_a3, tar_a4, tar_a5 = (
            targets["hm"],
            targets["a2"],
            targets["a3"],
            targets["a4"],
            targets["a5"],
        )

        # Prediction loss
        batch, h, w = tar_hm.shape
        pred_loss = self.pred_crit(hm, tar_hm, reduction="none")
        weights = self.cal_weights(tar_hm)
        pred_loss = (pred_loss * weights).sum() / (h * w)

        # Char attention loss
        a2_loss = self.attn_loss(a2, tar_a2)
        a3_loss = self.attn_loss(a3, tar_a3)
        a4_loss = self.attn_loss(a4, tar_a4)
        a5_loss = self.attn_loss(a5, tar_a5)

        loss = pred_loss + self.alpha * (a2_loss + a3_loss + a4_loss + a5_loss)
        return (
            loss,
            {
                "loss": loss.item(),
                "pred_loss": pred_loss.item(),
                "a2_loss": a2_loss.item(),
                "a3_loss": a3_loss.item(),
                "a4_loss": a4_loss.item(),
                "a5_loss": a5_loss.item(),
            },
        )
