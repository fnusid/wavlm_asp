import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LossWraper(nn.Module):
    """
    Drop-in replacement: training code can still call `loss = self.cosine_loss(emb, gt_embs)`.
    Loss components from the last call are stored in `self.last` for logging.

    lam = 0.0  -> plain PIT cosine distillation (identical to the old Hungarian version)
    lam > 0.0  -> PIT cosine + relational (inter-head) distillation
    """

    def __init__(self, lam: float = 0.1):
        super().__init__()
        # self.loss_fn = ArcFaceLoss(n_classes=num_class, emb_dim=emb_dim, s=s, m=m)
        self.loss_fn = PITCosineRelLoss(lam=lam)
        self.last = {}

    def forward(self, pred, gt):
        """
        pred: [B, 2, D]  student embeddings
        gt:   [B, 2, D]  teacher embeddings of the clean sources
        """
        loss, parts = self.loss_fn(pred, gt)
        self.last = parts
        return loss


class PITCosineRelLoss(nn.Module):
    """
    PIT cosine distillation + relational distillation.

    PIT term:  for each mixture, pick the better of the 2 head->speaker permutations
               and minimise 1 - mean cosine (same objective as Hungarian for 2x2).
    Rel term:  make cos(student head 1, student head 2) match
               cos(teacher spk 1, teacher spk 2) for the same mixture.
               Penalises head collapse without forcing similar speakers apart.
    """

    def __init__(self, lam: float = 0.1, dup_threshold: float = 0.7):
        super().__init__()
        self.lam = lam
        self.dup_threshold = dup_threshold

    def forward(self, pred, gt):
        assert pred.shape == gt.shape and pred.size(1) == 2, "expects [B, 2, D] for K=2"

        pred = F.normalize(pred, p=2, dim=-1)  # [B,2,D]
        gt = F.normalize(gt, p=2, dim=-1)      # [B,2,D]

        # cos[b, k, j] = cos(student head k, teacher speaker j)
        cos = torch.einsum("bkd,bjd->bkj", pred, gt)  # [B,2,2]

        # Two permutations for K=2
        perm_a = 0.5 * (cos[:, 0, 0] + cos[:, 1, 1])
        perm_b = 0.5 * (cos[:, 0, 1] + cos[:, 1, 0])
        pit = 1.0 - torch.maximum(perm_a, perm_b)     # [B]

        # Relational term: one cosine per mixture (NOT a [B, B] cross-batch matrix)
        cos_s = (pred[:, 0] * pred[:, 1]).sum(-1)     # [B] student inter-head cosine
        cos_t = (gt[:, 0] * gt[:, 1]).sum(-1)         # [B] teacher inter-speaker cosine
        # rel = (cos_s - cos_t).pow(2)                  # [B]
        rel = F.relu(cos_s - cos_t - 0.05).pow(2)
        # Both terms are batch means, so lam is NOT divided by batch size
        loss = pit.mean() + self.lam * rel.mean()

        parts = {
            "pit": pit.mean().detach(),
            "rel": rel.mean().detach(),
            "cos_student_heads": cos_s.mean().detach(),
            "cos_teacher_spk": cos_t.mean().detach(),
            "dup_proxy": (cos_s > self.dup_threshold).float().mean().detach(),
        }
        return loss, parts


class ArcFaceLoss(nn.Module):
    def __init__(self, n_classes, emb_dim=192, s=30.0, m=0.50):
        """
        n_classes: number of speakers (classes)
        emb_dim: dimension of embedding vector
        s: scale factor
        m: angular margin (in radians)
        """
        super().__init__()
        self.n_classes = n_classes
        self.emb_dim = emb_dim
        self.s = s
        self.m = m

        # Class weight matrix (each row is a class center)
        self.weight = nn.Parameter(torch.FloatTensor(n_classes, emb_dim))
        nn.init.xavier_normal_(self.weight)

        # Precompute constants
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings, labels):
        """
        embeddings: [B, D]  - model output embeddings
        labels:     [B]     - ground truth speaker IDs (ints)
        """
        x = F.normalize(embeddings, dim=1)
        W = F.normalize(self.weight, dim=1)

        cosine = F.linear(x, W)  # [B, n_classes]
        sine = torch.sqrt((1.0 - cosine ** 2).clamp(0, 1))

        # Add angular margin to target logits only
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        idx = torch.arange(embeddings.size(0), device=embeddings.device)
        logits = cosine.clone()
        logits[idx, labels] = phi[idx, labels]
        logits = logits * self.s

        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------
# Sanity check: python loss.py
# ---------------------------------------------------------------------
if __name__ == "__main__":
    from scipy.optimize import linear_sum_assignment

    torch.manual_seed(0)
    B, D = 64, 256
    pred = torch.randn(B, 2, D, requires_grad=True)
    gt = torch.randn(B, 2, D)

    # 1) PIT term matches the old Hungarian loop
    p, g = F.normalize(pred, dim=-1), F.normalize(gt, dim=-1)
    cos = torch.matmul(p, g.transpose(1, 2))
    old = 0.0
    for b in range(B):
        r, c = linear_sum_assignment(-cos[b].detach().numpy())
        old += 1.0 - cos[b, r, c].mean()
    old = old / B
    new = LossWraper(lam=0.0)(pred, gt)
    print(f"Hungarian PIT: {old.item():.6f} | vectorized PIT: {new.item():.6f}")
    assert torch.allclose(old, new, atol=1e-6)

    # 2) Relational term penalises collapse
    crit = LossWraper(lam=1.0)
    distinct = gt.clone()                                  # heads = teacher targets
    collapsed = gt[:, [0, 0], :].clone()                   # both heads = speaker 1
    crit(distinct, gt); print("distinct heads :", {k: round(v.item(), 4) for k, v in crit.last.items()})
    crit(collapsed, gt); print("collapsed heads:", {k: round(v.item(), 4) for k, v in crit.last.items()})

    # 3) Gradients flow
    LossWraper(lam=0.5)(pred, gt).backward()
    print("grad norm:", pred.grad.norm().item())
    print("OK")