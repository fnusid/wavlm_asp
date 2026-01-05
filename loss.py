import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import math




class LossWraper(nn.Module):
    def __init__(self):
        super().__init__()

        # self.loss_fn = ArcFaceLoss(n_classes = num_class, emb_dim = emb_dim, s = s, m=m)
        self.loss_fn = CosineSimilarityLoss()
        


    def forward(self, pred, gt):
        """
        pred: [B, 2, D]
        gt:   [B, 2, D]
        """
        loss = self.loss_fn(pred, gt)
        return loss


class CosineSimilarityLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        """
        pred: [B, 2, D]
        gt:   [B, 2, D]
        """
        # Normalize
        pred = F.normalize(pred, p=2, dim=-1)   # [B,2,D]
        gt   = F.normalize(gt,   p=2, dim=-1)   # [B,2,D]

        # Cosine similarity matrix: [B,2,2]
        cos = torch.matmul(pred, gt.transpose(1, 2))

        batch_size = cos.size(0)
        loss_total = 0.0

        for b in range(batch_size):

            # Hungarian on CPU (no gradients needed)
            cost = -cos[b].detach().cpu().numpy()      # [2,2]
            row_ind, col_ind = linear_sum_assignment(cost)

            # selected cosines with gradient
            sim = cos[b, row_ind, col_ind]             # 2 values

            # loss = 1 - mean(sim)
            loss_b = 1.0 - sim.mean()

            loss_total += loss_b

        return loss_total / batch_size




import torch
import torch.nn as nn
import torch.nn.functional as F
import math




class LossWraper(nn.Module):
    def __init__(self, num_class, emb_dim=768, s=30, m=0.2):
        super().__init__()

        self.loss_fn = ArcFaceLoss(n_classes = num_class, emb_dim = emb_dim, s = s, m=m)
        


    def forward(self, embeddings, labels):
        loss = self.loss_fn(embeddings, labels)
        return loss


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

    def forward(self, embeddings, labels, reduction="mean") :
        """
        embeddings: [B, D]  - model output embeddings
        labels:     [B]     - ground truth speaker IDs (ints)
        """
        # Normalize features and weights

        x = F.normalize(embeddings, dim=1)
        W = F.normalize(self.weight, dim=1)

        # Cosine similarity between embeddings and class weights
        cosine = F.linear(x, W)  # [B, n_classes]
        sine = torch.sqrt((1.0 - cosine ** 2).clamp(0, 1))

        # Add angular margin to target logits only
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Replace cosine value for the target class with phi

        # logits = (labels * phi) + ((1.0 - labels) * cosine)
        logits = cosine.clone()
        logits[torch.arange(embeddings.size(0)), labels] = phi[torch.arange(embeddings.size(0)), labels]
        logits *= self.s

        # Compute standard cross-entropy loss
        loss = F.cross_entropy(logits, labels, reduction='None')
        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        elif reduction == "none":
            return loss
        else:
            raise RuntimeError("Unknown reduction type: {}".format(reduction))

# implement PIT based ArcFace loss
class PITArcFaceLoss(nn.Module):
    def __init__(self, n_classes, emb_dim=192, s=30.0, m=0.50):
        super().__init__()
        self.arcface_loss = ArcFaceLoss(n_classes, emb_dim, s, m)
    
    def forward(self, embeddings, labels):
        """
        embeddings: [B, 2, D]
        labels:     [B, 2]  (Long)
        """
        e0, e1 = embeddings[:, 0, :], embeddings[:, 1, :]
        y0, y1 = labels[:, 0].long(), labels[:, 1].long()

        # per-sample losses
        L_id   = self.arcface(e0, y0, reduction="none") + self.arcface(e1, y1, reduction="none")  # [B]
        L_swap = self.arcface(e0, y1, reduction="none") + self.arcface(e1, y0, reduction="none")  # [B]

        L_min, pick = torch.min(torch.stack([L_id, L_swap], dim=0), dim=0)  # L_min: [B], pick: [B] (0=id,1=swap)
        loss = L_min.mean()
        return loss
