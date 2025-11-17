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

    def forward(self, embeddings, labels):
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
        loss = F.cross_entropy(logits, labels)

        preds = torch.argmax(logits, dim = 1)
        return loss