import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


class DualQueryLoss(nn.Module):
    def __init__(self, w_distill=0.2, w_orth=0.01):
        super().__init__()
        self.w_distill = w_distill
        self.w_orth = w_orth

    def forward(self, pred_student, pred_teacher, teacher_embs, Q_student, Q_teacher):
        """
        pred_student : [B, 2, D]
        pred_teacher : [B, 2, D]
        teacher_embs : [B, 2, D]     (teacher target embeddings)
        Q_student    : [B, H, 2, Hd]
        Q_teacher    : [B, H, 2, Hd]
        """

        B = pred_student.size(0)

        # Normalize embeddings
        pred_student = F.normalize(pred_student, p=2, dim=-1)
        pred_teacher = F.normalize(pred_teacher, p=2, dim=-1)
        teacher_embs = F.normalize(teacher_embs, p=2, dim=-1)

        # ---------------------------------------------------------
        # 1. Hungarian alignment per sample (teacher vs GT)
        # ---------------------------------------------------------
        align_cols = []   # list of length B, each is array of length 2

        sup_loss = 0.0
        for b in range(B):
            sim = pred_teacher[b] @ teacher_embs[b].T       # [2,2]
            cost = -sim.detach().cpu().numpy()
            _, col = linear_sum_assignment(cost)
            align_cols.append(col)
            sup_loss += (1.0 - sim[0, col[0]]*0.5 - sim[1, col[1]]*0.5)
        sup_loss /= B

        # ---------------------------------------------------------
        # 2. Distillation (student → teacher), using SAME perm
        # ---------------------------------------------------------
        distill_loss = 0.0
        for b in range(B):
            sim = pred_student[b] @ pred_teacher[b].T       # [2,2]
            col = align_cols[b]
            distill_loss += (1.0 - (sim[0, col[0]] + sim[1, col[1]]) * 0.5)
        distill_loss /= B

        # ---------------------------------------------------------
        # 3. Orthogonality of student queries
        # ---------------------------------------------------------
        # Q_student: [B,H,2,Hd]
        Qs = F.normalize(Q_student, dim=-1)
        sim_q = torch.einsum("bhqd,bhpd->bhqp", Qs, Qs)  # [B,H,2,2]

        I = torch.eye(2, device=sim_q.device).view(1,1,2,2)
        ortho_loss = ((sim_q - I)**2).mean()

        # ---------------------------------------------------------
        # 4. Query distillation (correct per-sample alignment!)
        # ---------------------------------------------------------
        Qt = Q_teacher.detach()   # [B,H,2,Hd]
        Qs = Q_student            # [B,H,2,Hd]

        Qt_aligned = torch.zeros_like(Qt)

        for b in range(B):
            col = align_cols[b]
            Qt_aligned[b,:,0,:] = Qt[b,:,col[0],:]
            Qt_aligned[b,:,1,:] = Qt[b,:,col[1],:]

        query_distill = ((Qs - Qt_aligned)**2).mean()

        # ---------------------------------------------------------
        total = (
            sup_loss +
            self.w_distill * distill_loss +
            self.w_orth * ortho_loss +
            query_distill
        )

        return {
            "total_loss": total,
            "sup_loss": sup_loss,
            "distill_loss": distill_loss,
            "ortho_loss": ortho_loss,
            "query_distill": query_distill,
        }
