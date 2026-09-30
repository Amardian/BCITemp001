"""Stage 3: Saliency-Preserving Knowledge Distillation (SPKD) - Sec 4.3.3.

Eq. (10): Hinton loss (CE + T^2 * KL of softened outputs)
Eq. (11): softplus sub-gradient surrogate for the Grad-CAM ReLU
Eq. (12): L_SAL = (1/N) sum_i MSE(S_t(x_i), S_s(x_i))
Eq. (13): L_SPKD = alpha*L_CE + (1-alpha-beta)*T^2*L_KL + beta*L_SAL
          alpha = 0.3, beta = 0.3, T = 4

The teacher's saliency S_t is pre-computed and detached (Alg. 3 line 4);
the student's saliency S_s is computed on the fly with the softplus
sub-gradient (line 5) so that dL_SAL/dtheta_s exists everywhere
(Proposition 4.1). Setting beta = 0 recovers Hinton KD (Table 16 row
"NAP + QAT + Hinton KD").
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .gradcam import channel_saliency
from .train import batches


def spkd_loss(logits_s, y, logits_t, S_t, S_s, alpha, beta, T):
    """Eq. (13) with Eq. (12) inside."""
    l_ce = F.cross_entropy(logits_s, y)
    p_t = F.log_softmax(logits_t / T, dim=1)
    p_s = F.log_softmax(logits_s / T, dim=1)
    # KL(t || s) with log-space targets (handles exact zeros safely)
    l_kl = F.kl_div(p_s, p_t, reduction="batchmean", log_target=True)
    l_sal = F.mse_loss(S_s, S_t)
    total = (alpha * l_ce
             + (1.0 - alpha - beta) * (T ** 2) * l_kl
             + beta * l_sal)
    return total, dict(l_ce=l_ce.item(), l_kl=l_kl.item(),
                       l_sal=l_sal.item())


def precompute_teacher_saliency(teacher, X, y, device, batch=32):
    """Teacher Grad-CAM channel saliency for every trial (detached)."""
    S_all = []
    teacher.eval()
    for s in range(0, len(y), batch):
        xb = X[s:s + batch].to(device)
        yb = y[s:s + batch].to(device)
        S = channel_saliency(teacher, xb, class_idx=yb,
                             normalize=True).detach()
        S_all.append(S.cpu())
    return torch.cat(S_all)


def run_spkd(teacher, student, X, y, device, cfg, beta=None,
             logger=print):
    """Train the student with L_SPKD (Alg. 3). Returns the distilled student.

    beta=cfg.spkd_beta by default; beta=0 gives the Hinton-KD ablation.
    """
    beta = cfg.spkd_beta if beta is None else beta
    student = copy.deepcopy(student).to(device)
    teacher = teacher.to(device).eval()

    # teacher saliency needs an autograd graph: (re-)enable grads, compute
    # the detached saliency maps, THEN freeze the parameters for distill
    for p in teacher.parameters():
        p.requires_grad_(True)
    S_teacher = precompute_teacher_saliency(teacher, X, y, device,
                                            batch=cfg.spkd_batch)
    for p in teacher.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(
        [p for p in student.parameters() if p.requires_grad], lr=cfg.spkd_lr)
    best_loss, best_state, bad = float("inf"), None, 0
    n = len(y)
    for ep in range(1, cfg.spkd_epochs + 1):
        student.train()
        # shuffle
        perm = torch.randperm(n)
        Xe, ye, Se = X[perm], y[perm], S_teacher[perm]
        ep_stats = []
        for s in range(0, n, cfg.spkd_batch):
            xb = Xe[s:s + cfg.spkd_batch].to(device)
            yb = ye[s:s + cfg.spkd_batch].to(device)
            Sb = Se[s:s + cfg.spkd_batch].to(device)
            opt.zero_grad()
            logits_s = student(xb)
            with torch.no_grad():
                logits_t = teacher(xb)
            # student saliency with softplus sub-gradient, differentiable
            S_s = channel_saliency(student, xb, class_idx=yb,
                                   subgrad="softplus",
                                   beta_soft=cfg.beta_soft,
                                   create_graph=True, detach=False)
            loss, stats = spkd_loss(logits_s, yb, logits_t, Sb, S_s,
                                    alpha=cfg.spkd_alpha, beta=beta,
                                    T=cfg.spkd_T)
            loss.backward()
            opt.step()
            ep_stats.append((loss.item(), stats))
        mean_loss = float(np.mean([a for a, _ in ep_stats]))
        mean_stats = {k: float(np.mean([st[k] for _, st in ep_stats]))
                      for k in ep_stats[0][1]}
        logger(f"    SPKD epoch {ep:3d}/{cfg.spkd_epochs}  "
               f"L {mean_loss:.4f}  (CE {mean_stats['l_ce']:.3f}  "
               f"KL {mean_stats['l_kl']:.3f}  SAL {mean_stats['l_sal']:.4f})")
        if mean_loss < best_loss - 1e-5:
            best_loss, bad = mean_loss, 0
            best_state = copy.deepcopy(student.state_dict())
        else:
            bad += 1
            if bad >= cfg.spkd_patience:
                logger("    SPKD early stop (patience)")
                break
    if best_state is not None:
        student.load_state_dict(best_state)
    return student
