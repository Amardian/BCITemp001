"""Multi-method explainability framework (Section 4.4).

(i)   Grad-CAM saliency analysis              -> gradcam.py
(ii)  Integrated Gradients (zero baseline)    -> integrated_gradients
(iii) EigenCAM cross-check (gradient-free PCA)-> eigencam_channels
(iv)  conditional permutation importance      -> conditional_perm_importance
(v)   causal channel ablation with dose-response -> dose_response
(vi)  time-frequency saliency vs band-shuffled null -> tf_null_test
(vii) Adebayo sanity checks                   -> adebayo_sanity
(viii)neurophysiological ground-truth spatial agreement -> jaccard_roi
"""

from __future__ import annotations

import copy
import warnings

import numpy as np
import torch
from scipy.stats import spearmanr

from .gradcam import channel_saliency


# (ii) Integrated Gradients -----------------------------------------------
@torch.enable_grad()
def integrated_gradients(model, X, y, device, steps=32, batch=32):
    """IG per channel: sum over (f, t) of |IG| (Eq. 16, zero baseline).

    Returns (attribution maps (N, C, F, T), per-channel scores (N, C)).
    """
    model.eval()
    maps, chan = [], []
    for s in range(0, len(y), batch):
        xb = X[s:s + batch].to(device)
        yb = y[s:s + batch].to(device)
        base = torch.zeros_like(xb)
        total = torch.zeros_like(xb)
        for a in range(1, steps + 1):
            xi = (base + (a / steps) * (xb - base)).detach().requires_grad_(True)
            logits = model(xi)
            score = logits.gather(1, yb.view(-1, 1)).sum()
            g = torch.autograd.grad(score, xi)[0]
            total += g.detach()
        ig = (xb - base) * total / steps
        maps.append(ig.cpu())
        chan.append(ig.abs().sum(dim=(2, 3)).cpu())
    return torch.cat(maps), torch.cat(chan)


# (iii) EigenCAM -----------------------------------------------------------
@torch.no_grad()
def eigencam_channels(model, X, device, batch=32):
    """Gradient-free PCA cross-check: first principal component of the
    last-conv activation matrix; channel score = energy of the component
    restricted to that channel's positions."""
    model.eval()
    out = []
    for s in range(0, len(X), batch):
        xb = X[s:s + batch].to(device)
        feats = model.forward_features(xb)              # (B, C, K, f, t)
        B, C, K, f, t = feats.shape
        z = feats.permute(0, 2, 1, 3, 4).flatten(2)      # (B, K, C*f*t)
        for i in range(B):
            m = z[i]
            m = m - m.mean(dim=1, keepdim=True)
            _, _, Vt = torch.linalg.svd(m, full_matrices=False)
            v = Vt[0]                                    # (P,) first PC
            v_ch = v.reshape(C, f * t)
            out.append(v_ch.pow(2).sum(dim=1).cpu())     # (C,)
    S = torch.stack(out)
    S = S / (S.sum(dim=1, keepdim=True) + 1e-8)
    return S


# (iv) Conditional permutation importance ----------------------------------
def conditional_perm_importance(model, X, y, device, channels,
                                k_neighbors=3, seed=0):
    """Permute each channel conditional on its k=3 spatially adjacent
    channels (ridge reconstruction + residual shuffle), measure accuracy
    drop. Controls EEG spatial correlation (Section 4.4)."""
    rng = np.random.default_rng(seed)
    from .train import accuracy
    base_acc = accuracy(model, X, y, device)
    C = X.shape[1]
    N, F, T = X.shape[0], X.shape[2], X.shape[3]
    drops = np.zeros(C)
    for c in range(C):
        neigh = [channels[(c - 1) % C], channels[(c + 1) % C],
                 channels[(c + C // 2) % C]][:k_neighbors]
        idx = np.array([i for i, n in enumerate(channels)
                        if n in neigh and i != c], dtype=int)
        if len(idx) < 2:
            idx = np.array([c], dtype=int)
        # design matrix: rows = (trial, freq, time) positions, cols = neighbors
        A = (X[:, idx].permute(1, 0, 2, 3).reshape(len(idx), -1)
             .numpy().astype(np.float64).T)
        b = X[:, c].reshape(-1).numpy().astype(np.float64)
        lam = 1e-2
        coef = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)
        pred = A @ coef
        resid = b - pred
        rng.shuffle(resid)
        Xc = X.clone()
        Xc[:, c] = torch.from_numpy(
            (pred + resid).reshape(N, F, T)).float()
        drops[c] = base_acc - accuracy(model, Xc, y, device)
    return drops, base_acc


# (v) Causal channel ablation with dose-response ---------------------------
def dose_response(model, X, y, device, S, kmax=10, seed=0):
    """Zero the top-k / bottom-k / random-k saliency channels per trial and
    measure the accuracy drop (k = 1..kmax) -> dose-response (Figure 6)."""
    from .train import accuracy
    rng = np.random.default_rng(seed)
    order_desc = torch.argsort(S, dim=1, descending=True)
    C = X.shape[1]
    base = accuracy(model, X, y, device)
    accs = {"top": [], "bottom": [], "random": []}
    for k in range(1, kmax + 1):
        for name in accs:
            Xk = X.clone()
            if name == "top":
                chs = order_desc[:, :k]
            elif name == "bottom":
                chs = order_desc[:, -k:]
            else:
                chs = torch.stack([
                    torch.from_numpy(rng.choice(C, size=k, replace=False))
                    for _ in range(len(X))])
            for i in range(len(X)):
                Xk[i, chs[i].tolist()] = 0.0
            accs[name].append(accuracy(model, Xk, y, device))
    drops = {name: [base - a for a in v] for name, v in accs.items()}
    return dict(accuracy=accs, drop=drops), base


# (vi) Time-frequency saliency vs band-shuffled null ------------------------
def tf_null_test(ig_maps, roi_idx, band_idx, n_shuffles=1000, seed=0):
    """Compare observed mu/beta-band saliency mass over ROI channels
    against a frequency-band-shuffled null (Section 4.4)."""
    rng = np.random.default_rng(seed)
    maps = np.abs(ig_maps.numpy())
    N, C, Fr, T = maps.shape
    roi = np.asarray(roi_idx)
    total = maps.sum() + 1e-12

    def mass(band_idx_):
        return maps[:, roi][:, :, band_idx_, :].sum() / total

    observed = mass(np.asarray(band_idx))
    null = np.array([mass(rng.permutation(Fr)[:len(band_idx)])
                     for _ in range(n_shuffles)])
    p = float((np.sum(null >= observed) + 1) / (n_shuffles + 1))
    z = (observed - null.mean()) / (null.std() + 1e-12)
    return dict(observed=float(observed), null_mean=float(null.mean()),
                null_std=float(null.std()), z=float(z), p=p,
                n_shuffles=n_shuffles)


# (vii) Adebayo sanity checks ----------------------------------------------
def adebayo_sanity(model, X, y, device, mode="full"):
    """Model-parameter randomization test: saliency should NOT stay similar
    after weight randomization (else it is insensitive to the model).

    mode='full'    : randomize every weight tensor
    mode='cascade' : randomize head + embedding + last conv block
    """
    S_orig = channel_saliency(model, X.to(device), class_idx=y.to(device))
    rand = copy.deepcopy(model)
    with torch.no_grad():
        if mode == "full":
            mods = [rand.head, rand.embedding, rand.tower[-1]]
            mods += list(rand.tower[:-1]) + [rand.gru]
        else:  # cascade
            mods = [rand.head, rand.embedding, rand.tower[-1]]
        for m in mods:
            for p in m.parameters():
                p.normal_(0, 0.1)
    S_rand = channel_saliency(rand, X.to(device), class_idx=y.to(device))
    rhos = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(y)):
            r = spearmanr(S_orig[i].numpy(), S_rand[i].numpy()).statistic
            rhos.append(0.0 if np.isnan(r) else float(r))
    return dict(mode=mode, spearman_mean=float(np.mean(rhos)),
                spearman_std=float(np.std(rhos)))


# (viii) Ground-truth spatial agreement -------------------------------------
def jaccard_roi(S, roi_idx, top_frac=0.33):
    """Jaccard between binarized top-33% saliency channels and the
    literature ROI mask (Eq. 17)."""
    S = S.numpy() if torch.is_tensor(S) else S
    roi = set(int(i) for i in roi_idx)
    C = S.shape[1]
    k = max(1, int(round(top_frac * C)))
    js = []
    for i in range(S.shape[0]):
        top = set(np.argsort(-S[i])[:k].tolist())
        union = top | roi
        js.append(len(top & roi) / len(union))
    return float(np.mean(js))
