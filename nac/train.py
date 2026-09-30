"""Training / evaluation loops shared by all pipeline stages."""

from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def batches(X, y, batch, shuffle=True, seed=0):
    n = len(y)
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed + int(time.time() * 1000 % 10000))
        rng.shuffle(idx)
    for s in range(0, n, batch):
        j = idx[s:s + batch]
        yield X[torch.as_tensor(j)], y[torch.as_tensor(j)]


def train_model(model, X_train, y_train, X_val, y_val, device,
                epochs=10, lr=1e-3, weight_decay=1e-4, batch=32,
                patience=None, logger=print, seed=0):
    """Adam training with optional early stopping (patience on val acc)."""
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay)
    best_acc, best_state, bad = -1.0, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        tot, correct, losses = 0, 0, []
        for xb, yb in batches(X_train, y_train, batch, seed=seed + ep):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            correct += (logits.argmax(1) == yb).sum().item()
            tot += len(yb)
        tr_acc = correct / max(tot, 1)
        if len(y_val):
            va = accuracy(model, X_val, y_val, device, batch=128)
        else:
            va = tr_acc
        msg = (f"    epoch {ep:3d}/{epochs}  loss {np.mean(losses):.4f}  "
               f"train-acc {100 * tr_acc:.2f}%  val-acc {100 * va:.2f}%")
        if patience is not None:
            if va > best_acc + 1e-4:
                best_acc, bad = va, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
            if bad >= patience:
                msg += "  (early stop)"
                logger(msg)
                break
        logger(msg)
    if patience is not None and best_state is not None:
        model.load_state_dict(best_state)
    return model


def finetune(model, loader, device, epochs=2, lr=1e-4, logger=print):
    """Fine-tune on a (X, y) tuple loader (Alg. 1 line 8 / Alg. 2 line 6)."""
    X, y = loader
    return train_model(model, X, y, X[:0], y[:0], device, epochs=epochs,
                       lr=lr, weight_decay=0.0, batch=32, logger=logger)


@torch.no_grad()
def accuracy(model, X, y, device, batch=128):
    model.eval().to(device)
    correct = 0
    for s in range(0, len(y), batch):
        xb = X[s:s + batch].to(device)
        logits = model(xb)
        correct += (logits.argmax(1).cpu() == y[s:s + batch]).sum().item()
    return correct / len(y)


@torch.no_grad()
def collect_logits(model, X, device, batch=128):
    """Full logits + per-subject accuracy helper."""
    model.eval().to(device)
    outs = []
    for s in range(0, len(X), batch):
        outs.append(model(X[s:s + batch].to(device)).cpu())
    return torch.cat(outs)


def per_subject_accuracy(logits, y, subj):
    """Subject-level unit of analysis (Reviewer R1.3)."""
    out = {}
    for s in np.unique(subj):
        m = subj == s
        pred = logits[m].argmax(1).numpy()
        out[int(s)] = float((pred == y[m].numpy()).mean())
    return out
