"""Expert-system decision layer (Section 4.6).

Component 1: temperature-scaling calibration (Eq. 20)
    p_cal = softmax(z / T_cal), T_cal optimized by NLL on a held-out set.
Component 2: conformal prediction with abstention (Eq. 21)
    C(x) = { k : 1 - p_cal(k) <= tau_alpha }
    tau_alpha = finite-sample-corrected quantile of nonconformity
    (Definition 4.4: coverage >= 1 - alpha under exchangeability,
    alpha = 0.1 -> 90% nominal coverage).
Decision rule: |C(x)| == 1 and max p >= tau_commit (0.7) -> commit,
otherwise abstain (ambiguous / non-conforming / low-confidence).
Component 3: clinical decision rule mapping committed labels to
communication actions (KaraOne phonemes -> spelling interface;
BCI IV-2a classes -> cursor commands).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .metrics import expected_calibration_error


def fit_temperature(logits, y, max_iter=200):
    """Optimize the scalar temperature on the calibration NLL (Eq. 20)."""
    logits = torch.as_tensor(logits, dtype=torch.float)
    y = torch.as_tensor(y, dtype=torch.long)
    log_T = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.05, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_T.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.exp().item())


def apply_temperature(logits, T):
    return torch.as_tensor(logits, dtype=torch.float) / T


def conformal_threshold(scores, alpha=0.1):
    """Finite-sample-corrected empirical quantile (Definition 4.4).

    k = ceil((n+1)(1-alpha)); tau = k-th smallest calibration score
    (or +inf when k > n, which guarantees coverage conservatively).
    """
    scores = np.sort(np.asarray(scores))
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if k > n:
        return float("inf")
    return float(scores[k - 1])


def prediction_sets(probs, tau_alpha):
    """C(x) = {k : 1 - p(k) <= tau_alpha} (Eq. 21)."""
    probs = np.asarray(probs)
    return [np.where(1 - p <= tau_alpha + 1e-12)[0].tolist() for p in probs]


def expert_decisions(probs, tau_alpha, tau_commit, action_map=None):
    """Full decision layer: conformal set + commit threshold + clinical
    action mapping.

    Returns per-trial dict rows: set_size, committed, correct, action.
    """
    probs = np.asarray(probs)
    pred = probs.argmax(axis=1)
    rows = []
    sets = prediction_sets(probs, tau_alpha)
    for i, cs in enumerate(sets):
        committed = (len(cs) == 1) and (probs[i, pred[i]] >= tau_commit)
        action = (action_map or {}).get(int(pred[i]), None) if committed else "ABSTAIN"
        rows.append(dict(i=i, set_size=len(cs), committed=bool(committed),
                         pred=int(pred[i]), action=action))
    return rows


def evaluate_expert_layer(logits_calib, y_calib, logits_test, y_test,
                          alpha=0.1, tau_commit=0.7, action_map=None):
    """Tables 17 / 18 quantities.

    Temperature is fitted on the first half of the held-out calibration
    set (paper: 'optimized on a held-out calibration set'); conformal
    nonconformity scores use the second half, avoiding double use of the
    same trials (falls back to the full set when it is too small).
    """
    logits_calib = torch.as_tensor(logits_calib, dtype=torch.float)
    y_calib = torch.as_tensor(y_calib, dtype=torch.long)
    n = len(y_calib)
    n_t = n // 2 if n >= 20 else n
    T = fit_temperature(logits_calib[:n_t], y_calib[:n_t])

    p_raw = F.softmax(torch.as_tensor(logits_test, dtype=torch.float), dim=1).numpy()
    z_cal = apply_temperature(logits_calib[n_t:], T)
    z_test = apply_temperature(logits_test, T)
    p_cal = F.softmax(z_test, dim=1).numpy()
    p_calib_cal = F.softmax(z_cal, dim=1).numpy()

    scores = 1.0 - p_calib_cal[np.arange(len(y_calib[n_t:])),
                               np.asarray(y_calib[n_t:])]
    tau_alpha = conformal_threshold(scores, alpha=alpha)

    y_test = np.asarray(y_test)
    p_test_cal = p_cal
    scores_test = 1.0 - p_test_cal[np.arange(len(y_test)), y_test]
    coverage = float(np.mean(scores_test <= tau_alpha + 1e-12))

    rows = expert_decisions(p_test_cal, tau_alpha, tau_commit, action_map)
    committed = np.array([r["committed"] for r in rows])
    correct = (p_test_cal.argmax(axis=1) == y_test)
    committed_acc = float(correct[committed].mean()) if committed.any() else 0.0
    raw_acc = float(correct.mean())
    abstain = float(1.0 - committed.mean())

    # tau_commit sweep (Table 18)
    sweep = []
    for tc in [0.5, 0.6, 0.7, 0.8, 0.9]:
        rows_t = expert_decisions(p_test_cal, tau_alpha, tc, None)
        com = np.array([r["committed"] for r in rows_t])
        cov = float(np.mean([
            len(np.where(1 - p_test_cal[i] <= tau_alpha + 1e-12)[0]) > 0
            and com[i] for i in range(len(y_test))]))
        ca = float(correct[com].mean()) if com.any() else 0.0
        sweep.append(dict(tau_commit=tc, coverage=100 * cov,
                          committed_accuracy=100 * ca,
                          abstention=100 * (1 - com.mean())))

    return dict(
        temperature=T, tau_alpha=tau_alpha,
        ece_uncalibrated=expected_calibration_error(p_raw, y_test),
        ece_calibrated=expected_calibration_error(p_test_cal, y_test),
        conformal_coverage=100 * coverage,
        abstention_rate=100 * abstain,
        committed_accuracy=100 * committed_acc,
        raw_accuracy=100 * raw_acc,
        n_committed=int(committed.sum()),
        sweep=sweep,
    )


# Component 3: clinical decision rule ---------------------------------------
KARAONE_ACTIONS = {
    0: "spell:'A'", 1: "spell:'E'", 2: "spell:'I'", 3: "spell:'O'",
    4: "spell:'U'", 5: "spell:'S'", 6: "spell:'SH'",
    7: "spell:word'PAT'", 8: "spell:word'POT'",
    9: "spell:word'NEW'", 10: "spell:word'GNAW'",
}

BCI2A_ACTIONS = {
    0: "cursor:LEFT", 1: "cursor:RIGHT", 2: "cursor:DOWN", 3: "cursor:UP",
}


def clinical_rule_examples(rows, n=5, classes=None):
    """A few committed decisions with their safe communication actions."""
    out = []
    for r in rows:
        if r["committed"]:
            out.append(dict(trial=r["i"], pred=classes[r["pred"]] if classes
                            else r["pred"], action=r["action"]))
        if len(out) >= n:
            break
    return out
