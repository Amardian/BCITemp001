"""End-to-end NAC pipeline orchestrator (Stages 1-6 of the paper).

For each dataset (KaraOne-like SMI target, BCI IV-2a-like transfer
source) it executes the paper's system:
  1. data (simulator by default, real loaders available)
  2. full FP32 CNN-GRU teacher (Section 4.2)
  3. NAC pipeline: NAP -> INT8 QAT -> SPKD (Section 4.3) + baselines
  4. CSI / CSI-S / SPR at every stage (Section 4.5, Table 16)
  5. multi-method XAI battery (Section 4.4)
  6. expert-system decision layer (Section 4.6, Tables 17-18)
  7. benchmarking + pre-registered statistics (Sections 4.7, 5.2)

Execution is STAGE-CHECKPOINTED: `run_dataset(ds, ..., stages=[...])`
runs the requested stages and saves every model + metric into
`outdir/ckpt_{ds}.pth`, so a long run can be split across several
processes (e.g. `python run_all.py --stages teacher,nap` then
`--stages distill,baselines`, ...). `--stages all` runs everything in
one process.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from . import figures as figs
from . import xai
from .bench import (export_onnx, latency_benchmark, sliding_window_online,
                    sustained_drift)
from .config import Config
from .data import generate_dataset, split_by_subject
from .expert import (BCI2A_ACTIONS, KARAONE_ACTIONS, clinical_rule_examples,
                     evaluate_expert_layer, expert_decisions)
from .gradcam import channel_saliency
from .metrics import (bca_bootstrap, cohens_dz, compression_stability,
                      noninferiority, percentile_ci, permutation_test,
                      wilcoxon_signed)
from .models import (CNNGRU, EEGNet, ShallowConvNet, count_parameters,
                     model_size_mb)
from .nap import l1_prune, lottery_ticket, nap
from .qat import int8_size_mb, quantize_model
from .spkd import run_spkd
from .train import (accuracy, collect_logits, per_subject_accuracy,
                    train_model)

STAGE_ORDER = ["teacher", "nap", "distill", "baselines", "tables",
               "xai", "expert", "bench", "stats"]


def make_logger(path):
    def log(msg):
        print(msg, flush=True)
        with open(path, "a") as f:
            f.write(msg + "\n")
    return log


def _safe_spearman_mean(A, B):
    """Mean Spearman row-correlation, NaN-safe (constant rows -> 0)."""
    rs = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(len(A)):
            r = spearmanr(A[i], B[i]).statistic
            rs.append(0.0 if np.isnan(r) else float(r))
    return float(np.mean(rs))


def _sal(model, X, y, device, batch=32):
    """Batched Grad-CAM channel saliency with true classes."""
    outs = []
    model.eval()
    for s in range(0, len(y), batch):
        S = channel_saliency(model, X[s:s + batch].to(device),
                             class_idx=y[s:s + batch].to(device),
                             normalize=True)
        outs.append(S.detach().cpu())
    return torch.cat(outs)


def run_dataset(dataset: str, cfg: Config, outdir: str, logger=print,
                stages=None) -> dict:
    t_start = time.time()
    device = cfg.device
    ds = dataset
    os.makedirs(f"{outdir}/tables", exist_ok=True)
    os.makedirs(f"{outdir}/figures", exist_ok=True)
    stages = STAGE_ORDER if stages is None or stages == ["all"] else stages

    # ------------------------------------------------------ data (always)
    data = generate_dataset(ds, cfg, seed=cfg.seed)
    splits = split_by_subject(data, seed=cfg.seed)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_calib, y_calib = splits["X_calib"], splits["y_calib"]
    X_test, y_test = splits["X_test"], splits["y_test"]
    C, F, T = data["X"].shape[1:]
    n_classes = len(data["classes"])
    channels, roi_idx = data["channels"], data["roi_idx"]
    subj_test = splits["subj_test"]
    sds_X = X_test[:cfg.nap_sds_trials].to(device)
    sds_y = y_test[:cfg.nap_sds_trials].to(device)
    calib_X = X_calib[:cfg.qat_calib]

    # ------------------------------------------------- checkpoint state
    ckpt_path = f"{outdir}/ckpt_{ds}.pth"
    state = dict(models={}, results={})
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, weights_only=False)
        logger(f"=== [{ds}] loaded checkpoint with variants: "
               f"{list(state['models'].keys())} ===")

    def save_ckpt():
        torch.save(state, ckpt_path)

    def register(name, model, stage_note="", quantized=False):
        state["models"][name] = model
        acc = accuracy(model, X_test, y_test, device)
        logits = collect_logits(model, X_test, device)
        psa = per_subject_accuracy(logits, y_test, subj_test)
        S_m = _sal(model, X_test, y_test, device)
        cs = compression_stability(S_full_test_ref, S_m.numpy(), roi_idx)
        size = (int8_size_mb(model) if quantized
                else model_size_mb(model, "fp32"))
        lb = latency_benchmark(model, X_test, device,
                               warmup=cfg.latency_warmup,
                               runs=min(cfg.latency_runs, 100))
        state["results"][name] = dict(
            variant=name, note=stage_note,
            accuracy=100 * acc, acc_drop_pp=100 * (acc - acc_teacher_ref),
            size_mb=size, params_m=count_parameters(model) / 1e6,
            params=count_parameters(model),
            latency_ms=lb["mean_ms"], latency_p99_ms=lb["p99_ms"],
            csi=cs["csi_mean"], csi_s=cs["csi_s_mean"], spr=cs["spr_mean"],
            csi_ci=bca_bootstrap(cs["per_trial"]["csi"],
                                 n_boot=cfg.n_bootstrap),
            spr_ci=percentile_ci(cs["per_trial"]["spr"],
                                 n_boot=cfg.n_bootstrap),
            per_subject_acc=psa, sal=S_m, n_csi_valid=cs.get("n_valid"),
        )
        logger(f"  [{name}] acc={100 * acc:.2f}%  size={size:.2f}MB  "
               f"CSI={cs['csi_mean']:.3f}  SPR={cs['spr_mean']:.3f}  "
               f"lat={lb['mean_ms']:.1f}ms")
        return model

    # reference values filled by the teacher stage (or checkpoint)
    S_full_test_ref = state.get("S_full_test")
    acc_teacher_ref = state.get("acc_teacher")
    acc_full_ref = state.get("acc_teacher", 0.0)

    # ================================================================ 1
    if "teacher" in stages:
        logger(f"\n=== [{ds}] Stage 1-2: data + CWT features ===")
        logger(f"  subjects={len(np.unique(data['subject']))}  "
               f"trials={len(data['y'])} channels={C} freqs={F} times={T} "
               f"classes={n_classes} "
               f"ROI={list(np.array(channels)[roi_idx])}")
        logger(f"=== [{ds}] Stage 3a: full FP32 teacher (CNN-GRU) ===")
        torch.manual_seed(cfg.seed)
        teacher = CNNGRU(C, n_classes, n_freqs=F, n_times=T,
                         filters=cfg.filters, gru_hidden=cfg.gru_hidden,
                         gru_layers=cfg.gru_layers,
                         embedding_dim=cfg.embedding_dim,
                         spatial_dim=cfg.spatial_dim)
        logger(f"  params={count_parameters(teacher) / 1e6:.3f}M  "
               f"size={model_size_mb(teacher):.2f} MB (FP32)")
        teacher = train_model(teacher, X_train, y_train, X_calib, y_calib,
                              device, epochs=cfg.teacher_epochs, lr=cfg.lr,
                              weight_decay=cfg.weight_decay, batch=cfg.batch,
                              patience=max(2, cfg.teacher_epochs // 2),
                              logger=logger, seed=cfg.seed)
        acc_full = accuracy(teacher, X_test, y_test, device)
        logits_full_test = collect_logits(teacher, X_test, device)
        logits_full_train = collect_logits(teacher, X_train, device)
        psa_full = per_subject_accuracy(logits_full_test, y_test, subj_test)
        logger(f"  teacher test accuracy = {100 * acc_full:.2f}%  "
               f"(chance {100 / n_classes:.1f}%)")
        S_full_test = _sal(teacher, X_test, y_test, device)

        # fill references, then register
        S_full_test_ref = S_full_test
        acc_teacher_ref = acc_full
        acc_full_ref = acc_full
        state["S_full_test"] = S_full_test
        state["acc_teacher"] = acc_full
        state["psa_full"] = psa_full
        state["logits_full_train"] = logits_full_train
        register("Full FP32 (teacher)", teacher, "no compression", False)
        save_ckpt()

    # ================================================================ 2
    if "nap" in stages and "NAP only" not in state["models"]:
        teacher = state["models"]["Full FP32 (teacher)"]
        logger(f"=== [{ds}] Stage 3b: NAP (lambda={cfg.nap_lambda}, "
               f"K={cfg.nap_iterations}, s={cfg.nap_sparsity}) ===")
        nap_model, nap_info = nap(teacher, (X_train, y_train), sds_X, sds_y,
                                  roi_idx, cfg, device, criterion="nap",
                                  logger=logger)
        register("NAP only", nap_model, "structured pruning, SDS criterion")
        save_ckpt()

        logger(f"=== [{ds}] Stage 3c: INT8 QAT (b={cfg.qat_bits}) ===")
        napq = quantize_model(nap_model, calib_X, mode="qat", device=device,
                              logger=logger)
        if cfg.qat_epochs > 0:
            from .train import finetune
            napq = finetune(napq, (X_train, y_train), device,
                            epochs=cfg.qat_epochs, lr=cfg.qat_lr,
                            logger=lambda m: None)
        register("NAP + QAT", napq, "pruned + INT8 QAT (STE)", quantized=True)
        save_ckpt()

    # ================================================================ 3
    if "distill" in stages and "NAC (NAP+QAT+SPKD)" not in state["models"]:
        teacher = state["models"]["Full FP32 (teacher)"]
        napq = state["models"]["NAP + QAT"]
        nap_model = state["models"].get("NAP only")
        logger(f"=== [{ds}] Stage 3d: SPKD (alpha={cfg.spkd_alpha}, "
               f"beta={cfg.spkd_beta}, T={cfg.spkd_T}) ===")
        nac = run_spkd(teacher, napq, X_train, y_train, device, cfg,
                       logger=logger)
        register("NAC (NAP+QAT+SPKD)", nac, "full NAC pipeline",
                 quantized=True)
        save_ckpt()

        if cfg.run_hinton and "NAP+QAT+Hinton (beta=0)" not in state["models"]:
            logger(f"=== [{ds}] SPKD ablation: Hinton KD (beta=0) ===")
            hinton = run_spkd(teacher, napq, X_train, y_train, device, cfg,
                              beta=0.0, logger=logger)
            register("NAP+QAT+Hinton (beta=0)", hinton,
                     "distillation without saliency term", quantized=True)
            save_ckpt()

        if cfg.run_alt_ordering and "NAP->SPKD->QAT" not in state["models"] \
                and nap_model is not None:
            logger(f"=== [{ds}] Alternative ordering: NAP -> SPKD -> QAT ===")
            pdq_student = run_spkd(teacher, nap_model, X_train, y_train,
                                   device, cfg, logger=logger)
            pdq = quantize_model(pdq_student, calib_X, mode="qat",
                                 device=device, logger=lambda m: None)
            register("NAP->SPKD->QAT", pdq, "alternative ordering",
                     quantized=True)
            save_ckpt()

    # ================================================================ 4
    if "baselines" in stages:
        if cfg.run_l1_baseline and "L1-only prune" not in state["models"]:
            teacher = state["models"]["Full FP32 (teacher)"]
            logger(f"=== [{ds}] Baseline: L1-only pruning ===")
            l1m = l1_prune(teacher, (X_train, y_train), cfg, device,
                           logger=logger)
            register("L1-only prune", l1m, "magnitude pruning baseline")
            l1_ptq = quantize_model(l1m, calib_X, mode="ptq", device=device,
                                    logger=logger)
            register("L1 + PTQ", l1_ptq, "post-training INT8",
                     quantized=True)
            l1_qat = quantize_model(l1m, calib_X, mode="qat", device=device,
                                    logger=lambda m: None)
            if cfg.qat_epochs > 0:
                from .train import finetune
                l1_qat = finetune(l1_qat, (X_train, y_train), device,
                                  epochs=cfg.qat_epochs, lr=cfg.qat_lr,
                                  logger=lambda m: None)
            register("L1 + QAT", l1_qat, "magnitude + INT8 QAT",
                     quantized=True)
            save_ckpt()

        if cfg.run_lth and "Lottery ticket" not in state["models"]:
            logger(f"=== [{ds}] Baseline: lottery-ticket hypothesis ===")
            torch.manual_seed(cfg.seed)
            teacher_init = CNNGRU(C, n_classes, n_freqs=F, n_times=T,
                                  filters=cfg.filters,
                                  gru_hidden=cfg.gru_hidden,
                                  gru_layers=cfg.gru_layers,
                                  embedding_dim=cfg.embedding_dim,
                                  spatial_dim=cfg.spatial_dim)
            lth = lottery_ticket(teacher_init, (X_train, y_train), cfg,
                                 device, logger=logger)
            register("Lottery ticket", lth, "L1-pruned init, retrained")
            save_ckpt()

    # ================================================================ 5
    if "tables" in stages:
        logger(f"=== [{ds}] Tables 6-7, 16 / Figures 2-3, 7a ===")
        results = state["results"]
        name_map = [
            ("Full FP32 (teacher)", "Full Model (FP32)"),
            ("L1-only prune", "Pruned (L1-only)"),
            ("L1 + PTQ", "Pruned + PTQ (INT8)"),
            ("L1 + QAT", "Pruned + QAT (INT8)"),
            ("NAP only", "NAP only (proposed)"),
            ("NAP + QAT", "NAP + QAT"),
            ("NAC (NAP+QAT+SPKD)", "NAP + QAT + SPKD (Full NAC)"),
            ("Lottery ticket", "Lottery Ticket (init prune)"),
        ]
        t6 = []
        for key, label in name_map:
            if key in results:
                r = results[key]
                t6.append(dict(
                    configuration=label, size_mb=round(r["size_mb"], 3),
                    params_m=round(r["params_m"], 3),
                    latency_ms=round(r["latency_ms"], 2),
                    latency_p99_ms=round(r["latency_p99_ms"], 2),
                    accuracy=round(r["accuracy"], 2),
                    acc_drop=round(r["acc_drop_pp"], 2),
                    csi=round(r["csi"], 3)))
        pd.DataFrame(t6).to_csv(
            f"{outdir}/tables/table6_ablation_{ds}.csv", index=False)

        t7 = []
        for key, label in [("NAC (NAP+QAT+SPKD)", "NAP -> QAT -> SPKD (proposed)"),
                           ("NAP->SPKD->QAT", "NAP -> SPKD -> QAT"),
                           ("NAP + QAT", "NAP -> QAT (no SPKD)"),
                           ("NAP+QAT+Hinton (beta=0)", "NAP -> QAT -> Hinton (beta=0)"),
                           ("Full FP32 (teacher)", "No compression (full FP32)")]:
            if key in results:
                r = results[key]
                t7.append(dict(ordering=label, accuracy=round(r["accuracy"], 2),
                               size_mb=round(r["size_mb"], 3),
                               csi=round(r["csi"], 3),
                               spr=round(r["spr"], 3)))
        pd.DataFrame(t7).to_csv(
            f"{outdir}/tables/table7_orderings_{ds}.csv", index=False)

        present = [k for k, _ in name_map if k in results]
        figs.fig_ablation(
            [lbl for k, lbl in name_map if k in results],
            [results[k]["accuracy"] for k in present],
            [results[k]["size_mb"] for k in present],
            [results[k]["csi"] for k in present],
            [results[k]["acc_drop_pp"] for k in present],
            f"{outdir}/figures/fig2_ablation_{ds}.png", cfg.csi_threshold)

        pareto = [(lbl, results[k]["accuracy"], results[k]["csi"],
                   "proposed" in lbl.lower())
                  for k, lbl in name_map if k in results
                  if k in ("NAC (NAP+QAT+SPKD)", "NAP->SPKD->QAT",
                           "NAP + QAT", "NAP+QAT+Hinton (beta=0)",
                           "Full FP32 (teacher)")]
        figs.fig_pareto(pareto, f"{outdir}/figures/fig3_pareto_{ds}.png",
                        cfg.csi_threshold)

        t16 = []
        for key, label in [("Full FP32 (teacher)", "Full model (FP32)"),
                           ("NAP only", "NAP only"),
                           ("NAP + QAT", "NAP + QAT (no SPKD)"),
                           ("NAP+QAT+Hinton (beta=0)", "NAP+QAT+Hinton KD (beta=0)"),
                           ("NAC (NAP+QAT+SPKD)", "NAP+QAT+SPKD (beta=0.3, full)")]:
            if key in results:
                r = results[key]
                t16.append(dict(
                    stage=label, csi=round(r["csi"], 3),
                    csi_ci=f"({r['csi_ci']['ci_low']:.3f}, {r['csi_ci']['ci_high']:.3f})",
                    csi_s=round(r["csi_s"], 3), spr=round(r["spr"], 3),
                    spr_ci=f"({r['spr_ci']['ci_low']:.3f}, {r['spr_ci']['ci_high']:.3f})"))
        pd.DataFrame(t16).to_csv(
            f"{outdir}/tables/table16_csi_spr_{ds}.csv", index=False)

        stage_keys = [k for k in
                      ["Full FP32 (teacher)", "NAP only", "NAP + QAT",
                       "NAP+QAT+Hinton (beta=0)", "NAC (NAP+QAT+SPKD)"]
                      if k in results]
        stage_label = {"Full FP32 (teacher)": "Full", "NAP only": "NAP",
                       "NAP + QAT": "NAP+QAT",
                       "NAP+QAT+Hinton (beta=0)": "+Hinton",
                       "NAC (NAP+QAT+SPKD)": "+SPKD (NAC)"}
        figs.fig_csi_progression(
            [stage_label[k] for k in stage_keys],
            [results[k]["csi"] for k in stage_keys],
            [results[k]["csi_ci"]["ci_low"] for k in stage_keys],
            [results[k]["csi_ci"]["ci_high"] for k in stage_keys],
            f"{outdir}/figures/fig7a_csi_progression_{ds}.png",
            cfg.csi_threshold)

    # ================================================================ 6
    if "xai" in stages:
        logger(f"=== [{ds}] Stage 4: multi-method XAI battery ===")
        results = state["results"]
        nac_model = state["models"]["NAC (NAP+QAT+SPKD)"]
        S_nac = results["NAC (NAP+QAT+SPKD)"]["sal"]
        xai_rows = []

        ig_maps, ig_chan = xai.integrated_gradients(
            nac_model, X_test, y_test, device, steps=cfg.ig_steps)
        ig_nac = ig_chan.numpy()
        ig_s = _sal(nac_model, X_test, y_test, device).numpy()
        rho_ig = _safe_spearman_mean(ig_nac, ig_s)
        xai_rows.append(dict(method="Integrated Gradients",
                             metric="Spearman(IG, Grad-CAM)",
                             value=round(rho_ig, 3)))

        ec = xai.eigencam_channels(nac_model, X_test, device).numpy()
        rho_ec = _safe_spearman_mean(ec, ig_s)
        xai_rows.append(dict(method="EigenCAM",
                             metric="Spearman(EigenCAM, Grad-CAM)",
                             value=round(rho_ec, 3)))

        n_perm_subset = min(128, len(y_test))
        drops, base_pi = xai.conditional_perm_importance(
            nac_model, X_test[:n_perm_subset], y_test[:n_perm_subset],
            device, channels, seed=cfg.seed)
        top_pi = np.argsort(-drops)[:5].tolist()
        xai_rows.append(dict(method="Conditional permutation",
                             metric="top-5 channels",
                             value=",".join(np.array(channels)[top_pi])))
        xai_rows.append(dict(method="Conditional permutation",
                             metric="max accuracy drop (pp)",
                             value=round(100 * drops.max(), 2)))

        n_dose = min(192, len(y_test))
        dose, base_dose = xai.dose_response(
            nac_model, X_test[:n_dose], y_test[:n_dose], device,
            S_nac[:n_dose], kmax=cfg.dose_kmax, seed=cfg.seed)
        figs.fig_dose(dose["drop"], base_dose,
                      f"{outdir}/figures/fig6_dose_response_{ds}.png",
                      cfg.dose_kmax)
        k_rep = min(5, len(dose["drop"]["top"])) - 1
        xai_rows.append(dict(method="Causal ablation (dose-response)",
                             metric=f"acc drop k={k_rep + 1} top channels (pp)",
                             value=round(100 * dose["drop"]["top"][k_rep], 2)))
        xai_rows.append(dict(method="Causal ablation (dose-response)",
                             metric=f"acc drop k={k_rep + 1} random channels (pp)",
                             value=round(100 * dose["drop"]["random"][k_rep], 2)))

        band = np.concatenate([data["mu_idx"], data["beta_idx"]])
        tf = xai.tf_null_test(ig_maps, roi_idx, band,
                              n_shuffles=cfg.n_null_shuffles, seed=cfg.seed)
        xai_rows.append(dict(method="TF null (band-shuffled)",
                             metric="ROI mu/beta mass (observed)",
                             value=round(tf["observed"], 4)))
        xai_rows.append(dict(method="TF null (band-shuffled)",
                             metric="z-score vs null", value=round(tf["z"], 2)))
        xai_rows.append(dict(method="TF null (band-shuffled)",
                             metric="p-value", value=tf["p"]))

        san_full = xai.adebayo_sanity(nac_model, X_test[:64],
                                      y_test[:64], device, mode="full")
        san_cas = xai.adebayo_sanity(nac_model, X_test[:64],
                                     y_test[:64], device, mode="cascade")
        xai_rows.append(dict(method="Adebayo sanity (full rand)",
                             metric="Spearman(orig, randomized)",
                             value=round(san_full["spearman_mean"], 3)))
        xai_rows.append(dict(method="Adebayo sanity (cascade rand)",
                             metric="Spearman(orig, randomized)",
                             value=round(san_cas["spearman_mean"], 3)))

        jac = xai.jaccard_roi(S_nac, roi_idx, top_frac=cfg.top_frac)
        jac_f = xai.jaccard_roi(state["S_full_test"], roi_idx,
                                top_frac=cfg.top_frac)
        xai_rows.append(dict(method="Neuro ground truth",
                             metric="Jaccard(top-33%, ROI) compressed",
                             value=round(jac, 3)))
        xai_rows.append(dict(method="Neuro ground truth",
                             metric="Jaccard(top-33%, ROI) full",
                             value=round(jac_f, 3)))
        pd.DataFrame(xai_rows).to_csv(
            f"{outdir}/tables/xai_summary_{ds}.csv", index=False)

        figs.fig_saliency(channels, state["S_full_test"].mean(0).numpy(),
                          S_nac.mean(0).numpy(), roi_idx,
                          f"{outdir}/figures/fig_saliency_{ds}.png")
        state["xai"] = dict(jaccard_compressed=jac, jaccard_full=jac_f,
                            ig_gradcam_spearman=rho_ig,
                            eigencam_spearman=rho_ec, tf_null=tf,
                            adebayo_full=san_full, adebayo_cascade=san_cas,
                            dose_response=dose["drop"],
                            perm_top=np.array(channels)[top_pi].tolist(),
                            perm_max_drop_pp=100 * float(drops.max()))
        save_ckpt()

    # ================================================================ 7
    if "expert" in stages:
        logger(f"=== [{ds}] Stages 4-6: expert-system decision layer ===")
        nac_model = state["models"]["NAC (NAP+QAT+SPKD)"]
        logits_nac_calib = collect_logits(nac_model, X_calib, device)
        logits_nac_test = collect_logits(nac_model, X_test, device)
        actions = KARAONE_ACTIONS if ds == "karaone" else BCI2A_ACTIONS
        exp = evaluate_expert_layer(
            logits_nac_calib, y_calib, logits_nac_test, y_test,
            alpha=cfg.conformal_alpha, tau_commit=cfg.tau_commit,
            action_map=actions)
        logger(f"  T_cal={exp['temperature']:.3f}  "
               f"ECE {exp['ece_uncalibrated']:.3f} -> "
               f"{exp['ece_calibrated']:.3f}  "
               f"coverage={exp['conformal_coverage']:.1f}%  "
               f"committed-acc={exp['committed_accuracy']:.1f}% "
               f"(raw {exp['raw_accuracy']:.1f}%)  "
               f"abstention={exp['abstention_rate']:.1f}%")
        pd.DataFrame([
            dict(metric="ECE (uncalibrated)", value=exp["ece_uncalibrated"]),
            dict(metric="ECE (temperature-scaled)",
                 value=exp["ece_calibrated"]),
            dict(metric="Conformal coverage (alpha=0.1)",
                 value=exp["conformal_coverage"]),
            dict(metric="Abstention rate (%)", value=exp["abstention_rate"]),
            dict(metric="Committed-trial accuracy (%)",
                 value=exp["committed_accuracy"]),
            dict(metric="Raw classifier accuracy (%)",
                 value=exp["raw_accuracy"]),
            dict(metric="Temperature T_cal", value=exp["temperature"]),
            dict(metric="Conformal threshold tau_alpha",
                 value=exp["tau_alpha"]),
        ]).to_csv(f"{outdir}/tables/table17_expert_{ds}.csv", index=False)
        pd.DataFrame(exp["sweep"]).to_csv(
            f"{outdir}/tables/table18_tradeoff_{ds}.csv", index=False)
        figs.fig_tradeoff(exp["sweep"],
                          f"{outdir}/figures/fig7b_tradeoff_{ds}.png")
        dec_rows = expert_decisions(
            torch.softmax(logits_nac_test, dim=1).numpy(), exp["tau_alpha"],
            cfg.tau_commit, actions)
        examples = clinical_rule_examples(dec_rows, n=5,
                                          classes=data["classes"])
        logger("  clinical decision examples: " + json.dumps(examples))
        state["expert_layer"] = exp
        state["clinical_examples"] = examples
        save_ckpt()

    # ================================================================ 8
    if "bench" in stages:
        logger(f"=== [{ds}] Stage 7: edge benchmarking (host CPU) ===")
        teacher = state["models"]["Full FP32 (teacher)"]
        nac_model = state["models"]["NAC (NAP+QAT+SPKD)"]
        lb_full = latency_benchmark(teacher, X_test, device,
                                    warmup=cfg.latency_warmup,
                                    runs=cfg.latency_runs)
        lb_nac = latency_benchmark(nac_model, X_test, device,
                                   warmup=cfg.latency_warmup,
                                   runs=cfg.latency_runs)
        drift = sustained_drift(nac_model, X_test, device,
                                runs=cfg.sustained_runs)
        online = sliding_window_online(nac_model, X_test, y_test, device,
                                       window_len=cfg.window_len,
                                       step=cfg.window_step,
                                       n_windows=cfg.n_vote_windows)
        bench_rows = [
            dict(model="Full FP32 (teacher)", platform="host-CPU",
                 latency_ms=round(lb_full["mean_ms"], 2),
                 p99_ms=round(lb_full["p99_ms"], 2),
                 throughput=round(lb_full["throughput_per_s"], 1)),
            dict(model="NAC (compressed)", platform="host-CPU",
                 latency_ms=round(lb_nac["mean_ms"], 2),
                 p99_ms=round(lb_nac["p99_ms"], 2),
                 throughput=round(lb_nac["throughput_per_s"], 1)),
            dict(model="NAC sustained", platform="host-CPU",
                 latency_ms=round(drift["last100_ms"], 2),
                 p99_ms=None, throughput=None),
            dict(model="NAC online (sliding window)", platform="host-CPU",
                 latency_ms=None, p99_ms=None, throughput=None),
        ]
        pd.DataFrame(bench_rows).to_csv(
            f"{outdir}/tables/benchmark_{ds}.csv", index=False)
        logger(f"  latency: full {lb_full['mean_ms']:.1f} ms  "
               f"NAC {lb_nac['mean_ms']:.1f} ms (p99 {lb_nac['p99_ms']:.1f})  "
               f"drift {drift['drift_pct']:+.1f}%  "
               f"online-acc {100 * online['online_accuracy']:.2f}%")
        state["benchmark"] = dict(teacher_latency=lb_full,
                                  nac_latency=lb_nac, sustained=drift,
                                  online=online)
        try:
            onnx_path = f"{outdir}/models_nac_{ds}.onnx"
            export_onnx(nac_model, X_test, onnx_path)
            logger(f"  ONNX export -> {onnx_path}")
        except Exception as e:  # pragma: no cover
            logger(f"  ONNX export skipped ({e})")
        save_ckpt()

    # ================================================================ 9
    if "stats" in stages:
        logger(f"=== [{ds}] Statistical validation (pre-registered) ===")
        results = state["results"]
        psa_full = state["psa_full"]
        psa_nac = results["NAC (NAP+QAT+SPKD)"]["per_subject_acc"]
        subj_ids = sorted(psa_full)
        diffs = np.array([psa_nac[s] - psa_full[s] for s in subj_ids])
        ni = noninferiority(diffs, margin=cfg.noninf_margin)
        wil = wilcoxon_signed([psa_nac[s] for s in subj_ids],
                              [psa_full[s] for s in subj_ids])
        dz = cohens_dz(diffs)
        r_nac = results["NAC (NAP+QAT+SPKD)"]
        stats_rows = [
            dict(test="Non-inferiority (delta=1%)", statistic=ni["mean"],
                 p=None,
                 detail=f"CI_low={ni['ci_low']:.4f}, noninferior={ni['noninferior']}"),
            dict(test="Wilcoxon signed-rank (full vs NAC)",
                 statistic=wil["statistic"], p=wil["p"],
                 detail="subject-level paired accuracies"),
            dict(test="Cohen's d_z (accuracy drop)", statistic=dz, p=None,
                 detail="paired per-subject"),
            dict(test="CSI bootstrap (NAC)", statistic=r_nac["csi_ci"]["point"],
                 p=None,
                 detail=f"BCa 95% CI ({r_nac['csi_ci']['ci_low']:.3f}, "
                        f"{r_nac['csi_ci']['ci_high']:.3f})"),
            dict(test="SPR bootstrap (NAC)", statistic=r_nac["spr_ci"]["point"],
                 p=None,
                 detail=f"95% CI ({r_nac['spr_ci']['ci_low']:.3f}, "
                        f"{r_nac['spr_ci']['ci_high']:.3f})"),
            dict(test="Permutation (TF null)", statistic=state["xai"]["tf_null"]["z"],
                 p=state["xai"]["tf_null"]["p"],
                 detail=f"observed={state['xai']['tf_null']['observed']:.4f}"),
        ]
        pd.DataFrame(stats_rows).to_csv(
            f"{outdir}/tables/stats_{ds}.csv", index=False)
        logger(f"  non-inferior: {ni['noninferior']} (mean diff "
               f"{100 * ni['mean']:+.2f}pp, CI_low {100 * ni['ci_low']:+.2f}pp)  "
               f"Wilcoxon p={wil['p']:.3f}  d_z={dz:.2f}")
        state["statistics"] = dict(noninferiority=ni, wilcoxon=wil,
                                   cohens_dz=dz)

        # ---- Figure 5 (accuracy vs size, incl. light baselines) ----
        acc_size_points = []
        for k, lbl in [
                ("Full FP32 (teacher)", "Full Model (FP32)"),
                ("L1-only prune", "L1-only prune"),
                ("L1 + PTQ", "L1 + PTQ"),
                ("L1 + QAT", "L1 + QAT"),
                ("NAP only", "NAP only"),
                ("NAP + QAT", "NAP + QAT"),
                ("NAC (NAP+QAT+SPKD)", "NAC (full)"),
                ("NAP->SPKD->QAT", "NAP-SPKD-QAT"),
                ("NAP+QAT+Hinton (beta=0)", "Hinton KD (beta=0)"),
                ("Lottery ticket", "Lottery ticket")]:
            if k in results:
                kind = "main" if k == "NAC (NAP+QAT+SPKD)" else "baseline"
                acc_size_points.append((lbl, results[k]["accuracy"],
                                        results[k]["size_mb"], kind))
        if cfg.run_light_baselines:
            logger(f"=== [{ds}] Lightweight baselines (Fig 5) ===")
            n_t_flat = F * T
            for cls, nm in ((EEGNet, "EEGNet"),
                            (ShallowConvNet, "ShallowConvNet")):
                torch.manual_seed(cfg.seed)
                m = cls(C, n_classes, n_times=n_t_flat).to(device)
                m = train_model(m, X_train, y_train, X_calib, y_calib,
                                device, epochs=cfg.baseline_epochs, lr=1e-3,
                                weight_decay=1e-4, batch=cfg.batch,
                                logger=lambda m_: None, seed=cfg.seed)
                acc_b = accuracy(m, X_test, y_test, device)
                size_b = model_size_mb(m)
                acc_size_points.append((nm, 100 * acc_b, size_b, "light"))
                logger(f"  {nm}: acc={100 * acc_b:.2f}%  size={size_b:.3f}MB  "
                       f"params={count_parameters(m)}")
        figs.fig_acc_size(acc_size_points,
                          f"{outdir}/figures/fig5_acc_size_{ds}.png")

        # ---- per-dataset summary ----
        teacher = state["models"]["Full FP32 (teacher)"]
        nac_model = state["models"]["NAC (NAP+QAT+SPKD)"]
        summary = dict(
            dataset=ds, preset=cfg.preset, seed=cfg.seed,
            n_subjects=int(len(np.unique(data["subject"]))),
            n_trials=int(len(data["y"])), n_channels=C, n_classes=n_classes,
            teacher=dict(params=count_parameters(teacher),
                         size_mb=model_size_mb(teacher),
                         accuracy=100 * state["acc_teacher"]),
            nac=dict(params=count_parameters(nac_model),
                     size_mb=int8_size_mb(nac_model),
                     accuracy=results["NAC (NAP+QAT+SPKD)"]["accuracy"],
                     size_reduction_pct=100 * (1 - int8_size_mb(nac_model)
                                               / model_size_mb(teacher)),
                     csi=r_nac["csi"], csi_s=r_nac["csi_s"], spr=r_nac["spr"],
                     latency_ms=state["benchmark"]["nac_latency"]["mean_ms"],
                     sustained_drift_pct=state["benchmark"]["sustained"]["drift_pct"],
                     online_accuracy=state["benchmark"]["online"]["online_accuracy"]),
            expert_layer=state["expert_layer"],
            xai=state["xai"],
            statistics=state["statistics"],
            benchmark=state["benchmark"],
            clinical_examples=state.get("clinical_examples", []),
            table6=pd.read_csv(f"{outdir}/tables/table6_ablation_{ds}.csv")
                .to_dict(orient="records"),
            table16=pd.read_csv(f"{outdir}/tables/table16_csi_spr_{ds}.csv")
                .to_dict(orient="records"),
        )
        with open(f"{outdir}/summary_{ds}.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger(f"=== [{ds}] done (summary written to {outdir}) ===")
    return state
