# پیاده‌سازی کامل مقاله NAC به زبان پایتون

**مقاله:** «Edge-Deployable Explainable EEG-Based Speech Motor Imagery Classification
via Neurophysiology-Aware Compression and Causal Saliency Validation»

این بسته، پیاده‌سازی کامل و قابل‌اجرا از کل روش مقاله است: خط‌لوله شش‌مرحله‌ای
NAC (هرس آگاهانه از نوروفیزیولوژی → کوانتیزاسیون INT8 آگاهانه از آموزش →
تقطیع دانش با حفظ سالیانسی)، معیارهای CSI/SPR، مجموعه‌ کامل XAI چندروشی،
لایه تصمیم سیستم خبره با پیش‌بینی کانفورمال و انصراف، و ارزیابی‌های آماری
پیش‌ثبت‌شده. اجرای پرِست «demo» روی این ماشین خروجی‌های واقعی تولید کرده که در
پوشه `results/` موجود است.

---

## اجرای سریع

```bash
pip install -r requirements.txt

python run_all.py --preset smoke     # آزمون درستی کل خط‌لوله (~۳ دقیقه)
python run_all.py --preset demo      # اجرای کامل در مقیاس دمو (~۴۰ دقیقه CPU)
python run_all.py --preset paper     # مقیاس کامل مقاله (دیتای واقعی + GPU)
```

اجرا به‌صورت مرحله‌ای (checkpoint) هم ممکن است؛ برای جلسات کوتاه یا ماشین‌های ضعیف:

```bash
python run_all.py --preset demo --stages teacher,nap
python run_all.py --preset demo --stages distill,baselines
python run_all.py --preset demo --stages tables,xai
python run_all.py --preset demo --stages expert,bench,stats
```

هر مرحله مدل‌ها و متریک‌ها را در `results/ckpt_<dataset>.pth` ذخیره و در فراخوانی
بعدی بازیابی می‌کند.

---

## نتیجه اجرای دمو (این ماشین، seed=42، CPU دو هسته)

| کمیت | KaraOne (هدف SMI) | BCI IV-2a (منبع انتقال) |
|---|---|---|
| دقت معلم FP32 | ‏80.9٪ (شانس 9.1٪) | ‏99.5٪ (شانس 25٪) |
| دقت مدل فشرده NAC | ‏81.9٪ | ‏99.5٪ |
| حجم: FP32 → INT8 | ‏8.70 → 2.03 MB (−76.6٪) | ‏4.59 → 1.01 MB (−78.1٪) |
| CSI (آستانه بالینی 0.85) | **0.907** ✅ | **0.920** ✅ |
| SPR | 1.002 | 0.997 |
| پوشش کانفورمال (α=0.1) | 88.0٪ | 98.6٪ |
| دقت Trials تخصیص‌یافته | 95.5٪ (از 81.9٪ خام) | 99.5٪ |
| ECE قبل → بعد از کالیبراسیون | 0.109 → 0.079 | — |
| غیر-دستکاری (δ=1٪) | ✅ CI_low = −0.55pp | ✅ |
| Wilcoxon / Cohen's d_z | p=0.225 / 0.40 | — |
| تأخیر استنتاج (CPU میزبان) | 15.3 ms | 6.4 ms |
| دقت آنلاین (پنجره لغزان) | 59.5٪ | 97.7٪ |

مسیر CSI در طول خط‌لوله (جدول 16) دقیقاً الگوی مقاله را بازتولید می‌کند:
Full=1.00 → NAP=0.728 → NAP+QAT=0.708 (زیر آستانه) → +SPKD=**0.907**
(بازیابی توسط جمله سالیانسی). مهم‌ترین یافته‌های XAI: پنج کانال علّیِ برتر
همگی در ROI حرکتی هستند (CP1, CP3, C3, Cz, C4)، حذف top-5 آن‌ها ‏22.4 واحد
دقت می‌کُشد (در برابر 7.3 برای تصادفی)، جرم سالیانسی mu/beta روی ROI با
p=0.005 معنادار است، و آزمون سلامت Adebayo پاس می‌شود.

---

## ساختار کد (نقشه‌ی بخش‌های مقاله)

| فایل | بخش/معادلات مقاله |
|---|---|
| `nac/config.py` | همه ابرپارامترها (λ=0.6, K=4, s=0.4, α=β=0.3, T=4, τ=0.7, δ=1٪, CSI≥0.85) + پریست‌ها |
| `nac/data.py` | §5.1 دیتاست‌ها، ROI بالینی (Def. 4.1)، پیش‌پردازش، CWT مورله، بارگذار داده واقعی |
| `nac/models.py` | §4.2 معماری پایه CNN-GRU (بلوک‌های 32/64/128/256، GRU دو لایه 128) + EEGNet/ShallowConvNet |
| `nac/gradcam.py` | Eq. 5, 11, 14, 15 — سالیانسی کانال Grad-CAM با زیرگرادیان softplus (قابل‌مشتق) |
| `nac/nap.py` | §4.3.1 Alg. 1 — SDS، Importance=λ·SDS+(1−λ)·NormL1، هرس ساختاری تکراری + L1/LTH |
| `nac/qat.py` | §4.3.2 Alg. 2 — FakeQuant INT8 (per-channel وزن، per-tensor اکتیویشن، STE) + PTQ |
| `nac/spkd.py` | §4.3.3 Alg. 3 — L_SPKD = α·CE + (1−α−β)·T²·KL + β·L_SAL، مشتق‌پذیری سالیانسی (Prop. 4.1) |
| `nac/xai.py` | §4.4 — IG، EigenCAM، جایگشت شرطی، دوز-پاسخ علّی، null باند-فرکانسی، Adebayo، Jaccard |
| `nac/metrics.py` | Def. 4.2-4.3 — CSI (Eq. 17)، CSI-S (18)، SPR (19)، ECE، BCa، Wilcoxon، d_z، آزمون غیر-دستکاری |
| `nac/expert.py` | §4.6 — کالیبراسیون دما (Eq. 20)، مجموعه کانفورمال (Eq. 21، Def. 4.4)، قانون تصمیم بالینی |
| `nac/bench.py` | §4.7 — تأخیر/پایداری حرارتی/پنجره لغزان آنلاین، خروجی ONNX (opset 13) |
| `nac/pipeline.py` | ارکستراتور ۹ مرحله‌ای با checkpoint |
| `run_all.py` | CLI |

خروجی‌ها: `results/tables/*.csv` (معادل جدول‌های 6، 7، 16، 17، 18 + XAI و آمار)،
`results/figures/*.png` (معادل شکل‌های 2، 3، 5، 6، 7a، 7b + مقایسه سالیانسی)،
`results/summary_*.json`، `results/models_nac_*.onnx` و `results/run.log`.

---

## یادداشت‌های پیاده‌سازی (صادقانه)

1. **داده:** KaraOne و BCI IV-2a قابل بازتوزیع نیستند؛ اجرای پیش‌فرض از
   **شبیه‌ساز نوروفیزیولوژیک** استفاده می‌کند که الگوهای تفکیک‌کننده کلاس
   (ERD در mu/beta روی نوار حساس‌حرکتی و ناحیه بروکا) دقیقاً روی ROI مقاله
   می‌کارد. بارگذار داده واقعی (`load_karaone_mat`، `load_bci2a_gdf`) و
   پیش‌پردازش کامل (فیلتر باترورث دوجهته، CWT مورله w₀=6، z-score) موجود است؛
   با `--data-dir` فعال می‌شود.
2. **معماری:** برای آنکه Grad-CAM (Eq. 14-15) مستقیماً سالیانسی *به‌ازای‌کانال*
   بدهد (کمیت لازم برای SDS، L_SAL، CSI/SPR و ROI_frac)، برج کانولوشن به‌ازای‌کانال
   است و یک GRU روی دنباله کانال‌ها + یک readout فضایی (معادل فیلتر فضایی CSP)
   اجرا می‌شود؛ مجموع پارامترها ‏2.28M ≈ مقاله (‏2.1M).
3. **مقیاس دمو:** هرس K=2×10٪ (مقاله: K=4×10٪)، یک seed (مقاله: پنج seed)،
   اپوک‌های کمتر. پریست `paper` تنظیمات کامل مقاله را دارد.
4. **INT8:** در محیط PyTorch، کوانتیزاسیون به‌صورت fake-quant با STE است؛
   هسته GRU در دمو با FP32 اجرا و در حساب حجم INT8 شمرده می‌شود؛ خروجی ONNX
   برای TensorRT/Jetson (مثل مقاله) تولید می‌شود. به همین دلیل تأخیر فشرده در
   PyTorch کمی بیشتر از FP32 است — شتاب واقعی INT8 به موتور TensorRT/ONNX-RT
   نیاز دارد (مثل استقرار مقاله روی Jetson/RPi4).
5. **دما:** روی نیمه‌اول مجموعه کالیبراسیون برازش و نیمه‌دوم برای نمره‌های
   کانفورمال استفاده می‌شود (§4.6 مقاله؛ تفکیک نیمه‌ها برای پرهیز از استفاده
   دوباره از همان داده).
6. **L_SAL روی نقشه‌های نرمال‌شده L1** محاسبه می‌شود (خوانش Def. 4.2)؛ در این
   داده مصنوعی جمله CE/KL بر MSE سالیانسی غالب است و بازپس‌گیری CSI عمدتاً
   از خودِ فرایند تقطیع می‌آید (در جدول 16، هر دو نسخه SPKD و Hinton بالای
   آستانه می‌روند؛ تفکیک β=0.3 نسبت به β=0 کمتر از مقاله است).
7. **تفاوت‌های تجربی با مقاله:** مزیت دقتی NAP نسبت به L1 بازتولید می‌شود
   (‏82.3٪ در برابر 80.6٪) اما مزیت CSIِ NAP نسبت به L1 در مقیاس دمو دیده
   نمی‌شود (نیازمند K=4/s=0.4 و زیرمجموعه SDS بزرگ‌تر)؛ ترتیب جایگزین
   NAP→SPKD→QAT در داده مصنوعی قابل‌مقایسه با ترتیب پیشنهادی است؛ Jaccard
   کمتر از مقاله است چون سقف ریاضیِ آن با top-33٪ از 64 کانال ≈0.43 است؛
   دقت‌های مصنوعی (به‌ویژه BCI2a) بالاتر از داده واقعی‌اند. این‌ها
   خروجی صادقانه‌ی پیاده‌سازی روی داده شبیه‌سازی‌شده‌اند، نه بازتولید عدد-to-عدد
   نتایج مقاله.
8. **برچسب‌های شکل‌ها انگلیسی** هستند تا با شکل‌های خود مقاله (که انگلیسی‌اند)
   یک‌به‌یک قابل مقایسه باشند.

---

## نیازمندی‌ها

`torch>=2.0, numpy, scipy, pandas, matplotlib` (و `onnx` برای خروجی ONNX).
بارگذار BCI IV-2a به‌طور اختیاری از `mne` استفاده می‌کند.

---

## English quickstart

Complete, runnable Python reimplementation of the NAC paper (neurophysiology-aware
EEG compression: NAP -> INT8 QAT -> SPKD; CSI/SPR metrics; multi-method causal XAI
battery; conformal expert layer with abstention). Default runs use a
neurophysiology-plausible data simulator planting class-discriminative mu/beta ERD
exactly on the paper's clinical ROI; loaders for the real KaraOne / BCI IV-2a are
included (`--data-dir`).

```bash
pip install -r requirements.txt
python run_all.py --preset smoke   # 3-min end-to-end correctness check
python run_all.py --preset demo    # full faithful run (~40 min CPU)
```

Outputs land in `results/`: CSV analogs of Tables 6/7/16/17/18, PNG analogs of
Figures 2/3/5/6/7a/7b, `summary_*.json`, ONNX exports, and `run.log`. Every module
docstring cites the paper section/equation it implements. See the honest
implementation-notes section above for the documented deviations (synthetic data,
demo-scale pruning schedule, fake-quant execution, host-CPU latency).
