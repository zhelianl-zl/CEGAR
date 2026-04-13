# CEGAR — Confident-Error Gradient Amplification for Robustness

CEGAR is a lightweight training method for deep classification that improves robustness by reallocating gradient budget toward **confident mistakes**. Instead of changing the forward objective, CEGAR keeps standard cross-entropy (CE) unchanged and rescales only the backward gradients on a per-sample basis. This makes it easy to plug into ordinary CE training while staying close to CE in wall time, memory use, and clean-error behavior.

The core idea is simple: not all mistakes are equally informative. A wrong prediction made with low confidence is often already near the decision boundary and is usually handled reasonably well by standard CE gradients. By contrast, a wrong prediction made with high confidence indicates a sharper and more dangerous failure: the model places a strong probability peak on the wrong class while appearing trustworthy. CEGAR detects such cases online and amplifies their training signal, while leaving other samples close to standard CE.

This repository also includes **Auto-CEGAR**, which removes most of the manual scheduling burden by adapting the confidence boundary and amplification strength automatically from running confidence statistics.

## Method Overview

For each sample in a minibatch, CEGAR computes a per-sample gate and uses it to scale the backward gradient:

```text
wrong_gate_i = 1 - p_y,i
conf_gate_i  = sigmoid(k · (conf_i - τ))
gate_i       = wrong_gate_i × conf_gate_i
scale_i      = 1 + λ · gate_i
```

where:

- `p_y,i` is the predicted probability of the true class.
- `conf_i` is the confidence signal, typically `pmax`.
- `τ` is the confidence boundary.
- `k` controls the sharpness of the confidence gate.
- `λ` controls the amplification strength.

The forward CE loss is unchanged. Only the backward gradient magnitude is rescaled through a custom autograd operator. An optional normalization step keeps the mean scale near 1, so CEGAR redistributes gradient mass without substantially changing the global step size.

## Auto-CEGAR

Manual CEGAR can require schedules for `λ`, `τ`, and `k`. Auto-CEGAR reduces this burden by:

- fixing `k` as a stable gate sharpness,
- adapting `τ` from the evolving confidence distribution,
- adapting `λ` from running gate and tail statistics.

This repository includes several automatic controllers, including:

- **`auto_q_valley`**: updates the confidence boundary from the valley of the per-epoch confidence histogram,
- **tail-based lambda control**: adjusts amplification strength from the active high-confidence-error region,
- **adaptive runtime cap**: limits amplification when the gate becomes unhealthy or overly sparse.

In practice, this turns CEGAR from a conceptually simple but manually awkward method into a usable training recipe.

## Key Features

- **Forward-unchanged CE**: keeps the standard CE objective and modifies only the backward gradient scale.
- **Confidence-aware emphasis**: focuses on samples that are both wrong and confident.
- **Auto-CEGAR controllers**: automatic boundary and amplification control from online confidence statistics.
- **Multi-dataset support**: CIFAR-10, CIFAR-100, SVHN, BinaryCIFAR-10, and ImageNet-32.
- **Robust-training baselines**: PGD-AT, TRADES, MART, and focal loss.
- **Adversarial evaluation**: FGSM, PGD-Linf, and PGD-Linf with random start.
- **Corruption evaluation**: CIFAR-C / CIFAR-100-C with configurable corruption lists and severity levels.
- **TSV-based sweep dispatch**: grid/list experiments via tab-separated config files; each row runs as an independent Slurm array task.

## Repository Structure

```text
train.py               # Main training entry point
ecg_loss.py            # CEGAR loss, confidence gate, Auto-CEGAR controllers
robust_losses.py       # TRADES / MART / PGD-AT / focal loss
models.py              # Model definitions (ResNet variants, WideResNet)
run.py                 # Local parallel launcher (multi-GPU, research use)
tools/
  eval_checkpoints.py  # Adversarial and corruption evaluation
  run_from_tsv.py      # TSV-to-CLI sweep dispatcher
scripts/
  cegs_array.sbatch    # Slurm array job template (PSC Bridges-2)
sweeps/                # Experiment config TSV files
configs/               # W&B / environment config files
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision wandb
```

ImageNet-32 requires the SmallImageNet pickle dataset. Set the environment variable `IMAGENET_DS_ROOT` to point to the directory containing `train_data_batch_1` ... `train_data_batch_10` and `val_data`.

## Quick Start

### Standard CEGAR training on CIFAR-10

```bash
python train.py \
  --dataset cifar10 \
  --stop epochs --stop_val 60 \
  --lr 0.01 --momentum 0.9 --batch 128 \
  --loss_stage1 ecg --loss_stage2 ecg \
  --stage1_epochs 60 --stage2_epochs 0 \
  --ecg_conf_type pmax \
  --ecg_lam_start auto_tr_autocap --ecg_lam_end 0.05 \
  --ecg_tau_start auto_q_valley --ecg_tau_end 0.xx \
  --ecg_k_start 20 --ecg_k_end 20 \
  --ecg_schedule linear
```

### TSV-based sweep

```bash
# Edit sweeps/your_config.tsv, then dispatch via Slurm
sbatch --array=0-N scripts/cegs_array.sbatch
```

## Evaluation

```bash
python tools/eval_checkpoints.py \
  --run_dir /path/to/run \
  --attacks fgsm,pgd_linf,pgd_linf_rs \
  --adv_eps 8 --adv_steps 20 \
  --c_corruptions gaussian_noise,shot_noise,... \
  --c_severity 5
```

All adversarial attacks are run in pixel space `[0, 1]` with proper clamping. The evaluation tool also supports checkpoint directories, Slurm job lookup, and optional corruption-suite evaluation.

## Citation

If you use this repository, please cite the associated paper once it is publicly available.
