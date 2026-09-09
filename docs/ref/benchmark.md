---
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Generated from benchmark results. Do not edit by hand.
{}
---

# Benchmarks

BioNeMo Inference Runtime (BioIR) measured against two OSS PyTorch baselines on
every GPU it is tested on. These are the numbers a run of the published
procedure should reproduce; they are not peak performance, and no attempt was
made to tune any configuration per GPU.

Both baselines are reported for every measurement, and the pair is the point.
Against OSS PyTorch eager, BioIR leads on every datacentre part here — 1.19x at
worst — but not on GB10, where Protenix is 0.90x and OpenFold3 0.99x. Against
OSS torch.compile the answer depends on the architecture: 1.5-3.0x on Ampere,
Hopper and Ada, 1.0-2.1x on Blackwell, and 0.8-1.7x on GB10. Either baseline
quoted alone overstates the result in one direction or understates it in the
other, so figures below 1.00x are printed rather than dropped.

`torch.compile` is ahead on 21% of individual measurements, and where it is
ahead is not random: never below 256 residues, and rising with length to around
29% above 769 — and only ever on Boltz-2 and OpenFold3, never on OpenFold2 or
Protenix. Long inputs on Blackwell are the case to check before quoting a single
figure.

GB10 is a desktop part and reads like one: 128 GB of LPDDR5x at 273 GB/s shared
between CPU and GPU, an order of magnitude under HBM. It runs the same spec
10-13x slower than an H100, and it is the only device here where BioIR's kernels
do not pay for themselves against eager on every model. It is reported because
it is a supported target, not as a comparison against the datacentre parts above
it.

![Speedup against input size on H100](../assets/speedup-vs-residues.png)

Per-sample speedup against input size, all four models on H100. The dashed line
is parity; crosses mark the largest input measured. The curves are far from flat
— high on the smallest inputs, where launch and dispatch overhead dominates and
BioIR has less of it, lowest in the few-hundred-residue range, then climbing
with length as the fused kernels start to matter. That shape is why every table
below is broken out by residue count, and why an overall geomean alone is not
enough to quote.

## Contents

- [Measurements](#measurements)
- [Boltz-2](#boltz-2)
- [OpenFold3](#openfold3)
- [OpenFold2 / AlphaFold2](#openfold2--alphafold2)
- [Protenix](#protenix)
- [Reproducing](#reproducing)

## Measurements

### Hardware

| Label          | Device                                            | SM  | Memory  | Power limit | Driver     |
| -------------- | ------------------------------------------------- | --- | ------- | ----------- | ---------- |
| `h100`         | NVIDIA H100 80GB HBM3                             | 90  | 80 GiB  | 700 W       | 580.105.08 |
| `h200`         | NVIDIA H200                                       | 90  | 140 GiB | 700 W       | 595.58.03  |
| `a100`         | NVIDIA A100-SXM4-80GB                             | 80  | 80 GiB  | 400 W       | 595.58.03  |
| `b200`         | NVIDIA B200                                       | 100 | 179 GiB | 1000 W      | 595.58.03  |
| `b300`         | NVIDIA B300 SXM6 AC                               | 103 | 269 GiB | 1100 W      | 595.58.03  |
| `gb200`        | NVIDIA GB200                                      | 100 | 185 GiB | 1200 W      | 595.58.03  |
| `gb300`        | NVIDIA GB300                                      | 103 | 278 GiB | 1400 W      | 595.58.03  |
| `l40s`         | NVIDIA L40S                                       | 89  | 45 GiB  | 350 W       | 595.58.03  |
| `l40`          | NVIDIA L40                                        | 89  | 45 GiB  | 300 W       | 595.58.03  |
| `l4`           | NVIDIA L4                                         | 89  | 22 GiB  | 72 W        | 595.58.03  |
| `rtx6000ada`   | NVIDIA RTX 6000 Ada Generation                    | 89  | 48 GiB  | 300 W       | 595.58.03  |
| `rtxpro6000sv` | NVIDIA RTX PRO 6000 Blackwell Server Edition      | 120 | 96 GiB  | 600 W       | 595.58.03  |
| `rtxpro6000ws` | NVIDIA RTX PRO 6000 Blackwell Workstation Edition | 120 | 96 GiB  | 600 W       | 595.58.03  |
| `gb10`         | NVIDIA GB10                                       | 121 | —       | —           | 595.58.03  |

One GPU per measurement, one process, no replicas. Absolute latencies move with
clocks, power limit and thermal headroom; the ratios are what should be
comparable.

### What Is Compared

Three backends fold the same inputs on the same GPU in the same container:

| Name              | What it is                                                    |
| ----------------- | ------------------------------------------------------------- |
| BioIR             | This runtime: fused kernels, `torch.compile` where it applies |
| OSS torch.compile | The reference implementation under `torch.compile`            |
| OSS PyTorch eager | The reference implementation, unmodified                      |

The OSS side runs its own inference script: eager always, and `torch.compile`
additionally when it passes a dynamic-shape probe. Protenix has no compile path,
which is why its compile column is empty.

### Units

Speedup is `OSS forward / BioIR forward` per sample; above 1 favours BioIR.
Every figure is a **geometric mean of per-sample speedups**, not a ratio of
totals, so each sample counts equally regardless of how long it runs. `n` is the
sample count and differs by model.

Speedups are broken out by input size in residues because the spread is wide --
Boltz-2 on H200 is 1.80x below 512 residues and 3.61x above 1024 against OSS
torch.compile. The overall geomean alone hides a factor of two.

Peak memory is `torch.cuda.max_memory_allocated` over the run. Accuracy is
OpenStructure lDDT over every sample and DockQ over the subset with a supported
protein interface.

## Boltz-2

| GPU            | vs OSS torch.compile | vs OSS PyTorch eager | &lt;512 res   | 512-1024 res  | &gt;1024 res  | n   |
| -------------- | -------------------- | -------------------- | ------------- | ------------- | ------------- | --- |
| `h100`         | 1.78x                | 2.65x                | 1.70x / 2.32x | 1.76x / 2.79x | 1.88x / 2.93x | 17  |
| `h200`         | 1.74x                | 2.54x                | 1.70x / 2.29x | 1.69x / 2.63x | 1.87x / 2.76x | 17  |
| `a100`         | 3.00x                | 2.80x                | 1.99x / 2.63x | 3.41x / 2.82x | 4.23x / 3.01x | 17  |
| `b200`         | 1.42x                | 1.46x                | 1.39x / 1.71x | 1.16x / 1.33x | 1.84x / 1.36x | 17  |
| `b300`         | 1.58x                | 1.62x                | 1.83x / 2.15x | 1.19x / 1.39x | 1.89x / 1.38x | 17  |
| `gb200`        | 1.13x                | 1.68x                | 1.75x / 2.47x | 0.90x / 1.36x | 0.87x / 1.36x | 17  |
| `gb300`        | 1.17x                | 1.71x                | 1.77x / 2.56x | 0.93x / 1.39x | 0.92x / 1.37x | 17  |
| `l40s`         | 1.88x                | 2.78x                | 1.77x / 2.39x | 1.84x / 2.85x | 2.04x / 3.23x | 17  |
| `l40`          | 2.66x                | 2.80x                | 2.01x / 2.76x | 2.22x / 2.71x | 4.62x / 2.96x | 17  |
| `l4`           | 1.71x                | 2.63x                | 1.75x / 2.24x | 1.71x / 2.93x | 1.61x / 3.04x | 14  |
| `rtx6000ada`   | 1.74x                | 2.46x                | 1.81x / 2.27x | 1.64x / 2.40x | 1.81x / 2.78x | 17  |
| `rtxpro6000sv` | 1.03x                | 1.50x                | 1.34x / 1.76x | 0.88x / 1.29x | 0.90x / 1.48x | 17  |
| `rtxpro6000ws` | 1.06x                | 1.54x                | 1.33x / 1.90x | 0.93x / 1.30x | 0.95x / 1.46x | 17  |
| `gb10`         | 1.65x                | 1.54x                | 1.21x / 1.43x | 1.88x / 1.68x | 2.04x / 1.53x | 17  |

Bucket cells are `OSS torch.compile / OSS PyTorch eager`. Below 1.00x means the
baseline was faster.

### Forward Latency

| GPU            | BioIR   | OSS torch.compile | OSS PyTorch eager |
| -------------- | ------- | ----------------- | ----------------- |
| `h100`         | 8.49 s  | 15.08 s           | 22.53 s           |
| `h200`         | 7.68 s  | 13.39 s           | 19.50 s           |
| `a100`         | 14.89 s | 44.70 s           | 41.75 s           |
| `b200`         | 9.83 s  | 13.95 s           | 14.39 s           |
| `b300`         | 10.06 s | 15.92 s           | 16.28 s           |
| `gb200`        | 9.83 s  | 11.09 s           | 16.49 s           |
| `gb300`        | 9.59 s  | 11.17 s           | 16.44 s           |
| `l40s`         | 16.82 s | 31.54 s           | 46.73 s           |
| `l40`          | 18.64 s | 49.52 s           | 52.17 s           |
| `l4`           | 48.24 s | 56.40 s           | 86.41 s           |
| `rtx6000ada`   | 17.13 s | 29.88 s           | 42.07 s           |
| `rtxpro6000sv` | 15.42 s | 15.84 s           | 23.12 s           |
| `rtxpro6000ws` | 14.13 s | 14.98 s           | 21.74 s           |
| `gb10`         | 85.78 s | 141.27 s          | 132.30 s          |

Geomean of the per-sample forward time, over the samples the ratios above are
taken over. Quote these when comparing two releases: a ratio that moved does not
say which side did.

### Peak Allocated Memory

| GPU            | BioIR     | OSS torch.compile | OSS PyTorch eager | BioIR / compile |
| -------------- | --------- | ----------------- | ----------------- | --------------- |
| `h100`         | 20.34 GiB | 32.27 GiB         | 32.27 GiB         | 0.63x           |
| `h200`         | 20.34 GiB | 32.27 GiB         | 32.27 GiB         | 0.63x           |
| `a100`         | 20.30 GiB | 32.25 GiB         | 32.25 GiB         | 0.63x           |
| `b200`         | 19.59 GiB | 32.27 GiB         | 32.27 GiB         | 0.61x           |
| `b300`         | 19.59 GiB | 32.27 GiB         | 32.27 GiB         | 0.61x           |
| `gb200`        | 19.59 GiB | 32.27 GiB         | 32.27 GiB         | 0.61x           |
| `gb300`        | 19.59 GiB | 32.27 GiB         | 32.27 GiB         | 0.61x           |
| `l40s`         | 20.30 GiB | 32.25 GiB         | 32.25 GiB         | 0.63x           |
| `l40`          | 20.30 GiB | 32.25 GiB         | 32.25 GiB         | 0.63x           |
| `l4`           | 15.30 GiB | 16.44 GiB         | 16.44 GiB         | 0.93x           |
| `rtx6000ada`   | 20.30 GiB | 32.25 GiB         | 32.25 GiB         | 0.63x           |
| `rtxpro6000sv` | 19.59 GiB | 32.27 GiB         | 32.27 GiB         | 0.61x           |
| `rtxpro6000ws` | 19.59 GiB | 32.27 GiB         | 32.27 GiB         | 0.61x           |
| `gb10`         | 19.59 GiB | 32.27 GiB         | 32.27 GiB         | 0.61x           |

Below 1.00x means BioIR allocates less than the baseline; above 1.00x means it
allocates more.

### Accuracy

| GPU            | lDDT                  | DockQ                 | n lDDT | n DockQ |
| -------------- | --------------------- | --------------------- | ------ | ------- |
| `h100`         | 0.667 / 0.665 / 0.664 | 0.611 / 0.606 / 0.601 | 17     | 9       |
| `h200`         | 0.667 / 0.664 / 0.663 | 0.611 / 0.609 / 0.617 | 17     | 9       |
| `a100`         | 0.662 / 0.660 / 0.663 | 0.613 / 0.608 / 0.610 | 17     | 9       |
| `b200`         | 0.663 / 0.661 / 0.663 | 0.613 / 0.611 / 0.610 | 17     | 9       |
| `b300`         | 0.669 / 0.660 / 0.664 | 0.606 / 0.601 / 0.605 | 17     | 9       |
| `gb200`        | 0.665 / 0.665 / 0.662 | 0.610 / 0.615 / 0.607 | 17     | 9       |
| `gb300`        | 0.669 / 0.667 / 0.665 | 0.608 / 0.614 / 0.611 | 17     | 9       |
| `l40s`         | 0.664 / 0.667 / 0.666 | 0.603 / 0.612 / 0.615 | 17     | 9       |
| `l40`          | 0.666 / 0.666 / 0.664 | 0.613 / 0.607 / 0.611 | 17     | 9       |
| `l4`           | 0.668 / 0.638 / 0.638 | 0.608 / 0.620 / 0.625 | 17     | 9       |
| `rtx6000ada`   | 0.663 / 0.665 / 0.663 | 0.613 / 0.616 / 0.608 | 17     | 9       |
| `rtxpro6000sv` | 0.668 / 0.661 / 0.666 | 0.610 / 0.603 / 0.610 | 17     | 9       |
| `rtxpro6000ws` | 0.667 / 0.661 / 0.668 | 0.609 / 0.609 / 0.619 | 17     | 9       |
| `gb10`         | 0.666 / 0.662 / 0.667 | 0.604 / 0.604 / 0.611 | 17     | 9       |

Each cell is `BioIR / OSS torch.compile / OSS PyTorch eager`. lDDT is scored
over every sample; DockQ covers the subset with a supported protein interface,
which is why its count is lower. The three should agree -- a speedup that moved
them would not be the same answer.

## OpenFold3

| GPU            | vs OSS torch.compile | vs OSS PyTorch eager | &lt;512 res   | 512-1024 res  | &gt;1024 res  | n   |
| -------------- | -------------------- | -------------------- | ------------- | ------------- | ------------- | --- |
| `h100`         | 1.55x                | 2.02x                | 2.01x / 2.38x | 1.25x / 1.76x | 1.48x / 1.96x | 17  |
| `h200`         | 1.54x                | 2.03x                | 2.01x / 2.37x | 1.24x / 1.78x | 1.46x / 1.97x | 17  |
| `a100`         | 1.56x                | 2.02x                | 1.98x / 2.52x | 1.24x / 1.66x | 1.54x / 1.95x | 17  |
| `b200`         | 1.18x                | 1.47x                | 1.82x / 2.10x | 0.92x / 1.21x | 0.94x / 1.22x | 17  |
| `b300`         | 1.06x                | 1.40x                | 1.57x / 1.88x | 0.84x / 1.18x | 0.86x / 1.21x | 17  |
| `gb200`        | 1.38x                | 1.67x                | 2.58x / 2.82x | 1.08x / 1.29x | 0.88x / 1.21x | 17  |
| `gb300`        | 1.33x                | 1.64x                | 2.46x / 2.73x | 1.04x / 1.28x | 0.86x / 1.21x | 17  |
| `l40s`         | 1.63x                | 1.90x                | 2.03x / 2.44x | 1.39x / 1.62x | 1.51x / 1.70x | 17  |
| `l40`          | 1.68x                | 2.01x                | 2.12x / 2.73x | 1.43x / 1.67x | 1.53x / 1.73x | 17  |
| `l4`           | 1.47x                | 1.85x                | 1.47x / 1.85x | — / —         | — / —         | 6   |
| `rtx6000ada`   | 1.63x                | 1.96x                | 1.95x / 2.37x | 1.41x / 1.72x | 1.54x / 1.82x | 17  |
| `rtxpro6000sv` | 1.07x                | 1.29x                | 1.64x / 1.92x | 0.83x / 1.03x | 0.86x / 1.04x | 17  |
| `rtxpro6000ws` | 1.06x                | 1.31x                | 1.55x / 1.98x | 0.84x / 1.03x | 0.90x / 1.05x | 17  |
| `gb10`         | 0.82x                | 0.99x                | 0.75x / 1.01x | 0.82x / 0.97x | 0.89x / 1.00x | 17  |

Bucket cells are `OSS torch.compile / OSS PyTorch eager`. Below 1.00x means the
baseline was faster.

### Forward Latency

| GPU            | BioIR    | OSS torch.compile | OSS PyTorch eager |
| -------------- | -------- | ----------------- | ----------------- |
| `h100`         | 11.12 s  | 17.26 s           | 22.46 s           |
| `h200`         | 10.23 s  | 15.78 s           | 20.76 s           |
| `a100`         | 18.93 s  | 29.49 s           | 38.21 s           |
| `b200`         | 11.87 s  | 14.00 s           | 17.49 s           |
| `b300`         | 12.14 s  | 12.83 s           | 17.00 s           |
| `gb200`        | 11.59 s  | 16.03 s           | 19.34 s           |
| `gb300`        | 11.64 s  | 15.51 s           | 19.13 s           |
| `l40s`         | 21.55 s  | 35.13 s           | 40.91 s           |
| `l40`          | 23.21 s  | 38.95 s           | 46.59 s           |
| `l4`           | 41.81 s  | 15.51 s           | 19.61 s           |
| `rtx6000ada`   | 20.75 s  | 33.73 s           | 40.68 s           |
| `rtxpro6000sv` | 17.95 s  | 19.16 s           | 23.07 s           |
| `rtxpro6000ws` | 16.48 s  | 17.53 s           | 21.56 s           |
| `gb10`         | 106.42 s | 86.81 s           | 105.61 s          |

Geomean of the per-sample forward time, over the samples the ratios above are
taken over. Quote these when comparing two releases: a ratio that moved does not
say which side did.

### Peak Allocated Memory

| GPU            | BioIR     | OSS torch.compile | OSS PyTorch eager | BioIR / compile |
| -------------- | --------- | ----------------- | ----------------- | --------------- |
| `h100`         | 22.14 GiB | 29.49 GiB         | 29.49 GiB         | 0.75x           |
| `h200`         | 22.14 GiB | 29.49 GiB         | 29.49 GiB         | 0.75x           |
| `a100`         | 22.10 GiB | 29.47 GiB         | 29.47 GiB         | 0.75x           |
| `b200`         | 21.39 GiB | 29.49 GiB         | 29.49 GiB         | 0.73x           |
| `b300`         | 21.39 GiB | 29.49 GiB         | 29.49 GiB         | 0.73x           |
| `gb200`        | 21.39 GiB | 29.49 GiB         | 29.49 GiB         | 0.73x           |
| `gb300`        | 21.39 GiB | 29.49 GiB         | 29.49 GiB         | 0.73x           |
| `l40s`         | 22.10 GiB | 29.47 GiB         | 29.47 GiB         | 0.75x           |
| `l40`          | 22.10 GiB | 29.47 GiB         | 29.47 GiB         | 0.75x           |
| `l4`           | 11.53 GiB | 8.40 GiB          | 8.40 GiB          | 1.37x           |
| `rtx6000ada`   | 22.10 GiB | 29.47 GiB         | 29.47 GiB         | 0.75x           |
| `rtxpro6000sv` | 21.39 GiB | 29.49 GiB         | 29.49 GiB         | 0.73x           |
| `rtxpro6000ws` | 21.39 GiB | 29.49 GiB         | 29.49 GiB         | 0.73x           |
| `gb10`         | 21.39 GiB | 29.49 GiB         | 29.49 GiB         | 0.73x           |

Below 1.00x means BioIR allocates less than the baseline; above 1.00x means it
allocates more.

### Accuracy

| GPU            | lDDT                  | DockQ         | n lDDT | n DockQ |
| -------------- | --------------------- | ------------- | ------ | ------- |
| `h100`         | 0.617 / 0.616 / 0.619 | 0.533 / — / — | 17     | 9       |
| `h200`         | 0.617 / 0.617 / 0.612 | 0.533 / — / — | 17     | 9       |
| `a100`         | 0.616 / 0.616 / 0.618 | 0.533 / — / — | 17     | 9       |
| `b200`         | 0.618 / 0.616 / 0.614 | 0.535 / — / — | 17     | 9       |
| `b300`         | 0.618 / 0.619 / 0.617 | 0.532 / — / — | 17     | 9       |
| `gb200`        | 0.613 / 0.606 / 0.613 | 0.531 / — / — | 17     | 9       |
| `gb300`        | 0.614 / 0.619 / 0.618 | 0.530 / — / — | 17     | 9       |
| `l40s`         | 0.617 / 0.617 / 0.615 | 0.544 / — / — | 17     | 9       |
| `l40`          | 0.617 / 0.622 / 0.618 | 0.535 / — / — | 17     | 9       |
| `l4`           | 0.575 / 0.407 / 0.407 | 0.529 / — / — | 14     | 7       |
| `rtx6000ada`   | 0.614 / 0.621 / 0.607 | 0.533 / — / — | 17     | 9       |
| `rtxpro6000sv` | 0.618 / 0.617 / 0.618 | 0.538 / — / — | 17     | 9       |
| `rtxpro6000ws` | 0.617 / 0.620 / 0.618 | 0.533 / — / — | 17     | 9       |
| `gb10`         | 0.611 / 0.620 / 0.619 | 0.538 / — / — | 17     | 9       |

Each cell is `BioIR / OSS torch.compile / OSS PyTorch eager`. lDDT is scored
over every sample; DockQ covers the subset with a supported protein interface,
which is why its count is lower. The three should agree -- a speedup that moved
them would not be the same answer.

## OpenFold2 / AlphaFold2

### Monomer

| GPU            | vs OSS torch.compile | vs OSS PyTorch eager | &lt;512 res   | 512-1024 res  | &gt;1024 res  | n   |
| -------------- | -------------------- | -------------------- | ------------- | ------------- | ------------- | --- |
| `h100`         | 2.55x                | 2.60x                | 2.17x / 2.11x | 2.67x / 2.83x | 3.23x / 3.35x | 15  |
| `h200`         | 2.61x                | 2.66x                | 2.24x / 2.16x | 2.73x / 2.90x | 3.25x / 3.37x | 15  |
| `a100`         | 2.58x                | 2.71x                | 2.15x / 2.30x | 2.70x / 2.84x | 3.35x / 3.46x | 15  |
| `b200`         | 1.82x                | 1.90x                | 1.73x / 1.72x | 1.84x / 2.00x | 1.99x / 2.07x | 15  |
| `b300`         | 1.83x                | 1.88x                | 1.74x / 1.69x | 1.84x / 1.99x | 2.04x / 2.05x | 15  |
| `gb200`        | 1.94x                | 2.03x                | 1.77x / 1.79x | 2.02x / 2.19x | 2.16x / 2.26x | 15  |
| `gb300`        | 2.05x                | 2.03x                | 1.91x / 1.79x | 2.13x / 2.18x | 2.21x / 2.25x | 15  |
| `l40s`         | 1.78x                | 1.77x                | 1.66x / 1.57x | 1.72x / 1.78x | 2.22x / 2.27x | 15  |
| `l40`          | 1.92x                | 1.95x                | 1.82x / 1.81x | 1.86x / 1.92x | 2.29x / 2.33x | 15  |
| `l4`           | 1.57x                | 1.69x                | 1.27x / 1.42x | 1.94x / 2.01x | — / —         | 12  |
| `rtx6000ada`   | 2.01x                | 2.08x                | 1.90x / 1.94x | 1.98x / 2.08x | 2.30x / 2.36x | 15  |
| `rtxpro6000sv` | 1.72x                | 1.71x                | 1.73x / 1.62x | 1.63x / 1.69x | 1.88x / 1.92x | 15  |
| `rtxpro6000ws` | 1.72x                | 1.77x                | 1.72x / 1.75x | 1.65x / 1.72x | 1.87x / 1.91x | 15  |
| `gb10`         | 1.27x                | 1.33x                | 1.13x / 1.25x | 1.55x / 1.55x | 1.09x / 1.11x | 15  |

Bucket cells are `OSS torch.compile / OSS PyTorch eager`. Below 1.00x means the
baseline was faster.

#### Forward Latency

| GPU            | BioIR   | OSS torch.compile | OSS PyTorch eager |
| -------------- | ------- | ----------------- | ----------------- |
| `h100`         | 3.05 s  | 7.78 s            | 7.93 s            |
| `h200`         | 2.89 s  | 7.56 s            | 7.69 s            |
| `a100`         | 5.33 s  | 13.74 s           | 14.46 s           |
| `b200`         | 3.51 s  | 6.40 s            | 6.67 s            |
| `b300`         | 3.38 s  | 6.20 s            | 6.34 s            |
| `gb200`        | 3.66 s  | 7.12 s            | 7.44 s            |
| `gb300`        | 3.70 s  | 7.59 s            | 7.49 s            |
| `l40s`         | 6.59 s  | 11.74 s           | 11.69 s           |
| `l40`          | 7.38 s  | 14.20 s           | 14.42 s           |
| `l4`           | 13.23 s | 34.05 s           | 36.30 s           |
| `rtx6000ada`   | 6.43 s  | 12.91 s           | 13.34 s           |
| `rtxpro6000sv` | 4.42 s  | 7.60 s            | 7.54 s            |
| `rtxpro6000ws` | 4.12 s  | 7.08 s            | 7.29 s            |
| `gb10`         | 29.71 s | 37.83 s           | 39.47 s           |

Geomean of the per-sample forward time, over the samples the ratios above are
taken over. Quote these when comparing two releases: a ratio that moved does not
say which side did.

#### Peak Allocated Memory

| GPU            | BioIR     | OSS torch.compile | OSS PyTorch eager | BioIR / compile |
| -------------- | --------- | ----------------- | ----------------- | --------------- |
| `h100`         | 29.69 GiB | 15.27 GiB         | 15.27 GiB         | 1.94x           |
| `h200`         | 29.69 GiB | 15.27 GiB         | 15.27 GiB         | 1.94x           |
| `a100`         | 29.67 GiB | 15.25 GiB         | 15.25 GiB         | 1.95x           |
| `b200`         | 18.77 GiB | 15.27 GiB         | 15.27 GiB         | 1.23x           |
| `b300`         | 18.77 GiB | 15.27 GiB         | 15.27 GiB         | 1.23x           |
| `gb200`        | 18.77 GiB | 15.27 GiB         | 15.27 GiB         | 1.23x           |
| `gb300`        | 18.77 GiB | 15.27 GiB         | 15.27 GiB         | 1.23x           |
| `l40s`         | 29.67 GiB | 15.25 GiB         | 15.25 GiB         | 1.95x           |
| `l40`          | 29.67 GiB | 13.00 GiB         | 13.00 GiB         | 2.28x           |
| `l4`           | 16.17 GiB | 13.00 GiB         | 13.00 GiB         | 1.24x           |
| `rtx6000ada`   | 29.67 GiB | 15.25 GiB         | 15.25 GiB         | 1.95x           |
| `rtxpro6000sv` | 23.56 GiB | 15.27 GiB         | 15.27 GiB         | 1.54x           |
| `rtxpro6000ws` | 23.56 GiB | 15.27 GiB         | 15.27 GiB         | 1.54x           |
| `gb10`         | 23.56 GiB | 15.27 GiB         | 15.27 GiB         | 1.54x           |

Below 1.00x means BioIR allocates less than the baseline; above 1.00x means it
allocates more.

#### Accuracy

| GPU            | lDDT                  | DockQ     | n lDDT | n DockQ |
| -------------- | --------------------- | --------- | ------ | ------- |
| `h100`         | 0.434 / 0.426 / 0.432 | — / — / — | 15     | —       |
| `h200`         | 0.434 / 0.421 / 0.426 | — / — / — | 15     | —       |
| `a100`         | 0.434 / 0.422 / 0.422 | — / — / — | 15     | —       |
| `b200`         | 0.434 / 0.429 / 0.433 | — / — / — | 15     | —       |
| `b300`         | 0.434 / 0.427 / 0.426 | — / — / — | 15     | —       |
| `gb200`        | 0.433 / 0.432 / 0.429 | — / — / — | 15     | —       |
| `gb300`        | 0.433 / 0.428 / 0.423 | — / — / — | 15     | —       |
| `l40s`         | 0.432 / 0.431 / 0.426 | — / — / — | 15     | —       |
| `l40`          | 0.434 / 0.419 / 0.433 | — / — / — | 15     | —       |
| `l4`           | 0.481 / 0.424 / 0.417 | — / — / — | 15     | —       |
| `rtx6000ada`   | 0.434 / 0.420 / 0.418 | — / — / — | 15     | —       |
| `rtxpro6000sv` | 0.435 / 0.429 / 0.418 | — / — / — | 15     | —       |
| `rtxpro6000ws` | 0.435 / 0.426 / 0.433 | — / — / — | 15     | —       |
| `gb10`         | 0.434 / 0.421 / 0.424 | — / — / — | 15     | —       |

Each cell is `BioIR / OSS torch.compile / OSS PyTorch eager`. lDDT is scored
over every sample; DockQ covers the subset with a supported protein interface,
which is why its count is lower. The three should agree -- a speedup that moved
them would not be the same answer.

### Multimer

| GPU            | vs OSS torch.compile | vs OSS PyTorch eager | &lt;512 res | 512-1024 res  | &gt;1024 res  | n   |
| -------------- | -------------------- | -------------------- | ----------- | ------------- | ------------- | --- |
| `h100`         | 2.66x                | 2.77x                | — / —       | 2.40x / 2.54x | 2.94x / 3.03x | 4   |
| `h200`         | 2.61x                | 2.75x                | — / —       | 2.35x / 2.51x | 2.90x / 3.00x | 4   |
| `a100`         | 2.64x                | 2.74x                | — / —       | 2.41x / 2.53x | 2.89x / 2.96x | 4   |
| `b200`         | 1.80x                | 1.90x                | — / —       | 1.68x / 1.82x | 1.92x / 2.00x | 4   |
| `b300`         | 1.79x                | 1.89x                | — / —       | 1.68x / 1.80x | 1.92x / 1.99x | 4   |
| `gb200`        | 1.80x                | 1.91x                | — / —       | 1.68x / 1.81x | 1.93x / 2.01x | 4   |
| `gb300`        | 1.80x                | 1.90x                | — / —       | 1.68x / 1.81x | 1.92x / 1.99x | 4   |
| `l40s`         | 2.13x                | 2.18x                | — / —       | 1.86x / 1.92x | 2.44x / 2.47x | 4   |
| `l40`          | 1.99x                | 2.04x                | — / —       | 1.79x / 1.86x | 2.20x / 2.24x | 4   |
| `l4`           | —                    | —                    | — / —       | — / —         | — / —         | —   |
| `rtx6000ada`   | 1.96x                | 2.03x                | — / —       | 1.78x / 1.87x | 2.16x / 2.21x | 4   |
| `rtxpro6000sv` | 1.83x                | 1.87x                | — / —       | 1.59x / 1.64x | 2.10x / 2.14x | 4   |
| `rtxpro6000ws` | 1.76x                | 1.81x                | — / —       | 1.55x / 1.62x | 2.00x / 2.03x | 4   |
| `gb10`         | 1.38x                | 1.40x                | — / —       | 1.61x / 1.66x | 1.18x / 1.19x | 4   |

Bucket cells are `OSS torch.compile / OSS PyTorch eager`. Below 1.00x means the
baseline was faster.

#### Forward Latency

| GPU            | BioIR    | OSS torch.compile | OSS PyTorch eager |
| -------------- | -------- | ----------------- | ----------------- |
| `h100`         | 8.15 s   | 21.64 s           | 22.59 s           |
| `h200`         | 7.69 s   | 20.10 s           | 21.12 s           |
| `a100`         | 14.66 s  | 38.68 s           | 40.11 s           |
| `b200`         | 9.35 s   | 16.78 s           | 17.80 s           |
| `b300`         | 9.08 s   | 16.29 s           | 17.16 s           |
| `gb200`        | 8.73 s   | 15.74 s           | 16.66 s           |
| `gb300`        | 8.69 s   | 15.62 s           | 16.50 s           |
| `l40s`         | 19.23 s  | 40.99 s           | 41.91 s           |
| `l40`          | 22.90 s  | 45.55 s           | 46.69 s           |
| `l4`           | —        | 137.80 s          | 141.20 s          |
| `rtx6000ada`   | 21.33 s  | 41.90 s           | 43.35 s           |
| `rtxpro6000sv` | 12.18 s  | 22.29 s           | 22.83 s           |
| `rtxpro6000ws` | 12.34 s  | 21.71 s           | 22.33 s           |
| `gb10`         | 104.29 s | 143.64 s          | 146.38 s          |

Geomean of the per-sample forward time, over the samples the ratios above are
taken over. Quote these when comparing two releases: a ratio that moved does not
say which side did.

#### Peak Allocated Memory

| GPU            | BioIR     | OSS torch.compile | OSS PyTorch eager | BioIR / compile |
| -------------- | --------- | ----------------- | ----------------- | --------------- |
| `h100`         | 18.01 GiB | 14.04 GiB         | 14.04 GiB         | 1.28x           |
| `h200`         | 18.01 GiB | 14.04 GiB         | 14.04 GiB         | 1.28x           |
| `a100`         | 17.99 GiB | 14.02 GiB         | 14.02 GiB         | 1.28x           |
| `b200`         | 16.31 GiB | 14.04 GiB         | 14.04 GiB         | 1.16x           |
| `b300`         | 16.31 GiB | 14.04 GiB         | 14.04 GiB         | 1.16x           |
| `gb200`        | 16.31 GiB | 14.04 GiB         | 14.04 GiB         | 1.16x           |
| `gb300`        | 16.31 GiB | 14.04 GiB         | 14.04 GiB         | 1.16x           |
| `l40s`         | 17.99 GiB | 14.02 GiB         | 14.02 GiB         | 1.28x           |
| `l40`          | 17.99 GiB | 11.84 GiB         | 11.84 GiB         | 1.52x           |
| `l4`           | —         | 11.84 GiB         | 11.84 GiB         | —               |
| `rtx6000ada`   | 17.99 GiB | 14.02 GiB         | 14.02 GiB         | 1.28x           |
| `rtxpro6000sv` | 21.43 GiB | 14.04 GiB         | 14.04 GiB         | 1.53x           |
| `rtxpro6000ws` | 21.43 GiB | 14.04 GiB         | 14.04 GiB         | 1.53x           |
| `gb10`         | 21.43 GiB | 14.04 GiB         | 14.04 GiB         | 1.53x           |

Below 1.00x means BioIR allocates less than the baseline; above 1.00x means it
allocates more.

#### Accuracy

| GPU            | lDDT                  | DockQ                 | n lDDT | n DockQ |
| -------------- | --------------------- | --------------------- | ------ | ------- |
| `h100`         | 0.784 / 0.736 / 0.766 | 0.603 / 0.601 / 0.610 | 4      | 4       |
| `h200`         | 0.784 / 0.762 / 0.761 | 0.603 / 0.591 / 0.606 | 4      | 4       |
| `a100`         | 0.777 / 0.750 / 0.777 | 0.597 / 0.617 / 0.620 | 4      | 4       |
| `b200`         | 0.783 / 0.766 / 0.774 | 0.607 / 0.610 / 0.606 | 4      | 4       |
| `b300`         | 0.783 / 0.756 / 0.779 | 0.607 / 0.609 / 0.620 | 4      | 4       |
| `gb200`        | 0.783 / 0.768 / 0.762 | 0.607 / 0.600 / 0.610 | 4      | 4       |
| `gb300`        | 0.783 / 0.768 / 0.774 | 0.607 / 0.635 / 0.621 | 4      | 4       |
| `l40s`         | 0.780 / 0.783 / 0.766 | 0.609 / 0.613 / 0.600 | 4      | 4       |
| `l40`          | 0.782 / 0.754 / 0.767 | 0.602 / 0.608 / 0.602 | 4      | 4       |
| `l4`           | — / 0.750 / 0.759     | — / 0.597 / 0.594     | 4      | 4       |
| `rtx6000ada`   | 0.782 / 0.769 / 0.766 | 0.602 / 0.622 / 0.608 | 4      | 4       |
| `rtxpro6000sv` | 0.776 / 0.746 / 0.762 | 0.601 / 0.594 / 0.608 | 4      | 4       |
| `rtxpro6000ws` | 0.776 / 0.758 / 0.759 | 0.601 / 0.595 / 0.603 | 4      | 4       |
| `gb10`         | 0.780 / 0.756 / 0.773 | 0.602 / 0.613 / 0.625 | 4      | 4       |

Each cell is `BioIR / OSS torch.compile / OSS PyTorch eager`. lDDT is scored
over every sample; DockQ covers the subset with a supported protein interface,
which is why its count is lower. The three should agree -- a speedup that moved
them would not be the same answer.

## Protenix

| GPU            | vs OSS torch.compile | vs OSS PyTorch eager | &lt;512 res | 512-1024 res | &gt;1024 res | n   |
| -------------- | -------------------- | -------------------- | ----------- | ------------ | ------------ | --- |
| `h100`         | —                    | 1.87x                | — / 2.06x   | — / 1.66x    | — / 1.92x    | 17  |
| `h200`         | —                    | 1.84x                | — / 2.03x   | — / 1.66x    | — / 1.83x    | 17  |
| `a100`         | —                    | 1.88x                | — / 2.21x   | — / 1.62x    | — / 1.85x    | 17  |
| `b200`         | —                    | 1.27x                | — / 1.85x   | — / 1.10x    | — / 0.96x    | 17  |
| `b300`         | —                    | 1.21x                | — / 1.59x   | — / 1.12x    | — / 0.96x    | 17  |
| `gb200`        | —                    | 1.49x                | — / 2.62x   | — / 1.20x    | — / 0.97x    | 17  |
| `gb300`        | —                    | 1.48x                | — / 2.57x   | — / 1.20x    | — / 0.98x    | 17  |
| `l40s`         | —                    | 1.76x                | — / 1.98x   | — / 1.64x    | — / 1.66x    | 17  |
| `l40`          | —                    | 1.71x                | — / 2.16x   | — / 1.48x    | — / 1.51x    | 17  |
| `l4`           | —                    | 1.53x                | — / 1.56x   | — / 1.48x    | — / 1.58x    | 14  |
| `rtx6000ada`   | —                    | 1.68x                | — / 2.26x   | — / 1.42x    | — / 1.45x    | 17  |
| `rtxpro6000sv` | —                    | 1.19x                | — / 1.61x   | — / 1.01x    | — / 1.00x    | 17  |
| `rtxpro6000ws` | —                    | 1.21x                | — / 1.68x   | — / 1.00x    | — / 1.01x    | 17  |
| `gb10`         | —                    | 0.90x                | — / 0.99x   | — / 0.83x    | — / 0.89x    | 17  |

Bucket cells are `OSS torch.compile / OSS PyTorch eager`. Below 1.00x means the
baseline was faster.

### Forward Latency

| GPU            | BioIR    | OSS torch.compile | OSS PyTorch eager |
| -------------- | -------- | ----------------- | ----------------- |
| `h100`         | 13.05 s  | —                 | 24.39 s           |
| `h200`         | 12.18 s  | —                 | 22.37 s           |
| `a100`         | 23.09 s  | —                 | 43.40 s           |
| `b200`         | 13.98 s  | —                 | 17.75 s           |
| `b300`         | 14.26 s  | —                 | 17.27 s           |
| `gb200`        | 13.73 s  | —                 | 20.43 s           |
| `gb300`        | 13.61 s  | —                 | 20.12 s           |
| `l40s`         | 25.89 s  | —                 | 45.56 s           |
| `l40`          | 31.08 s  | —                 | 53.00 s           |
| `l4`           | 62.01 s  | —                 | 82.79 s           |
| `rtx6000ada`   | 30.37 s  | —                 | 51.15 s           |
| `rtxpro6000sv` | 21.13 s  | —                 | 25.14 s           |
| `rtxpro6000ws` | 19.67 s  | —                 | 23.71 s           |
| `gb10`         | 154.76 s | —                 | 139.19 s          |

Geomean of the per-sample forward time, over the samples the ratios above are
taken over. Quote these when comparing two releases: a ratio that moved does not
say which side did.

### Peak Allocated Memory

| GPU            | BioIR     | OSS torch.compile | OSS PyTorch eager | BioIR / compile |
| -------------- | --------- | ----------------- | ----------------- | --------------- |
| `h100`         | 27.57 GiB | —                 | 35.16 GiB         | —               |
| `h200`         | 27.57 GiB | —                 | 35.16 GiB         | —               |
| `a100`         | 27.55 GiB | —                 | 35.14 GiB         | —               |
| `b200`         | 33.19 GiB | —                 | 35.16 GiB         | —               |
| `b300`         | 33.19 GiB | —                 | 35.16 GiB         | —               |
| `gb200`        | 33.19 GiB | —                 | 35.16 GiB         | —               |
| `gb300`        | 33.19 GiB | —                 | 35.16 GiB         | —               |
| `l40s`         | 27.55 GiB | —                 | 35.14 GiB         | —               |
| `l40`          | 27.55 GiB | —                 | 35.14 GiB         | —               |
| `l4`           | 17.42 GiB | —                 | 17.78 GiB         | —               |
| `rtx6000ada`   | 27.55 GiB | —                 | 35.14 GiB         | —               |
| `rtxpro6000sv` | 33.19 GiB | —                 | 35.16 GiB         | —               |
| `rtxpro6000ws` | 33.19 GiB | —                 | 35.16 GiB         | —               |
| `gb10`         | 33.19 GiB | —                 | 35.16 GiB         | —               |

Below 1.00x means BioIR allocates less than the baseline; above 1.00x means it
allocates more.

### Accuracy

| GPU            | lDDT              | DockQ             | n lDDT | n DockQ |
| -------------- | ----------------- | ----------------- | ------ | ------- |
| `h100`         | 0.658 / — / 0.650 | 0.591 / — / 0.545 | 17     | 9       |
| `h200`         | 0.657 / — / 0.650 | 0.591 / — / 0.568 | 17     | 9       |
| `a100`         | 0.659 / — / 0.649 | 0.599 / — / 0.541 | 17     | 9       |
| `b200`         | 0.659 / — / 0.648 | 0.590 / — / 0.562 | 17     | 9       |
| `b300`         | 0.658 / — / 0.654 | 0.589 / — / 0.549 | 17     | 9       |
| `gb200`        | 0.659 / — / 0.650 | 0.588 / — / 0.550 | 17     | 9       |
| `gb300`        | 0.659 / — / 0.653 | 0.589 / — / 0.555 | 17     | 9       |
| `l40s`         | 0.660 / — / 0.652 | 0.600 / — / 0.552 | 17     | 9       |
| `l40`          | 0.657 / — / 0.648 | 0.589 / — / 0.548 | 17     | 9       |
| `l4`           | 0.635 / — / 0.609 | 0.581 / — / 0.515 | 15     | 8       |
| `rtx6000ada`   | 0.658 / — / 0.650 | 0.601 / — / 0.554 | 17     | 9       |
| `rtxpro6000sv` | 0.658 / — / 0.649 | 0.598 / — / 0.548 | 17     | 9       |
| `rtxpro6000ws` | 0.658 / — / 0.646 | 0.588 / — / 0.542 | 17     | 9       |
| `gb10`         | 0.658 / — / 0.652 | 0.601 / — / 0.552 | 17     | 9       |

Each cell is `BioIR / OSS torch.compile / OSS PyTorch eager`. lDDT is scored
over every sample; DockQ covers the subset with a supported protein interface,
which is why its count is lower. The three should agree -- a speedup that moved
them would not be the same answer.

## Reproducing

These numbers are not tied to a build artifact: they come from `main`, on one
GPU, with public inputs and upstream baselines at pinned tags. The procedure is
the [`bench-perf-oss`](../../.agents/skills/bench-perf-oss/SKILL.md) agent skill
-- point an agent at it rather than following steps by hand. It installs BioIR
and each pinned baseline in isolation, runs one warmup and one timed forward per
sample, scores the output, and aggregates the geomeans above.

Have these ready first; the skill cannot supply them:

- **One NVIDIA GPU on driver 580 or newer.** The container ships a CUDA 13.2
  build of PyTorch. On a 535 driver it runs only through the
  forward-compatibility shim, and measuring on that path produced silent hangs
  and `SIGFPE` crashes part-way through a sweep -- wrong numbers, not merely
  slow ones. Check the ECC counters are zero while you are there.
- **`NGC_API_KEY` exported**, for the MSA Search NIM. `rebuild_dataset.py`
  builds the `dataset-0.3.0` inputs from RCSB and that endpoint rather than
  downloading a release; the structures need no credential at all
  (`--structures-only`). Free key at
  [build.nvidia.com](https://build.nvidia.com/).
- **`HF_TOKEN` exported.** OpenFold3's checkpoint is gated and an anonymous
  fetch returns `401`. Registering is free. Checkpoint staging generally is
  [Model Weights](model-weights.md).
- **A built extension**, per [Development Workflow](../dev.md).

When a ratio does not match, suspect the baseline environment before the code.
The most common cause by far is an upstream package resolved against a different
`torch` than BioIR's, at which point the ratio compares two PyTorch builds
rather than two implementations.

---

Container `nvcr.io/nvidia/pytorch:26.05-py3` · driver 580.105.08–610.57.04 ·
measured 2026-09-08
