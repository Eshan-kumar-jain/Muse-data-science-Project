# Machine Learning with MUSE Astro Data — HH399

**Self-supervised region mapping and zero-shot spectral classification of the irradiated Herbig–Haro jet HH399.**

This repository applies machine learning to a VLT/MUSE integral-field spectroscopy cube of **HH399**, an irradiated Herbig–Haro jet in the Trifid Nebula (M20). The central question: can learned embeddings, built **without labels**, recover the same physical structures — the jet, the diffuse background, and point sources — that astronomers normally identify by hand through line-ratio diagnostics?

The strongest result of the project is not any single accuracy score. It is that **two architecturally unrelated self-supervised models independently converge on the same jet/diffuse boundary** predicted by classical line-ratio diagnostics — stronger evidence that the boundary is real physical structure than any one method could give alone.

> MSc Data Science & Analytics thesis · Department of Computer Science, Maynooth University
> Author: Eshan Kumar Jain (25252963) · Supervisor: Prof. Rozenn Dahyot · Collaborator: Prof. Emma Whelan

---

## Table of contents

- [Overview](#overview)
- [The object and the data](#the-object-and-the-data)
- [The three models](#the-three-models)
- [Key results](#key-results)
- [Repository structure](#repository-structure)
- [Environment and setup](#environment-and-setup)
- [Running the pipeline](#running-the-pipeline)
- [Evaluation philosophy](#evaluation-philosophy)
- [Limitations](#limitations)
- [Future work](#future-work)
- [References](#references)
- [Acknowledgements](#acknowledgements)
- [GenAI use](#genai-use)

---

## Overview

HH399 is unusual among Herbig–Haro objects. Most sit in quiet regions where their emission comes almost entirely from the jet's own internal shocks. HH399 instead sits inside the Trifid Nebula, close to the region's O7.5 ionising star, so its line ratios are a blend of **shock excitation** and **external photoionisation** rather than shock physics alone. That blended signal is exactly what makes an unsupervised, physically-grounded approach worthwhile.

The project has four stages, all built on a physically-grounded exploratory baseline:

1. **Physical baseline** — continuum-subtracted narrow-band flux maps and line-ratio diagnostics (Hα, [N II], [S II], [O III]) for every spaxel, giving an independent, physics-based check no learned model is allowed to see during training.
2. **Model 1** — a self-supervised, CLIP-style spectral–spatial contrastive network whose clustering recovers the jet ridge without any labels.
3. **Model 2** — a supervised spectrum–text CLIP model enabling zero-shot classification against natural-language prompts.
4. **Model 3** — an AION-1-inspired shared transformer that fuses the MUSE cube with real FEROS spectra via masked-wavelength-bin reconstruction.

## The object and the data

The project combines two spectroscopic datasets that look nothing alike, plus a labelled external benchmark.

**VLT/MUSE integral-field cube** — the primary target. A full optical spectrum at every spatial pixel across the field, taken during the instrument's 2014 science-verification run (ESO programme 60.A-9322(A)). Roughly `3681 × 317 × 318` (wavelength × spatial × spatial), spanning ~4600 Å at medium resolution in air wavelengths, with **96,093** valid spaxels after masking. No labels exist for any spaxel — this is the whole premise.

**FEROS archival echelle spectra** — nine real exposures of the stars and nebular gas irradiating HH399: HD164492B/C, HD164514, and the surrounding NGC 6514 H II region. No spatial structure, but ~189,626 wavelength points per spectrum (roughly 50× the cube's sampling). These anchor the project to the specific irradiating system and serve as the real-world zero-shot / classification target for Models 2 and 3. They have known stellar/nebular identities used only for evaluation, not for training.

**SDSS spectra** — an external labelled benchmark for Model 2: 90 real spectra (30 each of STAR / GALAXY / QSO), de-redshifted to rest frame, used to validate the zero-shot idea before applying it to FEROS.

The ~50× resolution mismatch and the absence of any shared spatial axis mean the cube and FEROS **cannot** simply be forced onto one wavelength grid without destroying information. That mismatch is precisely why Model 3 exists.

## The three models

**Model 1 — Self-supervised spectral–spatial CLIP.** Two encoders — a 1-D CNN over each spaxel's spectrum and a 2-D CNN over the multi-channel image patch around it — trained jointly with a symmetric InfoNCE loss (the CLIP objective). No labels: the only constraint is that a spaxel's spectrum and its own spatial neighbourhood should be more similar to each other than to a randomly paired spectrum and patch. KMeans on the learned embeddings then paints the region map.

**Model 2 — Supervised zero-shot spectrum–text CLIP.** A second CLIP-style pair, this time a spectrum encoder aligned against a text encoder (a TF-IDF vectoriser feeding a small MLP) over short class-description sentences. Classification becomes a nearest-prompt-embedding lookup, so new classes can be named at test time without retraining a classifier head.

**Model 3 — AION-1-inspired shared transformer.** A different family entirely: no contrastive pairing. Per-modality tokenizers reduce each instrument's native spectrum to a fixed 48 bins, feeding one small shared transformer backbone. Because the transformer only ever sees learned token embeddings — never raw wavelengths — bin 10 of 48 can mean a different stretch of Ångströms in each instrument without breaking anything. The backbone trains self-supervised via **masked-wavelength-bin reconstruction** (the BERT / masked-autoencoder idea applied to spectra), then the same backbone is fine-tuned on the 9 FEROS spectra with its own tokenizer.

## Key results

| Stage | Metric | Result |
|---|---|---|
| Physical baseline | Hα S/N ≥ 3 coverage | 84.5% (96,093 spaxels); median [N II]/Hα = 0.22; [S II] 6716/6731 = 1.20 (low density) |
| Model 1 | Synthetic ARI / real-cube retrieval | 1.000 / ≈ chance (spectrum→patch 0.013 vs 0.016 chance) |
| Model 1 | K=3 region sizes | 80,303 diffuse / 12,974 jet / 1,470 point sources |
| Model 2 | SDSS accuracy | 0.889 (STAR & QSO perfect; 2 GALAXY→STAR, a physically sensible confusion) |
| Model 2 | FEROS accuracy | 0.000 — **inverted, not random** (clean separation, backwards labels) |
| Model 3 | Synthetic ARI / reconstruction MSE | 0.997 / 0.00598 → 0.00001 |
| Model 3 | K=3 region sizes | 81,606 diffuse / 13,341 jet / 1,146 point sources |
| Model 3 | FEROS accuracy | 0.889 (8/9) leave-one-out nearest-centroid |

Two takeaways carry the project:

The **convergence result** — Models 1 and 3 share no architecture, loss function, or training objective, yet both recover the same jet/diffuse boundary that the line-ratio baseline identifies independently. This is the headline evidence that the boundary is real.

The **diagnosable failure** — Model 2's inverted 0.000 on FEROS traced specifically to an untrained-vocabulary text encoder (FEROS-only words like "HII", "hot", "young" never received a gradient), not to a broken spectral representation. Model 3 has no text tower to fail that way, and its 8/9 on the identical 9 spectra is a direct, traceable answer to that failure rather than an unrelated fix.

## Repository structure

```
Muse-data-science-Project/
├── eda_line_ratios.py                  # Standalone EDA / line-ratio module
├── eda_line_ratios.ipynb               # Notebook version (cell-split, with narrative)
├── model1_spectral_spatial_clip.ipynb  # Model 1: spectral–spatial CLIP + InfoNCE
├── model2_spectrum_text_clip.ipynb     # Model 2: spectrum–text CLIP, zero-shot
├── model3_aion_transformer.ipynb       # Model 3: tokenizers + shared masked transformer
├── build_model3_notebook.py            # Builder script for the Model 3 notebook
├── eda_spaxels.csv                     # Per-spaxel table (96,093 rows) produced by the EDA
└── README.md
```

> Adjust the paths above to match your actual repository layout if it has changed.

## Environment and setup

Everything runs in Python inside a dedicated conda environment (`ml_env`), kept separate from system Python so the astronomy stack (which pins fairly specific `numpy`/`astropy` versions) does not collide with anything else.

```bash
# Create and activate the environment
conda create -n ml_env python=3.10
conda activate ml_env

# Core astronomy stack
pip install mpdaf astropy astroquery

# Modelling, clustering, data wrangling
pip install torch scikit-learn pandas numpy scipy matplotlib
```

Key libraries and why they are used:

- **mpdaf** — the MUSE instrument team's own package for reading the cube (DATA/STAT extensions, WCS, air-wavelength convention). Preferred over hand-rolled FITS parsing for a ~2.9 GB cube.
- **astropy** — underlying FITS I/O, units, and coordinate handling.
- **astroquery** — programmatic access to the SDSS spectra (Model 2) and FEROS/cube provenance. Note: `SDSS.get_spectra` needs the `run2d` column included in the query or it silently returns zero spectra.
- **PyTorch** — all three models, chosen for its explicit training loop.
- **scikit-learn** — KMeans (region maps) and ARI / confusion-matrix / classification-report utilities.
- **pandas** — the per-spaxel tables and results tables.

### Data access

The MUSE cube is from ESO programme **60.A-9322(A)** (2014 science-verification run) and is not redistributed here — obtain it from the [ESO Science Archive](https://archive.eso.org/). SDSS spectra are pulled at runtime via `astroquery`. FEROS exposures are archival ESO data for HD164492B/C, HD164514, and NGC 6514.

## Running the pipeline

Run the stages in order — each later stage checks itself against the baseline produced by the first.

```bash
# 1. Build the physical baseline and per-spaxel table (produces eda_spaxels.csv)
jupyter nbconvert --to notebook --execute eda_line_ratios.ipynb

# 2. Model 1 — self-supervised region mapping
jupyter nbconvert --to notebook --execute model1_spectral_spatial_clip.ipynb

# 3. Model 2 — zero-shot spectral classification
jupyter nbconvert --to notebook --execute model2_spectrum_text_clip.ipynb

# 4. Model 3 — shared masked-reconstruction transformer
jupyter nbconvert --to notebook --execute model3_aion_transformer.ipynb
```

> **Note on live Jupyter state.** If a notebook is open in the browser, Jupyter's autosave can overwrite on-disk edits with its in-memory copy. Edit notebooks either on disk **or** in the browser, not both at once.

## Evaluation philosophy

All three models share the same problem: nobody has labelled the HH399 cube spaxel-by-spaxel, so there is no real accuracy number for the region-mapping side. Following the approach AstroCLIP uses in the same situation, each model is checked against (a) a **synthetic dataset** with planted, known structure to confirm the pipeline is wired correctly, and (b) **qualitative agreement** with the line-ratio baseline. Only Model 2's SDSS classes and Model 3's FEROS labels give a genuine accuracy figure. Throughout, weak and inverted results are reported alongside the strong ones rather than hidden.

## Limitations

- **Model 1 retrieval sits at chance** on the real cube, even though the clustering is physically sensible — the embedding space separates coarse region types but does not encode fine per-spaxel identity.
- **Model 2's FEROS classification is inverted** (0.000), traced to an untrained-vocabulary text encoder rather than a broken spectral representation.
- **Model 3's jet cluster shows an unexplained negative Hα dip** that Model 1's equivalent cluster does not share — leading (unverified) guess is a bad column or sky-subtraction residual specific to that clustering split.
- **Single-object case study** — every result is built and validated on one cube and nine FEROS exposures; nothing here demonstrates that the pipeline generalises to a different jet or instrument without retraining.

## Future work

- Replace Model 2's TF-IDF + MLP text encoder with a pretrained sentence-transformer and re-run the FEROS test.
- Investigate Model 1's retrieval gap with harder in-batch negative sampling or a larger batch size.
- Chase down Model 3's Hα anomaly by re-running K=3 with a different random seed to test whether it is a per-spaxel artefact.
- Generalise the physical-baseline-plus-two-self-supervised-models pipeline to a second MUSE-observed HH jet.
- Scale Model 3 toward AION-1 proper by training the shared backbone across multiple objects.

## References

1. Radford, A. et al. (2021) *Learning Transferable Visual Models From Natural Language Supervision.* arXiv:2103.00020. https://arxiv.org/abs/2103.00020
2. Parker, L. et al. (2024) *AstroCLIP: a cross-modal foundation model for galaxies.* MNRAS, 531, 4990–5011. doi:10.1093/mnras/stae1450. https://academic.oup.com/mnras/article/531/4/4990/7697182
3. Parker, L. et al. (2025) *AION-1: Omnimodal Foundation Model for Astronomical Sciences.* arXiv:2510.17960. https://arxiv.org/abs/2510.17960
4. Murphy, G.C. et al. (2021) *A MUSE spectro-imaging study of the Th 28 jet: Precession in the inner jet?* A&A, 652, A119. doi:10.1051/0004-6361/202141315. https://www.aanda.org/articles/aa/full_html/2021/08/aa41315-21/aa41315-21.html
5. Baldwin, J.A., Phillips, M.M. & Terlevich, R. (1981) *Classification parameters for the emission-line spectra of extragalactic objects.* PASP, 93, 5–19. doi:10.1086/130766. https://iopscience.iop.org/article/10.1086/130766
6. Piqueras, L. et al. (2019) *MPDAF: A Python package for the analysis of VLT/MUSE data.* ADASS XXVII, ASP Conf. Ser. 521, 545. https://ui.adsabs.harvard.edu/abs/2019ASPC..521..545P
7. Yusef-Zadeh, F., Biretta, J. & Wardle, M. (2005) *Proper Motion of the Irradiated Jet HH 399 in the Trifid Nebula.* ApJ, 624, 246–253. doi:10.1086/428706. https://iopscience.iop.org/article/10.1086/428706

## Acknowledgements

With thanks to Prof. Rozenn Dahyot (supervisor) — whose suggestion to add the AION-1-inspired third model strengthened the core findings — and Prof. Emma Whelan (collaborator) for expertise in young stellar objects and jets. This project is part of the wider *Machine Learning with MUSE Astro Data* collaboration at Maynooth University.

## GenAI use

In accordance with Maynooth University's Student Working Guidelines for GenAI (v1.3), generative AI (Claude, via Claude Code) was used as a coding assistant for debugging, chapter structuring, and explanations of library functions and astronomical concepts. All scientific analysis, interpretation, and critical evaluation are the author's own work; all outputs were reviewed and verified.

---

*This README summarises an academic thesis project. Figures and full analysis live in the notebooks listed above.*
