---
description: Deep-ensemble (BNN) surrogate — a neural-network ensemble alternative to the GP surrogates, for larger or non-stationary datasets. Ships the `deep_ensemble` surrogate.
category: modifier
domain: both
---

## Overview

This bundle contributes a `deep_ensemble` surrogate: an ensemble of small neural
networks, each trained on a bootstrap resample, whose disagreement provides
epistemic uncertainty. It is a non-GP alternative for situations where the GP
surrogates underperform — and is *beyond* BoTorch's built-in toolkit, shipped as
a skill component rather than core.

## Surrogate

Select `surrogate: "deep_ensemble"` when the dataset is large enough that GP
cubic scaling hurts, or the response is strongly non-stationary / has sharp
feature interactions that a stationary kernel (even `dkl`) fits poorly, and you
do not need calibrated tail uncertainty. Prefer the GP surrogates
(`single_task` / `dkl` / `saas`) when data is scarce: with few points the
ensemble is under-trained and its variance is unreliable, which can mislead the
acquisition. It is single-objective.

## Acquisition

The ensemble exposes a Gaussian posterior (ensemble mean and variance), so the
standard acquisitions all apply — `log_ei` as the balanced default, `ucb` for
tunable exploration. Note the uncertainty is the ensemble's empirical spread,
not a calibrated GP posterior: be a little more exploratory (slightly higher
`ucb` beta) early, when the members have seen little data.

## Diagnostics

The deep ensemble has no LOO-CV panel; judge it by the trend and the acquisition
landscape. If the predicted optimum keeps jumping between unrelated regions, the
members disagree wildly (under-trained or too little data) — fall back to a GP
surrogate rather than trust the ensemble.
