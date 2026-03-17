# Predictive Alerting Demo

This project implements a small end-to-end prototype of a predictive alerting system for multivariate time-series data.

## Problem formulation

Given the previous `W` time steps of one or more metrics, the model predicts whether an incident will occur within the next `H` time steps.

This is implemented as a sliding-window binary classification problem:
- input: metrics from `[t-W+1, ..., t]`
- target: `1` if an incident is active at any point in `[t+1, ..., t+H]`, else `0`

## Dataset

The project uses a synthetic cloud-like dataset with four metrics:
- CPU usage
- P95 latency
- Error rate
- Request count

The generator includes:
- daily seasonality
- noise
- slow drift
- heavy-tailed spikes
- injected incident intervals

During incidents, latency and error rate rise, CPU typically increases, and traffic can shift.

## Model

Main model:
- GRU-based binary classifier

Loss:
- weighted binary cross-entropy (`BCEWithLogitsLoss` with `pos_weight`)

## Evaluation

### Window-level metrics
- Precision
- Recall
- F1
- PR-AUC
- ROC-AUC

### Incident-level alerting metrics
Predicted probabilities are converted to alerts using a threshold selected on the validation set.

An incident is considered detected if there is at least one alert in the `H` steps before or at the start of that incident.

Reported:
- Incident recall
- Lead time (mean / median)
- False alerts count
- Alerts per 1000 steps

## Run

```bash
python train_and_evaluate.py