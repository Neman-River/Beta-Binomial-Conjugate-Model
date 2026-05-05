# Beta-Binomial Bayesian Inference Simulator

An interactive Streamlit app for exploring Bayesian inference of a Binomial success rate θ using the Beta conjugate prior, plus an A/B testing case study notebook.

https://beta-binomial-simulator.streamlit.app

## What it does

The app walks through the full Bayesian workflow in four sections:

1. **Prior** — set your belief about θ via Beta(α, β) hyperparameters
2. **Data** — observe x successes in N trials; see the likelihood surface over θ
3. **Bayesian Update** — conjugate update yields a new Beta posterior, no MCMC needed
4. **Posterior Predictive** — marginalising over θ gives a Beta-Binomial distribution for future counts

## The model

```
Prior:     θ ~ Beta(α, β)
Likelihood: x | θ, N ~ Binomial(N, θ)
Posterior: θ | x ~ Beta(α + x, β + N − x)   ← closed-form conjugate update
Predictive: x̃ | x ~ Beta-Binomial(N_new, α + x, β + N − x)
```

**Conjugate update:**

$$\theta \mid x \sim \text{Beta}(\alpha + x,\; \beta + N - x)$$

The posterior mean is a weighted average of the prior mean and the MLE x/N. As N grows, the data dominates and the posterior concentrates around the observed frequency.

## App sections

| # | Section | Description |
|---|---------|-------------|
| 0 | Process Overview | Graphviz DAG of the full Bayesian workflow |
| 1 | Prior Distribution | PDF of Beta(α, β) with 95% credible interval |
| 2 | Observed Data | Likelihood of θ given x successes in N trials |
| 3 | Bayesian Update | Prior vs posterior overlay, weight breakdown |
| 4 | Posterior Predictive | Beta-Binomial vs Binomial — overdispersion from θ uncertainty |

## Sidebar controls

| Control | Description |
|---------|-------------|
| **α** | Prior pseudo-successes (α=1 with β=1 → uniform prior) |
| **β** | Prior pseudo-failures (higher β → prior belief toward lower θ) |
| **N** | Number of observed trials |
| **x** | Number of observed successes |
| **N_new** | Number of future trials for the predictive distribution |

## A/B testing case study

`AB-testing-betabinomial.ipynb` applies the Beta-Binomial model to a realistic A/B test:

**Scenario**: E-commerce checkout button — does Variant B convert more visitors?

- **Metric**: conversion rate θ ∈ [0, 1] (successes out of N visitors)
- **Prior**: Beta(α₀, β₀) over the conversion rate θ
- **Update**: closed-form conjugate posterior Beta(α₀ + x, β₀ + N − x)
- **Decision metric**: P(θ_B > θ_A) estimated via Monte Carlo
- **Posterior predictive**: Beta-Binomial PMF for each variant

## Run locally

**Prerequisites:** [uv](https://docs.astral.sh/uv/)

```bash
uv run streamlit run app.py
```

## Dependencies

Managed via `pyproject.toml` and `uv.lock`:

- `numpy`
- `scipy`
- `plotly`
- `streamlit`
- `matplotlib`
- `pandas`

## Project structure

```
.
├── app.py                           # Streamlit app
├── BetaBinomial.ipynb               # Concept notebook
├── AB-testing-betabinomial.ipynb    # A/B testing case study
├── pyproject.toml
└── uv.lock
```
