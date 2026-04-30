# Beta-Binomial Bayesian Inference Simulator

An interactive Streamlit app for exploring Bayesian inference with the Beta-Binomial conjugate model.

https://beta-binomial-simulator.streamlit.app

## What it does

Set prior beliefs and observed data via sliders — the app updates the posterior in real time and shows:

- **Prior & Posterior** — Beta distribution plots with means and 95% credible intervals
- **Predictive Distribution** — Beta-Binomial vs plain Binomial side-by-side, illustrating overdispersion from parameter uncertainty
- **Algorithm Summary** — step-by-step walkthrough of the Bayesian update with live values

## The model

```
Prior:     θ ~ Beta(α, β)
Likelihood: x | θ, N ~ Binomial(N, θ)
Posterior: θ | x, N ~ Beta(α + x, β + N − x)   ← closed-form, no MCMC needed
```

The posterior predictive samples θ from the posterior then draws from Binomial(N_new, θ), giving a Beta-Binomial distribution that correctly propagates uncertainty about θ.

## Run locally

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv run streamlit run app.py
```

## Project files

| File | Description |
|------|-------------|
| `app.py` | Streamlit app |
| `BetaBinomial.ipynb` | Jupiter Notebook with exercise |
| `model.stan` | Stan model (Beta prior + Binomial likelihood + posterior predictive) |
| `pyproject.toml` | Project dependencies (uv) |
| `requirements.txt` | Runtime dependencies for Streamlit Cloud |
