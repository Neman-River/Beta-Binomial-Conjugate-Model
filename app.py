import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist, binom, betabinom

st.set_page_config(page_title="Beta-Binomial Simulator", layout="wide")
st.title("Beta-Binomial Bayesian Inference Simulator")

# --- Sidebar controls ---
st.sidebar.header("Model Parameters")

alpha_prior = st.sidebar.slider("α (prior)", min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                                 help="Prior successes. α=1 with β=1 → uniform prior.")
beta_prior = st.sidebar.slider("β (prior)", min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                                help="Prior failures. Higher β → prior belief toward lower θ.")

st.sidebar.markdown("---")
N = st.sidebar.slider("N (observed trials)", min_value=1, max_value=200, value=50)
x = st.sidebar.slider("x (observed successes)", min_value=0, max_value=N, value=20)

st.sidebar.markdown("---")
N_new = st.sidebar.slider("N_new (future trials to predict)", min_value=1, max_value=200, value=50)

# --- Derived quantities ---
alpha_post = alpha_prior + x
beta_post = beta_prior + (N - x)

prior_mean = alpha_prior / (alpha_prior + beta_prior)
post_mean = alpha_post / (alpha_post + beta_post)

prior_ci = beta_dist.ppf([0.025, 0.975], alpha_prior, beta_prior)
post_ci = beta_dist.ppf([0.025, 0.975], alpha_post, beta_post)

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Prior & Posterior", "Predictive Distribution", "Algorithm Summary"])

# ======== Tab 1: Prior & Posterior ========
with tab1:
    st.subheader("Prior vs Posterior Distribution of θ")

    theta_range = np.linspace(0.001, 0.999, 500)
    prior_pdf = beta_dist.pdf(theta_range, alpha_prior, beta_prior)
    post_pdf = beta_dist.pdf(theta_range, alpha_post, beta_post)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(theta_range, prior_pdf, color="steelblue", linewidth=2,
            label=f"Prior: Beta(α={alpha_prior:.1f}, β={beta_prior:.1f})")
    ax.fill_between(theta_range, prior_pdf, alpha=0.2, color="steelblue")
    ax.plot(theta_range, post_pdf, color="darkorange", linewidth=2,
            label=f"Posterior: Beta(α={alpha_post:.1f}, β={beta_post:.1f})")
    ax.fill_between(theta_range, post_pdf, alpha=0.2, color="darkorange")
    ax.axvline(prior_mean, color="steelblue", linestyle="--", alpha=0.7, label=f"Prior mean = {prior_mean:.3f}")
    ax.axvline(post_mean, color="darkorange", linestyle="--", alpha=0.7, label=f"Posterior mean = {post_mean:.3f}")
    ax.set_xlabel("θ (response rate)")
    ax.set_ylabel("Density")
    ax.set_title("Bayesian Update: Prior → Posterior")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Prior mean", f"{prior_mean:.3f}")
        st.metric("Prior 95% CI", f"[{prior_ci[0]:.3f}, {prior_ci[1]:.3f}]")
        st.metric("Prior CI width", f"{prior_ci[1] - prior_ci[0]:.3f}")
    with col2:
        st.metric("Posterior mean", f"{post_mean:.3f}")
        st.metric("Posterior 95% CI", f"[{post_ci[0]:.3f}, {post_ci[1]:.3f}]")
        st.metric("Posterior CI width", f"{post_ci[1] - post_ci[0]:.3f}")

    st.info(
        f"After observing **{x} successes in {N} trials**, the posterior mean shifted from "
        f"**{prior_mean:.3f}** to **{post_mean:.3f}**, and the 95% CI narrowed from "
        f"**{prior_ci[1]-prior_ci[0]:.3f}** to **{post_ci[1]-post_ci[0]:.3f}** wide."
    )

# ======== Tab 2: Predictive Distribution ========
with tab2:
    st.subheader(f"Predictive Distribution for {N_new} New Trials")

    SIZE = 100_000
    theta_samples = beta_dist.rvs(alpha_post, beta_post, size=SIZE)
    bb_samples = binom.rvs(N_new, theta_samples, size=SIZE)  # Beta-Binomial via composition
    binom_samples = binom.rvs(N_new, post_mean, size=SIZE)   # Plain Binomial at fixed θ

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    axes[0].hist(bb_samples, bins=range(0, N_new + 2), density=True,
                 color="mediumseagreen", alpha=0.7)
    axes[0].set_title("Beta-Binomial\n(accounts for θ uncertainty)")
    axes[0].set_xlabel(f"Successes in {N_new} trials")
    axes[0].set_ylabel("Probability")

    axes[1].hist(binom_samples, bins=range(0, N_new + 2), density=True,
                 color="royalblue", alpha=0.7)
    axes[1].set_title(f"Binomial(θ = posterior mean = {post_mean:.3f})\n(ignores θ uncertainty)")
    axes[1].set_xlabel(f"Successes in {N_new} trials")

    fig.suptitle("Beta-Binomial has higher variance due to uncertainty about θ", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Beta-Binomial mean", f"{np.mean(bb_samples):.2f}")
        st.metric("Beta-Binomial std", f"{np.std(bb_samples):.2f}")
        st.metric("Beta-Binomial variance", f"{np.var(bb_samples):.2f}")
    with col2:
        st.metric("Binomial mean", f"{np.mean(binom_samples):.2f}")
        st.metric("Binomial std", f"{np.std(binom_samples):.2f}")
        st.metric("Binomial variance", f"{np.var(binom_samples):.2f}")

    excess = np.var(bb_samples) - np.var(binom_samples)
    st.warning(
        f"Beta-Binomial variance exceeds Binomial variance by **{excess:.2f}**. "
        f"This overdispersion comes from not knowing θ exactly — "
        f"use Beta-Binomial when making predictions with an uncertain parameter."
    )

# ======== Tab 3: Algorithm Summary ========
with tab3:
    st.subheader("Bayesian Inference Algorithm — Beta-Binomial Model")

    # Highlight current step based on slider state
    has_data = N > 0

    st.markdown("""
### Step-by-step workflow

**Step 1 — Define Prior: Beta(α, β)**
> Encodes beliefs about θ before seeing data.
> - `Beta(1, 1)` = uniform = complete uncertainty
> - `α > β` → prior belief that θ is high
> - `α + β` = "effective prior sample size" (larger → prior dominates more)
""")

    st.markdown(f"""
**Step 2 — Collect Data**
> Observe `x` successes in `N` trials.
> _Current values: x = {x}, N = {N}_
""")

    st.markdown(f"""
**Step 3 — Update (Bayes' Rule via conjugacy)**
> ```
> posterior = Beta(α + x,  β + (N − x))
>           = Beta({alpha_prior:.1f} + {x},  {beta_prior:.1f} + ({N} − {x}))
>           = Beta({alpha_post:.1f}, {beta_post:.1f})
> ```
> Each success increments α; each failure increments β.
> This is a **closed-form** update — no MCMC required.
""")

    st.markdown(f"""
**Step 4 — Summarize Posterior**
> - Mean = {post_mean:.4f}
> - 95% CI = [{post_ci[0]:.4f}, {post_ci[1]:.4f}]
""")

    st.markdown(f"""
**Step 5 — Predict New Data (Posterior Predictive)**
> - Draw θ* ~ Beta({alpha_post:.1f}, {beta_post:.1f}), 

> - Then y_new ~ Binomial({N_new}, θ*)

> - Result: Beta-Binomial({N_new}, {alpha_post:.1f}, {beta_post:.1f})

> ⚠ More spread than Binomial({N_new}, {post_mean:.3f}) because θ is uncertain.
""")

    st.markdown("""
**Step 6 — (Optional) Stan / MCMC**
> Use Stan when no conjugate prior exists.
> - Write `data`, `parameters`, `model`, `generated quantities` blocks
> - Run ≥ 4 chains; verify R̂ ≈ 1.0 for convergence
> - Extract samples with `fit.draws_pd()`

---

### Key concepts

| Concept | Insight |
|---|---|
| Conjugate prior | Beta × Binomial → Beta; exact closed-form posterior |
| Credible interval | Direct probability: "θ ∈ [a, b] with 95% probability" |
| Posterior predictive | Propagates both sampling variability **and** parameter uncertainty |
| Overdispersion | Beta-Binomial variance > Binomial variance when θ is unknown |
| Stan | General MCMC engine; same structure scales to complex models |
""")
