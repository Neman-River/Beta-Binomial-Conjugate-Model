import numpy as np
import streamlit as st
from scipy.stats import beta as beta_dist, binom
import plotly.graph_objects as go

st.set_page_config(page_title="Beta-Binomial Bayesian Inference Simulator", layout="wide")

# ── Global styles ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.step-banner {
    padding: 14px 22px;
    border-radius: 8px;
    margin: 56px 0 4px 0;
    color: white;
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}
.step-rule {
    border: none;
    border-top: 3px solid;
    margin: 0 0 20px 0;
    border-radius: 2px;
}
</style>
""", unsafe_allow_html=True)


def section(number: str, title: str, color: str) -> None:
    st.markdown(
        f'<div class="step-banner" style="background:{color};">'
        f'{number} &nbsp;·&nbsp; {title}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<hr class="step-rule" style="border-color:{color};">',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Prior Hyperparameters")
alpha_prior = st.sidebar.slider("α (prior successes)", min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                                 help="Prior successes. α=1 with β=1 → uniform prior.")
beta_prior = st.sidebar.slider("β (prior failures)", min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                                help="Prior failures. Higher β → prior belief toward lower θ.")

prior_mean = alpha_prior / (alpha_prior + beta_prior)
prior_ci = beta_dist.ppf([0.025, 0.975], alpha_prior, beta_prior)
st.sidebar.markdown(f"**E[θ]** = {prior_mean:.3f}")
st.sidebar.markdown(f"**95% CI**: [{prior_ci[0]:.3f}, {prior_ci[1]:.3f}]")

st.sidebar.divider()
st.sidebar.header("Observed Data")
N = st.sidebar.slider("N (observed trials)", min_value=1, max_value=200, value=50)
x = st.sidebar.slider("x (observed successes)", min_value=0, max_value=N, value=20)

st.sidebar.divider()
st.sidebar.header("Prediction")
N_new = st.sidebar.slider("N_new (future trials to predict)", min_value=1, max_value=200, value=50)

# ── Derived quantities ─────────────────────────────────────────────────────────
alpha_post = alpha_prior + x
beta_post = beta_prior + (N - x)

post_mean = alpha_post / (alpha_post + beta_post)
post_ci = beta_dist.ppf([0.025, 0.975], alpha_post, beta_post)

# ── Page title ────────────────────────────────────────────────────────────────
st.title("Beta-Binomial Conjugate Model")
st.markdown(
    "Bayesian inference for an unknown success probability θ using a Beta prior, "
    "when the data is a count of successes in N trials. "
    "Adjust the sliders on the left to explore how prior beliefs and observed data "
    "shape the posterior."
)

# ── Section 0 – Process overview ──────────────────────────────────────────────
section("0", "Process Overview", "#555E6B")
st.markdown(
    "The diagram below shows the full Bayesian workflow — from prior belief about θ, "
    "through observed Binomial data, to the posterior and predictive distribution."
)
st.graphviz_chart("""
digraph {
    rankdir=LR
    graph [bgcolor=transparent splines=ortho nodesep=0.6]
    node  [fontname="Helvetica" fontsize=13 style="filled,rounded" shape=box
           margin="0.25,0.15"]
    edge  [fontname="Helvetica" fontsize=11 color="#555555"]

    Prior [label="Prior\\nBeta(α, β)"
           fillcolor="#AED6F1" color="#2E86C1"]

    Theta [label="Latent rate  θ"
           fillcolor="#D5F5E3" color="#1E8449" shape=ellipse]

    Data [label="Observed counts\\nx successes in N trials"
          fillcolor="#E8DAEF" color="#7D3C98"]

    Likelihood [label="Likelihood\\nBinomial(N, θ)"
                fillcolor="#FDEBD0" color="#CA6F1E"]

    Update [label="Conjugate update\\nα* = α + x\\nβ* = β + (N − x)"
            fillcolor="#FDFEFE" color="#717D7E"]

    Posterior [label="Posterior\\nBeta(α*, β*)"
               fillcolor="#FADBD8" color="#C0392B"]

    Predictive [label="Posterior predictive\\nBeta-Binomial(N_new, α*, β*)"
                fillcolor="#FDEBD0" color="#D35400"]

    Prior    -> Theta      [label=" encodes belief\\n about θ"]
    Theta    -> Likelihood
    Likelihood -> Data     [label=" generates"]
    Prior    -> Update
    Data     -> Update
    Update   -> Posterior  [label=" yields"]
    Posterior -> Predictive [label=" marginalise\\n over θ"]
}
""")

# ── Section 1 – Prior ─────────────────────────────────────────────────────────
section("1", "Prior Distribution", "#2E86C1")
st.markdown("We encode our belief about the unknown success rate θ before seeing any data:")
st.latex(r"\theta \sim \text{Beta}(\alpha,\; \beta)")
st.markdown(
    "Beta(1, 1) is the uniform prior — complete uncertainty. "
    "Increasing α shifts belief toward higher θ; increasing β shifts it toward lower θ. "
    "The sum α + β acts as an *effective prior sample size*: the larger it is, the more the prior resists the data."
)

theta_range = np.linspace(0.001, 0.999, 500)
prior_pdf = beta_dist.pdf(theta_range, alpha_prior, beta_prior)

fig_prior = go.Figure()
fig_prior.add_trace(
    go.Scatter(
        x=theta_range, y=prior_pdf, mode="lines",
        line=dict(color="steelblue", width=2),
        fill="tozeroy", fillcolor="rgba(70,130,180,0.15)",
        name=f"Prior PDF  Beta({alpha_prior:.1f}, {beta_prior:.1f})",
    )
)
fig_prior.update_layout(
    xaxis_title="θ (success rate)", yaxis_title="Density",
    title=f"Beta({alpha_prior:.1f}, {beta_prior:.1f}) Prior",
    height=300, margin=dict(t=40, b=40),
)
st.plotly_chart(fig_prior, use_container_width=True)

c1, c2, c3 = st.columns(3)
c1.metric("Mean (E[θ])", f"{prior_mean:.3f}")
c2.metric("Eff. sample size (α+β)", f"{alpha_prior + beta_prior:.1f}")
c3.metric("95% CI", f"[{prior_ci[0]:.3f}, {prior_ci[1]:.3f}]")

# ── Section 2 – Observed Data ─────────────────────────────────────────────────
section("2", "Observed Data", "#7D3C98")
st.markdown(
    "We observe a **count of successes** x in N independent trials, "
    "each with the same unknown probability θ:"
)
st.latex(r"x \mid \theta,\, N \;\sim\; \text{Binomial}(N,\; \theta)")
st.markdown("The likelihood function is:")
st.latex(
    r"p(x \mid \theta) = \binom{N}{x}\,\theta^{x}\,(1-\theta)^{N-x}"
)

# Show the likelihood as a function of θ
likelihood = beta_dist.pdf(theta_range, x + 1, N - x + 1)  # proportional to binomial likelihood

fig_data = go.Figure()
fig_data.add_trace(
    go.Scatter(
        x=theta_range, y=likelihood, mode="lines",
        line=dict(color="mediumpurple", width=2),
        fill="tozeroy", fillcolor="rgba(147,112,219,0.15)",
        name=f"Likelihood  (x={x}, N={N})",
    )
)
fig_data.add_vline(
    x=x / N, line_dash="dot", line_color="mediumpurple",
    annotation_text=f"MLE = {x/N:.3f}",
    annotation_position="top right",
)
fig_data.update_layout(
    xaxis_title="θ", yaxis_title="Likelihood (unnormalised)",
    title=f"Likelihood of θ given x={x} successes in N={N} trials",
    height=280, margin=dict(t=40, b=40),
)
st.plotly_chart(fig_data, use_container_width=True)

d1, d2, d3, d4 = st.columns(4)
d1.metric("N (trials)", N)
d2.metric("x (successes)", x)
d3.metric("Failures (N − x)", N - x)
d4.metric("MLE of θ (x/N)", f"{x/N:.3f}")

# ── Section 3 – Bayesian Update ───────────────────────────────────────────────
section("3", "Bayesian Update", "#C0392B")
st.markdown(
    "The Beta prior is **conjugate** to the Binomial likelihood, "
    "so the posterior is also Beta — a closed-form update with no MCMC required:"
)
st.latex(r"\alpha^* = \alpha + x")
st.latex(r"\beta^* = \beta + (N - x)")
st.latex(r"\theta \mid x \sim \text{Beta}(\alpha^*,\; \beta^*)")
st.markdown(
    "Each success increments α; each failure increments β. "
    "As N grows, the posterior concentrates around the observed frequency x/N."
)

lo = min(beta_dist.ppf(0.0005, alpha_prior, beta_prior),
         beta_dist.ppf(0.0005, alpha_post, beta_post))
hi = max(beta_dist.ppf(0.9995, alpha_prior, beta_prior),
         beta_dist.ppf(0.9995, alpha_post, beta_post))
theta_grid = np.linspace(max(lo, 0.001), min(hi, 0.999), 500)

prior_pdf2 = beta_dist.pdf(theta_grid, alpha_prior, beta_prior)
post_pdf = beta_dist.pdf(theta_grid, alpha_post, beta_post)

fig_update = go.Figure()
fig_update.add_trace(
    go.Scatter(
        x=theta_grid, y=prior_pdf2, mode="lines",
        line=dict(color="steelblue", width=2, dash="dash"),
        name=f"Prior  Beta({alpha_prior:.1f}, {beta_prior:.1f})",
    )
)
fig_update.add_trace(
    go.Scatter(
        x=theta_grid, y=post_pdf, mode="lines",
        line=dict(color="crimson", width=2),
        fill="tozeroy", fillcolor="rgba(220,20,60,0.10)",
        name=f"Posterior  Beta({alpha_post:.1f}, {beta_post:.1f})",
    )
)
fig_update.add_vline(
    x=x / N, line_dash="dot", line_color="mediumpurple",
    annotation_text=f"x/N = {x/N:.3f}",
    annotation_position="top right",
)
fig_update.update_layout(
    xaxis_title="θ (success rate)", yaxis_title="Density",
    title="Prior vs Posterior for θ",
    height=340, margin=dict(t=40, b=40),
    legend=dict(x=0.02, y=0.95),
)
st.plotly_chart(fig_update, use_container_width=True)

p1, p2, p3 = st.columns(3)
p1.metric("Posterior mean (μ*)", f"{post_mean:.3f}")
p2.metric("Posterior 95% CI", f"[{post_ci[0]:.3f}, {post_ci[1]:.3f}]")
p3.metric("CI width", f"{post_ci[1] - post_ci[0]:.3f}")

st.markdown("---")
st.markdown("**How the posterior hyperparameters changed:**")
t1, t2 = st.columns(2)
with t1:
    st.markdown(
        f"| | Prior | Posterior |\n"
        f"|---|---|---|\n"
        f"| α | {alpha_prior:.1f} | {alpha_post:.1f} |\n"
        f"| β | {beta_prior:.1f} | {beta_post:.1f} |\n"
        f"| Mean | {prior_mean:.3f} | {post_mean:.3f} |\n"
        f"| CI width | {prior_ci[1]-prior_ci[0]:.3f} | {post_ci[1]-post_ci[0]:.3f} |"
    )
with t2:
    data_weight = N / (N + alpha_prior + beta_prior)
    prior_weight = (alpha_prior + beta_prior) / (N + alpha_prior + beta_prior)
    st.markdown("**Posterior mean as weighted average:**")
    st.latex(
        rf"\mu^* = {prior_weight:.3f} \cdot \frac{{\alpha}}{{\alpha+\beta}} + {data_weight:.3f} \cdot \frac{{x}}{{N}}"
    )
    st.markdown(
        f"Data weight = **{data_weight:.1%}**, Prior weight = **{prior_weight:.1%}**"
    )

# ── Section 4 – Posterior Predictive ─────────────────────────────────────────
section("4", "Posterior Predictive Distribution", "#D35400")
st.markdown(
    "Integrating out the uncertainty in θ over the posterior gives the "
    "**Beta-Binomial predictive distribution** for x̃ new successes in N_new trials:"
)
st.latex(
    r"\tilde{x} \mid x,\,N \;\sim\; \text{Beta-Binomial}(N_{\text{new}},\;\alpha^*,\;\beta^*)"
)
st.markdown(
    "The predictive variance has **two sources**: the Binomial sampling variability "
    "and the remaining uncertainty in θ. "
    "This makes it *overdispersed* compared to a plain Binomial at a fixed θ."
)

SIZE = 100_000
rng = np.random.default_rng(42)
theta_samples = beta_dist.rvs(alpha_post, beta_post, size=SIZE, random_state=rng)
bb_samples = binom.rvs(N_new, theta_samples, size=SIZE)
binom_samples = binom.rvs(N_new, post_mean, size=SIZE, random_state=np.random.default_rng(42))

k_range = np.arange(0, N_new + 1)

fig_pred = go.Figure()
fig_pred.add_trace(
    go.Bar(
        x=k_range,
        y=np.bincount(bb_samples, minlength=N_new + 1) / SIZE,
        name="Beta-Binomial (θ uncertain)",
        marker_color="darkorange", opacity=0.7,
    )
)
fig_pred.add_trace(
    go.Bar(
        x=k_range,
        y=np.bincount(binom_samples, minlength=N_new + 1) / SIZE,
        name=f"Binomial(θ = {post_mean:.3f})",
        marker_color="royalblue", opacity=0.7,
    )
)
fig_pred.update_layout(
    barmode="overlay",
    xaxis_title=f"Successes in {N_new} new trials",
    yaxis_title="Probability",
    title="Beta-Binomial vs Binomial Predictive Distribution",
    height=340, margin=dict(t=40, b=40),
    legend=dict(x=0.02, y=0.95),
)
st.plotly_chart(fig_pred, use_container_width=True)

excess = np.var(bb_samples) - np.var(binom_samples)

q1, q2, q3 = st.columns(3)
q1.metric("Predictive mean (BB)", f"{np.mean(bb_samples):.2f}")
q2.metric("Predictive std (BB)", f"{np.std(bb_samples):.2f}")
q3.metric("Overdispersion (extra var)", f"{excess:.2f}")

st.markdown("---")
t3, t4 = st.columns(2)
with t3:
    st.markdown(
        f"| | Beta-Binomial | Binomial |\n"
        f"|---|---|---|\n"
        f"| Mean | {np.mean(bb_samples):.2f} | {np.mean(binom_samples):.2f} |\n"
        f"| Std | {np.std(bb_samples):.2f} | {np.std(binom_samples):.2f} |\n"
        f"| Variance | {np.var(bb_samples):.2f} | {np.var(binom_samples):.2f} |"
    )
with t4:
    st.markdown("**Why overdispersion occurs:**")
    st.markdown(
        f"Beta-Binomial variance exceeds Binomial variance by **{excess:.2f}**. "
        f"When θ is uncertain, predictions must account for both "
        f"sampling noise *and* parameter uncertainty — use Beta-Binomial "
        f"whenever you haven't fully pinned down θ."
    )
