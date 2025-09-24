
import numpy as np
import pandas as pd
from scipy import stats
import streamlit as st

st.title("A/B Testing Playground — Frequentist & Bayesian")
st.write("Simulate conversions and (optionally) revenue, then analyze.")

with st.sidebar:
    st.header("Simulation Settings")
    nA = st.number_input("nA", 100, 200000, 4000, 100)
    nB = st.number_input("nB", 100, 200000, 4000, 100)
    pA = st.number_input("pA (baseline)", 0.0001, 0.9999, 0.08, 0.0001, format="%.4f")
    lift = st.number_input("Absolute lift (pB - pA)", -0.2, 0.2, 0.012, 0.0001, format="%.4f")
    use_revenue = st.checkbox("Simulate revenue (lognormal)", value=True)
    seed = st.number_input("Random seed", 0, 10**6, 42, 1)
    alpha = st.number_input("alpha", 0.0001, 0.2, 0.05, 0.0001, format="%.4f")
    power = st.number_input("power (1-β)", 0.5, 0.9999, 0.8, 0.01, format="%.2f")

np.random.seed(seed)
pB = pA + lift

# Simulate conversions
conv_A = np.random.binomial(1, pA, size=int(nA))
conv_B = np.random.binomial(1, pB, size=int(nB))

if use_revenue:
    rev_mu_A, rev_sigma_A = 1.8, 0.9
    rev_mu_B, rev_sigma_B = 1.9, 0.9
    revenue_A = np.where(conv_A==1, np.random.lognormal(rev_mu_A, rev_sigma_A, int(nA)), 0.0)
    revenue_B = np.where(conv_B==1, np.random.lognormal(rev_mu_B, rev_sigma_B, int(nB)), 0.0)
else:
    revenue_A = np.zeros(int(nA))
    revenue_B = np.zeros(int(nB))

xA, xB = int(conv_A.sum()), int(conv_B.sum())
crA, crB = xA/nA, xB/nB
st.subheader("Summary")
st.write(pd.DataFrame({
    "group":["A","B"],
    "n":[nA,nB],
    "conversions":[xA, xB],
    "conversion_rate":[crA, crB],
    "avg_revenue_per_user":[revenue_A.mean(), revenue_B.mean()]
}))

# Proportions z-test
p_pool = (xA + xB) / (nA + nB)
se = np.sqrt(p_pool*(1-p_pool)*(1/nA + 1/nB)) if (nA>0 and nB>0) else np.nan
z = (crB - crA)/se if (se is not None and np.isfinite(se) and se>0) else np.nan
pval_z = 2*(1-stats.norm.cdf(abs(z))) if (se is not None and np.isfinite(se) and se>0) else np.nan
zcrit = stats.norm.ppf(0.975)
ci = ((crB-crA)-zcrit*se, (crB-crA)+zcrit*se) if (se is not None and np.isfinite(se) and se>0) else (np.nan, np.nan)

# Chi-square
table = np.array([[xA, nA-xA],[xB, nB-xB]])
chi2, p_chi2, *_ = stats.chi2_contingency(table, correction=False)

# Wilson CI helper
def wilson_ci(x, n, alpha=0.05):
    if n==0: return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha/2)
    phat = x/n
    denom = 1 + z**2/n
    center = (phat + z*z/(2*n))/denom
    half = (z/denom)*np.sqrt((phat*(1-phat)/n) + (z*z/(4*n*n)))
    return center-half, center+half

wilson_A = wilson_ci(xA, nA, alpha)
wilson_B = wilson_ci(xB, nB, alpha)

st.subheader("Frequentist Tests")
rows = [
    {"metric":"diff (B-A) conv", "value": crB-crA, "ci_low": ci[0], "ci_high": ci[1]},
    {"metric":"z_stat",          "value": z,       "ci_low": None,   "ci_high": None},
    {"metric":"p_value_z",       "value": pval_z,  "ci_low": None,   "ci_high": None},
    {"metric":"chi2",            "value": chi2,    "ci_low": None,   "ci_high": None},
    {"metric":"p_value_chi2",    "value": p_chi2,  "ci_low": None,   "ci_high": None},
]
st.write(pd.DataFrame(rows))

# Revenue Welch t-test
if use_revenue:
    t_stat, p_t = stats.ttest_ind(revenue_B, revenue_A, equal_var=False)
    s2A = revenue_A.var(ddof=1); s2B = revenue_B.var(ddof=1)
    se_mean = np.sqrt(s2A/nA + s2B/nB)
    df_w = (s2A/nA + s2B/nB)**2 / ((s2A**2)/((nA**2)*(nA-1)) + (s2B**2)/((nB**2)*(nB-1)))
    tcrit = stats.t.ppf(0.975, df=df_w)
    mean_diff = revenue_B.mean() - revenue_A.mean()
    ci_rev = (mean_diff - tcrit*se_mean, mean_diff + tcrit*se_mean)

    rows_rev = [
        {"metric":"mean_rev_diff (B-A)", "value": mean_diff, "ci_low": ci_rev[0], "ci_high": ci_rev[1]},
        {"metric":"t_stat",              "value": t_stat,    "ci_low": None,      "ci_high": None},
        {"metric":"p_value_t",           "value": p_t,       "ci_low": None,      "ci_high": None},
    ]
    st.write(pd.DataFrame(rows_rev))

# Bayesian Beta–Bernoulli
draws = 100000
aA, bA = 1 + xA, 1 + (nA-xA)
aB, bB = 1 + xB, 1 + (nB-xB)
sA = np.random.beta(aA, bA, size=draws)
sB = np.random.beta(aB, bB, size=draws)
prob = (sB > sA).mean()
lift = (sB - sA)

st.subheader("Bayesian")
st.write(pd.DataFrame([{
    "metric":"P(p_B > p_A)",
    "value":prob,
    "ci_low": np.percentile(lift, 2.5),
    "ci_high":np.percentile(lift, 97.5)
}, {
    "metric":"posterior_mean_lift",
    "value":lift.mean(),
    "ci_low": None,
    "ci_high":None
}]))

# Power/MDE
def n_for_two_prop(p1, p2, alpha=0.05, power=0.8):
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    pbar = 0.5*(p1 + p2); qbar = 1 - pbar
    var = p1*(1-p1) + p2*(1-p2)
    n = ((z_alpha*np.sqrt(2*pbar*qbar) + z_beta*np.sqrt(var))**2) / ((p2 - p1)**2)
    return int(np.ceil(n))

def required_n_for_mde(pA, mde_abs, alpha=0.05, power=0.8):
    return n_for_two_prop(pA, pA + mde_abs, alpha, power)

def mde_for_n(pA, n, alpha=0.05, power=0.8):
    low, high = 1e-6, min(0.5, 0.9-pA)
    for _ in range(50):
        mid = 0.5*(low+high)
        if required_n_for_mde(pA, mid, alpha, power) > n:
            low = mid
        else:
            high = mid
    return 0.5*(low+high)

col1, col2 = st.columns(2)
with col1:
    target_mde = st.number_input("Target MDE (abs)", 0.0001, 0.5, 0.012, 0.0001, format="%.4f")
    st.write("Required n per arm:", required_n_for_mde(crA, target_mde, alpha, power))
with col2:
    st.write("Achievable MDE with current n:", mde_for_n(crA, nA, alpha, power))
