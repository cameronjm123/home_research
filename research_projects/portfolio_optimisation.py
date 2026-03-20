"""
Portfolio Optimisation using Markowitz Modern Portfolio Theory (MPT)

This script:
  1. Downloads monthly price data for a list of assets (stocks, ETFs, commodities, etc.)
  2. Estimates the covariance matrix using Ledoit-Wolf shrinkage (rather than the
     noisy sample covariance), and estimates expected returns from current valuation
     metrics (earnings yield for equities, SEC yield for bond ETFs) rather than
     historical means.
  3. Traces out the Efficient Frontier — the set of portfolios with the highest
     expected return for a given level of risk (volatility)
  4. Given a risk aversion parameter, finds the single optimal portfolio for that investor
  5. Optionally, given a risk-free rate, computes the Tangency Portfolio (highest Sharpe ratio)
     and draws the Capital Allocation Line (CAL)
  6. Enforces sector concentration constraints — no single sector may exceed a configurable
     weight (default 40%) of the total portfolio.

Dependencies: yfinance, numpy, scipy, matplotlib, pandas, scikit-learn
Install with: pip install yfinance numpy scipy matplotlib pandas scikit-learn
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf


# =============================================================================
# SECTOR MAP
# Default assignments for the 12-asset universe. Extend or override as needed.
# =============================================================================

DEFAULT_SECTOR_MAP = {
    "IBM":    "Technology",
    "GOOGL":  "Technology",
    "JPM":    "Financials",
    "GS":     "Financials",
    "TSCO.L": "ConsumerStaples",
    "WMT":    "ConsumerStaples",
    "GLD":    "Commodities",
    "PPLT":   "Commodities",
    "TLT":    "FixedIncome",
    "IEF":    "FixedIncome",
    "SHY":    "FixedIncome",
    "VOO":    "BroadEquity",
}


# =============================================================================
# 1. DATA COLLECTION
# =============================================================================

def get_monthly_returns(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted closing prices for the given tickers and convert to
    monthly percentage returns.

    Args:
        tickers : List of ticker symbols. These can be stocks (e.g. 'AAPL'),
                  ETFs (e.g. 'SPY'), or commodity ETFs (e.g. 'GLD' for gold).
                  yfinance uses Yahoo Finance tickers, so check Yahoo for the
                  correct symbol if unsure.
        start   : Start date string in 'YYYY-MM-DD' format.
        end     : End date string in 'YYYY-MM-DD' format.

    Returns:
        DataFrame of monthly returns, one column per ticker.
    """
    # Download daily adjusted closing prices for all tickers at once
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)["Close"]

    # Resample from daily to monthly by taking the last trading day of each month,
    # then compute the percentage change month-over-month.
    # pct_change() gives (P_t - P_{t-1}) / P_{t-1}, i.e. simple returns.
    monthly_returns = raw.resample("ME").last().pct_change().dropna()

    return monthly_returns


# =============================================================================
# 2. COVARIANCE ESTIMATION — LEDOIT-WOLF SHRINKAGE
# =============================================================================

def ledoit_wolf_covariance(monthly_returns: pd.DataFrame) -> np.ndarray:
    """
    Estimate the covariance matrix using Ledoit-Wolf analytical shrinkage.

    The sample covariance matrix is an unbiased estimator of the true covariance,
    but it suffers from severe estimation error when n_assets is large relative to
    the number of observations (the "curse of dimensionality"). This causes the
    matrix to be ill-conditioned: extreme eigenvalues are amplified, leading the
    optimiser to take massive, unstable bets on small estimated return differences.

    Ledoit-Wolf shrinkage addresses this by pulling ("shrinking") the sample
    covariance toward a structured target — typically a scaled identity matrix or
    a constant-correlation matrix — weighted by an analytically derived optimal
    shrinkage intensity alpha:

        Sigma_LW = (1 - alpha) * Sigma_sample + alpha * F

    where F is the shrinkage target. This reduces variance in the estimator at
    the cost of a small bias, yielding a well-conditioned matrix that produces
    more stable portfolio weights.

    sklearn's LedoitWolf implements the Oracle Approximating Shrinkage (OAS)
    variant with closed-form optimal alpha — no cross-validation required.

    Args:
        monthly_returns : DataFrame of monthly returns (T x n).

    Returns:
        Ledoit-Wolf covariance matrix as an (n x n) numpy array (monthly scale).
    """
    lw = LedoitWolf()
    lw.fit(monthly_returns.values)
    print(f"  Ledoit-Wolf shrinkage coefficient: {lw.shrinkage_:.4f}  "
          f"(0 = no shrinkage, 1 = full shrinkage to target)")
    return lw.covariance_


# =============================================================================
# 3. EXPECTED RETURNS — VALUATION-ADJUSTED FORWARD ESTIMATES
# =============================================================================

def get_valuation_adjusted_returns(tickers: list[str],
                                   monthly_returns: pd.DataFrame) -> np.ndarray:
    """
    Estimate annualised forward expected returns from current valuation metrics
    rather than historical means.

    Rationale: historical mean returns are extremely noisy estimators of true
    expected returns — a 10-year sample window has estimation error so large that
    the historical mean is often worse than a simple prior. Valuation-based
    estimates exploit the well-documented mean-reversion in valuations:

    - Equities & equity ETFs (stocks, VOO):
        Expected return ≈ 1 / forward_PE  (forward earnings yield)
        This is the forward version of the "earnings yield" model. Under the
        Gordon Growth Model with g ≈ 0, E[r] = E/P. With growth, E[r] = E/P + g,
        but the earnings yield alone is a strong predictor of 5–10 year returns.
        We prefer forwardPE; if unavailable we fall back to trailingPE.

    - Bond ETFs (TLT, IEF, SHY):
        Expected return ≈ current 30-day SEC yield.
        For a bond held to maturity the yield IS the expected return (ignoring
        default, which is negligible for US Treasuries). The SEC yield is the
        best available proxy in yfinance's `info` dict.

    - Commodity ETFs (GLD, PPLT) and other assets without earnings/yield:
        No reliable valuation anchor exists. We fall back to the historical mean
        for these assets only, with a warning.

    Args:
        tickers         : List of ticker symbols (same order as monthly_returns columns).
        monthly_returns : Historical monthly returns DataFrame (used only for fallback).

    Returns:
        Array of annualised expected returns, one per ticker.
    """
    historical_means_annual = monthly_returns.mean().values * 12
    expected_returns = np.zeros(len(tickers))

    print("\nValuation-adjusted expected returns:")
    for i, ticker in enumerate(tickers):
        try:
            info = yf.Ticker(ticker).info

            forward_pe   = info.get("forwardPE")
            trailing_pe  = info.get("trailingPE")
            # yfinance returns 'yield' as a decimal for ETFs (e.g. 0.042 = 4.2%)
            etf_yield    = info.get("yield")

            if forward_pe and np.isfinite(forward_pe) and forward_pe > 0:
                er = 1.0 / forward_pe
                print(f"  {ticker:10s}: forward earnings yield = {er*100:.2f}%  "
                      f"(fwd PE = {forward_pe:.1f})")

            elif trailing_pe and np.isfinite(trailing_pe) and trailing_pe > 0:
                er = 1.0 / trailing_pe
                print(f"  {ticker:10s}: trailing earnings yield = {er*100:.2f}%  "
                      f"(trail PE = {trailing_pe:.1f})")

            elif etf_yield and np.isfinite(etf_yield) and etf_yield > 0:
                er = etf_yield   # already annualised by Yahoo
                print(f"  {ticker:10s}: ETF/bond SEC yield = {er*100:.2f}%")

            else:
                er = historical_means_annual[i]
                print(f"  {ticker:10s}: no valuation data — fallback to "
                      f"historical mean = {er*100:.2f}%")

            expected_returns[i] = er

        except Exception as exc:
            expected_returns[i] = historical_means_annual[i]
            print(f"  {ticker:10s}: fetch error ({exc}) — fallback to "
                  f"historical mean = {historical_means_annual[i]*100:.2f}%")

    return expected_returns


# =============================================================================
# 4. SECTOR CONCENTRATION CONSTRAINTS
# =============================================================================

def make_sector_constraints(tickers: list[str],
                             sector_map: dict,
                             max_sector_weight: float = 0.40) -> list[dict]:
    """
    Build SLSQP inequality constraints that cap each sector's total portfolio weight.

    For each sector S, the constraint is:
        sum(w_i  for all i in S) <= max_sector_weight

    In SLSQP form (type='ineq' means fun(w) >= 0):
        fun(w) = max_sector_weight - sum(w[sector_indices]) >= 0

    Args:
        tickers           : List of ticker symbols (defines the weight-array ordering).
        sector_map        : Dict mapping ticker -> sector label. Any ticker not in the
                            map is assigned to an "Other" catch-all sector.
        max_sector_weight : Maximum fraction any single sector may hold (default 0.40).

    Returns:
        List of constraint dicts compatible with scipy.optimize.minimize (SLSQP).
    """
    # Group asset indices by sector label
    sector_indices: dict[str, list[int]] = {}
    for i, t in enumerate(tickers):
        label = sector_map.get(t, "Other")
        sector_indices.setdefault(label, []).append(i)

    constraints = []
    for sector, indices in sector_indices.items():
        idx = np.array(indices)
        constraints.append({
            "type": "ineq",
            "fun":  lambda w, idx=idx: max_sector_weight - np.sum(w[idx]),
        })

    return constraints


# =============================================================================
# 5. PORTFOLIO STATISTICS
# =============================================================================

def portfolio_performance(weights: np.ndarray, mean_returns: np.ndarray,
                           cov_matrix: np.ndarray) -> tuple[float, float]:
    """
    Calculate the annualised expected return and volatility of a portfolio.

    In Markowitz theory:
      - Portfolio return  = w^T * mu        (dot product of weights and asset returns)
      - Portfolio variance = w^T * Sigma * w (quadratic form with covariance matrix)

    We annualise by multiplying monthly return by 12 and monthly variance by 12
    (variance scales linearly with time under i.i.d. return assumptions).

    Args:
        weights      : Array of portfolio weights (must sum to 1).
        mean_returns : Array of mean monthly returns per asset.
        cov_matrix   : Covariance matrix of monthly returns (n x n).

    Returns:
        (annualised_return, annualised_volatility) as floats.
    """
    port_return = np.dot(weights, mean_returns) * 12          # annualise
    port_variance = weights @ cov_matrix @ weights * 12       # annualise
    port_volatility = np.sqrt(port_variance)
    return port_return, port_volatility


# =============================================================================
# 6. EFFICIENT FRONTIER
# =============================================================================

def compute_efficient_frontier(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                                n_points: int = 100,
                                sector_constraints: list | None = None
                                ) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Trace the Efficient Frontier by solving a constrained optimisation for each
    target return level.

    The optimisation problem for each target return r* is:
        minimise    w^T * Sigma * w        (minimise portfolio variance)
        subject to  sum(w) = 1             (weights must sum to 100%)
                    w^T * mu = r*          (achieve the target return)
                    w_i >= 0 for all i     (no short selling)
                    [sector constraints]   (no sector exceeds max_sector_weight)

    Args:
        mean_returns       : Array of annualised mean returns per asset.
        cov_matrix         : Annualised covariance matrix (n x n).
        n_points           : Number of points to compute on the frontier.
        sector_constraints : Optional list of SLSQP constraint dicts from
                             make_sector_constraints(). Pass None to skip.

    Returns:
        frontier_vols    : Array of portfolio volatilities on the frontier.
        frontier_returns : Array of portfolio returns on the frontier.
        frontier_weights : List of weight arrays corresponding to each point.
    """
    n_assets = len(mean_returns)

    min_ret = mean_returns.min()
    max_ret = mean_returns.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)

    frontier_vols = []
    frontier_returns = []
    frontier_weights = []

    base_constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    ]
    if sector_constraints:
        base_constraints = base_constraints + sector_constraints

    for target in target_returns:
        def objective(w):
            return w @ cov_matrix @ w

        def gradient(w):
            return 2 * cov_matrix @ w

        constraints = base_constraints + [
            {"type": "eq", "fun": lambda w, t=target: np.dot(w, mean_returns) - t},
        ]

        bounds = [(0, 1)] * n_assets
        w0 = np.ones(n_assets) / n_assets

        result = minimize(objective, w0, jac=gradient, method="SLSQP",
                          bounds=bounds, constraints=constraints,
                          options={"ftol": 1e-12, "maxiter": 1000})

        if result.success:
            w_opt = result.x
            ret, vol = portfolio_performance(w_opt, mean_returns / 12, cov_matrix / 12)
            frontier_vols.append(vol)
            frontier_returns.append(ret)
            frontier_weights.append(w_opt)

    return np.array(frontier_vols), np.array(frontier_returns), frontier_weights


# =============================================================================
# 7. OPTIMAL PORTFOLIO FOR A GIVEN RISK AVERSION
# =============================================================================

def find_optimal_portfolio(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                            risk_aversion: float,
                            sector_constraints: list | None = None
                            ) -> tuple[np.ndarray, float, float]:
    """
    Find the portfolio that maximises a mean-variance utility function:

        U(w) = E[r_p] - (A / 2) * Var(r_p)

    where A is the risk aversion coefficient:
        - A = 0   : investor is risk-neutral (just maximise return)
        - A = 1-3 : relatively risk-tolerant
        - A = 5-10: moderately risk-averse (typical retail investor)
        - A > 10  : very risk-averse (favours near-minimum-variance portfolios)

    Args:
        mean_returns       : Array of annualised mean returns per asset.
        cov_matrix         : Annualised covariance matrix (n x n).
        risk_aversion      : Scalar A >= 0.
        sector_constraints : Optional list of SLSQP constraint dicts.

    Returns:
        (weights, expected_return, volatility) of the optimal portfolio.
    """
    n_assets = len(mean_returns)

    def neg_utility(w):
        port_ret = np.dot(w, mean_returns)
        port_var = w @ cov_matrix @ w
        return -(port_ret - (risk_aversion / 2) * port_var)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if sector_constraints:
        constraints = constraints + sector_constraints

    bounds = [(0, 1)] * n_assets
    w0 = np.ones(n_assets) / n_assets

    result = minimize(neg_utility, w0, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"ftol": 1e-12, "maxiter": 1000})

    w_opt = result.x
    ret, vol = portfolio_performance(w_opt, mean_returns / 12, cov_matrix / 12)
    return w_opt, ret, vol


# =============================================================================
# 8. TANGENCY PORTFOLIO AND CAPITAL ALLOCATION LINE (CAL)
# =============================================================================

def find_tangency_portfolio(mean_returns: np.ndarray, cov_matrix: np.ndarray,
                             risk_free_rate: float,
                             sector_constraints: list | None = None
                             ) -> tuple[np.ndarray, float, float]:
    """
    Find the Tangency Portfolio — the risky portfolio with the highest Sharpe ratio.

    The Sharpe ratio measures excess return per unit of risk:
        Sharpe = (E[r_p] - r_f) / sigma_p

    The tangency portfolio sits at the point where a straight line drawn from
    the risk-free rate (on the y-axis) is tangent to the Efficient Frontier.
    This line is the Capital Allocation Line (CAL).

    Any combination of the risk-free asset and the tangency portfolio lies on
    the CAL and dominates (in Sharpe ratio terms) any portfolio on the frontier
    alone. This is the "Two-Fund Separation Theorem": all investors should hold
    some mix of the risk-free asset and the tangency portfolio.

    Args:
        mean_returns       : Array of annualised mean returns per asset.
        cov_matrix         : Annualised covariance matrix (n x n).
        risk_free_rate     : Annualised risk-free rate (e.g. 0.04 for 4%).
        sector_constraints : Optional list of SLSQP constraint dicts.

    Returns:
        (weights, expected_return, volatility) of the tangency portfolio.
    """
    n_assets = len(mean_returns)

    def neg_sharpe(w):
        port_ret = np.dot(w, mean_returns)
        port_vol = np.sqrt(w @ cov_matrix @ w)
        if port_vol == 0:
            return 0.0
        return -(port_ret - risk_free_rate) / port_vol

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if sector_constraints:
        constraints = constraints + sector_constraints

    bounds = [(0, 1)] * n_assets
    w0 = np.ones(n_assets) / n_assets

    result = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds,
                      constraints=constraints, options={"ftol": 1e-12, "maxiter": 1000})

    w_tan = result.x
    ret, vol = portfolio_performance(w_tan, mean_returns / 12, cov_matrix / 12)
    return w_tan, ret, vol


# =============================================================================
# 9. PLOTTING
# =============================================================================

def plot_results(frontier_vols, frontier_returns, optimal_weights, optimal_ret,
                 optimal_vol, tickers, risk_free_rate=None, tangency_ret=None,
                 tangency_vol=None):
    """
    Plot the Efficient Frontier and, if a risk-free rate is provided, the CAL.

    The plot shows:
      - Efficient Frontier curve (risk vs return tradeoff)
      - The optimal portfolio for the given risk aversion (red star)
      - Optionally: the tangency portfolio (green diamond) and the CAL
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # --- Efficient Frontier ---
    ax.plot(frontier_vols * 100, frontier_returns * 100,
            "b-", linewidth=2, label="Efficient Frontier")

    # --- Optimal portfolio for given risk aversion ---
    ax.scatter(optimal_vol * 100, optimal_ret * 100,
               marker="*", color="red", s=300, zorder=5,
               label="Optimal Portfolio (risk aversion A)")

    # --- Capital Allocation Line + Tangency Portfolio (optional) ---
    if risk_free_rate is not None and tangency_ret is not None:
        cal_vols = np.linspace(0, tangency_vol * 1.3, 100)
        sharpe = (tangency_ret - risk_free_rate) / tangency_vol
        cal_rets = risk_free_rate + sharpe * cal_vols

        ax.plot(cal_vols * 100, cal_rets * 100,
                "g--", linewidth=1.5, label="Capital Allocation Line (CAL)")

        ax.scatter(tangency_vol * 100, tangency_ret * 100,
                   marker="D", color="green", s=150, zorder=5,
                   label="Tangency Portfolio (max Sharpe)")

        ax.scatter(0, risk_free_rate * 100,
                   marker="o", color="grey", s=100, zorder=5,
                   label=f"Risk-Free Rate ({risk_free_rate*100:.1f}%)")

    ax.set_xlabel("Annualised Volatility (%)", fontsize=12)
    ax.set_ylabel("Annualised Expected Return (%)", fontsize=12)
    ax.set_title("Markowitz Efficient Frontier", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 10. MAIN — WIRE EVERYTHING TOGETHER
# =============================================================================

def run_portfolio_optimisation(
    tickers: list[str],
    start: str,
    end: str,
    risk_aversion: float,
    risk_free_rate=None,       # float or None
    cash_weight=None,          # float or None — e.g. 0.4 for 40% cash, 60% tangency
    sector_map: dict | None = None,   # dict mapping ticker -> sector label, or None to skip
    max_sector_weight: float = 0.40,  # maximum weight per sector (default 40%)
    use_ledoit_wolf: bool = True,     # use LW shrinkage instead of sample covariance
    use_valuation_returns: bool = True,  # use earnings yield / bond yield instead of historical mean
):
    """
    Full pipeline: download data → estimate covariance (LW shrinkage) →
    estimate expected returns (valuation-adjusted) → trace frontier →
    find optimal portfolio → (optionally) compute CAL.

    Args:
        tickers               : List of Yahoo Finance ticker symbols to include.
        start                 : Start date 'YYYY-MM-DD'.
        end                   : End date 'YYYY-MM-DD'.
        risk_aversion         : Risk aversion coefficient A. Higher = more conservative.
                                Typical range: 1 (aggressive) to 10 (conservative).
        risk_free_rate        : Optional annualised risk-free rate (e.g. 0.04 = 4%).
                                If provided, the tangency portfolio and CAL are computed.
        cash_weight           : Optional fraction to hold in the risk-free asset (e.g. 0.4).
                                Only used when risk_free_rate is also provided.
        sector_map            : Dict mapping ticker -> sector string. Pass None to skip
                                sector constraints entirely. Defaults to DEFAULT_SECTOR_MAP.
        max_sector_weight     : Cap on any single sector's total weight (default 0.40 = 40%).
        use_ledoit_wolf       : If True, use Ledoit-Wolf shrinkage for covariance.
                                If False, fall back to the sample covariance matrix.
        use_valuation_returns : If True, use earnings yield / bond yield as expected returns.
                                If False, use historical mean returns.
    """
    print(f"Downloading monthly returns for: {tickers}")
    monthly_returns = get_monthly_returns(tickers, start, end)
    print(f"Got {len(monthly_returns)} months of data from {monthly_returns.index[0].date()} "
          f"to {monthly_returns.index[-1].date()}\n")

    # Reorder columns to match tickers list (yfinance may sort them alphabetically)
    monthly_returns = monthly_returns[tickers]

    # --- Covariance matrix ---
    if use_ledoit_wolf:
        print("Estimating covariance matrix (Ledoit-Wolf shrinkage)...")
        cov_monthly = ledoit_wolf_covariance(monthly_returns)
    else:
        print("Estimating covariance matrix (sample covariance)...")
        cov_monthly = monthly_returns.cov().values

    cov_annual = cov_monthly * 12

    # --- Expected returns ---
    if use_valuation_returns:
        print("\nEstimating expected returns (valuation-adjusted)...")
        mean_ret_annual = get_valuation_adjusted_returns(tickers, monthly_returns)
    else:
        print("Estimating expected returns (historical mean)...")
        mean_ret_annual = monthly_returns.mean().values * 12
        for t, r in zip(tickers, mean_ret_annual):
            print(f"  {t:10s}: {r*100:.2f}%")

    # --- Sector constraints ---
    sector_constraints = None
    if sector_map is not None:
        effective_map = DEFAULT_SECTOR_MAP.copy()
        effective_map.update(sector_map)
        sector_constraints = make_sector_constraints(tickers, effective_map, max_sector_weight)
        # Show which tickers got sector assignments
        print(f"\nSector constraints (max {max_sector_weight*100:.0f}% per sector):")
        sector_groups: dict[str, list[str]] = {}
        for t in tickers:
            s = effective_map.get(t, "Other")
            sector_groups.setdefault(s, []).append(t)
        for s, members in sector_groups.items():
            print(f"  {s}: {', '.join(members)}")

    # --- Efficient Frontier ---
    print("\nComputing efficient frontier...")
    frontier_vols, frontier_rets, frontier_weights = compute_efficient_frontier(
        mean_ret_annual, cov_annual, sector_constraints=sector_constraints
    )

    # --- Optimal portfolio for this investor's risk aversion ---
    print(f"Finding optimal portfolio for risk aversion A = {risk_aversion}...")
    opt_weights, opt_ret, opt_vol = find_optimal_portfolio(
        mean_ret_annual, cov_annual, risk_aversion,
        sector_constraints=sector_constraints
    )

    print("\n--- Optimal Portfolio ---")
    print(f"  Expected Return : {opt_ret*100:.2f}%")
    print(f"  Volatility      : {opt_vol*100:.2f}%")
    print(f"  Sharpe Ratio    : {(opt_ret / opt_vol):.3f}  (unadjusted, no rf)")
    print("  Weights:")
    for ticker, w in zip(tickers, opt_weights):
        print(f"    {ticker:10s}: {w*100:.1f}%")

    # --- Tangency portfolio + CAL (optional) ---
    tangency_ret = tangency_vol = tangency_weights = None

    if risk_free_rate is not None:
        print(f"\nComputing tangency portfolio (risk-free rate = {risk_free_rate*100:.1f}%)...")
        tangency_weights, tangency_ret, tangency_vol = find_tangency_portfolio(
            mean_ret_annual, cov_annual, risk_free_rate,
            sector_constraints=sector_constraints
        )

        sharpe_tan = (tangency_ret - risk_free_rate) / tangency_vol

        print("\n--- Tangency Portfolio (max Sharpe) ---")
        print(f"  Expected Return : {tangency_ret*100:.2f}%")
        print(f"  Volatility      : {tangency_vol*100:.2f}%")
        print(f"  Sharpe Ratio    : {sharpe_tan:.3f}")
        print("  Weights:")
        for ticker, w in zip(tickers, tangency_weights):
            print(f"    {ticker:10s}: {w*100:.1f}%")

        # --- CAL split: blend tangency portfolio with risk-free asset ---
        if cash_weight is not None:
            y = 1.0 - cash_weight  # fraction invested in tangency portfolio
            blended_ret = risk_free_rate + y * (tangency_ret - risk_free_rate)
            blended_vol = y * tangency_vol
            blended_sharpe = (blended_ret - risk_free_rate) / blended_vol

            print(f"\n--- CAL Blend ({cash_weight*100:.0f}% cash / {y*100:.0f}% tangency) ---")
            print(f"  Expected Return : {blended_ret*100:.2f}%")
            print(f"  Volatility      : {blended_vol*100:.2f}%")
            print(f"  Sharpe Ratio    : {blended_sharpe:.3f}  (same as tangency — Sharpe is preserved on the CAL)")
            print(f"  Effective asset weights (including cash):")
            print(f"    {'Cash (rf)':10s}: {cash_weight*100:.1f}%")
            for ticker, w in zip(tickers, tangency_weights):
                print(f"    {ticker:10s}: {w * y * 100:.1f}%")

    # --- Plot ---
    plot_results(
        frontier_vols, frontier_rets,
        opt_weights, opt_ret, opt_vol,
        tickers,
        risk_free_rate=risk_free_rate,
        tangency_ret=tangency_ret,
        tangency_vol=tangency_vol,
    )

    return {
        "optimal_weights":      dict(zip(tickers, opt_weights)),
        "optimal_return":       opt_ret,
        "optimal_volatility":   opt_vol,
        "tangency_weights":     dict(zip(tickers, tangency_weights)) if tangency_weights is not None else None,
        "tangency_return":      tangency_ret,
        "tangency_volatility":  tangency_vol,
    }


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    results = run_portfolio_optimisation(
        tickers=[
            "IBM",    # US large cap tech / IT services
            "GOOGL",  # US large cap tech (Alphabet)
            "JPM",    # US large cap bank
            "GS",     # US investment bank
            "TSCO.L", # Tesco (UK consumer staples)
            "WMT",    # Walmart (US consumer staples)
            "GLD",    # Gold ETF (commodity)
            "PPLT",   # Platinum ETF (commodity)
            "TLT",    # 20+ Year US Treasury Bond ETF
            "IEF",    # 7-10 Year US Treasury Bond ETF
            "SHY",    # 1-3 Year US Treasury Bond ETF
            "VOO",    # S&P 500 ETF (broad US equity)
        ],
        start="2015-01-01",
        end="2025-12-31",
        risk_aversion=5,            # moderate risk aversion (1=aggressive, 10=conservative)
        risk_free_rate=0.043,       # ~4.3% (approximate current 3-month T-bill yield, Mar 2025)
        cash_weight=0.40,           # 40% cash, 60% tangency portfolio
        sector_map={},              # use DEFAULT_SECTOR_MAP (pass None to disable constraints)
        max_sector_weight=0.40,     # no single sector may exceed 40%
        use_ledoit_wolf=True,       # Ledoit-Wolf shrinkage covariance
        use_valuation_returns=True, # earnings yield / bond yield instead of historical mean
    )
