"""
Explosiveness tests: SADF
"""

from typing import Tuple, Union

import numpy as np
import pandas as pd

from mlfinpy.util.multiprocess import mp_pandas_obj

# pylint: disable=invalid-name


def _get_sadf_at_t(X: pd.DataFrame, y: pd.DataFrame, min_length: int, model: str, phi: float) -> float:
    """
    SADF's Inner Loop (get SADF value at t)

    Parameters
    ----------
    X : pd.DataFrame
        Lagged values, constants, trend coefficients.
    y : pd.DataFrame
        Y values (either ``y`` or ``y.diff()``).
    min_length : int
        Minimum number of samples needed for estimation
    model : str
        Either 'native', 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    phi : float
        Coefficient to penalize large sample lengths when computing SMT, in [0, 1]

    Returns
    -------
    float
        SADF statistics for y.index[-1]

    Notes
    -----
        Advances in Financial Machine Learning, Snippet 17.2, page 258.
    """
    start_points, bsadf = range(0, y.shape[0] - min_length + 1), -np.inf
    for start in start_points:
        y_, X_ = y[start:], X[start:]
        if y_.shape[0] < 20:
            # A regression fit on fewer than 20 observations is too noisy to
            # trust; excluding it from the sup-search avoids the ADF stat
            # being driven by a single unreliable small-sample window.
            continue
        b_mean_, b_std_ = get_betas(X_, y_)
        if not np.isnan(b_mean_[0]):
            b_mean_, b_std_ = b_mean_[0, 0], b_std_[0, 0] ** 0.5
            # TODO: Rewrite logic of this module to avoid division by zero
            with np.errstate(invalid="ignore"):
                all_adf = b_mean_ / b_std_
            if model[:2] == "sm":
                all_adf = np.abs(all_adf) / (y.shape[0] ** phi)
            if all_adf > bsadf:
                bsadf = all_adf
    return bsadf


def _get_y_x(
    series: pd.Series, model: str, lags: Union[int, list], add_const: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preparing The Datasets

    Parameters
    ----------
    series : pd.Series
        Series to prepare for test statistics generation (for example log prices)
    model : str
        Either 'native', 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    lags : int or list
        Either number of lags to use or array of specified lags
    add_const : bool
        Flag to add constant

    Returns
    -------
    x : pd.DataFrame
        Prepared X for SADF generation
    y : pd.DataFrame
        Prepared y for SADF generation

    Notes
    -----
        Advances in Financial Machine Learning, Snippet 17.2, page 258-259.
    """
    series = pd.DataFrame(series)
    series_diff = series.diff().dropna()
    x = _lag_df(series_diff, lags).dropna()
    x["y_lagged"] = series.shift(1).loc[x.index]  # add y_(t-1) column
    y = series_diff.loc[x.index]

    if add_const is True:
        x["const"] = 1

    if model == "native":
        # Plain Phillips-Wu-Yu (2011) / AFML Ch17 base spec: constant + y_lagged + lagged
        # diffs only, no deterministic trend term (unlike "linear"/"quadratic" below).
        # This is the specification the canonical SADF bubble test actually uses.
        if "const" not in x.columns:
            x["const"] = 1
        beta_column = "y_lagged"
    elif model == "linear":
        x["trend"] = np.arange(x.shape[0])  # Add t to the model (0, 1, 2, 3, 4, 5, .... t)
        beta_column = "y_lagged"  # Column which is used to estimate test beta statistics
    elif model == "quadratic":
        x["trend"] = np.arange(x.shape[0])  # Add t to the model (0, 1, 2, 3, 4, 5, .... t)
        x["quad_trend"] = np.arange(x.shape[0]) ** 2  # Add t^2 to the model (0, 1, 4, 9, ....)
        beta_column = "y_lagged"  # Column which is used to estimate test beta statistics
    elif model == "sm_poly_1":
        y = series.loc[y.index]
        x = pd.DataFrame(index=y.index)
        x["const"] = 1
        x["trend"] = np.arange(x.shape[0])
        x["quad_trend"] = np.arange(x.shape[0]) ** 2
        beta_column = "quad_trend"
    elif model == "sm_poly_2":
        y = np.log(series.loc[y.index])
        x = pd.DataFrame(index=y.index)
        x["const"] = 1
        x["trend"] = np.arange(x.shape[0])
        x["quad_trend"] = np.arange(x.shape[0]) ** 2
        beta_column = "quad_trend"
    elif model == "sm_exp":
        y = np.log(series.loc[y.index])
        x = pd.DataFrame(index=y.index)
        x["const"] = 1
        x["trend"] = np.arange(x.shape[0])
        beta_column = "trend"
    elif model == "sm_power":
        y = np.log(series.loc[y.index])
        x = pd.DataFrame(index=y.index)
        x["const"] = 1
        # TODO: Rewrite logic of this module to avoid division by zero
        with np.errstate(divide="ignore"):
            x["log_trend"] = np.log(np.arange(x.shape[0]))
        beta_column = "log_trend"
    else:
        raise ValueError("Unknown model")

    # Move y_lagged column to the front for further extraction
    columns = list(x.columns)
    columns.insert(0, columns.pop(columns.index(beta_column)))
    x = x[columns]
    return x, y


def _lag_df(df: pd.DataFrame, lags: Union[int, list[int]]) -> pd.DataFrame:
    """
    Advances in Financial Machine Learning, Snipet 17.3, page 259.

    Apply Lags to DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to apply lags to
    lags : int or list
        Either number of lags to use or array of specified lags

    Returns
    -------
    pd.DataFrame
        Dataframe with lags
    """
    df_lagged = pd.DataFrame()
    if isinstance(lags, int):
        lags = range(1, lags + 1)
    else:
        lags = [int(lag) for lag in lags]

    for lag in lags:
        temp_df = df.shift(lag).copy(deep=True)
        temp_df.columns = [str(i) + "_" + str(lag) for i in temp_df.columns]
        df_lagged = df_lagged.join(temp_df, how="outer")
    return df_lagged


def get_betas(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:  # Features(factors)  # Outcomes
    """
    Fitting The ADF Specification (get beta estimate and estimate variance)

    Parameters
    ----------
    X : pd.DataFrame
        Features (factors)
    y : pd.DataFrame
        Outcomes

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Betas and variances of estimates

    Notes
    -----
        Advances in Financial Machine Learning, Snipet 17.4, page 259.
    """
    xy = np.dot(X.T, y)
    xx = np.dot(X.T, X)

    try:
        xx_inv = np.linalg.inv(xx)
    except np.linalg.LinAlgError:
        return [np.nan], [[np.nan, np.nan]]

    b_mean = np.dot(xx_inv, xy)
    err = y - np.dot(X, b_mean)
    b_var = np.dot(err.T, err) / (X.shape[0] - X.shape[1]) * xx_inv

    return b_mean, b_var


def _sadf_outer_loop(
    X: pd.DataFrame, y: pd.DataFrame, min_length: int, model: str, phi: float, molecule: list
) -> pd.Series:
    """
    This function gets SADF for t times from molecule

    Parameters
    ----------
    X : pd.DataFrame
        Features(factors) matrix.
    y : pd.DataFrame
        Target vector.
    min_length : int
        Minimum number of observations.
    model : str
        Either 'native', 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'.
    phi : float
        Coefficient to penalize large sample lengths when computing SMT, in [0, 1].
    molecule : list
        Indices to get SADF.

    Returns
    -------
    pd.Series
        SADF statistics.
    """
    sadf_series = pd.Series(index=molecule, dtype="float64")
    for index in molecule:
        X_subset = X.loc[:index].values
        y_subset = y.loc[:index].values.reshape(-1, 1)
        value = _get_sadf_at_t(X_subset, y_subset, min_length, model, phi)
        sadf_series[index] = value
    return sadf_series


def _sadf_native_fast(series: pd.Series, lags: int, min_length: int) -> pd.Series:
    """
    Vectorised model="native" SADF: plain Phillips-Wu-Yu (2011) / AFML Ch17 base spec
    (constant + y_lagged + lagged diffs, no deterministic trend).

    Every ADF regression for window [t0, t] uses rows i in [t0+lags, t-1], and each
    row's own values (y[i], and the lag-differences around it) don't depend on t0 at
    all -- only which contiguous slice of rows a window includes does. So instead of
    rebuilding and re-inverting X'X from scratch for every (t0, t) pair (what
    `_sadf_outer_loop`/`_get_sadf_at_t` do for the other models -- correct, but
    O(n^3) and ~25x slower on a realistic daily-bar series), each row is precomputed
    once and its running (cumulative) sum taken, so any window's X'X/X'y is a
    difference of two cumulative sums in O(1). That turns the unavoidable O(n^2)
    sup-search into O(n^2) total work, with numpy's batched linalg solving every t0
    for a given t in one vectorised call -- exact same numbers as a from-scratch
    refit, just without redoing shared work.

    Parameters
    ----------
    series : pd.Series
        Series (e.g. log prices) to compute SADF over.
    lags : int
        Number of lagged difference terms in the ADF regression.
    min_length : int
        Minimum number of post-diff-and-lag observations required for a candidate
        regression window (same units as `get_sadf`'s `min_length` for the other
        models).

    Returns
    -------
    pd.Series
        SADF statistics aligned to `series.index`.
    """
    min_sl = min_length + lags
    n = len(series)
    L = lags
    p = L + 2
    out = np.full(n, np.nan)
    y = series.astype(float).ffill().to_numpy()

    if n - 1 - L < 1:  # not even one valid row
        return pd.Series(out, index=series.index, name="sadf")

    # Row i (i = L .. n-2) of the ADF regression, expressed once -- identical
    # for every window that includes it (see docstring).
    dy = np.diff(y)  # dy[k] = y[k+1] - y[k]
    i_idx = np.arange(L, n - 1)
    m = len(i_idx)
    target = dy[i_idx]
    lag_cols = np.column_stack([dy[i_idx - j] for j in range(1, L + 1)]) if L > 0 else np.empty((m, 0))
    R = np.column_stack([y[i_idx], np.ones(m), lag_cols])  # (m, p)

    # Prefix sums over "row position" pos = i - L, so that rows for window
    # (t0, t) -- i in [t0+L, t-1], i.e. pos in [t0, t-1-L] -- are
    # S2[t-L] - S2[t0] etc. (k = t - L is the exclusive upper prefix index).
    outer = np.einsum("ma,mb->mab", R, R)
    S2 = np.concatenate([np.zeros((1, p, p)), np.cumsum(outer, axis=0)])
    S3 = np.concatenate([np.zeros((1, p)), np.cumsum(R * target[:, None], axis=0)])
    S4 = np.concatenate([[0.0], np.cumsum(target**2)])

    for t in range(min_sl, n):
        k = t - L
        if k < 0 or k > m:
            continue
        t0_hi = min(t - min_sl, k - 20)  # need n_obs = k - t0 >= 20 for a reliable fit
        if t0_hi < 0:
            continue
        t0 = np.arange(0, t0_hi + 1)

        XtX = S2[k] - S2[t0]          # (q, p, p)
        XtY = S3[k] - S3[t0]          # (q, p)
        n_obs = k - t0                 # (q,)
        dof = n_obs - p
        valid = dof > 0

        det = np.linalg.det(XtX)
        valid &= np.abs(det) > 1e-12  # matches the singular-matrix skip in the naive form
        if not valid.any():
            continue

        beta = np.zeros_like(XtY)
        # b needs an explicit trailing dim (p, 1) so numpy batches solve()
        # over the leading q axis instead of treating (q, p) as one system.
        beta[valid] = np.linalg.solve(XtX[valid], XtY[valid][..., None]).squeeze(-1)
        rss = (S4[k] - S4[t0]) - np.einsum("qp,qp->q", beta, XtY)
        sigma2 = np.where(valid & (dof > 0), rss / np.maximum(dof, 1), np.nan)

        xtx_inv00 = np.full(len(t0), np.nan)
        xtx_inv00[valid] = np.linalg.inv(XtX[valid])[:, 0, 0]
        var_beta_lag = sigma2 * xtx_inv00
        valid &= np.isfinite(var_beta_lag) & (var_beta_lag > 0)
        if not valid.any():
            continue

        stat = beta[:, 0][valid] / np.sqrt(var_beta_lag[valid])
        best = stat.max()
        out[t] = best if np.isfinite(best) else np.nan

    return pd.Series(out, index=series.index, name="sadf")


def get_sadf(
    series: pd.Series,
    model: str,
    lags: Union[int, list],
    min_length: int,
    add_const: bool = False,
    phi: float = 0,
    num_threads: int = 8,
    verbose: bool = True,
) -> pd.Series:
    """
    Multithread implementation of SADF

    SADF fits the ADF regression at each end point t with backwards expanding start points. For the estimation
    of SADF(t), the right side of the window is fixed at t. SADF recursively expands the beginning of the sample
    up to t - min_length, and returns the sup of this set.

    When doing with sub- or super-martingale test, the variance of beta of a weak long-run bubble may be smaller than
    one of a strong short-run bubble, hence biasing the method towards long-run bubbles. To correct for this bias,
    ADF statistic in samples with large lengths can be penalized with the coefficient phi in [0, 1] such that:

    ADF_penalized = ADF / (sample_length ^ phi)

    Parameters
    ----------
    series : pd.Series
        Series for which SADF statistics are generated.
    model : str
        Either 'native', 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'.
    lags : int or list
        Either number of lags to use or array of specified lags.
    min_length : int
        Minimum number of observations needed for estimation.
    add_const : bool
        Flag to add constant.
    phi : float
        Coefficient to penalize large sample lengths when computing SMT, in [0, 1].
    num_threads : int
        Number of cores to use.
    verbose : bool
        Flag to report progress on asynch jobs.

    Returns
    -------
    pd.Series
        SADF statistics

    Notes
    -----
        Advances in Financial Machine Learning, p. 258-259.
    """
    if model == "native":
        if not isinstance(lags, int):
            raise ValueError("model='native' only supports a single int for `lags` (contiguous 1..lags).")
        return _sadf_native_fast(series, lags=lags, min_length=min_length)

    X, y = _get_y_x(series, model, lags, add_const)
    molecule = y.index[min_length : y.shape[0]]

    sadf_series = mp_pandas_obj(
        func=_sadf_outer_loop,
        pd_obj=("molecule", molecule),
        X=X,
        y=y,
        min_length=min_length,
        model=model,
        phi=phi,
        num_threads=num_threads,
        verbose=verbose,
    )
    return sadf_series
