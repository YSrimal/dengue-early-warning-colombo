"""
Shared feature logic for the Colombo dengue early-warning model.
Imported by the notebook, the pickled pipeline and the web app so that
training and inference use *exactly* the same transformations.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

HORIZON = 2   # we predict dengue risk 2 weeks ahead (actionable lead time)

# ---- the 11 fields a user (or the web form) supplies for "this week" ------
RAW_FIELDS = [
    "cases_t",      # notified dengue cases this week
    "cases_t1",     # ... 1 week ago
    "cases_t2",     # ... 2 weeks ago
    "cases_t3",     # ... 3 weeks ago
    "epi_week",     # epidemiological week number (1-53)
    "rain_4wk",     # mean weekly precipitation (mm) over the last 4 weeks
    "rain_8wk",     # mean weekly precipitation (mm) over the last 8 weeks
    "humidity_4wk", # mean relative humidity (%) over the last 4 weeks
    "temp_4wk",     # mean temperature (C) over the last 4 weeks
    "wind_4wk",     # mean wind speed (km/h) over the last 4 weeks
    "visib_4wk",    # mean visibility (km) over the last 4 weeks
]
CLIMATE_FIELDS = ["rain_4wk", "rain_8wk", "humidity_4wk", "temp_4wk", "wind_4wk", "visib_4wk"]


def add_derived(X):
    """Derived features. Works on a DataFrame with RAW_FIELDS columns."""
    X = pd.DataFrame(X).copy()
    X["cases_mean4"] = X[["cases_t", "cases_t1", "cases_t2", "cases_t3"]].mean(axis=1)
    X["cases_growth"] = (X["cases_t"] + 1) / (X["cases_t1"] + 1)          # week-on-week ratio
    X["cases_trend3"] = X["cases_t"] - X["cases_t3"]                       # 3-week change
    # cyclic encoding: week 53 is next to week 1 (monsoon seasonality)
    X["week_sin"] = np.sin(2 * np.pi * X["epi_week"] / 52.0)
    X["week_cos"] = np.cos(2 * np.pi * X["epi_week"] / 52.0)
    return X.drop(columns=["epi_week"])


def load_clean_weekly(path):
    """Read the raw Ministry-of-Health CSV, fix data-quality problems and
    return a *continuous* weekly table (missing weeks re-inserted & imputed)."""
    df = pd.read_csv(path)
    df["SLP"] = pd.to_numeric(df["SLP"], errors="coerce")        # '#DIV/0!' -> NaN
    df["date"] = pd.to_datetime(
        df["Year"].astype(str) + "-W" + df["Week"].astype(str).str.zfill(2) + "-1",
        format="%G-W%V-%u")
    df = df.sort_values("date").set_index("date")
    full = pd.date_range(df.index.min(), df.index.max(), freq="7D")
    df = df.reindex(full)
    df.index.name = "date"
    # Weeks with exactly 0 notified cases (5 weeks in 2012, e.g. 38 -> 0 -> 0 -> 64) are
    # implausible for Colombo and are treated as un-reported weeks, not true zeros.
    df.loc[df["Cases"] == 0, "Cases"] = np.nan
    df["was_missing_week"] = df["Cases"].isna()
    iso = df.index.isocalendar()
    df["Year"], df["Week"] = iso["year"].astype(int).values, iso["week"].astype(int).values
    num = ["Cases", "TEM", "TMAX", "Tm", "SLP", "H", "PP", "VV", "V", "VM"]
    df[num] = df[num].interpolate(method="time", limit_direction="both")
    return df


def build_supervised(weekly, horizon=HORIZON):
    """Turn the weekly series into rows of (inputs at week t) -> (cases at week t+horizon)."""
    w = weekly
    X = pd.DataFrame(index=w.index)
    X["cases_t"] = w["Cases"]
    X["cases_t1"] = w["Cases"].shift(1)
    X["cases_t2"] = w["Cases"].shift(2)
    X["cases_t3"] = w["Cases"].shift(3)
    X["epi_week"] = w["Week"]
    X["rain_4wk"] = w["PP"].rolling(4).mean()
    X["rain_8wk"] = w["PP"].rolling(8).mean()
    X["humidity_4wk"] = w["H"].rolling(4).mean()
    X["temp_4wk"] = w["TEM"].rolling(4).mean()
    X["wind_4wk"] = w["V"].rolling(4).mean()
    X["visib_4wk"] = w["VV"].rolling(4).mean()
    y_cases_next = w["Cases"].shift(-horizon)
    ok = X.notna().all(axis=1) & y_cases_next.notna()
    return X[ok][RAW_FIELDS], y_cases_next[ok]


class TimeAwareStacking(ClassifierMixin, BaseEstimator):
    """Stacking that respects time order.

    Out-of-fold base-model probabilities are produced with an expanding-window
    TimeSeriesSplit (the past never sees the future), a logistic-regression
    meta-learner is trained on them, and the base models are finally refit on
    all training data.  sklearn's StackingClassifier cannot do this because
    TimeSeriesSplit is not a partition of the data."""

    def __init__(self, estimators, n_splits=5, gap=HORIZON, meta_C=1.0):
        self.estimators = estimators      # list of (name, estimator)
        self.n_splits = n_splits
        self.gap = gap
        self.meta_C = meta_C

    @staticmethod
    def _rows(X, idx):
        return X.iloc[idx] if hasattr(X, "iloc") else X[idx]

    def fit(self, X, y):
        y = np.asarray(y)
        n, k = len(y), len(self.estimators)
        oof = np.full((n, k), np.nan)
        for tr, va in TimeSeriesSplit(self.n_splits, gap=self.gap).split(np.arange(n)):
            if len(np.unique(y[tr])) < 2:      # early folds can contain a single class
                continue
            for j, (_, est) in enumerate(self.estimators):
                m = clone(est).fit(self._rows(X, tr), y[tr])
                oof[va, j] = m.predict_proba(self._rows(X, va))[:, 1]
        ok = ~np.isnan(oof).any(axis=1)
        if ok.sum() >= 10 and len(np.unique(y[ok])) == 2:
            self.meta_ = LogisticRegression(C=self.meta_C, class_weight="balanced", max_iter=1000).fit(oof[ok], y[ok])
        else:                                   # too little data for a meta-learner -> plain average
            self.meta_ = None
        self.fitted_ = [clone(est).fit(X, y) for _, est in self.estimators]
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        P = np.column_stack([m.predict_proba(X)[:, 1] for m in self.fitted_])
        if self.meta_ is None:
            p = P.mean(axis=1)
            return np.column_stack([1 - p, p])
        return self.meta_.predict_proba(P)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
