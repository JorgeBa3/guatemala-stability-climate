"""
How much faster would a medicine degrade in Guatemalan climates than at the
reference conditions used for stability testing?

For a product whose degradation rate follows the humidity-corrected Arrhenius
equation,  ln k = ln A - Ea/(R T) + B*RH,  the rate averaged over a climate
record, relative to the rate at a constant reference condition, is

    AF = mean_t exp( -Ea/R * (1/T_t - 1/T_ref) + B * (RH_t - RH_ref) )

AF > 1 means the location is harsher than the reference condition; the time to
reach any fixed degradation level is divided by AF. ln A cancels, so AF depends
only on the two sensitivities (Ea, B) and on the climate.

Inputs  : data/climate/ (written by download_climate.py)
Outputs : results/station_coverage.csv
          results/climate_summary.csv
          results/acceleration_factors.csv
          results/hybrid_validation.csv

Run from the project folder:  python analysis/climate_kinetics.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CLIM = ROOT / "data" / "climate"
OUT = ROOT / "results"

R = 8.314462618e-3                       # kJ/(mol K)
YEARS = list(range(2015, 2026))
OBSERVED = {"isd_lite", "metar", "synop"}  # forecast-model rows are discarded

EA_GRID = [60, 70, 80, 83.144, 90, 100, 110, 120, 130, 140, 150]   # kJ/mol
B_GRID = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]                        # per %RH
REFS = {"25C/60%RH": (25.0, 60.0),      # zone II
        "30C/65%RH": (30.0, 65.0),      # zone IVa, listed for Guatemala
        "30C/75%RH": (30.0, 75.0)}      # zone IVb

MIN_PER_CELL = 30        # observations needed in each (month, hour) cell
DAYTIME_UTC = set(range(12, 24)) | {0}   # hours reported by daytime-only stations
DAYS_IN_MONTH = np.array([31, 28.25, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
N_BOOT = 1000


# ───────────────────────────── loading ─────────────────────────────

def load_meteostat(sid: str) -> pd.DataFrame:
    frames = []
    for year in YEARS:
        f = CLIM / "meteostat" / f"{sid}_{year}.csv.gz"
        if f.exists():
            frames.append(pd.read_csv(f, usecols=[
                "year", "month", "day", "hour", "temp", "temp_source",
                "rhum", "rhum_source"]))
    if not frames:
        return pd.DataFrame(columns=["T_obs", "RH_obs"])
    d = pd.concat(frames)
    ok = (d["temp_source"].isin(OBSERVED) & d["rhum_source"].isin(OBSERVED)
          & d["temp"].between(-10, 50) & d["rhum"].between(5, 100))
    d = d[ok]
    idx = pd.to_datetime(d[["year", "month", "day", "hour"]])
    out = pd.DataFrame({"T_obs": d["temp"].to_numpy(float),
                        "RH_obs": d["rhum"].to_numpy(float)}, index=idx)
    return out[~out.index.duplicated()]


def load_power(sid: str) -> pd.DataFrame:
    frames = []
    for year in YEARS:
        f = CLIM / "nasa_power" / f"{sid}_{year}.csv"
        if not f.exists():
            continue
        with open(f, encoding="utf-8") as fh:
            skip = next(i for i, line in enumerate(fh) if line.startswith("-END HEADER-")) + 1
        frames.append(pd.read_csv(f, skiprows=skip))
    if not frames:
        return pd.DataFrame(columns=["T_pow", "RH_pow"])
    d = pd.concat(frames)
    d = d[(d["T2M"] > -900) & (d["RH2M"] > -900)]
    idx = pd.to_datetime(d.rename(columns={"YEAR": "year", "MO": "month",
                                           "DY": "day", "HR": "hour"})
                         [["year", "month", "day", "hour"]])
    return pd.DataFrame({"T_pow": d["T2M"].to_numpy(float),
                         "RH_pow": d["RH2M"].to_numpy(float)}, index=idx)


def hybrid(df: pd.DataFrame, obs_mask: pd.Series) -> pd.DataFrame:
    """Observed values where available; elsewhere the reanalysis shifted by the
    mean observed-minus-reanalysis difference of that calendar month. Used
    only as the fallback for long gaps; see `fill_gaps`."""
    both = df[obs_mask & df["T_pow"].notna()]
    dT = (both["T_obs"] - both["T_pow"]).groupby(both.index.month).mean()
    dRH = (both["RH_obs"] - both["RH_pow"]).groupby(both.index.month).mean()
    month = df.index.month
    T = df["T_pow"] + month.map(dT).to_numpy()
    RH = (df["RH_pow"] + month.map(dRH).to_numpy()).clip(5, 100)
    T = T.where(~obs_mask, df["T_obs"])
    RH = RH.where(~obs_mask, df["RH_obs"])
    return pd.DataFrame({"T": T, "RH": RH}).dropna()


MAX_GAP_HOURS = 13   # longest run of missing hours bridged by interpolation


def fill_gaps(df: pd.DataFrame, obs_mask: pd.Series) -> pd.DataFrame:
    """Observed values where available. In gaps of up to MAX_GAP_HOURS the
    reanalysis curve is shifted so that it passes through the observations on
    both sides of the gap: the observed-minus-reanalysis difference is
    interpolated linearly in time across the gap. Longer gaps fall back to the
    monthly mean difference (function `hybrid`)."""
    base = hybrid(df, obs_mask)
    out = base.copy()
    gap_id = obs_mask.cumsum()
    gap_len = (~obs_mask).groupby(gap_id).transform("sum")
    short = (~obs_mask) & (gap_len <= MAX_GAP_HOURS) & (gap_id > 0) \
        & (gap_id < gap_id.iloc[-1])
    for obs_col, pow_col, name in (("T_obs", "T_pow", "T"), ("RH_obs", "RH_pow", "RH")):
        resid = (df[obs_col] - df[pow_col]).where(obs_mask)
        resid = resid.interpolate(method="linear", limit_area="inside")
        value = df[pow_col] + resid
        ok = short & value.notna()
        idx = ok[ok].index.intersection(out.index)
        out.loc[idx, name] = value.loc[idx]
    out["RH"] = out["RH"].clip(5, 100)
    return out


# ─────────────────────────── calculation ───────────────────────────

def cell_tables(series: pd.DataFrame, values: np.ndarray):
    """Sum and count of `values` per (year, month, hour) -> arrays [year, 288]."""
    yi = series.index.year.to_numpy() - YEARS[0]
    ci = (series.index.month.to_numpy() - 1) * 24 + series.index.hour.to_numpy()
    sums = np.zeros((len(YEARS), 288))
    cnts = np.zeros((len(YEARS), 288))
    np.add.at(sums, (yi, ci), values)
    np.add.at(cnts, (yi, ci), 1.0)
    return sums, cnts


WEIGHTS = np.repeat(DAYS_IN_MONTH / DAYS_IN_MONTH.sum() / 24.0, 24)


def composite(sums, cnts):
    """Mean of the 288 (month, hour) cell means, weighted by month length, so
    that hours and seasons with more observations do not count more."""
    s, c = sums.sum(axis=0), cnts.sum(axis=0)
    if (c < MIN_PER_CELL).any():
        return np.nan
    return float(np.sum(WEIGHTS * s / c))


def bootstrap(sums, cnts, rng):
    out = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(YEARS), len(YEARS))
        s, c = sums[pick].sum(axis=0), cnts[pick].sum(axis=0)
        if (c == 0).any():
            continue
        out.append(np.sum(WEIGHTS * s / c))
    return (np.percentile(out, 2.5), np.percentile(out, 97.5)) if out else (np.nan, np.nan)


def rate_term(series, ea, b, t_ref=25.0, rh_ref=60.0):
    T = series["T"].to_numpy() + 273.15
    return np.exp(-ea / R * (1.0 / T - 1.0 / (t_ref + 273.15))
                  + b * (series["RH"].to_numpy() - rh_ref))


def ref_shift(ea, b, t_ref, rh_ref):
    """Rate at the reference condition relative to the rate at 25C/60%RH.
    Dividing an AF computed against 25C/60%RH by this gives the AF against
    the reference."""
    return np.exp(-ea / R * (1 / (t_ref + 273.15) - 1 / 298.15) + b * (rh_ref - 60.0))


def self_check():
    """A climate that is constant at a reference condition must give AF = 1
    against that reference, and a hotter reference must give a smaller AF."""
    idx = pd.date_range("2015-01-01", "2025-12-31 23:00", freq="h")
    for ref, (t_ref, rh_ref) in REFS.items():
        const = pd.DataFrame({"T": t_ref, "RH": rh_ref}, index=idx)
        for ea in (60, 150):
            for b in (0.0, 0.10):
                af = composite(*cell_tables(const, rate_term(const, ea, b)))
                af /= ref_shift(ea, b, t_ref, rh_ref)
                assert abs(af - 1) < 1e-9, (ref, ea, b, af)
    assert ref_shift(83.144, 0.0, 30, 60) > 1 > ref_shift(83.144, 0.0, 20, 60)
    assert ref_shift(83.144, 0.05, 25, 75) > 1


def acceleration_table(series, label, method, rng):
    rows = []
    for ea in EA_GRID:
        for b in B_GRID:
            sums, cnts = cell_tables(series, rate_term(series, ea, b))
            af = composite(sums, cnts)
            lo, hi = bootstrap(sums, cnts, rng) if np.isfinite(af) else (np.nan, np.nan)
            for ref, (t_ref, rh_ref) in REFS.items():
                f = ref_shift(ea, b, t_ref, rh_ref)
                rows.append({"station": label, "method": method, "Ea_kJ_mol": ea,
                             "B_per_pctRH": b, "reference": ref,
                             "AF": af / f, "AF_lo": lo / f, "AF_hi": hi / f})
    return rows


def climate_summary(series):
    def comp(v):
        return composite(*cell_tables(series, v))
    mean_T = comp(series["T"].to_numpy())
    mean_RH = comp(series["RH"].to_numpy())
    ea = 83.144
    m = comp(np.exp(-ea / R / (series["T"].to_numpy() + 273.15)))
    mkt = -ea / R / np.log(m) - 273.15 if np.isfinite(m) else np.nan
    return mean_T, mean_RH, mkt


# ────────────────────────────── main ──────────────────────────────

def main():
    self_check()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261004)
    stations = pd.read_csv(CLIM / "stations.csv", dtype={"station_id": str})
    coverage, summary, af_rows, valid = [], [], [], []

    for st in stations.itertuples():
        obs, pow_ = load_meteostat(st.station_id), load_power(st.station_id)
        if pow_.empty:   # same coordinates as another station share one file
            twin = stations[(stations.latitude == st.latitude)
                            & (stations.longitude == st.longitude)].station_id.iloc[0]
            pow_ = load_power(twin)
        df = pow_.join(obs, how="outer")
        has_obs = df["T_obs"].notna() & df["RH_obs"].notna()
        hours = df.index.hour
        n_total = len(pd.date_range(f"{YEARS[0]}-01-01", f"{YEARS[-1]}-12-31 23:00", freq="h"))
        night = ~pd.Index(hours).isin(DAYTIME_UTC)
        obs_series = df.loc[has_obs, ["T_obs", "RH_obs"]].rename(
            columns={"T_obs": "T", "RH_obs": "RH"})
        _, cnts = cell_tables(obs_series, np.ones(len(obs_series)))
        full_day = bool((cnts.sum(axis=0) >= MIN_PER_CELL).all())
        coverage.append({
            "station_id": st.station_id, "name": st.name, "elevation_m": st.elevation_m,
            "observed_hours": int(has_obs.sum()),
            "pct_of_all_hours": round(100 * has_obs.sum() / n_total, 1),
            "pct_of_night_hours": round(100 * (has_obs & night).sum() / max(night.sum(), 1), 1),
            "all_month_hour_cells_filled": full_day})

        usable = {}
        if full_day:
            usable["station observations"] = obs_series
        if has_obs.sum() > 20000 and not full_day:
            usable["observations + anchored reanalysis at night"] = fill_gaps(df, has_obs)
        power_only = pow_.rename(columns={"T_pow": "T", "RH_pow": "RH"}).dropna()

        for method, series in usable.items():
            mean_T, mean_RH, mkt = climate_summary(series)
            summary.append({"station": st.name, "elevation_m": st.elevation_m,
                            "method": method, "mean_T_C": mean_T,
                            "mean_RH_pct": mean_RH, "MKT_C": mkt})
            af_rows += acceleration_table(series, st.name, method, rng)

        # Validation of the night-filling method where the truth is known:
        # hide the night observations of a full-coverage station and compare.
        if full_day:
            masked = has_obs & pd.Series(pd.Index(hours).isin(DAYTIME_UTC), index=df.index)
            filled = fill_gaps(df, masked)
            monthly = hybrid(df, masked)
            for ea in (83.144, 120):
                for b in (0.0, 0.04, 0.08):
                    truth = composite(*cell_tables(obs_series, rate_term(obs_series, ea, b)))
                    hyb = composite(*cell_tables(filled, rate_term(filled, ea, b)))
                    mon = composite(*cell_tables(monthly, rate_term(monthly, ea, b)))
                    raw = composite(*cell_tables(power_only, rate_term(power_only, ea, b)))
                    day = obs_series[pd.Index(obs_series.index.hour).isin(DAYTIME_UTC)]
                    day_only = float(np.mean(rate_term(day, ea, b)))
                    valid.append({"station": st.name, "Ea_kJ_mol": ea, "B_per_pctRH": b,
                                  "AF_observed": truth, "AF_night_filled": hyb,
                                  "AF_monthly_bias": mon,
                                  "err_monthly_bias_pct": 100 * (mon / truth - 1),
                                  "AF_reanalysis_only": raw, "AF_daytime_only": day_only,
                                  "err_night_filled_pct": 100 * (hyb / truth - 1),
                                  "err_reanalysis_only_pct": 100 * (raw / truth - 1),
                                  "err_daytime_only_pct": 100 * (day_only / truth - 1)})

    pd.DataFrame(coverage).to_csv(OUT / "station_coverage.csv", index=False)
    pd.DataFrame(summary).round(2).to_csv(OUT / "climate_summary.csv", index=False)
    pd.DataFrame(af_rows).round(4).to_csv(OUT / "acceleration_factors.csv", index=False)
    pd.DataFrame(valid).round(4).to_csv(OUT / "hybrid_validation.csv", index=False)
    print(pd.DataFrame(coverage).to_string(index=False))
    print()
    print(pd.DataFrame(summary).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
