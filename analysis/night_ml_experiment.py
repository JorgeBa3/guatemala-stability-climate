"""
Experiment: can a learned model reconstruct night-time temperature and
humidity at stations that only report during the day better than simpler
rules?

Evaluated where the truth is known. At each station with night observations,
the night hours (01-11 UTC, 19:00-05:00 local) are hidden and reconstructed by
four methods:

  monthly   reanalysis + mean daytime bias of the calendar month (the method
            used in the first version of the paper)
  naive     straight line between the 18:00 and 06:00 observations
  anchored  reanalysis curve shifted so that it passes through the 18:00 and
            06:00 observations (bias interpolated linearly through the night)
  ml        anchored + a gradient-boosting correction, trained ONLY on the
            other stations (leave-one-station-out)

Run from the project folder:  python analysis/night_ml_experiment.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import climate_kinetics as ck

OUT = ck.OUT
NIGHT = list(range(1, 12))            # UTC hours without daytime reports
TEST_STATIONS = ["78641", "78627", "78637", "MGMM0", "78647"]
COMBOS = [(83.144, 0.0), (83.144, 0.04), (83.144, 0.08),
          (120.0, 0.0), (120.0, 0.04), (120.0, 0.08)]
FEATURES = ["h", "T_pow", "RH_pow", "rT_e", "rT_m", "rRH_e", "rRH_m",
            "T_e", "T_m", "RH_e", "RH_m", "dTpow_e", "dTpow_m",
            "doy_sin", "doy_cos", "Tmax_prev", "RHmin_prev"]


def dewpoint(T, RH):
    """Magnus formula; T in C, RH in %."""
    g = np.log(np.clip(RH, 1, 100) / 100.0) + 17.62 * T / (243.12 + T)
    return 243.12 * g / (17.62 - g)


def rel_humidity(T, Td):
    es = lambda x: np.exp(17.62 * x / (243.12 + x))
    return np.clip(100.0 * es(np.minimum(Td, T)) / es(T), 5, 100)


def station_frame(sid):
    obs, pw = ck.load_meteostat(sid), ck.load_power(sid)
    return pw.join(obs, how="left")


def night_table(df):
    """One row per (UTC date, night hour) with anchors at 00 and 12 UTC."""
    d = df.copy()
    d["date"] = d.index.normalize()
    d["h"] = d.index.hour
    e = d[d.h == 0].set_index("date")[["T_obs", "RH_obs", "T_pow", "RH_pow"]].add_suffix("_e0")
    m = d[d.h == 12].set_index("date")[["T_obs", "RH_obs", "T_pow", "RH_pow"]].add_suffix("_m0")
    day = d[(d.h >= 12)].copy()            # 06:00-17:00 local of the same UTC date
    day["next"] = day["date"] + pd.Timedelta(days=1)
    prev = day.groupby("next").agg(Tmax_prev=("T_obs", "max"), RHmin_prev=("RH_obs", "min"))
    n = d[d.h.isin(NIGHT)].join(e, on="date").join(m, on="date").join(prev, on="date")
    n = n.dropna(subset=["T_pow", "RH_pow", "T_obs_e0", "T_obs_m0", "RH_obs_e0", "RH_obs_m0",
                         "T_pow_e0", "T_pow_m0"])
    f = n.h / 12.0
    n["rT_e"], n["rT_m"] = n.T_obs_e0 - n.T_pow_e0, n.T_obs_m0 - n.T_pow_m0
    n["rRH_e"], n["rRH_m"] = n.RH_obs_e0 - n.RH_pow_e0, n.RH_obs_m0 - n.RH_pow_m0
    n["T_anch"] = n.T_pow + n.rT_e + f * (n.rT_m - n.rT_e)
    n["RH_anch"] = (n.RH_pow + n.rRH_e + f * (n.rRH_m - n.rRH_e)).clip(5, 100)
    # same anchoring done on dew point, with humidity derived afterwards
    td_pow = dewpoint(n.T_pow, n.RH_pow)
    rTd_e = dewpoint(n.T_obs_e0, n.RH_obs_e0) - dewpoint(n.T_pow_e0, n.RH_pow_e0)
    rTd_m = dewpoint(n.T_obs_m0, n.RH_obs_m0) - dewpoint(n.T_pow_m0, n.RH_pow_m0)
    n["Td_anch"] = td_pow + rTd_e + f * (rTd_m - rTd_e)
    n["Td_obs"] = dewpoint(n.T_obs, n.RH_obs)
    n["T_anchtd"] = n.T_anch
    n["RH_anchtd"] = rel_humidity(n.T_anch, n.Td_anch)
    n["T_naive"] = n.T_obs_e0 + f * (n.T_obs_m0 - n.T_obs_e0)
    n["RH_naive"] = n.RH_obs_e0 + f * (n.RH_obs_m0 - n.RH_obs_e0)
    n["T_e"], n["T_m"], n["RH_e"], n["RH_m"] = n.T_obs_e0, n.T_obs_m0, n.RH_obs_e0, n.RH_obs_m0
    n["dTpow_e"], n["dTpow_m"] = n.T_pow - n.T_pow_e0, n.T_pow_m0 - n.T_pow
    doy = n.index.dayofyear
    n["doy_sin"], n["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    return n


def fit_models(train, seed=0):
    tr = train.dropna(subset=["T_obs", "RH_obs"])
    kw = dict(max_iter=300, learning_rate=0.05, max_depth=6, min_samples_leaf=50,
              l2_regularization=1.0, random_state=seed)
    mT = HistGradientBoostingRegressor(**kw).fit(tr[FEATURES], tr.T_obs - tr.T_anch)
    mRH = HistGradientBoostingRegressor(**kw).fit(tr[FEATURES], tr.RH_obs - tr.RH_anch)
    mTd = HistGradientBoostingRegressor(**kw).fit(tr[FEATURES], tr.Td_obs - tr.Td_anch)
    return mT, mRH, mTd


def predict(models, n):
    mT, mRH, mTd = models
    T = n.T_anch + mT.predict(n[FEATURES])
    RH = (n.RH_anch + mRH.predict(n[FEATURES])).clip(5, 100)
    RH_td = rel_humidity(T, n.Td_anch + mTd.predict(n[FEATURES]))
    return T, RH, RH_td


def assemble(df, night, T_col, RH_col):
    """Daytime observations + reconstructed nights. Hours a method does not
    cover (a missing anchor, daytime gaps) are filled as in the main pipeline,
    so that only the treatment of anchored nights differs between methods."""
    has = df.T_obs.notna() & df.RH_obs.notna()
    day_mask = has & pd.Series(pd.Index(df.index.hour).isin(ck.DAYTIME_UTC), index=df.index)
    base = ck.hybrid(df, day_mask)                      # monthly method everywhere
    out = ck.fill_gaps(df, day_mask)                    # as in the main pipeline
    idx = night.index.intersection(out.index)
    out.loc[idx, "T"] = night.loc[idx, T_col]
    out.loc[idx, "RH"] = night.loc[idx, RH_col]
    return out, base


def af(series, ea, b):
    return ck.composite(*ck.cell_tables(series, ck.rate_term(series, ea, b)))


def main():
    frames = {s: station_frame(s) for s in TEST_STATIONS}
    nights = {s: night_table(frames[s]) for s in TEST_STATIONS}
    hourly, afrows = [], []
    for held in TEST_STATIONS:
        train = pd.concat([nights[s] for s in TEST_STATIONS if s != held])
        models = fit_models(train)
        n = nights[held].copy()
        n["T_ml"], n["RH_ml"], n["RH_mltd"] = predict(models, n)
        n["T_mltd"] = n["T_ml"]
        df = frames[held]
        truth = df.dropna(subset=["T_obs", "RH_obs"]).rename(
            columns={"T_obs": "T", "RH_obs": "RH"})[["T", "RH"]]
        series = {}
        for name in ("naive", "anch", "anchtd", "ml", "mltd"):
            series[name], base = assemble(df, n, f"T_{name}", f"RH_{name}")
        series["monthly"] = base
        # hourly error on night hours that were actually observed
        t = n.dropna(subset=["T_obs", "RH_obs"])
        mon = base.reindex(t.index)
        row = {"station": held, "n_night_obs": len(t),
               "T_mae_monthly": (mon["T"] - t.T_obs).abs().mean(),
               "RH_mae_monthly": (mon["RH"] - t.RH_obs).abs().mean()}
        for name in ("naive", "anch", "anchtd", "ml", "mltd"):
            row[f"T_mae_{name}"] = (t[f"T_{name}"] - t.T_obs).abs().mean()
            row[f"RH_mae_{name}"] = (t[f"RH_{name}"] - t.RH_obs).abs().mean()
        hourly.append(row)
        for ea, b in COMBOS:
            ref = af(truth, ea, b)
            r = {"station": held, "Ea": ea, "B": b, "AF_true": ref}
            for name, s in series.items():
                r[f"err_{name}_pct"] = 100 * (af(s, ea, b) / ref - 1) if np.isfinite(ref) else np.nan
            afrows.append(r)

    H = pd.DataFrame(hourly).round(2)
    A = pd.DataFrame(afrows).round(2)
    OUT.mkdir(exist_ok=True)
    H.to_csv(OUT / "night_reconstruction_hourly_error.csv", index=False)
    A.to_csv(OUT / "night_reconstruction_af_error.csv", index=False)
    pd.set_option("display.width", 220)
    print(H.to_string(index=False))
    print()
    print(A.to_string(index=False))
    e = A[[c for c in A.columns if c.startswith("err_")]].abs()
    print("\nabsolute AF error (%), all stations and sensitivities")
    print(e.agg(["mean", "median", "max"]).round(1).to_string())
    print("\nby station (mean absolute AF error, %)")
    print(A.assign(**{c: A[c].abs() for c in e.columns}).groupby("station")[list(e.columns)].mean().round(1).to_string())


if __name__ == "__main__":
    main()
