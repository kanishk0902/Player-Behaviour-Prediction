"""
Model code for the "Predicting Batter Aggression in the IPL" mini project.

This is the notebook (miniproject.ipynb) turned into reusable functions:
  load -> sort -> leak-free features -> chronological split -> Logistic Regression
  -> threshold tuning -> evaluation -> predict_ball() for the app.

Nothing here imports Streamlit, so it can also be run from a script or notebook.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRAIN_END_YEAR = 2021
BEST_C = 0.01                      # chosen by 5-fold GroupKFold in the notebook (Table 3.7)
K_BAT, K_BOWL, K_VEN, K_PH, K_ATT = 30, 60, 150, 40, 300   # smoothing strengths

FEATURES = ['chase', 'phase_MID', 'phase_DEATH', 'inn_wkts_before', 'crr', 'rrr', 'team_last6', 'last_ball',
            'bat_balls_log', 'new_bat', 'bat_sr_inn',
            'bat_hist_bound', 'bat_hist_dot', 'bat_phase_bound', 'bowl_hist_bound', 'bowl_hist_dot',
            'venue_bound', 'bat_exp', 'season']
TARGET = 'is_bound'

FEATURE_LABELS = {
    'chase': 'Chasing (2nd innings)', 'phase_MID': 'Middle overs (7-15)', 'phase_DEATH': 'Death overs (16-20)',
    'inn_wkts_before': 'Wickets lost', 'crr': 'Current run rate', 'rrr': 'Required run rate',
    'team_last6': 'Runs in last 6 balls', 'last_ball': 'Last ball of the over',
    'bat_balls_log': 'Balls faced (settling in)', 'new_bat': 'Just arrived at crease',
    'bat_sr_inn': 'Batter strike rate this innings', 'bat_hist_bound': 'Batter career boundary rate',
    'bat_hist_dot': 'Batter career dot-ball rate', 'bat_phase_bound': 'Batter boundary rate in this phase',
    'bowl_hist_bound': 'Bowler boundary rate conceded', 'bowl_hist_dot': 'Bowler dot-ball rate',
    'venue_bound': 'Venue boundary rate', 'bat_exp': 'Batter experience (log balls)',
    'season': 'Season trend',
}

REQUIRED_COLUMNS = ['Date', 'Match ID', 'Innings', 'Over', 'Ball', 'Batter', 'Bowler', 'Venue',
                    'Batter Runs', 'Runs From Ball', 'Wicket', 'Valid Ball', 'Target Score',
                    'Bat First', 'Bat Second']


def _pipe(C=BEST_C, **kw):
    return Pipeline([('scale', StandardScaler()),
                     ('lr', LogisticRegression(C=C, max_iter=2000, **kw))])


# --------------------------------------------------------------------------------------
# 1. Data preparation (notebook cells 2-7)
# --------------------------------------------------------------------------------------
def prepare(df: pd.DataFrame):
    """Return (d, globals) where d holds one row per legal delivery with all 19 features."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"The CSV is missing columns: {missing}")

    df = df.copy()
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'row_id'})
    elif 'row_id' not in df.columns:
        df['row_id'] = np.arange(len(df))
    df['Date'] = pd.to_datetime(df['Date'])

    # chronological order + target
    df = df.sort_values(['Date', 'Match ID', 'Innings', 'row_id']).reset_index(drop=True)
    df['is_bound'] = df['Batter Runs'].isin([4, 6]).astype(int)
    df['is_six'] = (df['Batter Runs'] == 6).astype(int)
    df['is_dot_bat'] = (df['Batter Runs'] == 0).astype(int)

    # leak-free match state = running total minus the current ball
    gi = df.groupby(['Match ID', 'Innings'])
    df['inn_runs_before'] = gi['Runs From Ball'].cumsum() - df['Runs From Ball']
    df['inn_wkts_before'] = gi['Wicket'].cumsum() - df['Wicket']
    df['legal_before'] = gi['Valid Ball'].cumsum() - df['Valid Ball']
    df['team_last6'] = (gi['Runs From Ball'].transform(lambda s: s.shift(1).rolling(6, min_periods=1).sum())
                        .fillna(0))
    gb = df.groupby(['Match ID', 'Innings', 'Batter'])
    df['bat_runs_before'] = gb['Batter Runs'].cumsum() - df['Batter Runs']
    df['bat_balls_before'] = gb['Valid Ball'].cumsum() - df['Valid Ball']

    d = df[df['Valid Ball'] == 1].copy()
    d['year'] = d['Date'].dt.year

    global_b = d.loc[d.year <= TRAIN_END_YEAR, 'is_bound'].mean()
    global_d = d.loc[d.year <= TRAIN_END_YEAR, 'is_dot_bat'].mean()
    d['phase'] = pd.cut(d['Over'], [0, 6, 15, 20], labels=['PP', 'MID', 'DEATH']).astype(str)

    def prior_stats(data, keys, prefix):
        """Per-match totals, then cumulative totals from PREVIOUS matches only."""
        m = (data.groupby(keys + ['Match ID', 'Date'])
                 .agg(n=('Valid Ball', 'sum'), b=('is_bound', 'sum'), z=('is_dot_bat', 'sum'))
                 .reset_index().sort_values(keys + ['Date', 'Match ID']))
        for c in ['n', 'b', 'z']:
            m[f'{prefix}_{c}'] = m.groupby(keys)[c].cumsum() - m[c]
        return m[keys + ['Match ID', f'{prefix}_n', f'{prefix}_b', f'{prefix}_z']]

    d = d.merge(prior_stats(d, ['Batter'], 'bat'), on=['Batter', 'Match ID'], how='left')
    d = d.merge(prior_stats(d, ['Bowler'], 'bowl'), on=['Bowler', 'Match ID'], how='left')
    d = d.merge(prior_stats(d, ['Batter', 'phase'], 'batph'), on=['Batter', 'phase', 'Match ID'], how='left')
    d = d.merge(prior_stats(d, ['Venue'], 'ven'), on=['Venue', 'Match ID'], how='left')

    d['bat_hist_bound'] = (d.bat_b + K_BAT * global_b) / (d.bat_n + K_BAT)
    d['bat_hist_dot'] = (d.bat_z + K_BAT * global_d) / (d.bat_n + K_BAT)
    d['bowl_hist_bound'] = (d.bowl_b + K_BOWL * global_b) / (d.bowl_n + K_BOWL)
    d['bowl_hist_dot'] = (d.bowl_z + K_BOWL * global_d) / (d.bowl_n + K_BOWL)
    d['bat_phase_bound'] = (d.batph_b + K_PH * d.bat_hist_bound) / (d.batph_n + K_PH)
    d['venue_bound'] = (d.ven_b + K_VEN * global_b) / (d.ven_n + K_VEN)
    d['bat_exp'] = np.log1p(d.bat_n)

    # match-situation features
    d['chase'] = (d['Innings'] == 2).astype(int)
    d['phase_MID'] = (d.phase == 'MID').astype(int)
    d['phase_DEATH'] = (d.phase == 'DEATH').astype(int)
    d['last_ball'] = (d['Ball'] == 6).astype(int)
    balls_left = (120 - d.legal_before).clip(lower=1)
    d['crr'] = np.where(d.legal_before > 0, d.inn_runs_before / d.legal_before.clip(lower=1) * 6, 0)
    runs_to_get = (d['Target Score'] - d.inn_runs_before).clip(lower=0)
    d['rrr'] = np.where(d.chase == 1, (runs_to_get / balls_left * 6).clip(upper=36), 0)
    d['bat_balls_log'] = np.log1p(d.bat_balls_before)
    d['new_bat'] = (d.bat_balls_before < 6).astype(int)
    d['bat_sr_inn'] = np.where(d.bat_balls_before > 0,
                               d.bat_runs_before / d.bat_balls_before.clip(lower=1) * 100, 0)
    d['season'] = d['year'] - 2008
    d[FEATURES] = d[FEATURES].fillna(0)
    return d, dict(GLOBAL_B=float(global_b), GLOBAL_D=float(global_d))


# --------------------------------------------------------------------------------------
# 2. Training + evaluation (notebook cells 15-25)
# --------------------------------------------------------------------------------------
def train_and_evaluate(d: pd.DataFrame, glob: dict) -> dict:
    train, test = d[d.year <= TRAIN_END_YEAR], d[d.year > TRAIN_END_YEAR]
    if len(test) == 0:
        raise ValueError("No deliveries after 2021 found - the test seasons are missing from the CSV.")
    X_tr, y_tr, g_tr = train[FEATURES], train[TARGET], train['Match ID']
    X_te, y_te = test[FEATURES], test[TARGET]

    model = _pipe().fit(X_tr, y_tr)
    six_model = _pipe().fit(X_tr, train['is_six'])
    bal_model = _pipe(class_weight='balanced').fit(X_tr, y_tr)

    # F1-optimal threshold from out-of-fold TRAINING predictions
    gkf = GroupKFold(n_splits=5)
    oof = cross_val_predict(_pipe(), X_tr, y_tr, cv=gkf, groups=g_tr, method='predict_proba')[:, 1]
    prec, rec, thr = precision_recall_curve(y_tr, oof)
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    best_thr = float(thr[np.argmax(f1s)])

    p_test = model.predict_proba(X_te)[:, 1]
    p_bal = bal_model.predict_proba(X_te)[:, 1]
    base_prob = np.full(len(y_te), y_tr.mean())

    variants = {'Default threshold (0.5)': (p_test, 0.5),
                'class_weight="balanced" (0.5)': (p_bal, 0.5),
                f'Tuned threshold ({best_thr:.3f})': (p_test, best_thr)}
    rows, cms = [], {}
    for name, (p, t) in variants.items():
        yhat = (p >= t).astype(int)
        rows.append({'Decision rule': name, 'Accuracy': accuracy_score(y_te, yhat),
                     'Precision': precision_score(y_te, yhat, zero_division=0),
                     'Recall': recall_score(y_te, yhat), 'F1': f1_score(y_te, yhat)})
        cms[name] = confusion_matrix(y_te, yhat)
    fpr, tpr, _ = roc_curve(y_te, p_test)
    frac_pos, mean_pred = calibration_curve(y_te, p_test, n_bins=10, strategy='quantile')

    coefs = pd.Series(model.named_steps['lr'].coef_[0], index=FEATURES)
    coef_df = pd.DataFrame({'coefficient': coefs, 'odds_ratio': np.exp(coefs)})

    tst = test.assign(pred=p_test)
    players = (tst.groupby('Batter').agg(balls=('Valid Ball', 'sum'), actual=(TARGET, 'mean'),
                                         predicted=('pred', 'mean')))
    players = players[players.balls >= 250].sort_values('actual', ascending=False)

    ll, ll_base = log_loss(y_te, p_test), log_loss(y_te, base_prob)
    return dict(
        model=model, six_model=six_model, best_thr=best_thr,
        f1_curve=(thr, f1s), metrics=pd.DataFrame(rows).set_index('Decision rule'), confusion=cms,
        roc=(fpr, tpr), calib=(mean_pred, frac_pos), coef_df=coef_df, players=players,
        auc=float(roc_auc_score(y_te, p_test)), logloss=float(ll), logloss_base=float(ll_base),
        base_acc=float(1 - y_te.mean()),
        n_train=len(train), n_test=len(test), rate_train=float(y_tr.mean()), rate_test=float(y_te.mean()),
        avg_six=float(train['is_six'].mean()), feat_mean=X_tr.mean(), feat_median=X_tr.median(),
    )


# --------------------------------------------------------------------------------------
# 3. Lookup tables for the predictor (notebook cell 27)
# --------------------------------------------------------------------------------------
def build_lookups(d: pd.DataFrame) -> dict:
    d = d.copy()
    d['bat_team'] = np.where(d.Innings == 1, d['Bat First'], d['Bat Second'])
    d['bowl_team'] = np.where(d.Innings == 1, d['Bat Second'], d['Bat First'])
    last_year = d.year.max()
    teams = sorted(d.loc[d.year == last_year, 'bat_team'].unique())
    recent = d[d.year >= last_year - 1]
    squad_bat = {t: recent[recent.bat_team == t].groupby('Batter').size().sort_values(ascending=False).index.tolist()
                 for t in teams}
    squad_bowl = {t: recent[recent.bowl_team == t].groupby('Bowler').size().sort_values(ascending=False).index.tolist()
                  for t in teams}
    vc = d[d.year >= last_year - 2]['Venue'].value_counts()
    venues = sorted(vc[vc >= 1000].index.tolist()) or sorted(vc.index.tolist())

    def tot(keys, data=d):
        return data.groupby(keys).agg(n=('Valid Ball', 'sum'), b=('is_bound', 'sum'), z=('is_dot_bat', 'sum'))

    return dict(teams=teams, squad_bat=squad_bat, squad_bowl=squad_bowl, venues=venues,
                BAT=tot('Batter'), BOWL=tot('Bowler'), VEN=tot('Venue'), BATPH=tot(['Batter', 'phase']),
                ATTACK=tot('bowl_team', d[d.year >= last_year - 2]),
                batters=sorted(d['Batter'].unique()), bowlers=sorted(d['Bowler'].unique()),
                last_year=int(last_year))


# --------------------------------------------------------------------------------------
# 4. Prediction (notebook cell 27: predict_ball)
# --------------------------------------------------------------------------------------
def build_row(bundle, batter, bowler='', venue='', innings=1, over=10, ball=1, wickets=2, balls_faced=0,
              strike_rate=0.0, run_rate=8.0, required_rate=0.0, last6_runs=7, opposition=''):
    L, G = bundle['lookups'], bundle['glob']
    GB, GD = G['GLOBAL_B'], G['GLOBAL_D']
    BAT, BOWL, VEN, BATPH, ATTACK = L['BAT'], L['BOWL'], L['VEN'], L['BATPH'], L['ATTACK']

    ph = 'PP' if over <= 6 else ('MID' if over <= 15 else 'DEATH')
    n, b, z = BAT.loc[batter]
    bat_b, bat_z = (b + K_BAT * GB) / (n + K_BAT), (z + K_BAT * GD) / (n + K_BAT)
    pn, pb = BATPH.loc[(batter, ph), ['n', 'b']] if (batter, ph) in BATPH.index else (0, 0)

    if bowler:
        wn, wb, wz = BOWL.loc[bowler]
        bowl_b, bowl_z = (wb + K_BOWL * GB) / (wn + K_BOWL), (wz + K_BOWL * GD) / (wn + K_BOWL)
        bowler_name = bowler
    elif opposition and opposition in ATTACK.index:
        wn, wb, wz = ATTACK.loc[opposition]
        bowl_b, bowl_z = (wb + K_ATT * GB) / (wn + K_ATT), (wz + K_ATT * GD) / (wn + K_ATT)
        bowler_name = f'{opposition} attack (average)'
    else:
        bowl_b, bowl_z, bowler_name = GB, GD, 'average bowler'

    if venue:
        vn, vb = VEN.loc[venue, ['n', 'b']]
        ven_b, venue_name = (vb + K_VEN * GB) / (vn + K_VEN), venue
    else:
        ven_b, venue_name = GB, 'average ground'

    row = dict(chase=int(innings == 2), phase_MID=int(ph == 'MID'), phase_DEATH=int(ph == 'DEATH'),
               inn_wkts_before=wickets, crr=run_rate, rrr=required_rate if innings == 2 else 0.0,
               team_last6=last6_runs, last_ball=int(ball == 6),
               bat_balls_log=np.log1p(balls_faced), new_bat=int(balls_faced < 6),
               bat_sr_inn=strike_rate if balls_faced > 0 else 0.0,
               bat_hist_bound=bat_b, bat_hist_dot=bat_z, bat_phase_bound=(pb + K_PH * bat_b) / (pn + K_PH),
               bowl_hist_bound=bowl_b, bowl_hist_dot=bowl_z, venue_bound=ven_b, bat_exp=np.log1p(n),
               season=L['last_year'] - 2008)
    return row, dict(batter=batter, bowler=bowler_name, venue=venue_name, phase=ph,
                     career_balls=int(n), career_bound=float(b / n) if n else float('nan'))


def predict_ball(bundle, **kw) -> dict:
    """Probability of a boundary / six for one hypothetical delivery, plus a per-feature explanation."""
    row, info = build_row(bundle, **kw)
    X = pd.DataFrame([row])[FEATURES]
    R = bundle['results']
    p_bound = float(R['model'].predict_proba(X)[0, 1])
    p_six = float(R['six_model'].predict_proba(X)[0, 1])
    avg_b, avg_s = bundle['glob']['GLOBAL_B'], R['avg_six']
    ratio = p_six / avg_s
    verdict = ('Above-average chance of a six' if ratio > 1.15
               else 'Below-average chance of a six' if ratio < 0.85 else 'About an average chance of a six')

    # why: contribution of each feature to the log-odds relative to an average training ball
    scaler, lr = R['model'].named_steps['scale'], R['model'].named_steps['lr']
    contrib = lr.coef_[0] * (X.values[0] - scaler.mean_) / scaler.scale_
    contrib = pd.Series(contrib, index=FEATURES)

    return dict(info, p_bound=p_bound, p_six=p_six, p_six_over=1 - (1 - p_six) ** 6, verdict=verdict,
                avg_bound=avg_b, avg_six=avg_s, six_ratio=ratio, bound_ratio=p_bound / avg_b,
                flagged=p_bound >= R['best_thr'], contrib=contrib, row=row)
