# IPL Batter Aggression – Streamlit prototype

Interactive version of the mini project *Predicting Batter Aggression in the IPL using Logistic Regression*.
It re-runs the notebook pipeline (leak-free features, chronological split, StandardScaler + LogisticRegression, C = 0.01,
F1-tuned threshold) and wraps it in a web app.

## Run
```bash
pip install -r requirements.txt
# put ball_by_ball_ipl.csv next to app.py (Kaggle: jamiewelsh2/ball-by-ball-ipl), or upload it in the sidebar
streamlit run app.py
```
First launch takes about a minute (feature building + training); the result is cached in `.cache/`, so later launches are instant.
Delete `.cache/` if you change the model code.

## Tabs
| Tab | What it shows |
|---|---|
| Predict a ball | Pick batter, bowler, venue and match situation; get P(boundary), P(six), P(≥1 six in next 6 balls), a verdict, a "why" chart of feature contributions and an over-by-over sweep |
| Model performance | ROC-AUC, log loss, the three decision rules, confusion matrices, ROC, calibration, F1-vs-threshold, player-level predicted vs actual |
| What drives aggression | Standardised coefficients and odds ratios, preset scenarios |
| Data explorer | Class balance, boundary % by over / balls faced / season, top batters |
| About | Method, leakage control, limitations |

## Files
- `ipl_model.py` – notebook logic as functions (no Streamlit dependency)
- `app.py` – the UI
