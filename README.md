# Colombo Dengue Early-Warning (Applied ML: ensemble learning + explainability)

Predicts whether **Colombo district will record a high-dengue week (> ~150 notified cases) two weeks from now**, from the latest four weeks of case counts, the epidemiological week and 4-/8-week climate averages.
Covers the Lecture 7 class activity (RF vs XGBoost, non-default parameters, stacking, inference, `.pkl`, GenAI front end) and the final assignment (local problem, data analysis, model selection, comparison with existing approaches, SHAP/LIME, deployment).

## Data
Weekly notified dengue cases and climate variables, Colombo district, 2009-2014 (279 weeks) - Epidemiology Unit, Ministry of Health Sri Lanka, distributed in the R package [`colmozzie`](https://github.com/thiyangt/colmozzie) (CC0). Converted to `data/colmozzie_dengue_colombo.csv`.
Data problems found and fixed: an Excel `#DIV/0!` in `SLP`, 3 missing calendar weeks, and 5 weeks in 2012 with an implausible 0 cases (treated as unreported).

## Results (chronological test: 66 weeks, Jan 2013 - Apr 2014; CV = 5-fold expanding window on training)
| Model | CV AUC | Test AUC (95% block-bootstrap CI) |
|---|---|---|
| **XGBoost (deployed)** | **0.832** | 0.792 (0.65-0.90) |
| Logistic regression (baseline) | 0.822 | 0.752 (0.57-0.89) |
| SVM RBF (baseline) | 0.820 | 0.760 (0.59-0.89) |
| Soft voting RF+XGB | 0.815 | 0.801 (0.65-0.91) |
| Random Forest | 0.809 | 0.810 (0.65-0.93) |
| LightGBM | 0.806 | 0.773 (0.60-0.90) |
| ARIMA + RF hybrid | 0.802 | 0.800 (0.64-0.92) |
| Time-aware stacking (4 models) | 0.797 | 0.799 (0.64-0.91) |
| CatBoost | 0.782 | 0.806 (0.66-0.92) |
| Persistence rule | 0.758 | 0.716 (0.54-0.86) |
| Decision tree | 0.707 | 0.696 (0.58-0.80) |

**Honest reading:** ensembles clearly beat naive rules and a single tree, but their edge over regularised linear baselines is small and the intervals overlap (small data, ~66 test weeks). Case history and monsoon seasonality drive the model (SHAP); climate adds no demonstrable value (ablation). The model **missed** the explosive surge in the left-out week (p = 11% vs 329 actual cases) - it is a decision-support signal, not an autonomous alarm.

## Repository layout
```
data/                 CSV (+ original .rda)
dengue_features.py    cleaning, feature building, TimeAwareStacking (shared by notebook + app + pickle)
notebooks/            dengue_ensemble_analysis.ipynb (executed) + .html copy
models/               dengue_model.pkl, model_meta.json
streamlit_app.py      Streamlit front end (main UI)
app/app.py            Flask front end (form + JSON API, used by the Dockerfile)
prompts/              GenAI prompt used for the front end
figures/              all plots + model_comparison.csv
Dockerfile, Procfile, requirements*.txt
```

## Run the Streamlit app (recommended)
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py        # http://localhost:8501
```
Live demo: *(add your Streamlit Community Cloud link here after deploying)*

## Run the notebook / Flask app
```bash
pip install -r requirements-notebook.txt
jupyter notebook notebooks/dengue_ensemble_analysis.ipynb   # re-creates models/ and figures/ (~5 min on 1 CPU)
python app/app.py                                           # http://localhost:5000
curl -X POST localhost:5000/predict -H "Content-Type: application/json" \
     -d '{"cases_t":160,"cases_t1":150,"cases_t2":120,"cases_t3":100,"epi_week":50,"rain_4wk":10,"rain_8wk":8,"humidity_4wk":82,"temp_4wk":27,"wind_4wk":5,"visib_4wk":19.5}'
```
In Colab: upload the repo folder, `!pip install -q shap lime catboost lightgbm xgboost statsmodels`, open the notebook from `notebooks/`.

## Deploy (bonus)
The Docker image is cloud-neutral (listens on `$PORT`).
```bash
docker build -t dengue-ew . && docker run -p 8080:8080 dengue-ew          # local test
gcloud run deploy dengue-ew --source . --allow-unauthenticated             # Google Cloud Run
```
Render / Azure Container Apps / AWS App Runner: point them at this repo's Dockerfile.
The `.pkl` was produced with scikit-learn 1.8 / xgboost 3.4; keep `requirements-app.txt` versions when serving.

## Limitations
~270 weekly rows, one district, one chronological split; notified cases are not true incidence; thresholds drift with the upward trend; coursework prototype, not validated for public-health use.
