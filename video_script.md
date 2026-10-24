# 5-minute video script (screen-record the notebook + web app)

**0:00-0:30 Problem.** "Dengue is a major public-health burden in Colombo. Hospitals and the Epidemiology Unit need lead time, so I predict whether a week two weeks from now will be a high-dengue week, more than 150 cases. Data: weekly Ministry of Health notifications and climate for Colombo, 2009-2014." *(show title slide / notebook section 2)*

**0:30-1:15 Data and preprocessing.** "I audited the data before modelling and found three problems that `isna()` does not show: an Excel #DIV/0! hiding in the pressure column, three missing calendar weeks, and five weeks reporting exactly zero cases - reporting gaps, so I imputed them. I rebuilt a continuous weekly calendar so that lags are real lags." *(section 3-4)*

**1:15-1:50 EDA.** "Cases peak in June-July and December-January, matching the two monsoons. Autocorrelation is strong, 0.7 at one week. Climate correlations are weak and lagged." *(figures 02, 03)*

**1:50-2:30 Design.** "Chronological split with a two-week embargo, expanding-window cross-validation, threshold from training only. The last week of the data is left out for inference. Parameters are not defaults: shallow trees, large leaf sizes, class weights and L2 regularisation because I only have 204 training weeks." *(sections 6, 8 - show the parameter table)*

**2:30-3:30 Results.** "I compared Random Forest, XGBoost, LightGBM, CatBoost, soft voting, time-aware stacking and an ARIMA hybrid against persistence, a decision tree, logistic regression and SVM. XGBoost has the best CV AUC, 0.83, test 0.79. Ensembles clearly beat naive rules, but intervals overlap and linear baselines are close - so I do not overclaim." *(figure 06, table)*

**3:30-4:15 Explainability.** "SHAP says last week's cases and season drive predictions, with a step at the 150-case threshold. Climate gets some attribution, but my ablation shows removing it does not hurt - SHAP shows what the model uses, not what helps." *(figures 11, 12)*

**4:15-5:00 Inference and app.** "On the left-out week the model said 11% but 329 cases followed - it misses explosive surges; I report that as a limitation. Here is the web app generated with a GenAI prompt: three real example weeks, probability and risk band. It is packaged in Docker for cloud deployment." *(live demo of the three example buttons)*
