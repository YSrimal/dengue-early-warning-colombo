# GenAI prompt used to generate the front end (Step 7 of the activity)

Tool: Claude (the same prompt works in Gemini / ChatGPT).

```
I have developed a machine-learning model and I need a small web front end for it.

MODEL
- scikit-learn Pipeline saved with pickle at models/dengue_model.pkl
  (first step is a FunctionTransformer that builds derived features from dengue_features.add_derived,
   so the app must `sys.path.insert` the project root before unpickling).
- Binary classifier: predict_proba(df)[:,1] = probability that Colombo district records a
  HIGH-dengue week (> 148 notified cases) two weeks from now.
- A JSON file models/model_meta.json holds the training range of every field, the example inputs,
  the model name, and test ROC-AUC.

INPUT FIELDS (all numeric, one text box each, must be passed to the model as a one-row DataFrame in this column order)
cases_t, cases_t1, cases_t2, cases_t3   - notified dengue cases this week and 1/2/3 weeks ago
epi_week                                 - epidemiological week number, 1-53
rain_4wk, rain_8wk                       - mean weekly rainfall (mm) over last 4 / 8 weeks
humidity_4wk (%), temp_4wk (C), wind_4wk (km/h), visib_4wk (km) - 4-week averages

REQUIREMENTS
1. Flask app, single file app/app.py, no front-end build step.
2. Form with a labelled text box (with unit) for every field, a "Fill example" button, and a Predict button.
3. Validate: numbers only, non-negative, epi_week 1-53. Warn (do not block) if a value is outside the training range.
4. Show the probability as a large percentage plus a LOW / ELEVATED / HIGH label and one sentence of plain-language advice.
5. Also expose POST /predict (JSON in, JSON out) and GET /health.
6. Responsive, light/dark mode, readable by public-health officers (no ML jargon on the page).
7. Add a footer saying this is a coursework decision-support prototype.
Return the full code and the commands to run it and to deploy it with Docker.
```

## Follow-up prompts used
- "Add input-range warnings using model_meta.json field_ranges."
- "Give me a Dockerfile that runs this with gunicorn on $PORT (Cloud Run / Render / Azure Container Apps)."
