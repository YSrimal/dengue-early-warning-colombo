"""
Colombo Dengue Early-Warning - web front end.
Loads the pickled scikit-learn pipeline (models/dengue_model.pkl) and exposes
  GET  /          HTML form (one text box per model input field)
  POST /          form submit -> probability + alert level
  POST /predict   JSON API   {"cases_t": 120, ...} -> {"probability": ...}
  GET  /health    liveness probe (used by cloud platforms)
Run:  python app/app.py        (dev)     |     gunicorn app.app:app   (production)
"""
import json, pickle, sys
from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))                 # makes `dengue_features` importable for unpickling
from dengue_features import RAW_FIELDS        # noqa: E402

with open(ROOT / "models" / "dengue_model.pkl", "rb") as f:
    MODEL = pickle.load(f)
META = json.load(open(ROOT / "models" / "model_meta.json"))
THRESH, HORIZON = META["high_week_threshold_cases"], META["horizon_weeks"]

FIELDS = [  # name, label, unit, help
    ("cases_t",  "Dengue cases - this week",   "cases", "Notified cases in Colombo district, latest week"),
    ("cases_t1", "Dengue cases - 1 week ago",  "cases", ""),
    ("cases_t2", "Dengue cases - 2 weeks ago", "cases", ""),
    ("cases_t3", "Dengue cases - 3 weeks ago", "cases", ""),
    ("epi_week", "Epidemiological week number", "1-53", "Week of the year of the latest report"),
    ("rain_4wk", "Rainfall, 4-week average",   "mm/week", "Mean of the last 4 weekly precipitation values"),
    ("rain_8wk", "Rainfall, 8-week average",   "mm/week", "Mean of the last 8 weekly precipitation values"),
    ("humidity_4wk", "Relative humidity, 4-week average", "%", ""),
    ("temp_4wk", "Temperature, 4-week average", "°C", ""),
    ("wind_4wk", "Wind speed, 4-week average", "km/h", ""),
    ("visib_4wk", "Visibility, 4-week average", "km", ""),
]
assert [f[0] for f in FIELDS] == RAW_FIELDS

def band(p):
    if p >= 0.5:  return ("HIGH", "Likely high-dengue week ahead: consider stepping up vector control and hospital readiness.")
    if p >= 0.25: return ("ELEVATED", "Some risk of a high week: keep monitoring closely.")
    return ("LOW", "A high-dengue week is unlikely on current indicators.")

def predict(values):
    X = pd.DataFrame([values])[RAW_FIELDS].astype(float)
    p = float(MODEL.predict_proba(X)[0, 1])
    warn = [f"{k} = {values[k]:g} is outside the range seen in training ({META['field_ranges'][k]['min']:.1f} to {META['field_ranges'][k]['max']:.1f}); treat the result with extra caution."
            for k in RAW_FIELDS if not META["field_ranges"][k]["min"] <= float(values[k]) <= META["field_ranges"][k]["max"]]
    level, advice = band(p)
    return dict(probability=round(p, 4), level=level, advice=advice, warnings=warn)

def parse(form):
    vals, errs = {}, []
    for name, label, *_ in FIELDS:
        raw = (form.get(name) or "").strip()
        try:
            v = float(raw)
            if v < 0: raise ValueError
            vals[name] = v
        except ValueError:
            errs.append(f"“{label}” must be a non-negative number.")
    if "epi_week" in vals and not 1 <= vals["epi_week"] <= 53:
        errs.append("Epidemiological week must be between 1 and 53.")
    return vals, errs

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Colombo Dengue Early Warning</title>
<style>
:root{--bg:#f6f4ef;--card:#fff;--ink:#1d2330;--mute:#667085;--line:#e3dfd6;--accent:#a8382a;--ok:#2f7d4f;--mid:#c58a12}
@media (prefers-color-scheme:dark){:root{--bg:#14161b;--card:#1d2027;--ink:#eceff4;--mute:#9aa3b2;--line:#2c313b;--accent:#e0684f;--ok:#58b57c;--mid:#e0ac3a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
main{max-width:860px;margin:0 auto;padding:28px 18px 60px}h1{font-size:1.65rem;margin:0 0 4px}.sub{color:var(--mute);margin:0 0 22px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:18px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}
label{display:block;font-size:.88rem;font-weight:600;margin-bottom:4px}label span{font-weight:400;color:var(--mute)}
input{width:100%;padding:9px 11px;border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink);font-size:1rem}
input:focus{outline:2px solid var(--accent);outline-offset:1px}small{color:var(--mute);font-size:.78rem}
h2{font-size:1rem;margin:0 0 12px;text-transform:uppercase;letter-spacing:.05em;color:var(--mute)}
.row{display:flex;gap:10px;flex-wrap:wrap;margin-top:18px}
button{border:0;border-radius:9px;padding:11px 20px;font-size:1rem;font-weight:600;cursor:pointer}
.go{background:var(--accent);color:#fff}.alt{background:transparent;border:1px solid var(--line);color:var(--ink)}
.result{border-left:6px solid var(--accent)}.LOW{border-color:var(--ok)}.ELEVATED{border-color:var(--mid)}.HIGH{border-color:var(--accent)}
.big{font-size:2.4rem;font-weight:700;line-height:1}.tag{display:inline-block;padding:2px 10px;border-radius:99px;font-size:.8rem;font-weight:700;color:#fff;margin-left:8px;vertical-align:middle}
.tag.LOW{background:var(--ok)}.tag.ELEVATED{background:var(--mid)}.tag.HIGH{background:var(--accent)}
.err{color:var(--accent);margin:0 0 6px}.warn{color:var(--mid);font-size:.88rem;margin:6px 0 0}.foot{color:var(--mute);font-size:.8rem}
</style></head><body><main>
<h1>Colombo Dengue Early Warning</h1>
<p class="sub">Probability that <b>{{h}} weeks from now</b> Colombo district records a <b>high-dengue week</b> (more than {{t}} notified cases).</p>
{% if errors %}<div class="card">{% for e in errors %}<p class="err">{{e}}</p>{% endfor %}</div>{% endif %}
{% if r %}<div class="card result {{r.level}}"><h2>Prediction</h2>
<div><span class="big">{{ '%.0f'|format(r.probability*100) }}%</span><span class="tag {{r.level}}">{{r.level}}</span></div>
<p>{{r.advice}}</p>{% for w in r.warnings %}<p class="warn">⚠ {{w}}</p>{% endfor %}</div>{% endif %}
<form method="post" class="card" id="f"><h2>Enter the latest weekly figures</h2><div class="grid">
{% for n,l,u,hp in fields %}<div><label for="{{n}}">{{l}} <span>({{u}})</span></label>
<input id="{{n}}" name="{{n}}" type="number" step="any" min="0" required value="{{ vals.get(n,'') }}"><small>{{hp}}</small></div>{% endfor %}
</div><div class="row"><button class="go" type="submit">Predict</button>
{% for e in examples %}<button class="alt" type="button" onclick="ex({{loop.index0}})" title="{{e.outcome}}">{{e.label}}</button>{% endfor %}</div></form>
<p class="foot">Model: {{m}} · test ROC-AUC {{auc}} (95% CI {{lo}}–{{hi}}) · trained on Colombo weekly data {{d0}} to {{d1}} (Epidemiology Unit, Ministry of Health).
Decision-support prototype for a coursework project, not an operational forecasting system.</p></main>
<script>const EX={{examples|tojson}};function ex(i){const v=EX[i].inputs;for(const k in v){document.getElementById(k).value=v[k]}}</script></body></html>"""

app = Flask(__name__)

def render(vals=None, r=None, errors=None):
    return render_template_string(PAGE, fields=FIELDS, vals=vals or {}, r=r, errors=errors or [], h=HORIZON, t=int(THRESH),
        m=META["model"], auc=META["test_auc"], lo=META["test_auc_ci"][0], hi=META["test_auc_ci"][1],
        d0=META["data_range"][0], d1=META["data_range"][1], examples=META["examples"])

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render()
    vals, errs = parse(request.form)
    if errs:
        return render(request.form, None, errs), 400
    return render(vals, predict(vals))

@app.post("/predict")
def api_predict():
    body = request.get_json(silent=True) or {}
    vals, errs = parse({k: str(body.get(k, "")) for k in RAW_FIELDS})
    if errs:
        return jsonify(error=errs), 400
    return jsonify(predict(vals) | {"horizon_weeks": HORIZON, "high_week_threshold": THRESH})

@app.get("/health")
def health():
    return jsonify(status="ok", model=META["model"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(__import__("os").environ.get("PORT", 5000)), debug=False)
