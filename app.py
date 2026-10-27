"""
Sleep Disorder Prediction System — Flask web app
Pages: Dashboard | Predict | Model Performance | History
Run:  python app.py   then open http://127.0.0.1:5000
"""
import csv, io, sqlite3
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template, request, send_from_directory, Response, redirect, url_for, jsonify

# ---------- locate project files (same folder as app.py, or its parent) ----------
HERE = Path(__file__).resolve().parent
def find(name):
    for d in (HERE, HERE.parent):
        if (d / name).exists():
            return d / name
    return None

MODEL_PATH = find("sleep_model.joblib")
DATA_PATH = find("Sleep_health_and_lifestyle_dataset.csv")
if MODEL_PATH is None or DATA_PATH is None:
    raise SystemExit("Put sleep_model.joblib and Sleep_health_and_lifestyle_dataset.csv next to app.py")
FIG_DIR = MODEL_PATH.parent
DB_PATH = HERE / "predictions.db"

# works whether HTML/CSS are in templates/ & static/ folders or loose next to app.py
_tpl = "templates" if (HERE / "templates").is_dir() else "."
_sta = "static" if (HERE / "static").is_dir() else "."
app = Flask(__name__, template_folder=_tpl, static_folder=_sta, static_url_path="/static")
model = joblib.load(MODEL_PATH)
CLASSES = list(model.classes_)

# ---------- dataset (cleaned the same way as in training) ----------
raw = pd.read_csv(DATA_PATH, keep_default_na=False)
df = raw.drop(columns=["Person ID"]).copy()
df["BMI Category"] = df["BMI Category"].replace({"Normal Weight": "Normal"})
bp = df["Blood Pressure"].str.split("/", expand=True).astype(int)
df["Systolic_BP"], df["Diastolic_BP"] = bp[0], bp[1]
df = df.drop(columns=["Blood Pressure"])
dedup = df.drop_duplicates()
FEATURES = list(model.feature_names_in_)          # Occupation is NOT used by the model
OCCUPATIONS = sorted(df["Occupation"].unique()) + ["Student", "Homemaker", "Business Owner", "Retired", "Other"]
# "typical healthy person" baseline used for explanations
BASELINE = {c: (float(dedup[c].median()) if pd.api.types.is_numeric_dtype(dedup[c]) else dedup[c].mode()[0]) for c in FEATURES}

# training-data range of each numeric feature (for out-of-distribution checks)
RANGES = {c: (float(dedup[c].min()), float(dedup[c].max())) for c in FEATURES
          if pd.api.types.is_numeric_dtype(dedup[c])}
LOW_CONF = 60.0  # below this confidence the prediction is flagged as uncertain

def reliability(row, conf):
    """Flag inputs outside the training range and low-confidence predictions."""
    ood = []
    for f, (lo, hi) in RANGES.items():
        v = float(row.at[0, f])
        if v < lo or v > hi:
            ood.append({"feature": f, "value": v, "lo": lo, "hi": hi})
    level = "high" if not ood and conf >= LOW_CONF else ("low" if ood else "medium")
    return {"ood": ood, "level": level}

# ---------- Knowledge-guided hybrid: ML (SVM) + clinical screening layer ----------
def clinical_scores(row):
    """A-priori screening scores, not learned from data.
    Insomnia: short sleep, poor sleep quality, high stress.
    Sleep apnea: STOP-BANG-style factors available here (Pressure, BMI, Age, Gender) + raised heart rate."""
    r = row.iloc[0]
    ins = (min(max((7 - r["Sleep Duration"]) / 2, 0), 1) + min(max((7 - r["Quality of Sleep"]) / 3, 0), 1)
           + min(max((r["Stress Level"] - 5) / 4, 0), 1)) / 3
    hyp = 1.0 if (r["Systolic_BP"] >= 130 or r["Diastolic_BP"] >= 85) else 0.0
    bmi = {"Normal": 0, "Overweight": 0.5, "Obese": 1}[r["BMI Category"]]
    apn = (hyp + bmi + (1.0 if r["Age"] > 50 else 0.0) + (0.5 if r["Gender"] == "Male" else 0.0)
           + (0.5 if r["Heart Rate"] >= 80 else 0.0)) / 4
    p = {"Insomnia": ins, "None": (1 - ins) * (1 - apn), "Sleep Apnea": apn}
    tot = sum(p.values()) + 1e-9
    return [p[c] / tot for c in CLASSES]

ALPHA_IN, ALPHA_OOD = 0.8, 0.25  # ML weight: 80% inside the training range; 25% outside it (ML cannot extrapolate)

def out_of_range(row):
    return any(not (lo <= float(row.at[0, f]) <= hi) for f, (lo, hi) in RANGES.items())

def hybrid_proba(row):
    a = ALPHA_OOD if out_of_range(row) else ALPHA_IN
    ml = model.predict_proba(row[FEATURES])[0]
    cl = clinical_scores(row)
    return [a * m + (1 - a) * c for m, c in zip(ml, cl)], list(ml), cl, a

# ---------- database ----------
def db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS predictions(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, name TEXT, gender TEXT, age INT,
        occupation TEXT, bmi TEXT, sleep REAL, quality INT, activity INT, stress INT,
        systolic INT, diastolic INT, heart INT, steps INT, prediction TEXT, confidence REAL)""")
    return con

# ---------- helpers ----------
ADVICE = {
    "None": ["Your profile does not show signs of a sleep disorder.",
             "Keep a consistent sleep schedule of 7–9 hours.",
             "Stay active and keep stress in check."],
    "Insomnia": ["Go to bed and wake up at the same time every day.",
                 "Avoid screens, caffeine and heavy meals 1–2 hours before bed.",
                 "Try relaxation techniques to lower stress.",
                 "See a doctor if poor sleep lasts more than 3 weeks."],
    "Sleep Apnea": ["Consult a doctor; a sleep study (polysomnography) confirms apnea.",
                    "Work on blood pressure and weight; both are strongly linked to apnea.",
                    "Sleep on your side and avoid alcohol before bed.",
                    "Loud snoring or gasping at night are warning signs."],
}

def risk_flags(r):
    flags = []
    if r["Systolic_BP"] >= 140 or r["Diastolic_BP"] >= 90:
        flags.append(("High blood pressure (Stage 2)", "danger"))
    elif r["Systolic_BP"] >= 130 or r["Diastolic_BP"] >= 80:
        flags.append(("Elevated blood pressure (Stage 1)", "warn"))
    if r["Sleep Duration"] < 6:
        flags.append(("Short sleep (< 6 h)", "danger"))
    elif r["Sleep Duration"] < 7:
        flags.append(("Below recommended sleep (< 7 h)", "warn"))
    if r["Stress Level"] >= 7:
        flags.append(("High stress level", "warn"))
    if r["BMI Category"] in ("Overweight", "Obese"):
        flags.append((f"BMI: {r['BMI Category']}", "danger" if r["BMI Category"] == "Obese" else "warn"))
    if r["Quality of Sleep"] <= 5:
        flags.append(("Poor sleep quality", "warn"))
    if r["Daily Steps"] < 5000:
        flags.append(("Low daily steps (< 5000)", "warn"))
    if not flags:
        flags.append(("No major risk factors", "ok"))
    return flags

def explain(row, cls):
    """Model-agnostic explanation: replace each feature with a typical value
    and measure how much the predicted-class probability drops."""
    i = CLASSES.index(cls)
    base_p = hybrid_proba(row)[0][i]
    out = []
    for f in FEATURES:
        r2 = row.copy()
        r2[f] = BASELINE[f]
        out.append((f, round((base_p - hybrid_proba(r2)[0][i]) * 100, 1)))
    out.sort(key=lambda x: abs(x[1]), reverse=True)
    return out[:6]

# ---------- NOVEL FEATURE: Personalised Risk-Reduction Advisor (counterfactual search) ----------
# Only lifestyle / clinically modifiable features are changed; age, gender, occupation stay fixed.
ACTIONS = {
    "Sleep Duration":          [("+0.5 h sleep", 0.5), ("+1 h sleep", 1.0), ("+1.5 h sleep", 1.5)],
    "Quality of Sleep":        [("+1 sleep quality", 1), ("+2 sleep quality", 2)],
    "Stress Level":            [("-1 stress", -1), ("-2 stress", -2), ("-3 stress", -3)],
    "Physical Activity Level": [("+15 min activity", 15), ("+30 min activity", 30)],
    "Daily Steps":             [("+1000 steps", 1000), ("+2000 steps", 2000), ("+3000 steps", 3000)],
    "Systolic_BP":             [("-5 systolic BP", -5), ("-10 systolic BP", -10), ("-15 systolic BP", -15)],
    "Diastolic_BP":            [("-5 diastolic BP", -5), ("-10 diastolic BP", -10)],
    "BMI Category":            [("Reach normal BMI", "Normal")],
}
LIMITS = {"Sleep Duration": (4, 9), "Quality of Sleep": (1, 10), "Stress Level": (1, 10),
          "Physical Activity Level": (0, 120), "Daily Steps": (0, 15000),
          "Systolic_BP": (110, 200), "Diastolic_BP": (70, 130)}

def apply(row, feat, delta):
    r = row.copy()
    if feat == "BMI Category":
        if r.at[0, feat] == "Normal":
            return None
        r.at[0, feat] = delta
        return r
    lo, hi = LIMITS[feat]
    new = r.at[0, feat] + delta
    if new < lo or new > hi:
        return None
    r.at[0, feat] = round(new, 1) if feat == "Sleep Duration" else int(new)
    return r

def counterfactual(row, max_steps=4):
    """Greedy search for the smallest set of lifestyle changes that moves the
    prediction to 'None'. Each step picks the change that raises P(None) most."""
    i_none = CLASSES.index("None")
    cur = row.copy()
    p0 = hybrid_proba(cur)[0][i_none]
    plan, used = [], set()
    for _ in range(max_steps):
        if max(range(3), key=lambda k: hybrid_proba(cur)[0][k]) == i_none:
            break
        best = None
        for feat, opts in ACTIONS.items():
            if feat in used:
                continue
            for label, d in opts:
                cand = apply(cur, feat, d)
                if cand is None:
                    continue
                p = hybrid_proba(cand)[0][i_none]
                cost = abs(d) if not isinstance(d, str) else 0
                if best is None or p > best[0] + 1e-9:
                    best = (p, feat, label, cand)
        if best is None or best[0] <= hybrid_proba(cur)[0][i_none] + 1e-9:
            break
        p, feat, label, cand = best
        plan.append({"feature": feat, "change": label,
                     "from": row.at[0, feat], "to": cand.at[0, feat], "p_none": round(p * 100, 1)})
        used.add(feat); cur = cand
    final = hybrid_proba(cur)[0]
    k = max(range(3), key=lambda j: final[j])
    return {"plan": plan, "start_p": round(p0 * 100, 1),
            "end_p": round(final[i_none] * 100, 1),
            "end_label": CLASSES[k], "success": CLASSES[k] == "None"}

def form_to_row(f):
    return pd.DataFrame([{
        "Gender": f["gender"], "Age": int(f["age"]), "Occupation": f["occupation"],
        "Sleep Duration": float(f["sleep_duration"]), "Quality of Sleep": int(f["sleep_quality"]),
        "Physical Activity Level": int(f["activity"]), "Stress Level": int(f["stress"]),
        "BMI Category": f["bmi"], "Heart Rate": int(f["heart_rate"]),
        "Daily Steps": int(f["steps"]), "Systolic_BP": int(f["systolic"]),
        "Diastolic_BP": int(f["diastolic"]),
    }])

def read_csv_table(name):
    p = find(name)
    if not p:
        return None, None
    t = pd.read_csv(p)
    return list(t.columns), t.values.tolist()

# ---------- routes ----------
@app.route("/")
def dashboard():
    counts = dedup["Sleep Disorder"].value_counts().reindex(CLASSES).fillna(0).astype(int)
    by_bmi = pd.crosstab(dedup["BMI Category"], dedup["Sleep Disorder"]).reindex(columns=CLASSES, fill_value=0)
    occ = pd.crosstab(dedup["Occupation"], dedup["Sleep Disorder"]).reindex(columns=CLASSES, fill_value=0)
    sleep_avg = dedup.groupby("Sleep Disorder")["Sleep Duration"].mean().reindex(CLASSES).round(2)
    stress_avg = dedup.groupby("Sleep Disorder")["Stress Level"].mean().reindex(CLASSES).round(2)
    sys_avg = dedup.groupby("Sleep Disorder")["Systolic_BP"].mean().reindex(CLASSES).round(1)
    con = db()
    n_pred = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    con.close()
    return render_template("dashboard.html", page="dashboard",
        total=len(raw), unique=len(dedup), dup_pct=round((1 - len(dedup) / len(raw)) * 100),
        n_pred=n_pred, classes=CLASSES, counts=counts.tolist(),
        bmi_labels=list(by_bmi.index), bmi_data={c: by_bmi[c].tolist() for c in CLASSES},
        occ_labels=list(occ.index), occ_data={c: occ[c].tolist() for c in CLASSES},
        sleep_avg=sleep_avg.tolist(), stress_avg=stress_avg.tolist(), sys_avg=sys_avg.tolist())

@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    form = request.form
    if request.method == "POST":
        if int(form["systolic"]) <= int(form["diastolic"]):
            return render_template("predict.html", page="predict", result=None, form=form,
                                   occupations=OCCUPATIONS, error="Systolic BP must be higher than diastolic BP.")
        row = form_to_row(form)
        probs, ml_p, cl_p, alpha = hybrid_proba(row)
        k = max(range(3), key=lambda j: probs[j])
        pred = CLASSES[k]
        conf = round(probs[k] * 100, 1)
        r = row.iloc[0]
        rel = reliability(row, conf)
        flags = risk_flags(r)
        advice = ADVICE[pred]
        if pred == "None" and any(lvl == "danger" for _, lvl in flags):
            advice = ["The model does not detect a sleep disorder, but serious risk indicators are present.",
                      "Please get blood pressure and sleep checked by a doctor.",
                      "Aim for 7–9 hours of sleep and reduce stress."]
        result = dict(rel=rel, alpha=int(alpha * 100),
                      ml_probs=[(c, round(p * 100, 1)) for c, p in zip(CLASSES, ml_p)],
                      cl_probs=[(c, round(p * 100, 1)) for c, p in zip(CLASSES, cl_p)], label=pred, confidence=conf,
                      probs=[(c, round(p * 100, 1)) for c, p in zip(CLASSES, probs)],
                      advice=advice, flags=flags, factors=explain(row, pred),
                      cf=counterfactual(row) if pred != "None" else None)
        con = db()
        with con:
            con.execute("""INSERT INTO predictions(ts,name,gender,age,occupation,bmi,sleep,quality,activity,
                stress,systolic,diastolic,heart,steps,prediction,confidence) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (datetime.now().strftime("%Y-%m-%d %H:%M"), form.get("name") or "Anonymous", r["Gender"],
                 int(r["Age"]), r["Occupation"], r["BMI Category"], float(r["Sleep Duration"]),
                 int(r["Quality of Sleep"]), int(r["Physical Activity Level"]), int(r["Stress Level"]),
                 int(r["Systolic_BP"]), int(r["Diastolic_BP"]), int(r["Heart Rate"]), int(r["Daily Steps"]),
                 pred, conf))
        con.close()
    return render_template("predict.html", page="predict", result=result, form=form, occupations=OCCUPATIONS)

@app.route("/performance")
def performance():
    cv_cols, cv_rows = read_csv_table("results_cv.csv")
    cmp_cols, cmp_rows = read_csv_table("results_comparison.csv")
    figs = [f for f in ["fig_naive_vs_honest.png", "fig_confusion.png", "fig_shap.png", "fig_eda.png", "fig_corr.png"]
            if (FIG_DIR / f).exists()]
    best = None
    if cv_rows:
        accs = [float(str(r[1]).split()[0]) for r in cv_rows]
        best = accs.index(max(accs))
    return render_template("performance.html", page="performance", cv_cols=cv_cols, cv_rows=cv_rows, best=best,
                           cmp_cols=cmp_cols, cmp_rows=cmp_rows, figs=figs,
                           total=len(raw), unique=len(dedup))

@app.route("/history")
def history():
    con = db()
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    con.close()
    summary = {c: sum(1 for r in rows if r["prediction"] == c) for c in CLASSES}
    return render_template("history.html", page="history", rows=rows, summary=summary, classes=CLASSES)

def _ids(raw):
    return [int(x) for x in raw if str(x).strip().isdigit()]

@app.route("/history/export")
def export():
    ids = _ids(request.args.get("ids", "").split(","))
    con = db()
    with con:
        if ids:
            q = "SELECT * FROM predictions WHERE id IN (%s) ORDER BY id" % ",".join("?" * len(ids))
            cur = con.execute(q, ids)
        else:
            cur = con.execute("SELECT * FROM predictions ORDER BY id")
        buf = io.StringIO(); w = csv.writer(buf)
        w.writerow([d[0] for d in cur.description]); w.writerows(cur.fetchall())
    con.close()
    fname = "prediction_history_selected.csv" if ids else "prediction_history.csv"
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={fname}"})

@app.route("/history/delete", methods=["POST"])
def delete_selected():
    ids = _ids(request.form.getlist("ids"))
    if ids:
        con = db()
        with con:
            con.execute("DELETE FROM predictions WHERE id IN (%s)" % ",".join("?" * len(ids)), ids)
        con.close()
    return redirect(url_for("history"))

@app.route("/history/clear", methods=["POST"])
def clear():
    con = db()
    with con:
        con.execute("DELETE FROM predictions")
    con.close()
    return redirect(url_for("history"))

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Used by the live What-If sliders (no saving to history)."""
    row = form_to_row(request.get_json())
    probs = hybrid_proba(row)[0]
    return jsonify(label=CLASSES[max(range(3), key=lambda j: probs[j])],
                   probs={c: round(p * 100, 1) for c, p in zip(CLASSES, probs)})

@app.route("/figures/<path:name>")
def figures(name):
    return send_from_directory(FIG_DIR, name)

if __name__ == "__main__":
    app.run(debug=True)
