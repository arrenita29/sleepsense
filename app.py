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

app = Flask(__name__, template_folder=".", static_folder=".", static_url_path="/static")
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
FEATURES = [c for c in df.columns if c != "Sleep Disorder"]
OCCUPATIONS = sorted(df["Occupation"].unique())
# "typical healthy person" baseline used for explanations
BASELINE = {c: (float(dedup[c].median()) if pd.api.types.is_numeric_dtype(dedup[c]) else dedup[c].mode()[0]) for c in FEATURES}

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
    base_p = model.predict_proba(row)[0][i]
    out = []
    for f in FEATURES:
        r2 = row.copy()
        r2[f] = BASELINE[f]
        out.append((f, round((base_p - model.predict_proba(r2)[0][i]) * 100, 1)))
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
    p0 = model.predict_proba(cur)[0][i_none]
    plan, used = [], set()
    for _ in range(max_steps):
        if CLASSES[model.predict_proba(cur)[0].argmax()] == "None":
            break
        best = None
        for feat, opts in ACTIONS.items():
            if feat in used:
                continue
            for label, d in opts:
                cand = apply(cur, feat, d)
                if cand is None:
                    continue
                p = model.predict_proba(cand)[0][i_none]
                cost = abs(d) if not isinstance(d, str) else 0
                if best is None or p > best[0] + 1e-9:
                    best = (p, feat, label, cand)
        if best is None or best[0] <= model.predict_proba(cur)[0][i_none] + 1e-9:
            break
        p, feat, label, cand = best
        plan.append({"feature": feat, "change": label,
                     "from": row.at[0, feat], "to": cand.at[0, feat], "p_none": round(p * 100, 1)})
        used.add(feat); cur = cand
    final = model.predict_proba(cur)[0]
    return {"plan": plan, "start_p": round(p0 * 100, 1),
            "end_p": round(final[i_none] * 100, 1),
            "end_label": CLASSES[final.argmax()],
            "success": CLASSES[final.argmax()] == "None"}

def form_to_row(f):
    return pd.DataFrame([{
        "Gender": f["gender"], "Age": int(f["age"]), "Occupation": f["occupation"],
        "Sleep Duration": float(f["sleep_duration"]), "Quality of Sleep": int(f["sleep_quality"]),
        "Physical Activity Level": int(f["activity"]), "Stress Level": int(f["stress"]),
        "BMI Category": f["bmi"], "Heart Rate": int(f["heart_rate"]),
        "Daily Steps": int(f["steps"]), "Systolic_BP": int(f["systolic"]),
        "Diastolic_BP": int(f["diastolic"]),
    }])[FEATURES]

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
    with db() as con:
        n_pred = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
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
        row = form_to_row(form)
        probs = model.predict_proba(row)[0]
        pred = CLASSES[probs.argmax()]
        conf = round(probs.max() * 100, 1)
        r = row.iloc[0]
        result = dict(label=pred, confidence=conf,
                      probs=[(c, round(p * 100, 1)) for c, p in zip(CLASSES, probs)],
                      advice=ADVICE[pred], flags=risk_flags(r), factors=explain(row, pred),
                      cf=counterfactual(row) if pred != "None" else None)
        with db() as con:
            con.execute("""INSERT INTO predictions(ts,name,gender,age,occupation,bmi,sleep,quality,activity,
                stress,systolic,diastolic,heart,steps,prediction,confidence) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (datetime.now().strftime("%Y-%m-%d %H:%M"), form.get("name") or "Anonymous", r["Gender"],
                 int(r["Age"]), r["Occupation"], r["BMI Category"], float(r["Sleep Duration"]),
                 int(r["Quality of Sleep"]), int(r["Physical Activity Level"]), int(r["Stress Level"]),
                 int(r["Systolic_BP"]), int(r["Diastolic_BP"]), int(r["Heart Rate"]), int(r["Daily Steps"]),
                 pred, conf))
    return render_template("predict.html", page="predict", result=result, form=form, occupations=OCCUPATIONS)

@app.route("/performance")
def performance():
    cv_cols, cv_rows = read_csv_table("results_cv.csv")
    cmp_cols, cmp_rows = read_csv_table("results_comparison.csv")
    figs = [f for f in ["fig_naive_vs_honest.png", "fig_confusion.png", "fig_shap.png", "fig_eda.png", "fig_corr.png"]
            if (FIG_DIR / f).exists()]
    return render_template("performance.html", page="performance", cv_cols=cv_cols, cv_rows=cv_rows,
                           cmp_cols=cmp_cols, cmp_rows=cmp_rows, figs=figs,
                           total=len(raw), unique=len(dedup))

@app.route("/history")
def history():
    with db() as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM predictions ORDER BY id DESC").fetchall()
    summary = {c: sum(1 for r in rows if r["prediction"] == c) for c in CLASSES}
    return render_template("history.html", page="history", rows=rows, summary=summary, classes=CLASSES)

@app.route("/history/export")
def export():
    with db() as con:
        cur = con.execute("SELECT * FROM predictions ORDER BY id")
        buf = io.StringIO(); w = csv.writer(buf)
        w.writerow([d[0] for d in cur.description]); w.writerows(cur.fetchall())
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=prediction_history.csv"})

@app.route("/history/clear", methods=["POST"])
def clear():
    with db() as con:
        con.execute("DELETE FROM predictions")
    return redirect(url_for("history"))

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Used by the live What-If sliders (no saving to history)."""
    row = form_to_row(request.get_json())
    probs = model.predict_proba(row)[0]
    return jsonify(label=CLASSES[probs.argmax()],
                   probs={c: round(p * 100, 1) for c, p in zip(CLASSES, probs)})

@app.route("/figures/<path:name>")
def figures(name):
    return send_from_directory(FIG_DIR, name)

if __name__ == "__main__":
    app.run(debug=True)
