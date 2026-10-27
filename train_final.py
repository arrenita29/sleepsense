"""
Train the improved SleepSense model (run once:  python train_final.py)
- Occupation removed (it reduced honest accuracy and blocks new jobs like 'Student')
- ML core: SVM (RBF), the best model under honest evaluation (a voting ensemble is also compared)
- Evaluated on deduplicated data with repeated stratified 5-fold CV
"""
import warnings; warnings.filterwarnings("ignore")
import joblib, numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

SEED = 42
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv", keep_default_na=False)
df = df.drop(columns=["Person ID", "Occupation"])
df["BMI Category"] = df["BMI Category"].replace({"Normal Weight": "Normal"})
bp = df["Blood Pressure"].str.split("/", expand=True).astype(int)
df["Systolic_BP"], df["Diastolic_BP"] = bp[0], bp[1]
df = df.drop(columns=["Blood Pressure"]).drop_duplicates().reset_index(drop=True)
y = df.pop("Sleep Disorder"); X = df
CAT = ["Gender", "BMI Category"]; NUM = [c for c in X.columns if c not in CAT]

def make(clf):
    pre = ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
    return Pipeline([("pre", pre), ("clf", clf)])

models = {
    "KNN": make(KNeighborsClassifier(5)),
    "SVM (RBF)": make(SVC(C=10, probability=True, random_state=SEED)),
    "Decision Tree": make(DecisionTreeClassifier(max_depth=4, random_state=SEED)),
    "Random Forest": make(RandomForestClassifier(200, max_depth=9, random_state=SEED)),
    "ANN-MLP": make(MLPClassifier((24,), max_iter=2000, random_state=SEED)),
    "Proposed: Voting (SVM + LR + RF)": make(VotingClassifier([
        ("svm", SVC(C=10, probability=True, random_state=SEED)),
        ("lr", LogisticRegression(class_weight="balanced", max_iter=3000)),
        ("rf", RandomForestClassifier(300, max_depth=5, class_weight="balanced", random_state=SEED))],
        voting="soft")),
}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
rows = []
for name, m in models.items():
    s = cross_validate(m, X, y, cv=cv, scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"])
    rows.append({"Model": name,
                 "Accuracy": f"{s['test_accuracy'].mean():.3f} ± {s['test_accuracy'].std():.3f}",
                 "Precision (macro)": f"{s['test_precision_macro'].mean():.3f}",
                 "Recall (macro)": f"{s['test_recall_macro'].mean():.3f}",
                 "F1 (macro)": f"{s['test_f1_macro'].mean():.3f} ± {s['test_f1_macro'].std():.3f}"})
    print(rows[-1])
pd.DataFrame(rows).to_csv("results_cv.csv", index=False)
final = models["SVM (RBF)"].fit(X, y)   # best honest accuracy -> ML core of the hybrid system
joblib.dump(final, "sleep_model.joblib")
print("Saved sleep_model.joblib  features:", list(X.columns))
