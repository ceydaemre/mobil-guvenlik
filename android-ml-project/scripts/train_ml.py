import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Grafik ayarları
plt.rcParams["figure.figsize"] = (10, 6)

# === VERİYİ YÜKLE ===
df = pd.read_csv("output/features.csv")

# sınıf etiketlerini sayısallaştır
X = df.drop(["class", "apk_name"], axis=1)
y = df["class"].map({"benign": 0, "malware": 1})  # binary mapping

# === 1) Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# === 2) RandomForest Eğitimi ===
rf = RandomForestClassifier(n_estimators=250, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("\n=== RandomForest Classification Report ===")
print(classification_report(y_test, y_pred))

print("\n=== RandomForest Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))


# === 3) 10-Fold Cross Validation ===
print("\n=== 10-Fold Cross Validation (RandomForest) ===")
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(rf, X, y, cv=kfold, scoring="accuracy")

print("Scores:", cv_scores)
print("Mean Accuracy:", cv_scores.mean())
print("Std:", cv_scores.std())


# === 4) ROC Curve + AUC ===
y_prob = rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — RandomForest")
plt.legend()
plt.savefig("output/roc_curve_rf.png")
plt.close()


# === 5) Feature Importance (Top 20) ===
importances = rf.feature_importances_
indices = np.argsort(importances)[-20:]  # en önemli 20 özellik

plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel("Importance")
plt.title("Feature Importance (Top 20 Features)")
plt.savefig("output/feature_importance.png")
plt.close()


# === 6) XGBoost Karşılaştırması ===
xgb = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42,
)

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("\n=== XGBoost Classification Report ===")
print(classification_report(y_test, y_pred_xgb))

print("\n=== XGBoost Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred_xgb))


# === 7) Correlation Heatmap ===
plt.figure(figsize=(14, 10))
sns.heatmap(X.corr(), cmap="coolwarm", vmax=1.0, vmin=-1.0)
plt.title("Feature Correlation Heatmap")
plt.savefig("output/correlation_heatmap.png")
plt.close()

print("\n[OK] Tüm analizler tamamlandı. Çıktılar 'output' klasörüne kaydedildi.\n")
