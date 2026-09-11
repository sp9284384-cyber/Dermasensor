"""
DermaSensor - Model Training
=============================
Trains a RandomForestClassifier and a DecisionTreeClassifier on the
rule-based synthetic dataset, compares them, saves the better model as
model.joblib, and exports feature-importance plots used by the
explainability layer in the backend.
"""

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

FEATURES = ["temperature", "ph_level", "moisture"]
TARGET = "risk_class"
CLASS_ORDER = ["Low", "Medium", "High"]


def main():
    df = pd.read_csv("dermasensor_dataset.csv")
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Decision Tree (simple, highly interpretable) ---
    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt.predict(X_test))

    # --- Random Forest (stronger, still supports feature importance) ---
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))

    print(f"Decision Tree accuracy: {dt_acc:.4f}")
    print(f"Random Forest accuracy: {rf_acc:.4f}")
    print("\nRandom Forest classification report:\n")
    print(classification_report(y_test, rf.predict(X_test)))

    # Pick the better model to serve in production
    best_model, best_name = (rf, "RandomForest") if rf_acc >= dt_acc else (dt, "DecisionTree")
    print(f"\nSelected model for deployment: {best_name}")

    joblib.dump({"model": best_model, "features": FEATURES, "classes": list(best_model.classes_)}, "model.joblib")
    print("Saved -> model.joblib")

    # --- Feature importance plot (Random Forest) ---
    importances = rf.feature_importances_
    plt.figure(figsize=(6, 4))
    plt.bar(FEATURES, importances, color=["#2b8a3e", "#1971c2", "#e8590c"])
    plt.title("Feature Importance - Random Forest")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance_rf.png", dpi=150)
    plt.close()

    # --- Feature importance plot (Decision Tree) ---
    importances_dt = dt.feature_importances_
    plt.figure(figsize=(6, 4))
    plt.bar(FEATURES, importances_dt, color=["#2b8a3e", "#1971c2", "#e8590c"])
    plt.title("Feature Importance - Decision Tree")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance_tree.png", dpi=150)
    plt.close()

    # --- Decision tree visualization (for explainability docs) ---
    plt.figure(figsize=(16, 8))
    plot_tree(
        dt, feature_names=FEATURES, class_names=list(dt.classes_),
        filled=True, rounded=True, fontsize=8, max_depth=3
    )
    plt.tight_layout()
    plt.savefig("decision_tree.png", dpi=150)
    plt.close()

    # --- Confusion matrix (best model) ---
    cm = confusion_matrix(y_test, best_model.predict(X_test), labels=CLASS_ORDER)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.xticks(range(3), CLASS_ORDER)
    plt.yticks(range(3), CLASS_ORDER)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(3):
        for j in range(3):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    print("Saved plots: feature_importance_rf.png, feature_importance_tree.png, decision_tree.png, confusion_matrix.png")


if __name__ == "__main__":
    main()
