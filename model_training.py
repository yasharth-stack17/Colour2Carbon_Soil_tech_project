import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error,
    median_absolute_error,
)
from sklearn.preprocessing import StandardScaler
import joblib

sns.set_theme(style="whitegrid", context="talk")

# ──────────────────────────────────────────────
# 1. Load & Prepare Data
# ──────────────────────────────────────────────
print("=" * 60)
print("  SOIL ORGANIC CARBON (SOC) PREDICTION MODEL")
print("=" * 60)

df = pd.read_csv("soil_RGB_SOC_data.csv")
print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Feature engineering – add Brightness as an extra feature
df["Brightness"] = (df["R"] + df["G"] + df["B"]) / 3.0

X = df[["R", "G", "B", "Brightness"]]
y = df["SOC"]

print(f"Features used: {list(X.columns)}")
print(f"Target: SOC\n")

# ──────────────────────────────────────────────
# 2. Train / Test Split (80-20)
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}\n")

# ──────────────────────────────────────────────
# 3. Feature Scaling
# ──────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ──────────────────────────────────────────────
# 4. Train Random Forest Regressor
# ──────────────────────────────────────────────
print("Training Random Forest Regressor...")
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
)
rf_model.fit(X_train_scaled, y_train)
print("Training complete.\n")

# ──────────────────────────────────────────────
# 5. Predictions
# ──────────────────────────────────────────────
y_pred_train = rf_model.predict(X_train_scaled)
y_pred_test = rf_model.predict(X_test_scaled)

# ──────────────────────────────────────────────
# 6. Evaluation Metrics (7 metrics)
# ──────────────────────────────────────────────
metrics = {
    "R² Score": r2_score(y_test, y_pred_test),
    "Adjusted R²": 1 - (1 - r2_score(y_test, y_pred_test)) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1),
    "Mean Absolute Error (MAE)": mean_absolute_error(y_test, y_pred_test),
    "Mean Squared Error (MSE)": mean_squared_error(y_test, y_pred_test),
    "Root Mean Squared Error (RMSE)": np.sqrt(mean_squared_error(y_test, y_pred_test)),
    "Mean Absolute Percentage Error (MAPE)": mean_absolute_percentage_error(y_test, y_pred_test),
    "Median Absolute Error": median_absolute_error(y_test, y_pred_test),
    "Explained Variance Score": explained_variance_score(y_test, y_pred_test),
    "Max Error": max_error(y_test, y_pred_test),
}

print("=" * 60)
print("  MODEL EVALUATION METRICS (Test Set)")
print("=" * 60)
for name, value in metrics.items():
    print(f"  {name:<42}: {value:.6f}")
print("=" * 60)

# Also print train metrics for overfitting check
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = metrics["R² Score"]
print(f"\n  Train R²: {train_r2:.6f}  |  Test R²: {test_r2:.6f}")
if train_r2 - test_r2 > 0.05:
    print("  [WARNING] Possible overfitting detected (Train R2 >> Test R2)")
else:
    print("  [OK] No significant overfitting detected.")
print()

# ──────────────────────────────────────────────
# 7. Feature Importance
# ──────────────────────────────────────────────
importances = rf_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
importance_df = importance_df.sort_values("Importance", ascending=False)

print("Feature Importances:")
for _, row in importance_df.iterrows():
    bar = "#" * int(row["Importance"] * 50)
    print(f"  {row['Feature']:<12} {row['Importance']:.4f}  {bar}")
print()

# ──────────────────────────────────────────────
# 8. Visualizations
# ──────────────────────────────────────────────
print("Generating model evaluation plots...")

# 8a. Actual vs Predicted
plt.figure(figsize=(8, 8))
plt.hexbin(y_test, y_pred_test, gridsize=40, cmap="inferno", mincnt=1)
plt.colorbar(label="Count")
plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=2, label="Perfect Prediction")
plt.xlabel("Actual SOC")
plt.ylabel("Predicted SOC")
plt.title("Actual vs Predicted SOC (Test Set)")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
print("Saved 'actual_vs_predicted.png'")
plt.close()

# 8b. Residual Distribution
residuals = y_test - y_pred_test

plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=50, color="coral")
plt.axvline(0, color="black", linestyle="--", lw=1.5)
plt.xlabel("Residual (Actual - Predicted)")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.tight_layout()
plt.savefig("residual_distribution.png")
print("Saved 'residual_distribution.png'")
plt.close()

# 8c. Residuals vs Predicted
plt.figure(figsize=(10, 6))
plt.hexbin(y_pred_test, residuals, gridsize=40, cmap="coolwarm", mincnt=1)
plt.colorbar(label="Count")
plt.axhline(0, color="black", linestyle="--", lw=1.5)
plt.xlabel("Predicted SOC")
plt.ylabel("Residual")
plt.title("Residuals vs Predicted SOC")
plt.tight_layout()
plt.savefig("residuals_vs_predicted.png")
print("Saved 'residuals_vs_predicted.png'")
plt.close()

# 8d. Feature Importance Bar Chart
plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=importance_df, palette="viridis", hue="Feature", legend=False)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig("feature_importance.png")
print("Saved 'feature_importance.png'")
plt.close()

# 8e. Metrics Summary Bar Chart
metric_names = list(metrics.keys())
metric_values = list(metrics.values())

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.barh(metric_names, metric_values, color=sns.color_palette("mako", len(metric_names)))
ax.set_xlabel("Value")
ax.set_title("Model Evaluation Metrics Summary")
for bar, val in zip(bars, metric_values):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center")
plt.tight_layout()
plt.savefig("metrics_summary.png")
print("Saved 'metrics_summary.png'")
plt.close()

# ──────────────────────────────────────────────
# 9. Save Model & Scaler
# ──────────────────────────────────────────────
joblib.dump(rf_model, "random_forest_SOC_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\nSaved trained model  -> 'random_forest_SOC_model.pkl'")
print("Saved fitted scaler  -> 'scaler.pkl'")

print("\n" + "=" * 60)
print("  ALL DONE!")
print("=" * 60)
