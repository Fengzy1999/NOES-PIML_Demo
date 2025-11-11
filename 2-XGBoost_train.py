import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

# Set font for Chinese characters (optional)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("Loading data...")
# Read data
df = pd.read_excel('train_data6.xlsx')
print(f"Data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Define features and target variable
# Assume last column 'P' is the target; others are features
X = df.drop('P15', axis=1)
y = df['P']

print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")
print(f"Target variable stats: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}")

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Split train and test sets
print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")

# Create XGBoost model
print("\nStart training XGBoost model...")
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    early_stopping_rounds=50,
    eval_metric='rmse'
)

# 训练模型
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=100
)

# 预测
y_train_pred = xgb_model.predict(X_train)
y_test_pred = xgb_model.predict(X_test)

# 计算评估指标
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Model evaluation results
print("\n========== Model evaluation results ==========")
print(f"Train RMSE: {train_rmse:.6f}")
print(f"Test RMSE: {test_rmse:.6f}")
print(f"Train MAE: {train_mae:.6f}")
print(f"Test MAE: {test_mae:.6f}")
print(f"Train R²: {train_r2:.6f}")
print(f"Test R²: {test_r2:.6f}")

# 保存模型
# Save model
model_path = 'xgboost_train_data5_model.pkl'
joblib.dump(xgb_model, model_path)
print(f"\nModel saved to: {model_path}")

# 创建可视化
plt.style.use('default')

# 计算特征重要性
feature_importance = xgb_model.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=True)

print("\nGenerating visualization charts...")


# 1. Predicted vs True scatter plot
plt.figure(1)
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_test_pred, alpha=0.6, s=30)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('True Value', fontsize=12)
plt.ylabel('Predicted Value', fontsize=12)
plt.title(f'Predicted vs True\nR² = {test_r2:.4f}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('1_Predicted_vs_True.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Residuals plot
plt.figure(2)
plt.figure(figsize=(10, 8))
residuals = y_test - y_test_pred
plt.scatter(y_test_pred, residuals, alpha=0.6, s=30)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Value', fontsize=12)
plt.ylabel('Residual (True - Predicted)', fontsize=12)
plt.title('Residual Distribution', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('2_Residuals.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Feature importance
plt.figure(3)
plt.figure(figsize=(10, 8))
plt.barh(importance_df['feature'], importance_df['importance'])
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('3_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Training loss curve
plt.figure(4)
plt.figure(figsize=(10, 8))
eval_result = xgb_model.evals_result()
epochs = len(eval_result['validation_0']['rmse'])
x_axis = range(0, epochs)
plt.plot(x_axis, eval_result['validation_0']['rmse'], label='Train', linewidth=2)
plt.plot(x_axis, eval_result['validation_1']['rmse'], label='Validation', linewidth=2)
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('RMSE', fontsize=12)
plt.title('Training Loss Curve', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('4_training_loss_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. Prediction error distribution
plt.figure(5)
plt.figure(figsize=(10, 8))
errors = np.abs(y_test - y_test_pred)
plt.hist(errors, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Absolute Error', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title(f'Prediction Error Distribution\nMAE = {test_mae:.4f}', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('5_prediction_error_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Correlation heatmap (top 10 important features)
plt.figure(6)
plt.figure(figsize=(12, 10))
top_features = importance_df.tail(10)['feature'].tolist()
top_features_data = df[top_features + ['P']]
correlation_matrix = top_features_data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            fmt='.3f', square=True, cbar_kws={'shrink': 0.8})
plt.title('Top Feature Correlation Heatmap', fontsize=14)
plt.tight_layout()
plt.savefig('6_top_feature_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

print("All visualization charts have been generated and saved!")

# SHAP analysis
print("\nPerforming SHAP analysis...")

# Initialize SHAP explainer
explainer = shap.TreeExplainer(xgb_model)
# compute SHAP values (use first 1000 samples to speed up)
shap_values = explainer.shap_values(X_test[:1000])

print("SHAP values calculation completed!")

# 7. SHAP Summary Plot
plt.figure(7)
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test[:1000], plot_type="dot", show=False)
plt.title('SHAP Summary Plot - Feature importance and impact', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('7_SHAP_summary_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. SHAP Summary Plot (bar)
plt.figure(8)
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test[:1000], plot_type="bar", show=False)
plt.title('SHAP Summary Plot - Mean feature importance', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('8_SHAP_summary_bar.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. SHAP Waterfall Plot (select one sample)
plt.figure(9)
sample_idx = 0
plt.figure(figsize=(12, 8))
shap.waterfall_plot(shap_values[sample_idx], features=X_test.iloc[sample_idx])
# shap.waterfall_plot(explainer.expected_value, shap_values[sample_idx], X_test.iloc[sample_idx], show=False)
plt.title(f'SHAP Waterfall Plot - Sample {sample_idx} explanation', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('9_SHAP_waterfall_plot.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. SHAP Partial Dependence Plots (select top 4 features)
plt.figure(10)
top_4_features = importance_df.tail(4)['feature'].tolist()
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.ravel()

for i, feature in enumerate(top_4_features):
    feature_idx = list(X.columns).index(feature)
    shap.partial_dependence_plot(
        feature_idx, xgb_model.predict, X_test[:1000], ice=False,
        model_expected_value=True, feature_expected_value=True, ax=axes[i], show=False
    )
    axes[i].set_title(f'{feature} - Partial dependence', fontsize=12)

plt.suptitle('SHAP Partial Dependence Plots - Main feature effects', fontsize=14)
plt.tight_layout()
plt.savefig('10_SHAP_partial_dependence.png', dpi=300, bbox_inches='tight')
plt.show()

# 11. SHAP Force Plot (save as HTML)
force_plot = shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0], show=False)
shap.save_html("11_SHAP_force_plot.html", force_plot)
print("SHAP Force Plot saved as HTML file: 11_SHAP_force_plot.html")

# 12. SHAP Decision Plot
plt.figure(11)
plt.figure(figsize=(12, 10))
shap.decision_plot(explainer.expected_value, shap_values[:50], X_test[:50], show=False)
plt.title('SHAP Decision Plot - decision paths for first 50 samples', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('12_SHAP_decision_plot.png', dpi=300, bbox_inches='tight')
plt.show()

print("SHAP analysis completed! All SHAP charts have been generated and saved!")

# Output detailed feature importance table
print("\n========== Feature importance ranking ==========")
importance_df_sorted = importance_df.sort_values('importance', ascending=False)
for idx, row in importance_df_sorted.iterrows():
    print(f"{row['feature']:>8}: {row['importance']:.6f}")

# Output prediction statistics
print("\n========== Prediction statistics ==========")
print(f"True value range: [{y_test.min():.4f}, {y_test.max():.4f}]")
print(f"Predicted value range: [{y_test_pred.min():.4f}, {y_test_pred.max():.4f}]")
print(f"Mean absolute error rate: {(test_mae / y_test.mean() * 100):.2f}%")

# Save prediction results
results_df = pd.DataFrame({
    'TrueValue': y_test.values,
    'PredictedValue': y_test_pred,
    'AbsoluteError': np.abs(y_test.values - y_test_pred)
})
results_df.to_csv('prediction_results_XGBoost.csv', index=False, encoding='utf-8-sig')
print(f"\nPrediction results saved to: prediction_results_XGBoost.csv")

print("\n========== Program execution completed ==========")
