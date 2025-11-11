import pandas as pd
import matplotlib.pyplot as plt

# Font settings
plt.rcParams['font.sans-serif'] = ['Arial']  
plt.rcParams['axes.unicode_minus'] = False    

# Read data
data = pd.read_excel('pso_prediction_result.xlsx')

# Load true values and predictions
true_values = data.iloc[:, 0]
xgb_pred = data.iloc[:, 1]
piml_pred = data.iloc[:, 2]

# Calculate residuals
xgb_residual = xgb_pred - true_values
piml_residual = piml_pred - true_values

# Plot residuals vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(true_values, xgb_residual, color='blue', alpha=0.7, label='XGBoost Residual')
plt.scatter(true_values, piml_residual, color='red', alpha=0.7, label='NO-PIML Residual')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlabel('Actual value',fontsize=12)
plt.ylabel('Residual',fontsize=12)

plt.legend(fontsize=10)

plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()