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

# Plot predicted vs actual values
plt.figure(figsize=(8, 6))
plt.scatter(true_values, xgb_pred, color='blue', label='XGBoost prediction value', alpha=0.7)
plt.scatter(true_values, piml_pred, color='red', label='NO-PIML predicted value', alpha=0.7)
plt.plot([true_values.min(), true_values.max()], [true_values.min(), true_values.max()], 'k--', lw=2, label='Ideal Prediction')
plt.xlabel('Actual value',fontsize=12)
plt.ylabel('Predicted value',fontsize=12)
plt.legend(fontsize=10)


plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
