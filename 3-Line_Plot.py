import pandas as pd
import matplotlib.pyplot as plt

# Set font (keeps negative sign display correct)
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Read data
data = pd.read_excel('Line_data_P15_3.xlsx')

# Extract true and predicted values
true_values = data.iloc[:, 0]
xgb_pred = data.iloc[:, 1]
piml_pred = data.iloc[:, 2]


# Plot line chart of actual and predicted distributions
plt.figure(figsize=(10, 6))
plt.plot(true_values.values, label='Actual value', color='black', linewidth=2)
plt.plot(xgb_pred.values, label='XGBoost prediction value', color='blue', linestyle='--')
plt.plot(piml_pred.values, label='NO-PIML predicted value', color='red', linestyle='-.')
plt.xlabel('Sample', fontsize=12)
plt.ylabel('Value', fontsize=12)

# Adjust legend font size
plt.legend(fontsize=10)

# Adjust tick label font sizes
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()