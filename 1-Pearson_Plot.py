import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read data
df = pd.read_excel("NOES_traindata_P15.xlsx")

# Compute correlation matrix
corr = df.corr(numeric_only=True)

# Plot
plt.figure(figsize=(10, 10), dpi=150)

sns.heatmap(corr,
            cmap="RdBu_r",  # colormap
            center=-0.4,      # center the colormap at -0.4 for symmetry
            square=True,      # square cells
            linewidths=0.3,   # line width between cells
            linecolor='black',# grid line color
            cbar_kws={"shrink": 0.7},  # shrink colorbar
            xticklabels=True,
            yticklabels=True)

plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)
# plt.title("Correlation Heatmap", fontsize=14)
plt.tight_layout()
plt.show()
