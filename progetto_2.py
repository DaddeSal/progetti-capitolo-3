import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_linnerud
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


# 1. Data Exploration & Preprocessing

linnerud = load_linnerud()

X = pd.DataFrame(linnerud.data, columns=linnerud.feature_names)
y = pd.DataFrame(linnerud.target, columns=linnerud.target_names)


print(X.head())
print(X.describe())

print(y.head())
print(y.describe())


scaler_X = StandardScaler()
X_scaled = pd.DataFrame(scaler_X.fit_transform(X), columns=X.columns)


scaler_y = StandardScaler()
y_scaled = pd.DataFrame(scaler_y.fit_transform(y), columns=y.columns)


# 2. Target Engineering

pca_target = PCA(n_components=1)
y_pca = pca_target.fit_transform(y_scaled).ravel()

y_manual = y_scaled['Waist'].values


# 3. Addestramento & Valutazione

lin_reg = LinearRegression()

lin_reg.fit(X_scaled, y_pca)
y_pred_pca = lin_reg.predict(X_scaled)

mse_pca = ((y_pca - y_pred_pca) ** 2).mean()
r2_pca = lin_reg.score(X_scaled, y_pca)

print("\n=== RISULTATI SCENARIO 1 (PCA Target) ===")
print(f"MSE: {mse_pca:.4f}")
print(f"R2 Score: {r2_pca:.4f}")

# --- Scenario 2: Target Manuale ('Waist') ---
lin_reg.fit(X_scaled, y_manual)
y_pred_manual = lin_reg.predict(X_scaled)

mse_manual = ((y_manual - y_pred_manual) ** 2).mean()
r2_manual = lin_reg.score(X_scaled, y_manual)

print("\n=== RISULTATI SCENARIO 2 (Target Manuale: Waist) ===")
print(f"MSE: {mse_manual:.4f}")
print(f"R2 Score: {r2_manual:.4f}")


# 4. Analisi & Visualizzazione


pca_feat = PCA(n_components=2)
X_pca_feat = pca_feat.fit_transform(X_scaled)

PC1_feat = X_pca_feat[:, 0]
PC2_feat = X_pca_feat[:, 1]

X_pc1_only = PC1_feat.reshape(-1, 1)

lin_reg_vis = LinearRegression()
lin_reg_vis.fit(X_pc1_only, y_pca)

r2_vis = lin_reg_vis.score(X_pc1_only, y_pca)

x_grid = np.linspace(PC1_feat.min(), PC1_feat.max(), 100).reshape(-1, 1)
y_grid_pred = lin_reg_vis.predict(x_grid)

plt.figure(figsize=(8, 5))
plt.scatter(PC1_feat, y_pca, color='purple', alpha=0.5, label='Dati Reali')
plt.plot(x_grid, y_grid_pred, color='red', label=f'Linear Regression (R2 = {r2_vis:.3f})')

plt.xlabel('PC1 Feature')
plt.ylabel('PC1 Target (n_components=1)')
plt.title('Relazione tra PC1 Feature e PC1 Target')
plt.legend()
plt.show()