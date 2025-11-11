import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
from pyswarms import single as ps
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== XGBoost direct training and evaluation ==========
def xgboost_train_and_plot(df, target_col, result_prefix='XGBoost'):
    print("\n========== XGBoost direct training ==========")
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    print(f"Train RMSE: {train_rmse:.6f}")
    print(f"Test RMSE: {test_rmse:.6f}")
    print(f"Train MAE: {train_mae:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print(f"Train R²: {train_r2:.6f}")
    print(f"Test R²: {test_r2:.6f}")

    # Visualization
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_test_pred, alpha=0.7, s=50, c='blue', edgecolors='black')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('True Value')
    plt.ylabel('XGBoost Prediction')
    plt.title(f'XGBoost model: Predicted vs True\nR² = {test_r2:.4f}, RMSE = {test_rmse:.4f}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{result_prefix}_Predicted_vs_True.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Save results
    results_df = X_test.copy()
    results_df['TrueValue'] = y_test.values
    results_df['XGBoostPrediction'] = y_test_pred
    results_df['AbsoluteError'] = np.abs(y_test.values - y_test_pred)
    results_df.to_csv(f'{result_prefix}_prediction_results_full.csv', index=False, encoding='utf-8-sig')
    print(f"XGBoost prediction results (with input features) saved to: {result_prefix}_prediction_results_full.csv")
    
    # Feature importance ranking
    importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importance
    }).sort_values('importance', ascending=False)
    print("\n========== Feature importance ranking ==========")
    for idx, row in importance_df.iterrows():
        print(f"{row['feature']:>12}: {row['importance']:.6f}")
    # Save as csv
    importance_df.to_csv(f'{result_prefix}_feature_importance_sorted.csv', index=False, encoding='utf-8-sig')
    print(f"Feature importance saved to: {result_prefix}_feature_importance_sorted.csv")
    
    return model, results_df, y_test, y_test_pred

# ========== PSO-XGBoost optimizer ==========
class PSOXGBoostOptimizer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_model = None
        self.best_params = None
        self.swarm_history = []
        self.fitness_history = []

    def objective_function(self, params_array):
        scores = []
        for params in params_array:
            try:
                n_estimators = int(params[0])
                max_depth = int(params[1])
                learning_rate = params[2]
                subsample = params[3]
                colsample_bytree = params[4]
                model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    random_state=42,
                    verbosity=0
                )
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
                scores.append(rmse)
            except Exception:
                scores.append(999999)
        return np.array(scores)

    def optimize(self, n_particles=10, n_iterations=20):
        print("Start PSO optimization...")
        bounds = (
            np.array([50, 3, 0.01, 0.5, 0.5]),
            np.array([1000, 10, 0.3, 1.0, 1.0])
        )
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        optimizer = ps.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=5,
            options=options,
            bounds=bounds
        )
        self.swarm_history = []
        self.fitness_history = []
        cost_history = []
        for i in range(n_iterations):
            optimizer.optimize(self.objective_function, iters=1, verbose=False)
            self.swarm_history.append(optimizer.swarm.position.copy())
            self.fitness_history.append(optimizer.swarm.current_cost.copy())
            cost_history.append(optimizer.swarm.best_cost)
            print(f"Iteration {i+1}/{n_iterations}, current best RMSE: {optimizer.swarm.best_cost:.6f}")
        best_pos = optimizer.swarm.best_pos
        best_cost = optimizer.swarm.best_cost
        self.best_params = {
            'n_estimators': int(best_pos[0]),
            'max_depth': int(best_pos[1]),
            'learning_rate': best_pos[2],
            'subsample': best_pos[3],
            'colsample_bytree': best_pos[4]
        }
        print(f"\nPSO optimization finished!")
        print(f"Best parameters: {self.best_params}")
        print(f"Best RMSE: {best_cost:.6f}")
        return best_cost, self.best_params, cost_history

    def train_best_model(self):
        if self.best_params is None:
            raise ValueError("Please run optimize() first.")
        print("\nTraining final model with best parameters...")
        self.best_model = xgb.XGBRegressor(
            **self.best_params,
            random_state=42,
            verbosity=1
        )
        self.best_model.fit(self.X_train, self.y_train)
        return self.best_model

    def evaluate_and_plot(self):
        if self.best_model is None:
            raise ValueError("Please train the model first.")
        y_train_pred = self.best_model.predict(self.X_train)
        y_test_pred = self.best_model.predict(self.X_test)
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        print("\n========== PSO-optimized model evaluation results ==========")
        print(f"Train RMSE: {train_rmse:.6f}")
        print(f"Test RMSE: {test_rmse:.6f}")
        print(f"Train MAE: {train_mae:.6f}")
        print(f"Test MAE: {test_mae:.6f}")
        print(f"Train R²: {train_r2:.6f}")
        print(f"Test R²: {test_r2:.6f}")
        self._plot_predictions(y_test_pred, test_r2, test_rmse, test_mae)
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'predictions': y_test_pred
        }

    def _plot_predictions(self, y_pred, r2, rmse, mae):
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 2, 1)
        plt.scatter(self.y_test, y_pred, alpha=0.6, s=50, c='blue', edgecolors='black', linewidth=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('True Value', fontsize=12)
        plt.ylabel('Predicted Value', fontsize=12)
        plt.title(f'PSO-optimized XGBoost: Predicted vs True\nR² = {r2:.4f}', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Residual plot
        plt.subplot(2, 2, 2)
        residuals = self.y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6, s=50, c='green', edgecolors='black', linewidth=0.5)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Value', fontsize=12)
        plt.ylabel('Residuals', fontsize=12)
        plt.title(f'Residual Distribution\nRMSE = {rmse:.4f}', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Error distribution histogram
        plt.subplot(2, 2, 3)
        errors = np.abs(self.y_test - y_pred)
        plt.hist(errors, bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Absolute Error', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title(f'Prediction Error Distribution\nMAE = {mae:.4f}', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Residual Q-Q plot
        plt.subplot(2, 2, 4)
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Residual Q-Q plot', fontsize=14)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('PSO_XGBoost_results_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

        plt.figure(figsize=(10, 8))
        plt.scatter(self.y_test, y_pred, alpha=0.7, s=60, c='blue', edgecolors='black', linewidth=0.8)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()],
                 'r--', lw=3, label='Perfect prediction line')
        plt.xlabel('True Value', fontsize=14)
        plt.ylabel('PSO-optimized XGBoost Prediction', fontsize=14)
        plt.title(f'PSO-optimized XGBoost model: Predicted vs True\nR² = {r2:.4f}, RMSE = {rmse:.4f}', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        textstr = f'Samples: {len(self.y_test)}\nR² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)
        plt.tight_layout()
        plt.savefig('PSO_XGBoost_Predicted_vs_True.png', dpi=300, bbox_inches='tight')
        plt.show()

    def plot_pso_process(self, cost_history):
        # 1. Convergence curve
        plt.figure(figsize=(8, 5))
        plt.plot(cost_history, marker='o')
        plt.xlabel('Iterations')
        plt.ylabel('Best RMSE')
        plt.title('PSO optimization convergence')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('PSO_convergence.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 2. Particle evolution (example using two main parameters)
        swarm_history = np.array(self.swarm_history)
        plt.figure(figsize=(10, 6))
        for i in range(swarm_history.shape[1]):
            plt.plot(swarm_history[:, i, 0], swarm_history[:, i, 1], marker='o', alpha=0.5, label=f'Particle {i+1}')
        plt.xlabel('n_estimators')
        plt.ylabel('max_depth')
        plt.title('PSO particle trajectories (n_estimators vs max_depth)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('PSO_particle_trajectories.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 3. Parameter trends (global best parameter per generation)
        best_params_trend = []
        for gen in range(swarm_history.shape[0]):
            best_idx = np.argmin(self.fitness_history[gen])
            best_params_trend.append(swarm_history[gen, best_idx, :])
        best_params_trend = np.array(best_params_trend)
        param_names = ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree']
        plt.figure(figsize=(10, 7))
        for i in range(5):
            plt.plot(best_params_trend[:, i], label=param_names[i])
        plt.xlabel('Iterations')
        plt.ylabel('Parameter value')
        plt.title('PSO global best parameter trends')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('PSO_parameter_trends.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 4. Fitness distribution (final generation)
        last_fitness = self.fitness_history[-1]
        plt.figure(figsize=(8, 5))
        plt.hist(last_fitness, bins=15, color='purple', alpha=0.7, edgecolor='black')
        plt.xlabel('RMSE')
        plt.ylabel('Number of particles')
        plt.title('PSO final generation particle fitness distribution')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('PSO_fitness_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("Loading data...")
    df = pd.read_excel('NG_traindata5_P15.xlsx')
    print(f"Data shape: {df.shape}")

    # Add physics-formula features
    df['K1'] = (9.5484 +
                5.0273 * df['C'].clip(lower=0) +
                5.8212 * df['Mn'].clip(lower=0) +
                13.0808 * df['Si'].clip(lower=0) +
                45.4141 * df['P'].clip(lower=0) +
                18.5418 * df['S'].clip(lower=0))
    df['K2'] = 7.650 - 65 * (df['Si'].clip(lower=0) + 1.7 * df['Als'].clip(lower=0))
    df['K3'] = df['B']**2 / (df['K1'] * df['K2'])

    # XGBoost direct training
    xgb_model, xgb_results_df, y_test, xgb_pred = xgboost_train_and_plot(df, target_col='P', result_prefix='XGBoost')

    # PSO-XGBoost optimization
    X = df.drop('P', axis=1)
    y = df['P']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pso_optimizer = PSOXGBoostOptimizer(X_train, X_test, y_train, y_test)
    best_cost, best_params, cost_history = pso_optimizer.optimize(n_particles=15, n_iterations=30)
    pso_optimizer.plot_pso_process(cost_history)
    best_model = pso_optimizer.train_best_model()
    results = pso_optimizer.evaluate_and_plot()

    # Save PSO model and results
    model_path = 'NG_PIML_traindata5_P15.pkl'
    joblib.dump(best_model, model_path)
    print(f"\nThe PSO-optimized XGBoost model has been saved to: {model_path}")
    params_df = pd.DataFrame([best_params])
    params_df.to_csv('pso_best_parameters.csv', index=False, encoding='utf-8-sig')
    print(f"Best parameters saved to: pso_best_parameters.csv")
    results_df = X_test.copy()
    results_df['TrueValue'] = y_test.values
    results_df['PSO_Optimized_Prediction'] = results['predictions']
    results_df['AbsoluteError'] = np.abs(y_test.values - results['predictions'])
    results_df.to_csv('pso_prediction_results_full.csv', index=False, encoding='utf-8-sig')
    print(f"PSO optimized prediction results (with input features) saved to: pso_prediction_results_full.csv")

    print("\n========== All processes completed ==========")
    print("Generated files:")
    print("1. XGBoost_Predicted_vs_True.png - XGBoost prediction comparison")
    print("2. XGBoost_prediction_results_full.csv - XGBoost prediction results")
    print("3. PSO_XGBoost_results_analysis.png - PSO 4-in-1 analysis figure")
    print("4. PSO_XGBoost_Predicted_vs_True.png - PSO main prediction comparison")
    print("5. NG_PIML_traindata5_P15.pkl - PSO-optimized model")
    print("6. pso_best_parameters.csv - PSO best parameters")
    print("7. pso_prediction_results_full.csv - PSO prediction results (with input features)")
    print("8. PSO_convergence.png - PSO convergence curve")
    print("9. PSO_particle_trajectories.png - particle trajectories")
    print("10. PSO_parameter_trends.png - parameter trends")
    print("11. PSO_fitness_distribution.png - fitness distribution")

if __name__ == "__main__":
    main()