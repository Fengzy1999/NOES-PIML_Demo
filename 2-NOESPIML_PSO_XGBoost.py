import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import shap
from pyswarms import single as ps
import warnings
warnings.filterwarnings('ignore')

# Set font for Chinese characters (keeps display compatible if Chinese appears)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class PSOXGBoostOptimizer:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.best_model = None
        self.best_params = None
        
    def objective_function(self, params_array):
        """
        PSO Objective Function: Optimizing 
        XGBoost HyperparametersParameter Order:
        [n_estimators, max_depth, learning_rate, subsample, colsample_bytree]
        """
        scores = []
        
        for params in params_array:
            try:
                # Analyze Parameters
                n_estimators = int(params[0])
                max_depth = int(params[1])
                learning_rate = params[2]
                subsample = params[3]
                colsample_bytree = params[4]
                
                # XGBoost model
                model = xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    random_state=42,
                    verbosity=0
                )
                
                # Train
                model.fit(self.X_train, self.y_train)
                
                # Predict
                y_pred = model.predict(self.X_test)
                
                # RMSE
                rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
                scores.append(rmse)
                
            except Exception as e:
                # False check
                scores.append(999999)
                
        return np.array(scores)
    
    def optimize(self, n_particles=10, n_iterations=20):
        
        print("Start PSO...")
        
        # Parameter boundaries
        # [n_estimators, max_depth, learning_rate, subsample, colsample_bytree]
        bounds = (
            np.array([50, 3, 0.01, 0.5, 0.5]),      # Downbound
            np.array([1000, 10, 0.3, 1.0, 1.0])     # uPpbound
        )
        
        # PSO options
        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        
        # PSO optimizer
        optimizer = ps.GlobalBestPSO(
            n_particles=n_particles,
            dimensions=5,
            options=options,
            bounds=bounds
        )
        
        # Perform optimization
        best_cost, best_pos = optimizer.optimize(
            self.objective_function,
            iters=n_iterations,
            verbose=True
        )
        
        # Store best parameters
        self.best_params = {
            'n_estimators': int(best_pos[0]),
            'max_depth': int(best_pos[1]),
            'learning_rate': best_pos[2],
            'subsample': best_pos[3],
            'colsample_bytree': best_pos[4]
        }
        
        print(f"\nPSO Finished.")
        print(f"Best parameters: {self.best_params}")
        print(f"Best RMSE: {best_cost:.6f}")
        
        return best_cost, self.best_params
    
    def train_best_model(self):
       
        if self.best_params is None:
            raise ValueError("No optimized parameters found. Please run optimize() first.")
            
        print("\nUse the best parameters...")
        
        self.best_model = xgb.XGBRegressor(
            **self.best_params,
            random_state=42,
            verbosity=1
        )
        
        self.best_model.fit(self.X_train, self.y_train)
        
        return self.best_model
    
    def evaluate_and_plot(self):
        """
        Model Evaluation and Plotting Results
        """
        if self.best_model is None:
            raise ValueError("No trained model found. Please run train_best_model() first.")
            
        # predictions
        y_train_pred = self.best_model.predict(self.X_train)
        y_test_pred = self.best_model.predict(self.X_test)
        
        # calculate metrics
        train_rmse = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_test_pred))
        train_mae = mean_absolute_error(self.y_train, y_train_pred)
        test_mae = mean_absolute_error(self.y_test, y_test_pred)
        train_r2 = r2_score(self.y_train, y_train_pred)
        test_r2 = r2_score(self.y_test, y_test_pred)
        
        print("\n========== Model results ==========")
        print(f"train RMSE: {train_rmse:.6f}")
        print(f"test RMSE: {test_rmse:.6f}")
        print(f"train MAE: {train_mae:.6f}")
        print(f"test MAE: {test_mae:.6f}")
        print(f"train R²: {train_r2:.6f}")
        print(f"test R²: {test_r2:.6f}")
        
        # Scatter plots
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

        # Add statistics textbox
        textstr = f'Samples: {len(self.y_test)}\nR² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=props)

        plt.tight_layout()
        plt.savefig('PSO_XGBoost_Predicted_vs_True.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("Load data...")
    
    # Read data
    df = pd.read_excel('NOES_traindata_P15.xlsx')
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
    
    # Define features and target variable
    X = df.drop('P15', axis=1)
    y = df['P15']
    
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    
    # Split train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Create PSO optimizer
    pso_optimizer = PSOXGBoostOptimizer(X_train, X_test, y_train, y_test)
    
    # Run PSO optimization
    best_cost, best_params = pso_optimizer.optimize(n_particles=15, n_iterations=30)
    
    # Train the best model
    best_model = pso_optimizer.train_best_model()
    
    # Evaluate and generate plots
    results = pso_optimizer.evaluate_and_plot()
    
    # Save best model
    model_path = 'NOES-PIML_model_p.pkl'
    joblib.dump(best_model, model_path)
    print(f"\nThe PSO-optimized XGBoost model has been saved to: {model_path}")
    
    # Save optimized parameters
    params_df = pd.DataFrame([best_params])
    params_df.to_csv('pso_best_parameters.csv', index=False, encoding='utf-8-sig')
    print(f"Best parameters saved to: pso_best_parameters.csv")
    
    # Save prediction results (including input features, true values, predictions, absolute errors)
    results_df = X_test.copy()
    results_df['TrueValue'] = y_test.values
    results_df['PSO_Optimized_Prediction'] = results['predictions']
    results_df['AbsoluteError'] = np.abs(y_test.values - results['predictions'])
    results_df.to_csv('pso_prediction_results_full.csv', index=False, encoding='utf-8-sig')
    print(f"PSO optimized prediction results (with input features) saved to: pso_prediction_results_full.csv")

    # ========== SHAP analysis ==========
    print("\nPerforming SHAP analysis...")

    # Compute feature importance DataFrame
    feature_importance = best_model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=True)

    # Initialize SHAP explainer
    explainer = shap.TreeExplainer(best_model)
    # compute SHAP values (use up to first 1000 samples to limit time/memory)
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
            feature_idx, best_model.predict, X_test[:1000], ice=False,
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
    
    print("\n========== PSO-optimized XGBoost completed ==========")


if __name__ == "__main__":
    main()
