# # Gold Price Prediction - Modeling Phase
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.svm import SVR
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import warnings
# warnings.filterwarnings('ignore')

# def main():
#     print("=== Gold Price Prediction - Modeling Phase ===\n")
    
#     # 1. Load preprocessed data
#     print("1. Loading data...")
#     df = pd.read_csv("data/filtered_data.csv", index_col='Date', parse_dates=True)
#     print(f"Data shape: {df.shape}")
#     print(f"Columns: {df.columns.tolist()}")
    
#     # 2. Feature Engineering
#     print("\n2. Feature Engineering...")
#     df_engineered = create_features(df)
#     print(f"Engineered data shape: {df_engineered.shape}")
    
#     # 3. Prepare data for modeling
#     print("\n3. Preparing data for modeling...")
#     X, y = prepare_data(df_engineered)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Scale features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
    
#     print(f"Training set: {X_train.shape}")
#     print(f"Test set: {X_test.shape}")
    
#     # 4. Train and evaluate models
#     print("\n4. Training and evaluating models...")
#     results = train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
    
#     # 5. Compare models
#     print("\n5. Model comparison:")
#     compare_models(results)
    
#     # 6. Save best model
#     print("\n6. Saving best model...")
#     save_best_model(results, scaler)
    
#     print("\n=== Modeling completed! ===")

# def create_features(df):
#     """Create lag and rolling features for time series"""
#     df_engineered = df.copy()
    
#     # Create lag features
#     target_col = 'Gold_Spot'
#     for lag in [1, 7, 30]:
#         df_engineered[f'{target_col}_lag_{lag}'] = df_engineered[target_col].shift(lag)
    
#     # Create rolling statistics
#     for window in [7, 30, 90]:
#         df_engineered[f'{target_col}_rolling_mean_{window}'] = df_engineered[target_col].rolling(window=window).mean()
#         df_engineered[f'{target_col}_rolling_std_{window}'] = df_engineered[target_col].rolling(window=window).std()
    
#     # Drop NaN values
#     df_engineered = df_engineered.dropna()
#     return df_engineered

# def prepare_data(df):
#     """Prepare features and target"""
#     target = 'Gold_Spot'
#     features = [col for col in df.columns if col != target]
    
#     X = df[features]
#     y = df[target]
#     return X, y

# def train_models(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled):
#     """Train multiple models and evaluate them"""
#     models = {
#         'Linear Regression': LinearRegression(),
#         'Ridge Regression': Ridge(),
#         'Lasso Regression': Lasso(),
#         'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
#         'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
#         'SVR': SVR()
#     }
    
#     results = {}
#     for name, model in models.items():
#         print(f"Training {name}...")
        
#         if name == 'SVR':
#             model.fit(X_train_scaled, y_train)
#             y_pred = model.predict(X_test_scaled)
#         else:
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)
        
#         # Calculate metrics
#         mse = mean_squared_error(y_test, y_pred)
#         mae = mean_absolute_error(y_test, y_pred)
#         r2 = r2_score(y_test, y_pred)
        
#         results[name] = {
#             'model': model,
#             'MSE': mse,
#             'MAE': mae,
#             'R2': r2,
#             'RMSE': np.sqrt(mse),
#             'predictions': y_pred
#         }
        
#         print(f"  R² Score: {r2:.4f}")
#         print(f"  RMSE: {np.sqrt(mse):.2f}")
#         print(f"  MAE: {mae:.2f}\n")
    
#     return results

# def compare_models(results):
#     """Compare and display model performance"""
#     results_df = pd.DataFrame({name: {k: v for k, v in data.items() if k != 'model' and k != 'predictions'} 
#                               for name, data in results.items()}).T
    
#     print("Model Performance Comparison:")
#     print(results_df.sort_values('R2', ascending=False))
    
#     # Plot results
#     fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
#     # R² Score
#     axes[0,0].bar(results_df.index, results_df['R2'])
#     axes[0,0].set_title('R² Score Comparison')
#     axes[0,0].set_ylabel('R² Score')
#     axes[0,0].tick_params(axis='x', rotation=45)
    
#     # RMSE
#     axes[0,1].bar(results_df.index, results_df['RMSE'])
#     axes[0,1].set_title('RMSE Comparison')
#     axes[0,1].set_ylabel('RMSE')
#     axes[0,1].tick_params(axis='x', rotation=45)
    
#     # MAE
#     axes[1,0].bar(results_df.index, results_df['MAE'])
#     axes[1,0].set_title('MAE Comparison')
#     axes[1,0].set_ylabel('MAE')
#     axes[1,0].tick_params(axis='x', rotation=45)
    
#     plt.tight_layout()
#     plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
#     plt.show()

# def save_best_model(results, scaler):
#     """Save the best performing model"""
#     best_model_name = max(results.keys(), key=lambda x: results[x]['R2'])
#     best_model = results[best_model_name]['model']
    
#     print(f"Best model: {best_model_name} (R² = {results[best_model_name]['R2']:.4f})")
    
#     # Save model and scaler
#     import joblib
#     joblib.dump(best_model, 'best_model.pkl')
#     joblib.dump(scaler, 'scaler.pkl')
#     print("Model and scaler saved as 'best_model.pkl' and 'scaler.pkl'")

# if __name__ == "__main__":
#     main()

