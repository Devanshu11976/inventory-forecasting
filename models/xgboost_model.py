import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from models.base import BaseModel
from features.engineering import engineer_features

class XGBoostModel(BaseModel):
    """
    XGBoost Model (Section III-C: 200 estimators, max_depth=4, learning_rate=0.05).
    Also includes a tune=True mode using GridSearchCV.
    """
    def __init__(self, tune=False, n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42):
        super().__init__()
        self.tune = tune
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': random_state,
            'objective': 'reg:absoluteerror',  # Optimizes for Mean Absolute Error
            'n_jobs': -1
        }
        self.model = None
        self.history_df = None
        self.feature_cols = [
            'lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14',
            'rolling_mean_7', 'rolling_mean_14',
            'day_of_week', 'day_of_year'
        ]
        
    def fit(self, train_df: pd.DataFrame):
        """
        Fits the XGBoost model on the training dataframe after feature engineering.
        """
        self.history_df = train_df.copy()
        
        # Engineer features
        engineered = engineer_features(self.history_df)
        
        # Drop rows with NaN (the first 14 days are NaNs because of lags/rolling windows)
        train_data = engineered.dropna(subset=self.feature_cols).copy()
        
        X_train = train_data[self.feature_cols]
        y_train = train_data['units_sold']
        
        if self.tune:
            # Hyperparameter tuning over grid (Section III-C)
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 5],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 1.0]
            }
            grid_search = GridSearchCV(
                estimator=XGBRegressor(objective='reg:absoluteerror', random_state=self.params['random_state'], n_jobs=-1),
                param_grid=param_grid,
                scoring='neg_mean_absolute_error',
                cv=3,
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_params = grid_search.best_params_
            self.params.update(best_params)
            
        self.model = XGBRegressor(**self.params)
        self.model.fit(X_train, y_train)
        
    def predict(self, n_days: int) -> list:
        """
        Forecasts n_days recursively. If n_days = 1, it forecasts the next day.
        """
        if self.model is None:
            raise ValueError("XGBoost model must be fitted before predicting.")
            
        predictions = []
        # Keep a copy of original history so we don't permanently mutate it during multi-step recursive forecasting
        original_history = self.history_df.copy()
        
        for _ in range(n_days):
            # 1. Determine the next date and product ID
            last_row = self.history_df.iloc[-1]
            next_date = pd.to_datetime(last_row['date']) + pd.Timedelta(days=1)
            product_id = last_row['product_id']
            
            # 2. Append placeholder row
            next_row = pd.DataFrame([{
                'product_id': product_id,
                'date': next_date,
                'units_sold': np.nan
            }])
            self.history_df = pd.concat([self.history_df, next_row], ignore_index=True)
            
            # 3. Re-engineer features to compute lags/rolling windows for the new row
            engineered = engineer_features(self.history_df)
            
            # 4. Extract feature vector of the last row
            X_pred = engineered[self.feature_cols].iloc[[-1]]
            
            # 5. Predict demand
            pred = self.model.predict(X_pred)[0]
            pred = max(0.0, float(pred))  # Clip demand at zero (Section III-B)
            predictions.append(pred)
            
            # 6. Fill in the prediction so it can act as lags for subsequent steps
            self.history_df.loc[self.history_df.index[-1], 'units_sold'] = pred
            
        # Restore original history
        self.history_df = original_history
        return predictions
        
    def update_history(self, actual_date, actual_units_sold: float):
        """
        Appends actual sales to history during walk-forward evaluation.
        """
        last_row = self.history_df.iloc[-1]
        next_row = pd.DataFrame([{
            'product_id': last_row['product_id'],
            'date': pd.to_datetime(actual_date),
            'units_sold': actual_units_sold
        }])
        self.history_df = pd.concat([self.history_df, next_row], ignore_index=True)
