import pandas as pd
import numpy as np
import warnings
from models.base import BaseModel
from statsmodels.tsa.arima.model import ARIMA

class ARIMAModel(BaseModel):
    """
    ARIMA Model (Section III-C: fixed order (2,1,2), MLE fit, forecasts full test horizon directly).
    Also includes auto_arima mode to tune p,d,q on train data.
    """
    def __init__(self, order=(2, 1, 2), auto_arima=False):
        super().__init__()
        self.order = order
        self.auto_arima = auto_arima
        self.model_fit = None
        self.train_series = None
        
    def fit(self, train_df: pd.DataFrame):
        """
        Fits the ARIMA model on the training series.
        """
        self.train_series = train_df['units_sold'].values.astype(float)
        
        if self.auto_arima:
            # Tuned ARIMA mode (AIC grid search over p, d, q in [0, 3])
            try:
                import pmdarima as pm
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    auto_model = pm.auto_arima(
                        self.train_series,
                        start_p=0, start_q=0,
                        max_p=3, max_d=3, max_q=3,
                        seasonal=False,
                        stepwise=True,
                        suppress_warnings=True,
                        error_action="ignore"
                    )
                self.order = auto_model.order
            except Exception as e:
                # Fallback to manual grid search if pmdarima is not available
                best_aic = float('inf')
                best_order = self.order
                for p in range(4):
                    for d in range(2):  # Limit d to 0 or 1 for model stability
                        for q in range(4):
                            try:
                                with warnings.catch_warnings():
                                    warnings.simplefilter("ignore")
                                    model = ARIMA(self.train_series, order=(p, d, q))
                                    res = model.fit()
                                    if res.aic < best_aic:
                                        best_aic = res.aic
                                        best_order = (p, d, q)
                            except:
                                continue
                self.order = best_order
                
        # Fit final ARIMA model using maximum likelihood
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(self.train_series, order=self.order)
                self.model_fit = model.fit()
        except Exception as e:
            # Fallback to ARIMA(1, 0, 0) for numerical stability if the fit fails to converge
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(self.train_series, order=(1, 0, 0))
                self.model_fit = model.fit()
            
    def predict(self, n_days: int) -> list:
        """
        Forecasts full test horizon directly.
        """
        if self.model_fit is None:
            raise ValueError("ARIMA model must be fitted before predicting.")
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            forecast = self.model_fit.forecast(steps=n_days)
            
        # Sales cannot be negative, clip at zero (Section III-B / III-C consistency)
        return list(np.clip(forecast, 0.0, None))
