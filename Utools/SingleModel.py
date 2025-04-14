import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .draw import model_performance  # use . to import the draw module from the same package

class SingleModel:
    """Class for training and evaluation of the model which be passed in.
       For the train dataset, we used the StandardScaler to scale the data.
    """
    
    def __init__(self, model, random_state=42):
        """
        Initialize the model wrapper
        
        Args:
            model: Any scikit-learn compatible model (e.g., RandomForestRegressor, LinearRegression)
            random_state: Random seed for reproducibility
        """
        self.pipe = Pipeline([
        ('scaler', StandardScaler()), # StandardScaler's mean is 0 and std is 1, Z-score normalization
        ('model', model)], verbose=True)

        self.random_state = random_state
        self.is_trained = False
    
    def get_model(self):
        """
        Get the underlying model
        
        Returns:
            model: The underlying model
        """
        return self.pipe.named_steps['model']
    
    def train(self, X_train, y_train):
        """
        Train the model with provided training data
        
        Args:
            X_train: Training features
            y_train: Training target values
        
        Returns:
            self: For method chaining
        """
        print("Starting model training...")
        
        # Fit the model
        self.pipe.fit(X_train, y_train)
        self.is_trained = True
        
        print("Model training completed!")
        return self
    
    def predict(self, X_test):
        """
        Make predictions using the trained model
        
        Args:
            X_test: Test data
            
        Returns:
            predictions: Model predictions
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained. Call train() first.")
            
        return self.pipe.predict(X_test)
    
    def evaluate(self, X_test, y_test, fig_path=None, fig_show=True):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test target values
            fig_path: Path to save the performance plot
            fig_show: Whether to show the plot
        Returns:
            metrics: Dictionary with performance metrics
        """

        # 
        def mean_absolute_percentage_error(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            # Avoid division by zero
            mask = y_true != 0
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        

        if not self.is_trained:
            raise ValueError("Model has not been trained. Call train() first.")
        
        # Test set evaluation
        test_pred = self.predict(X_test)
        test_r2, test_rmse, test_mae = model_performance(y_test, test_pred, fig_path=fig_path, fig_show=fig_show)
        test_mape = mean_absolute_percentage_error(y_test, test_pred)
        metrics = {
            'mae': test_mae,
            'rmse': test_rmse,
            'r2': test_r2,
            'mape': test_mape
        }
        
        # Print test metrics
        print("\nModel Evaluation Results:")
        print(f"Test set size: {len(X_test)}")
        print(f"Test set: R²: {test_r2:.4f}", f"RMSE: {test_rmse:.4f}", f"MAE: {test_mae:.4f}", f"MAPE: {test_mape:.4f}%")

        return metrics
    
    
    def save_model(self, file_path):
        """
        Save the trained model to a file
        
        Args:
            file_path: Path to save the model
            
        Returns:
            self: For method chaining
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained. Call train() first.")
        
        joblib.dump(self.pipe, file_path)    # save the scaler and model together as a pipeline
        print(f"Model saved to: {file_path}")
        return self
    
    def save_prediction(self, X_test, y_test, file_path, save_df=None):
        """
        Save predictions to a CSV file
        
        Args:
            X_test: Test features
            y_test: Test target values
            file_path: Path to save the predictions
            
        Returns:
            self: For method chaining
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained. Call train() first.")
        
        y_pred = self.predict(X_test)
        
        if save_df is not None:
            y_pred_df = save_df.copy()
            y_pred_df['predicted_band_gap'] = y_pred
            y_pred_df.to_csv(file_path, index=False)
        else:
            # Create DataFrame for predictions and true values
            df = pd.DataFrame({
                'True': y_test,
                'Predicted': y_pred
            })
        
            df.to_csv(file_path, index=False)
        
        print(f"Predictions saved to: {file_path}")
        return self
    

# Test
if __name__ == "__main__":

    from sklearn.ensemble import RandomForestRegressor
    
    # Example usage
    file_dir = os.path.join(os.getcwd(), '../Data/composition_data/feature_data')
    # exp data
    exp_train = pd.read_csv(os.path.join(file_dir, 'exp', 'train.csv'))
    exp_test = pd.read_csv(os.path.join(file_dir, 'exp', 'test.csv'))

    exp_train_X = exp_train.drop(columns=['composition', 'band_gap'])
    exp_train_y = exp_train['band_gap']    

    exp_test_X = exp_test.drop(columns=['composition', 'band_gap'])
    exp_test_y = exp_test['band_gap']
    
    # model
    rf_model = RandomForestRegressor(n_estimators=300, max_depth=None, max_features=0.25, random_state=42, n_jobs=-1)
    model = SingleModel(rf_model, random_state=42)
    model.train(exp_train_X, exp_train_y)
    model.evaluate(exp_test_X, exp_test_y, fig_path=os.path.join(file_dir, 'exp', 'rf_model.png'))
    model.save_model(os.path.join(file_dir, 'exp', 'rf_model.pkl'))
    model.save_prediction(exp_test_X, exp_test_y, os.path.join(file_dir, 'exp', 'rf_model_pred.csv'), save_df=exp_test)