# The version has been deprecated, please use the latest version in Utools/draw.py
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error


def parity_plot(y_test, y_pred, r2_score=None, rmse=None, mae=None, fig_path=None):
    """
    Generate a parity plot to visualize the model's predictions against the true values.
    """
    fig = plt.figure(figsize=(8, 8), dpi=300)
    plt.scatter(y_test, y_pred, s=2, c="#9cc3e5", marker="^")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', c="#aad390", zorder=5)

    # inward ticks
    plt.tick_params(axis='both', direction='in', length=4) 
    plt.xlabel('Actual Band Gap(eV)', fontsize=12)
    plt.ylabel('Predicted Band Gap(eV)', fontsize=12)
    plt.title('Band Gap of Actual vs Predicted', fontsize=14)

    plt.grid(True, linestyle="--", alpha=0.4)

    # display the r2_score, rmse, mae in the plot
    if r2_score is not None and rmse is not None and mae is not None:
        bbox_props = dict(boxstyle="round,pad=0.5", ec="#aad390", lw=0.8, 
                        facecolor="white", alpha=0.8)
    
        plt.text(0.05, 0.95, 
                r"$R^2$: {:.3f}" "\n"
                r"$RMSE$: {:.3f} eV" "\n" 
                r"$MAE$: {:.3f} eV".format(r2_score, rmse, mae),
                fontsize=10, 
                color='#9cc3e5',
                transform=plt.gca().transAxes,  
                verticalalignment='top',  
                horizontalalignment='left', 
                bbox=bbox_props)
        
    # save figure
    if fig_path:
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()

# Model performance metrics
def model_performance(y_test, y_pred, fig_path=None):
    """
    Calculate and print model performance metrics.
    """
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # draw
    parity_plot(y_test, y_pred, r2_score=r2, rmse=rmse, mae=mae, fig_path=fig_path)
    return r2, rmse, mae

def plot_feature_importance(model, feature_names, top_n=10, fig_path=None):
    """plot the feature importance of the model"""
    importances = model.feature_importances_
    importances_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    importances_df = importances_df.sort_values(by='Importance', ascending=False).head(top_n)

    fig = plt.figure(figsize=(10, 6), dpi=300)
    sns.barplot(x='Importance', y='Feature', data=importances_df.iloc[:top_n], palette='GnBu_r', hue='Feature')
    # show importance value
    for i in range(top_n):
        plt.text(importances_df.iloc[i, 1], i, f'{importances_df.iloc[i, 1]:.3f}')
    plt.xlabel('Importance', size=13)
    plt.ylabel('Feature', size=13)
    plt.title('Feature Importance of Band Gap', size=15)
    plt.show()
    # save the figure
    if fig_path:
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    return importances_df