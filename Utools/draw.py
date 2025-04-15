import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error


import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error


import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error


def parity_plot(y_test, y_pred, r2_score=None, rmse=None, mae=None, fig_path=None, fig_show=True):
    """
    Generate a parity plot to visualize the model's predictions against the true values.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
    ax0 = axes[0]
    ax0.scatter(y_test, y_pred, s=2, c="#9cc3e5", marker="^")
    ax0.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', c="#aad390", zorder=5)

    # inward ticks
    ax0.tick_params(axis='both', direction='in', length=4) 
    ax0.set_xlabel('Actual Band Gap(eV)', fontsize=12)
    ax0.set_ylabel('Predicted Band Gap(eV)', fontsize=12)
    ax0.set_title('Band Gap of Actual vs Predicted', fontsize=14)
    ax0.set_xlim([y_test.min()-0.5, y_test.max()+0.5])
    ax0.set_ylim([y_test.min()-0.5, y_test.max()+0.5])
    ax0.set_aspect('equal', adjustable='box') 



    ax0.grid(True, linestyle="--", alpha=0.4)


    # display the r2_score, rmse, mae in the plot
    if r2_score is not None and rmse is not None and mae is not None:
        bbox_props = dict(boxstyle="round,pad=0.5", ec="#aad390", lw=0.8, 
                        facecolor="white", alpha=0.8)
    
        ax0.text(0.05, 0.95, 
                r"$R^2$: {:.3f}" "\n"
                r"$RMSE$: {:.3f} eV" "\n" 
                r"$MAE$: {:.3f} eV".format(r2_score, rmse, mae),
                fontsize=10, 
                color='#9cc3e5',
                transform=ax0.transAxes,  
                verticalalignment='top',  
                horizontalalignment='left', 
                bbox=bbox_props)
        
    # draw the histogram of the residuals
    ax1 = axes[1]
    # calculate the residuals: y_true - y_pred
    residuals = y_test - y_pred
    # 
    max_resid = round(residuals.max())
    min_resid = round(residuals.min())
    if (max_resid - min_resid) < 6:
        # if the range of residuals is less than 5, set the bin size to 0.5
        bin_edges = np.arange(min_resid - 0.25, max_resid + 0.25 + 1e-10, 0.5)
    else:
        # if the range of residuals is greater than 5, set the bin size to 1
        bin_edges = np.arange(min_resid - 0.5, max_resid + 0.5, 1)  
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # bin centers
    
    # draw the histogram
    sns.histplot(residuals, bins=bin_edges, kde=True, color="#9cc3e5", ax=ax1, stat="probability", alpha=0.5)
    ax1.set_xlabel('Residuals(eV)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Residuals Distribution', fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.tick_params(axis='both', direction='in', length=4)
    # make the ticks in the middle of the bins
    ax1.set_xticks(bin_centers)   # set x-ticks to bin centers

    # save figure
    if fig_path:
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {fig_path}")
    if fig_show:
        plt.show()
    else:
        plt.close(fig)

        
# Model performance metrics
def model_performance(y_test, y_pred, fig_path=None, fig_show=True):
    """
    Calculate and print model performance metrics.
    """
    r2 = r2_score(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # draw
    parity_plot(y_test, y_pred, r2_score=r2, rmse=rmse, mae=mae, fig_path=fig_path, fig_show=fig_show)
    return r2, rmse, mae



def plot_feature_importance(model, feature_names, top_n=10, fig_path=None, fig_show=True):
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
    # save the figure
    if fig_path:
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {fig_path}")
    if fig_show:
        plt.show()
    else:
        plt.close(fig)
    return importances_df