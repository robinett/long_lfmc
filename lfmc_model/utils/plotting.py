import seaborn as sns
import matplotlib.pyplot as plt
import sys
import numpy as np
import matplotlib.dates as mdates

def kde_plot(
    data, data_names, save_name, title=None,
    xlabel=None, ylabel=None
):
    """
    Create a Kernel Density Estimate (KDE) plot for a specified column in the data.

    Parameters:
    - data: List of np arrays containing the data.
    - data_names: the names of the different data sets.
    - xlabel: Label for the x-axis (optional).
    - ylabel: Label for the y-axis (optional).

    Returns:
    - ax: The axes object of the plot.
    """
    plt.figure(figsize=(10, 6))
    for d,dat in enumerate(data):
        sns.kdeplot(dat, label=data_names[d])
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

def plot_multiple_timeseries_from_df(
    df,
    date_col,
    x_label,
    y_label,
    save_name
):
    # get the columns that are not date
    columns = df.columns[df.columns != date_col]
    dates = df[date_col].values
    fig,ax = plt.subplots(figsize=(10, 6))
    for c,col in enumerate(columns):
        ax.plot(dates, df[col].values, label=col)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.legend()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.xticks(rotation=45)
    plt.savefig(save_name,bbox_inches='tight')
    plt.close()

def pred_obs_scatter(
    preds,
    obs,
    plot_path,
    mae=None,
    rmse=None,
    r2=None
):
    plt.figure(figsize=(6, 6))
    plt.scatter(obs, preds, alpha=0.5)
    max_val = max(np.max(obs), np.max(preds))
    min_val = min(np.min(obs), np.min(preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')
    plt.xlabel('Observed')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Observed')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend()
    if mae is not None:
        plt.text(0.05, 0.95, f'MAE: {mae:.3f}', transform=plt.gca().transAxes, verticalalignment='top')
    if rmse is not None:
        plt.text(0.05, 0.90, f'RMSE: {rmse:.3f}', transform=plt.gca().transAxes, verticalalignment='top')
    if r2 is not None:
        plt.text(0.05, 0.85, f'R²: {r2:.3f}', transform=plt.gca().transAxes, verticalalignment='top')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()