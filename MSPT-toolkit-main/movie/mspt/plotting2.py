import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.neighbors import KernelDensity
from IPython.display import clear_output

def generate_2D_KDE(data, x='median_mass', y='Deff_global', x_range=(0,200), y_range=(-1,1),
                    figsize=(5,5), traj_length=5, density=None, n_levels=12,
                    cmap=mpl.cm.gray_r, alpha=1.0, show=True):
    """
    Calculate 2D-KDE to visualize the distribution of molecular mass and diffusion of single particles.

    Parameters:
    - data: pandas DataFrame containing the mass and diffusion information obtained from MSPT analysis.
    - x: string, x variable (default 'median_mass').
    - y: string, y variable (default 'Deff_global').
    - x_range: tuple, x axis limit (default (0, 200)).
    - y_range: tuple, y axis limit in log scale (default (-1, 1)).
    - figsize: tuple, size of figure frame in inches (default (5,5)).
    - traj_length: int or None, minimum trajectory length in frames (default 5).
    - density: int, (int, int), or None, density limits for filtering data (default None).
    - n_levels: int, number of contour levels (default 12).
    - cmap: matplotlib colormap or registered colormap name (default mpl.cm.gray_r).
    - alpha: float, alpha blending value, between 0 (transparent) and 1 (opaque) (default 1.0).
    - show: bool, whether to display the figure (default True).

    Returns:
    - fig: matplotlib figure instance.
    - ax: matplotlib axes instance.
    - Z: 2D array of KDE results.
    - data: filtered DataFrame used in the KDE.
    """
    # Filter data based on trajectory length and density requirements, and drop NaNs
    data = data.dropna(subset=[x, y])
    if traj_length is not None:
        data = data[data['len'] >= traj_length]
    if density is not None:
        if isinstance(density, tuple) and len(density) == 2:
            data = data[data['particle number (linked)'].between(density[0], density[1])]
        else:
            data = data[data['particle number (linked)'] <= density]

    # Ensure there are no NaNs before transformation
    data = data.dropna(subset=[x, y])
    values = data[[x, y]]
    values[y] = np.log10(values[y])  # Log transformation of y

    # Prepare grid for KDE
    x_lin = np.linspace(x_range[0], x_range[1], 500)
    y_lin = np.logspace(y_range[0], y_range[1], 500)
    X, Y = np.meshgrid(x_lin, y_lin)
    sample_points = np.vstack([X.ravel(), np.log10(Y.ravel())]).T

    # Perform KDE
    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
    kde.fit(values)
    log_dens = kde.score_samples(sample_points)
    Z = np.exp(log_dens).reshape(X.shape)

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    cset = ax.contourf(X, Y, Z, levels=np.linspace(Z.min(), Z.max(), n_levels), cmap=cmap, alpha=alpha)
    ax.set_xlabel(x)
    ax.set_ylabel('Log10(' + y + ')')
    ax.set_xlim(x_range)
    ax.set_ylim(10**np.array(y_range))

    # Colorbar
    cbar = fig.colorbar(cset)
    cbar.ax.set_ylabel('Density')

    # Display the plot if required
    if show:
        plt.show()

    return fig, ax, Z, data
