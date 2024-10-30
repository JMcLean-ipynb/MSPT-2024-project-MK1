import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from IPython.display import clear_output

def generate_2D_KDE(data, x='median_mass', y='Deff_global', x_range=(0,200),
                    y_range=(-1,1), figsize=(5,5), traj_length=5, density=None,
                    n_levels=12, cmap=mpl.cm.gray_r, alpha=1.0, show=True):
    '''
    Calculate 2D-KDE to visualize the distribution of molecular mass and diffusion of single particles.

    KDE is performed using scikit-learn's KernelDensity.

    Parameters
    ----------
    data : pandas DataFrame
        DataFrame containing the mass and diffusion information obtained from MSPT analysis.
    x : str, optional
        x variable. Choose between 'median_mass', 'mean_mass', or their contrast analogues ('med_c', 'mean_c'). The default is 'median_mass'.
    y : str, optional
        y variable. Choose between 'D_MSD' (MSD fit), 'D_JDD' (JDD fit), 
        'Deff_JDD_2c' (2-component JDD fit), 'Deff_JDD' (1 or 2-component JDD fit),
        'D_MSD_JDD' (global MSD and JDD fit), 'Deff_MSD_JDD_2c'
        (global MSD and JDD fit, 2 components), 'Deff_MSD_JDD'
        (global MSD and JDD fit, 1 or 2 components).
        The default is 'D_MSD'.
    x_range : (float, float), optional
        x axis limit. The default is (0,200).
    y_range : (float, float), optional
        y axis limit in log scale. The default is (-1,1).
    figsize : (float, float), optional
        Size of figure frame in inches. The default is (5,5).
    traj_length : int or None, optional
        Minimum trajectory length in frames. The default is 5.
    density : int, (int, int), or None, optional
        Upper limit (if int) or interval (if tuple) of membrane particle
        density in units of trajectory numbers. Each trajectory is assigned
        to an apparent membrane protein density determined as the median number
        of all trajectories detected during the trajectory’s lifetime.
        The default is None.
    n_levels : int, optional
        Determines the number of contour lines. The default is 12.
    cmap : str or Colormap, optional
        A Colormap instance or registered colormap name.
        The default is mpl.cm.gray_r.
    alpha : float, optional
        The alpha blending value, between 0 (transparent) and 1 (opaque).
        The default is 1.0.
    show : bool, optional
        Display figure. The default is True.

    Returns
    -------
    fig : Figure instance

    list
        axes objects of the plot. The list contains the main axis containing
        the 2D-KDE image (axs0), marginal distributions on top (axs1) and
        right (axs2), and the colorbar (cax).
    df_kde : pandas DataFrame
        DataFrame containing 2D-KDE result.
    data_plotted : pandas DataFrame
        filtered DataFrame.

    '''
    assert x in ('median_mass', 'mean_mass', 'med_c', 'mean_c'), 'Choose between median_mass, mean_mass, med_c, or mean_c'
    assert y in ('D_MSD', 'D_JDD', 'Deff_JDD_2c', 'Deff_JDD', 'D_MSD_JDD', 'Deff_MSD_JDD_2c', 'Deff_MSD_JDD'), 'Choose between D_MSD, D_JDD, Deff_JDD_2c, Deff_JDD, D_MSD_JDD, Deff_MSD_JDD_2c, or Deff_MSD_JDD'
    
    fig = plt.figure(figsize=figsize)
    
    im_fraction = 0.55
    axs0 = fig.add_axes((0.25, 0.2, im_fraction*5.0/6.0, im_fraction*5.0/6.0))
    axs1 = fig.add_axes((0.25, 0.2+im_fraction*5.0/6.0, im_fraction*5.0/6.0, im_fraction*1.0/6.0), xticklabels=[])
    axs2 = fig.add_axes((0.25 + im_fraction*5.0/6.0, 0.2, im_fraction*1.0/6.0, im_fraction*5.0/6.0), yticklabels=[])
    cax = fig.add_axes([0.825, 0.2, 0.025, im_fraction*5.0/6.0])
    
    mmin, mmax = x_range
    Dmin, Dmax = y_range
    
    dat = data.copy()
    
    if density:
        if len(density) == 1:
            dat = dat[dat['particle number (linked)'] <= density]
        if len(density) == 2:
            dat = dat[dat['particle number (linked)'].between(density[0], density[1], inclusive='both')]

    if traj_length:
        dat = dat[dat['len'] >= traj_length]

    # Filter out results with a negative median mass
    dat = dat[dat['median_mass'] >= 0]

    # Filter out unsuccessful fits
    if y == 'D_JDD':
        dat = dat[dat['fit_JDD_success'] == 1]
    elif y == 'Deff_JDD_2c':
        dat = dat[dat['fit_JDD_2c_success'] == 1]
    elif y == 'Deff_JDD':
        dat = dat[(dat['fit_JDD_success'] == 1) | (dat['fit_JDD_2c_success'] == 1)]
    elif y == 'D_MSD_JDD':
        dat = dat[dat['fit_MSD_JDD_1c_success'] == 1]
    elif y == 'Deff_MSD_JDD_2c':
        dat = dat[dat['fit_MSD_JDD_2c_success'] == 1]
    elif y == 'Deff_MSD_JDD':
        dat = dat[(dat['fit_MSD_JDD_1c_success'] == 1) | (dat['fit_MSD_JDD_2c_success'] == 1)]
    
    # Filter out NaNs in y variable originating from non-physical fit results
    dat = dat[~dat[y].isna()]
    
    # Filter out very slow particles (and negative diffusion coeffs)
    dat = dat[dat[y] > 0.0001]
    
    # Filter out particles where first 3 mean squared displacements are not
    # monotonously increasing
    dat = dat[dat['MSD_check'] == True]

    # Prepare the data for KDE
    X = np.vstack([dat[x].values, np.log10(dat[y].values)]).T

    # Define the grid
    x_grid = np.linspace(mmin, mmax, 512)
    y_grid = np.linspace(Dmin, Dmax, 512)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    grid_samples = np.vstack([X_grid.ravel(), Y_grid.ravel()]).T

    # Fit the KDE model
    kde = KernelDensity(bandwidth=1.0, kernel='gaussian')
    kde.fit(X)

    # Evaluate the density model on the grid
    log_density = kde.score_samples(grid_samples)
    density = np.exp(log_density).reshape(X_grid.shape)

    # Plot the KDE result
    fig, ax = plt.subplots(figsize=figsize)
    cs = ax.contourf(X_grid, Y_grid, density, levels=n_levels, cmap=cmap, alpha=alpha)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    # Plot marginal distributions
    axs1.plot(x_grid, density.sum(axis=0))
    axs2.plot(density.sum(axis=1), y_grid)

    if show:
        plt.show()

    # Prepare the results DataFrame
    df_kde = pd.DataFrame(grid_samples, columns=[x, y])
    df_kde['density'] = np.exp(log_density)

    return fig, [ax, axs1, axs2, cax], df_kde, dat
