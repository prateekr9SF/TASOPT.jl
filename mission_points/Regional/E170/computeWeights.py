"""
Script for visualizing frequency-weighted integration points over a Mach–CL 2D histogram.

This script reads a CSV file containing frequency data binned over Mach and lift coefficient (CL),
then selects a specified number of integration points uniformly distributed over a bounding box.
It computes weights for each integration point by summing nearby frequency bins and normalizes
these weights for numerical integration. The resulting integration points and their weights are
overlaid on the 2D histogram and saved as figures.

Modules:
- `get_bin_sizes(df)`:
    Determines the minimum bin sizes in Mach and CL directions.

- `generate_uniform_points(mach_min, mach_max, cl_min, cl_max, grid_dim)`:
    Generates a grid of uniformly spaced integration points within the Mach–CL domain.

- `compute_frequencies_and_weights(mach_points, cl_points, df, mach_bin_size, cl_bin_size, filter_radius)`:
    Computes weights for integration points based on surrounding bin frequencies.

- `plot_2d_histogram(...)`:
    Plots the 2D histogram of frequency data and optionally overlays the bounding box
    and integration points with size-scaled markers.

- `main()`:
    Executes the full pipeline using 1, 4, and 9 integration points and outputs plots
    and printed weight tables.

Expected CSV Format:
    The input CSV must contain the columns:
        - "Mach_bin": float
        - "CL_bin": float
        - "Frequency": float

Output:
    - PNG plots saved to `Plots/` directory for each integration scheme.
    - Printed DataFrames showing integration points and their corresponding weights.

Author:
    Prateek Ranjan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_bin_sizes(df):
    mach_bins = np.sort(df["Mach_bin"].unique())
    cl_bins = np.sort(df["CL_bin"].unique())
    mach_bin_size = np.min(np.diff(mach_bins))
    cl_bin_size = np.min(np.diff(cl_bins))
    return mach_bin_size, cl_bin_size


def generate_uniform_points(mach_min, mach_max, cl_min, cl_max, grid_dim):
    """
    Generate uniformly spaced integration points within a bounding box.
    
    Parameters:
        mach_min, mach_max: float — bounds of the Mach axis
        cl_min, cl_max: float — bounds of the CL axis
        grid_dim: int — number of total points (should be 1, 4, 9, etc. for square grids)

    Returns:
        mach_points: np.ndarray of Mach values
        cl_points: np.ndarray of CL values
    """
    if grid_dim == 1:
        mach_points = np.array([(mach_min + mach_max) / 2])
        cl_points = np.array([(cl_min + cl_max) / 2])
    else:
        n_per_axis = int(np.sqrt(grid_dim))
        if n_per_axis ** 2 != grid_dim:
            raise ValueError("grid_dim must be a perfect square (e.g., 1, 4, 9, 16, ...)")
        mach_vals = np.linspace(mach_min, mach_max, n_per_axis)
        cl_vals = np.linspace(cl_min, cl_max, n_per_axis)
        mach_grid, cl_grid = np.meshgrid(mach_vals, cl_vals)
        mach_points = mach_grid.flatten()
        cl_points = cl_grid.flatten()
    
    return mach_points, cl_points


def compute_frequencies_and_weights(mach_points, cl_points, df, mach_bin_size, cl_bin_size, filter_radius):
    total_frequency = df["Frequency"].sum()
    weights = []
    surrounding_bins_list = []

    for mach, cl in zip(mach_points, cl_points):
        surrounding = df[
            (df["Mach_bin"] >= mach - filter_radius * mach_bin_size) & (df["Mach_bin"] <= mach + filter_radius * mach_bin_size) &
            (df["CL_bin"] >= cl - filter_radius * cl_bin_size) & (df["CL_bin"] <= cl + filter_radius * cl_bin_size)
        ]
        surrounding_bins_list.append(surrounding)
        relative_freq = surrounding["Frequency"].sum() / total_frequency if not surrounding.empty else 0
        weights.append(relative_freq)

    weights = np.array(weights)
    weights /= np.sum(weights)

    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("Sum of weights ≠ 1. Check normalization.")

    return mach_points, cl_points, weights, surrounding_bins_list


def plot_2d_histogram(df, mach_min, mach_max, cl_min, cl_max,
                      mach_points, cl_points, surrounding_bins_list,
                      weights, plot_points, plot_box, n_points):
    
    # Set up plot
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    MMo = 0.82
    fig, ax1 = plt.subplots()

    histogram_data = df.pivot(index="CL_bin", columns="Mach_bin", values="Frequency")
    mach_bins = df["Mach_bin"].unique()
    cl_bins = df["CL_bin"].unique()

    pcm = ax1.pcolor(mach_bins, cl_bins, histogram_data, cmap=cmap,
                     shading='auto', vmin=0, vmax=200, alpha=0.8)

    cbar = fig.colorbar(pcm, ax=ax1, label='Frequency')
    ticks = [0, 25, 50, 75, 100, 125, 150, 175, 200]
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=20, labelrotation=0, labelcolor='black')
    cbar.set_label('Frequency', fontsize=20, fontname="Times New Roman")
    for label in cbar.ax.get_yticklabels():
        label.set_fontname("Times New Roman")

    # Bounding box
    if plot_box or plot_points:
        bbox_x = [mach_min, mach_max, mach_max, mach_min, mach_min]
        bbox_y = [cl_min, cl_min, cl_max, cl_max, cl_min]
        ax1.plot(bbox_x, bbox_y, color='gray', linewidth=2, linestyle='dashed')

    # Integration points
    if plot_points:
        integration_point_sizes = weights * 1000
        ax1.scatter(mach_points, cl_points, color='C0', marker='P',
                    edgecolor='black', s=integration_point_sizes)

    # Vertical line at Mach = 0.89
    ax1.axvline(x=MMo, color='black', linewidth=2.5, linestyle=':')

    # Axes settings
    ax1.set_xlabel('Mach', fontsize=22, fontname="Times New Roman")
    ax1.set_ylabel(r'C$_L$', fontsize=22, fontname="Times New Roman")
    ax1.set_xlim(0.59, 0.90)
    ax1.set_ylim(0.44, 0.58)

    ax1.tick_params(bottom=True, top=False, left=True, right=True)
    ax1.tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=False)
    ax1.tick_params(which='major', length=10, width=1.2, direction='in')
    ax1.tick_params(which='minor', length=5, width=1.2, direction='in')

    for axis in ['top', 'bottom', 'left', 'right']:
        ax1.spines[axis].set_linewidth(1.5)

    plt.xticks(fontname="Times New Roman", fontsize=20)
    plt.yticks(fontname="Times New Roman", fontsize=20)

    F = plt.gcf()
    Size = F.get_size_inches()
    F.set_size_inches(Size[0] * 1.5, Size[1] * 1.5, forward=True)

    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.tight_layout()

    # Save plots
    save_dir = "Plots"
    filename = f"{save_dir}/E170_Mach_CL_hist_weights_{n_points}pts.png"
    plt.savefig(filename)
    plt.close()


def main():
    # Load data
    df = pd.read_csv("data/E170_histogram_data.csv")
    mach_bin_size, cl_bin_size = get_bin_sizes(df)

    # Domain bounds
    mach_min, mach_max = 0.70, 0.82
    cl_min, cl_max = 0.48, 0.54
    filter_radius = 4

    for n_points in [1, 4, 9]:
        mach_pts, cl_pts = generate_uniform_points(mach_min, mach_max, cl_min, cl_max, n_points)
        mach_pts, cl_pts, weights, surrounding_bins_list = compute_frequencies_and_weights(
            mach_pts, cl_pts, df, mach_bin_size, cl_bin_size, filter_radius
        )

        plot_box = False
        plot_points = False
        plot_2d_histogram(df, mach_min, mach_max, cl_min, cl_max,
                          mach_pts, cl_pts, surrounding_bins_list,
                          weights, plot_points, plot_box, n_points)

        print(f"\nNumerical Weights for {n_points} Integration Point(s):")
        print(pd.DataFrame({
            "Mach": mach_pts,
            "CL": cl_pts,
            "Weight": np.round(weights, 4)
        }))


if __name__ == "__main__":
    main()
