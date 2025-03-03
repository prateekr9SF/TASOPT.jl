import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from numpy.polynomial.legendre import leggauss

"""
Script for analyzing and visualizing Mach-CL histograms with integration points.

This script processes a dataset of Mach and CL frequency distributions, computes bin sizes,
generates integration points, and visualizes the data using 2D histograms. It supports Gaussian quadrature-based integration point selection and frequency-based weight normalization.

Dependencies:
    - numpy
    - pandas
    - matplotlib
    - seaborn
    - scipy

Functions:
    - get_bin_sizes(df): Computes the bin size for Mach and CL from the dataset.
    - plot_2d_histogram(df, ...): Plots a 2D histogram with integration points and bounding box.
    - compute_frequencies_and_weights(mach_points, cl_points, df, ...): Computes normalized weights based on frequency distribution.
    - generate_9_uniform_points(mach_min, mach_max, cl_min, cl_max): Generates 9 evenly spaced integration points in the bounding box.

Example Usage:
    ```python
    file_path = "data/E170_histogram_data.csv"
    df = pd.read_csv(file_path)
    mach_bin_size, cl_bin_size = get_bin_sizes(df)
    mach_points_9, cl_points_9, weights_9 = generate_9_uniform_points(mach_min, mach_max, cl_min, cl_max)
    mach_points_9, cl_points_9, weights_9, surrounding_bins_list = compute_frequencies_and_weights(mach_points_9, cl_points_9, df, mach_bin_size, cl_bin_size, filter_radius)
    plot_2d_histogram(df, mach_min, mach_max, cl_min, cl_max, mach_points_9, cl_points_9, surrounding_bins_list, weights_9, plot_box, plot_points)
    ```

Author: Prateek Ranjan, Ph.D
Date: March 2025
"""


def get_bin_sizes(df):
    """Return the bin size for Mach and CL from the CSV file."""
    mach_bins = np.sort(df["Mach_bin"].unique())
    cl_bins = np.sort(df["CL_bin"].unique())
    mach_bin_size = np.min(np.diff(mach_bins))
    cl_bin_size = np.min(np.diff(cl_bins))
    return mach_bin_size, cl_bin_size

def plot_2d_histogram(df, mach_min, mach_max, cl_min, cl_max, mach_points_5, cl_points_5, surrounding_bins_list, weights_5,plot_box, plot_points):
    """Plot the 2D histogram with integration points using exact frequency values."""
    
    # Use Seaborn rocket_r colormap
    cmap = sns.color_palette("rocket_r", as_cmap=True)
    
    MMo = 0.82
    
    fig, ax1 = plt.subplots()
    histogram_data = df.pivot(index="CL_bin", columns="Mach_bin", values="Frequency")
    mach_bins = df["Mach_bin"].unique()
    cl_bins = df["CL_bin"].unique()
    
    pcm = ax1.pcolor(mach_bins, cl_bins, histogram_data, cmap=cmap, shading='auto', vmin = 0, vmax = 200, alpha = 0.8)
    
    # Configure the colorbar
    cbar = fig.colorbar(pcm, ax=ax1, label='Frequency')
    
    # Define linear ticks for the colorbar
    ticks = [0, 25, 50, 75, 100, 125, 150, 175, 200]  # Linear tick positions
    cbar.set_ticks(ticks)
    
    # Customize the colorbar appearance
    cbar.ax.tick_params(labelsize=20, labelrotation=0, labelcolor='black')
    cbar.set_label('Frequency', fontsize=20, fontname="Times New Roman")
    
    # Ensure tick labels use the desired font
    for label in cbar.ax.get_yticklabels():
        label.set_fontname("Times New Roman")
    
    if plot_box and plot_points == False:     
    # Add the bounding box
        bbox_x = [mach_min, mach_max, mach_max, mach_min, mach_min]
        bbox_y = [cl_min, cl_min, cl_max, cl_max, cl_min]
        plt.plot(bbox_x, bbox_y, color='gray', linewidth=3, linestyle='dashed')
    
    # Add the 5 integration points
    #plt.scatter(mach_points_5, cl_points_5, color='C0', marker='o', edgecolor='black', s=100, label="Integration Points")
    
    # Add a vertical dashed line at Mach = 0.80 for all Reynolds numbers
    ax1.axvline(x=MMo, color='black', linewidth=2.5, linestyle=':')
    
    
    if plot_points and plot_box == False:
        
        # Add the bounding box
        bbox_x = [mach_min, mach_max, mach_max, mach_min, mach_min]
        bbox_y = [cl_min, cl_min, cl_max, cl_max, cl_min]
        plt.plot(bbox_x, bbox_y, color='gray', linewidth=2, linestyle='dashed')
        
       # Highlight surrounding bins in gray
       # for surrounding_bins in surrounding_bins_list:
        #    plt.scatter(surrounding_bins["Mach_bin"], surrounding_bins["CL_bin"], color='gray', marker='s', s=50, alpha=0.5)
       
        # Add the 5 integration points with size proportional to weight
        integration_point_sizes = weights_5 * 1000  # Scale weights for better visibility
        plt.scatter(mach_points_5, cl_points_5, color='C0', marker='P', edgecolor='black', s=integration_point_sizes)    

    # Add labels and title
    ax1.set_xlabel('Mach', fontsize=22, fontname="Times New Roman")
    ax1.set_ylabel(r'C$_L$', fontsize=22, fontname="Times New Roman")
    
    # Adjust axis ticks and spines
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
    
    ax1.set_xlim(0.59, 0.90)
    ax1.set_ylim(0.44, 0.58)
    plt.tight_layout()

    if plot_points:
        plt.savefig("Plots/B738_Mach_CL_hist_weights.png")
    if plot_box:
        plt.savefig("Plots/B738_Mach_CL_hist_box.png")
    else:
        plt.savefig("Plots/B738_Mach_CL_hist.png")


def compute_frequencies_and_weights(mach_points, cl_points, df, mach_bin_size, cl_bin_size, filter_radius):
    """Compute average frequencies from surrounding bins and normalize weights. Ensure sum of weights is 1."""
    frequencies = []
    surrounding_bins_list = []
    total_frequency = df["Frequency"].sum()
    
    for mach, cl in zip(mach_points, cl_points):
        surrounding_bins = df[(df["Mach_bin"] >= mach - filter_radius * mach_bin_size) & (df["Mach_bin"] <= mach + filter_radius * mach_bin_size) &
                              (df["CL_bin"] >= cl - filter_radius * cl_bin_size) & (df["CL_bin"] <= cl + filter_radius * cl_bin_size)]
        surrounding_bins_list.append(surrounding_bins)
        relative_freq = surrounding_bins["Frequency"].sum() / total_frequency if not surrounding_bins.empty else 0
        frequencies.append(relative_freq)
    
    frequencies = np.array(frequencies)
    weights = frequencies / np.sum(frequencies)
    
    if not np.isclose(np.sum(weights), 1.0):
        raise ValueError("Sum of weights does not equal 1. Check frequency data and normalization.")
    
    return mach_points, cl_points, weights, surrounding_bins_list


def generate_9_uniform_points(mach_min, mach_max, cl_min, cl_max):
    """Generate 9 uniformly spaced integration points within the bounding box."""
    mach_points = np.linspace(mach_min, mach_max, 3)
    cl_points = np.linspace(cl_min, cl_max, 3)
    
    # Generate 9-point grid
    mach_grid, cl_grid = np.meshgrid(mach_points, cl_points)
    mach_points_9 = mach_grid.flatten()
    cl_points_9 = cl_grid.flatten()
    
    # Assign equal weights to all points
    weights_9 = np.full_like(mach_points_9, 1 / len(mach_points_9))
    
    return mach_points_9, cl_points_9, weights_9


# Load the dataset
file_path = "data/B738_histogram_data.csv"
df = pd.read_csv(file_path)

# Get bin sizes
mach_bin_size, cl_bin_size = get_bin_sizes(df)

# Define the bounding box limits
mach_min, mach_max = 0.70, 0.82
cl_min, cl_max = 0.48, 0.54
filter_radius = 4

# Generate 9 integration points using Gaussian quadrature
mach_points_9, cl_points_9, weights_9 = generate_9_uniform_points(mach_min, mach_max, cl_min, cl_max)

# Compute frequencies and adjusted weights
mach_points_9, cl_points_9, weights_9, surrounding_bins_list = compute_frequencies_and_weights(mach_points_9, cl_points_9, df, mach_bin_size, cl_bin_size, filter_radius)


plot_box = False
plot_points = True


# Plot the 2D histogram with integration points
plot_2d_histogram(df, mach_min, mach_max, cl_min, cl_max, mach_points_9, cl_points_9, surrounding_bins_list, weights_9, plot_box, plot_points)

# Display numerical weights
print("\nNumerical Weights for 9 Integration Points:")
print(pd.DataFrame({"Mach": mach_points_9, "CL": cl_points_9, "Weight": np.round(weights_9, 4)}))
