import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.tri as tri

# Import the data csv file using pandas
df1 = pd.read_csv('QE_48E-06.csv', skiprows=0)

Arr1 = df1.to_numpy()

# Extract all performance properties
Ma = Arr1[:, 0]
CL = Arr1[:, 1]
MLD = Arr1[:, 3]

# Define the number of levels for the contour plot
num_levels = 50

# Define the min and max values for the colorbar
vmin = np.min(MLD)
vmax = np.max(MLD)

# Create the contour plot
fig1 = plt.figure()
ax1 = fig1.gca()

triang = tri.Triangulation(Ma, CL)
refiner = tri.UniformTriRefiner(triang)

tri_refi, MLD_refi = refiner.refine_field(MLD, triinterpolator=None, subdiv=5)

# Create contour plot
contour = ax1.tricontour(tri_refi, MLD_refi, levels=num_levels, cmap="RdBu_r", vmin=vmin, vmax=vmax)
#ax1.clabel(contour, contour.levels, fontsize=12, fmt='%1.4f', colors='k')

# Create filled contour plot
contour_filled = ax1.tricontourf(tri_refi, MLD_refi, levels=num_levels, cmap="RdBu_r",  vmin=vmin, vmax=vmax)
cbar = fig1.colorbar(contour_filled, ax=ax1)
cbar.set_label('ML/D', fontsize=16, fontname="Times New Roman")
cbar.ax.tick_params(labelsize=12)

# Set axis labels
ax1.set_xlabel('Mach', fontsize=20, fontname="Times New Roman")
ax1.set_ylabel('$C_L$', fontname="Times New Roman", fontsize=20)

# Modify axis spines
ax1.spines['top'].set_visible(True)
ax1.spines['right'].set_visible(True)

# Modify axes ticks properties
plt.xticks(fontname="Times New Roman", fontsize=16)
plt.yticks(fontname="Times New Roman", fontsize=16)

# Adjust figure size
F = plt.gcf()
Size = F.get_size_inches()
F.set_size_inches(Size[0] * 1.5, Size[1] * 1.5, forward=True)

plt.savefig("B772-MLD.png")

plt.show()


