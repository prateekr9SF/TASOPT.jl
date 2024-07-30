import pandas as pd
import matplotlib.pyplot as plt
import csv
from matplotlib.colors import Normalize

plt.style.use('seaborn-v0_8-deep')

num_rows  =1000000

# Load the CSV file
file_path1 = 'A332/mach_number_climb_A332.csv'
file_path2 = 'A332/mach_number_cruise_A332.csv'
file_path3 = 'A332/mach_number_descent_A332.csv'

df1 = pd.read_csv(file_path1, nrows = num_rows)
df2 = pd.read_csv(file_path2, nrows = num_rows)
df3 = pd.read_csv(file_path3, nrows  =num_rows)


# Plot the histogram
fig1 = plt.figure()
ax1 = fig1.gca()


plt.hist(df1['mach'], bins=100, color='C0', edgecolor='black', label='Climb', alpha = 0.4)
plt.hist(df2['mach'], bins=100, color='C1', edgecolor='black', label='Cruise', alpha = 0.8)
plt.hist(df3['mach'], bins=100, color='C2', edgecolor='black', label='Descent', alpha = 0.5)

plt.xlabel('Mach Number' ,fontname = "Times New Roman", fontsize = 20)
plt.ylabel('Frequency',  fontname = "Times New Roman", fontsize = 20)

for axis in ['top', 'bottom', 'left', 'right']:
            ax1.spines[axis].set_linewidth(1.5)

 # Modify axes ticks properties
plt.xticks(fontname="Times New Roman", fontsize=20)
plt.yticks(fontname="Times New Roman", fontsize=20)

# Set legend properties
plt.legend(fontsize=22, frameon=False, prop={'family': 'Times New Roman'}, ncol = 3)

plt.xlim([0.2, 1])
plt.ylim([0, 300000])

F = plt.gcf()
Size = F.get_size_inches()
F.set_size_inches(Size[0]*1.7, Size[1]*1.5, forward=True)

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.savefig('Plots/A332_Distance_bin.png')

plt.show()


