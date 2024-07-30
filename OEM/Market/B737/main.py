import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

# Apply the seaborn dark style sheet
plt.style.use('seaborn-v0_8-deep')

# Load the Excel file to examine its contents
file_path = '737CurrentFleet.xlsx'
xls = pd.ExcelFile(file_path)

# Setting font properties for axis labels and titles
font_properties = {'family': 'Times New Roman', 'size': 12}

# Display sheet names to understand the structure of the file
sheet_names = xls.sheet_names

# Load the sheets into DataFrames to analyze their contents
aircraft_detail_df = pd.read_excel(xls, sheet_name='Aircraft Detail')
export_detail_df = pd.read_excel(xls, sheet_name='Export Detail')

# Display the first few rows of each sheet to understand their structure
aircraft_detail_df_head = aircraft_detail_df.head()
export_detail_df_head = export_detail_df.head()

# Extracting the 'Series' column from the aircraft detail DataFrame
series_counts = aircraft_detail_df['Series'].value_counts()

fig1 = plt.figure()
ax1 = fig1.gca()
top_5_series_counts = series_counts.head(12)


F = plt.gcf()
Size = F.get_size_inches()
F.set_size_inches(Size[0]*1.7, Size[1]*1.5, forward=True)

plt.xticks(fontname="Times New Roman", fontsize=20)
plt.yticks(fontname="Times New Roman", fontsize=20)


top_5_series_counts.plot(kind='bar', color='C0', edgecolor='black', alpha = 0.8)
plt.xlabel('Aircraft Series',fontname = "Times New Roman", fontsize = 20)
plt.ylabel('Frequency', fontname = "Times New Roman", fontsize = 20)

plt.xticks(rotation=45, ha='right')

for axis in ['top', 'bottom', 'left', 'right']:
    ax1.spines[axis].set_linewidth(1.5)

plt.tight_layout()
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.savefig('B737_family.png')
plt.show()

