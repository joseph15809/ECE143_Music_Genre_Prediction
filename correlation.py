
# Author: Rachel Handran in Group 10 ECE 143
# Music Genre Correlation Analysis 
# Do certain music genres rise and fall together over time?
#   aka, when pop rises, does R&B also rise? when rock falls, does metal also fall?

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

# Global Variables 
INPUT_CSV = "BIG_DATA1.csv" # Input csv must be of format: 
TOP_N_GENRES = 5 # Top N genres to analyze, default = 5
AGG_PERIOD = "W" # Aggregate period: Choose "Y" for yearly or "W" for weekly

# Asserts
# TODO assert csv columns
assert(isinstance(TOP_N_GENRES, int) and TOP_N_GENRES > 0)
assert(isinstance(INPUT_CSV, str) and ".csv" in INPUT_CSV)
assert(isinstance(AGG_PERIOD,str) and AGG_PERIOD in ["Y","W"])

# 1. Load Dataset: 
# Input columns: date,rank,song,artist,last-week,peak-rank,weeks-on-board,genre
df = pd.read_csv(INPUT_CSV, dtype=str)

# print("\nInitial dataframe\n", df) # print initial df

# Parse date as datetime (if not already), coerce data parsing errors
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d',errors='coerce') 
df = df.dropna(subset=['date'])  # Drop rows that can't be parsed/with null values in date

# 2. Clean data:
df = df.dropna(subset=['genre']) # Remove rows missing genres 
# Normalization of genres names if needed here (rnb = r&b = R&B) TODO
print("\nAfter dropping missing genres:\n",df) 

# 3. Choose aggregation type, yearly or weekly TODO
if(AGG_PERIOD == 'Y'):
    df['period'] = df['date'].dt.to_period('Y').dt.to_timestamp() # Use Jan 1 YYYY as index
elif(AGG_PERIOD == 'W'):
    # Use ISO Calendar numbering (ex: 2005-01-01	ISO:2004-W53-6) TODO
    iso = df['date'].dt.isocalendar() # returns df week, year, day
    df['period'] = iso['year'].astype(str) + "-W" +iso['week'].astype(str).str.zfill(2)
else:
    raise ValueError("Invalid AGG_PERIOD, must choose 'Y' or 'W'")

# 4. Build new df: proportion table
# Group by period and genre, count how many songs in each period per genre
# essentially asks: how many pop songs were on the chart during this period? how about rap? how aout pop in the next period? etc

genre_counts = df.groupby(['period','genre']).size().reset_index(name='count')
genre_counts['total-in-period'] = genre_counts.groupby('period')['count'].transform('sum')
genre_counts['proportion'] = genre_counts['count'] / genre_counts['total-in-period']
# Now genre_counts looks like form:
# period   genre	count total-in-period proportion
# 2000-W03	pop 	3     100              3/100
# 2010    	r&b 	120   5200            120/5200
proportion_table = genre_counts.pivot(index='period', columns='genre', values='proportion').fillna(0) 
# Make pivot table form:
# period   pop   rap  ...
# 2000-W03  0.03 0.02

# 5. Select top N genres to analyze
genre_totals = df['genre'].value_counts()
top_genres = genre_totals.nlargest(TOP_N_GENRES).index.tolist()
proportion_table_top = proportion_table.reindex(columns=top_genres).fillna(0) # keep only top n genres

print("\nProportion table: \n",proportion_table) # Print to terminal
pivot_filename = 'genre_proportions_by_' + AGG_PERIOD + '_top_' + str(TOP_N_GENRES) + '.csv'
proportion_table_top.to_csv(pivot_filename) # Save pivot table, output to csv
print("\nSaved to {}.csv".format(pivot_filename))

# 6. Correlation Analysis: Pearson/Linear
correlation = proportion_table_top.corr(method='pearson') # Compute Pearson linear correlation  

print("\nCorrelation dataframe\n",correlation) # Print to terminal
correlation_filename = 'correlation_by_' + AGG_PERIOD + '_top_' + str(TOP_N_GENRES) + '.csv'
correlation.to_csv(correlation_filename) # Save to csv
print("\nSaved to {}.csv".format(correlation_filename))

# 7. Visualization: Clustered Heatmap
cmap = mcolors.LinearSegmentedColormap.from_list("custom_corr", ["red","white","green"]) # custom colormap, red -1 white +0 green +1

sns.set_theme(context='notebook', style='white') # Set visual theme as notebook

cluster = sns.clustermap(data=correlation, 
                         figsize=(10,10), 
                         metric='euclidean', 
                         method='average',
                         annot=True,
                         cmap=cmap,
                         vmin=-1, vmax=1,center=0, # ensure symmetry
                         cbar_kws={'label': 'Pearson r'},
                         linewidths=0.5
                        )
plt.suptitle('Genre Correlation Clustered Heatmap')
fig_filname = 'clustermap_by_' + AGG_PERIOD + '_top_' + str(TOP_N_GENRES) + '.png'
plt.savefig(fig_filname, dpi=300)
plt.show()
