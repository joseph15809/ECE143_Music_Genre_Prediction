
# Author: Rachel Handran in Group 10 ECE 143
# Music Genre Correlation Analysis 
# Do certain music genres rise and fall together over time?
#   aka, when pop rises, does R&B also rise? when rock falls, does metal also fall?
# Positive correl: >0.7
# No correl: 0
# Negative correl: < -0.6

# Input CSV format
# date,rank,song,artist,last-week,peak-rank,weeks-on-board,genre
# 2010-12-25,1,Firework,Katy Perry,1,1,8,pop
# 2010-12-25,2,What's My Name?,Rihanna Featuring Drake,3,1,8,R&B
# 2010-12-25,3,Grenade,Bruno Mars,5,3,11,pop

# Note about input: do not drop duplicates of songs, 
# we need to acccount for multiple appearances across weeks
# also, billboard csv will never have duplicate rows by nature

# Need to make a table in this format
# week-yr,pop,rock,hip-hop,country,r&b
# 1-2000,0.34,0.28,0.15,0.12,0.11
# 3-2001,0.36,0.24,0.18,0.11,0.09

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Global Variables 
INPUT_CSV = "BIG_DATA1.csv" # Input csv must be of format: 
TOP_N_GENRES = 5 # Top N genres to analyze, default = 5
AGG_PERIOD = "Y" # Aggregate period: Choose "Y" for yearly or "W" for weekly (warning, much longer runtime)

# Asserts
# TODO assert csv columns
# TODO assert input types
assert(isinstance(TOP_N_GENRES, int) and TOP_N_GENRES > 0)
assert(isinstance(INPUT_CSV, str) and ".csv" in INPUT_CSV)
assert(isinstance(AGG_PERIOD,str) and AGG_PERIOD in ["Y","W"])

# 1. Load Dataset: 
# Input columns: date,rank,song,artist,last-week,peak-rank,weeks-on-board,genre
df = pd.read_csv(INPUT_CSV, dtype=str)

# peek at data TODO
#print("\nInitial dataframe\n", df)

# Parse date as datetime (if not already), coerce data parsing errors
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d',errors='coerce') 
df = df.dropna(subset=['date'])  # Drop rows that can't be parsed/with null values in date

# 2. Clean data:
df = df.dropna(subset=['genre']) # Remove rows missing genres 
# Normalization of genres names if needed here (rnb = r&b = R&B) TODO
print("\nAfter dropping missing genres:\n",df) 

# 3. Choose aggregation TODO
if(AGG_PERIOD == 'Y'):
    df['period'] = df['date'].dt.to_period('Y').dt.to_timestamp() # Use Jan 1 YYYY as index
elif(AGG_PERIOD == 'W'):
    # Use ISO Calendar numbering (ex: 2005-01-01	ISO:2004-W53-6) TODO
    iso = df['date'].dt.isocalendar() # returns df week, year, day
    df['period'] = iso['year'].astype(str) + "-W" +iso['week'].astype(int).str.zfill(2)
    # OLD Create 2 new columns assigns week and year according to iso TODO delete
        #df['year'] = iso['year'].astype(int)
        #df['week'] = iso['week'].astype(int)
        # Create week_year column so each week is unique
        #df['week-yr'] = df['week'].astype(str) + '-' + df['year'].astype(str)
else:
    raise ValueError("Invalid AGG_PERIOD, must choose 'Y' or 'W'")

# 4. Build new df: proportion table
#if(AGG_PERIOD == 'Y'): # period: year
genre_counts = df.groupby(['period','genre']).size().reset_index(name='count')
genre_counts['total-in-period'] = genre_counts.groupby('period')['count'].transform('sum')
genre_counts['proportion'] = genre_counts['count'] / genre_counts['total-in-period']
proportion_table = genre_counts.pivot(index='period', columns='genre', values='proportion').fillna(0)
'''
    else: # period: week-yr 
    # Group by week-yr and genre, count how many songs in each week per genre
    # essentially asks: how many pop songs were on the chart during this week? how about rap? how aout pop in the next week? etc
    genre_counts = df.groupby(['period','genre']).size().reset_index(name='count')
    # Now genre_counts looks like form:
    # week-yr   genre	count
    # 52-2010	pop	3
    # 01-2011	r&b	1

    # Compute total # songs that week, transform to copy total-in-week for 
    # all rows marked with that week
    # Takes row: 52-2010	pop	3
    # repeats 3 for all rows marked 52-2010
    genre_counts['total-in-period'] = genre_counts.groupby('week-yr')['count'].transform('sum')

    # Compute the fraction songs in this genre / all genres
    # ex: 52-2010 has 3 pop 1 rap 4 total, pop:3/4=0.75, rap = 0.25
    genre_counts['proportion'] = genre_counts['count'] / genre_counts['total-in-week']

    # Create pivot table using those values
    proportion_table = genre_counts.pivot(index='period', columns='genre', values='proportion')'''

# 5. Select top N genres to analyze
genre_totals = df['genre'].value_counts()
top_genres = genre_totals.nlargest(TOP_N_GENRES).index.tolist()
proportion_table_top = proportion_table.reindex(columns=top_genres).fillna(0) # keep only top n genres

print("\nProportion table: \n",proportion_table) # Print to terminal
proportion_table_top.to_csv('genre_proportions_by_year_fake.csv') # Save output to csv

# Save proportion df to file TODO
#proportion_table.to_csv('genre_proportions_by_weekyr_fake.csv')
#print("\nSaved to genre_proportions_by_weekyr_fake.csv")
#df.to_csv('billboard_cleaned_with_weekyr.csv')

# 6. Correlation Analysis: Pearson/Linear
correlation = proportion_table_top.corr(method='pearson') # Compute Pearson linear correlation  

print("\nCorrelation dataframe\n",correlation) # Print to terminal
correlation.to_csv('correlation_fake.csv') # Save to csv

# 7. Visualization: Clustered Heatmap
sns.set_theme(context='notebook', style='white') # Set visual theme as notebook

cluster = sns.clustermap(data=correlation, 
                         figsize=(10,10), 
                         metric='euclidean', 
                         method='average')

plt.show()
