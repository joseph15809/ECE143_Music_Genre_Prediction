# Sources: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
# https://en.wikipedia.org/wiki/ISO_week_date
# https://www.geeksforgeeks.org/pandas/python-pandas-dataframe-groupby/
# Author: Rachel Handran in Group 10 ECE 143
# Music Genre Correlation Analysis 
# Do certain music genres rise and fall together over time?
#   aka, when pop rises, does R&B also rise? when rock falls, does metal also fall?

import pandas as pd


# 1. Load Dataset: Billboard + genres (namisha)
df = pd.read_csv("billboard_fake.csv")
# columns: 'year', 'artist', 'song', 'genre', 'rank'

# peek at data
print("Initial df", df)

# Input CSV
# date,rank,song,artist,last-week,peak-rank,weeks-on-board,genre
#2010-12-25,1,Firework,Katy Perry,1,1,8,pop
#2010-12-25,2,What's My Name?,Rihanna Featuring Drake,3,1,8,R&B
#2010-12-25,3,Grenade,Bruno Mars,5,3,11,pop
#2010-12-25,4,Raise Your Glass,P!nk,2,1,10,pop

# Note about input: do not drop duplicates of songs, 
# we need to acccount for multiple appearances across weeks
# also, billboard csv will never have duplicate rows by nature

# Need this Format
# | Week-Yr| Pop  | Rock | Hip-Hop| Country | R&B | ... |
# |------  |------|------|------|------|------|
# | 1-2000 | 0.34 | 0.28 | 0.15 | 0.12 | 0.11 |
# | 3-2001 | 0.36 | 0.24 | 0.18 | 0.11 | 0.09 |

# 2. Parse Dates, add 3 Columns year, week, week-yr
# Parse date as datetime (if not already)
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
# Drop rows that can't be parsed/with null values in date
df = df.dropna(subset=['date']) 

# Use ISO Calendar numbering (ex: 2005-01-01	ISO:2004-W53-6) TODO
iso = df['date'].dt.isocalendar() # returns df week, year, day
# Create 2 new columns assigns week and year according to iso
df['year'] = iso['year'].astype(int)
df['week'] = iso['week'].astype(int)
# Create week_year column so each week is unique
df['week-yr'] = df['week'].astype(str) + '-' + df['year'].astype(str)

# 3. Clean genre column
# Remove rows missing genres or years 
df = df.dropna(subset=['genre','year'])
# Normalization of genres names if needed here (rnb = r&b = R&B) TODO
print("\nAfter cleaning, sample rows:")
print(df)

# 4. Build new df: proportion table
# week-yr,pop,rock,hip-hop,country,r&b
# 1-2000,0.34,0.28,0.15,0.12,0.11
# 3-2001,0.36,0.24,0.18,0.11,0.09
# Group by week-yr and genre, count how many songs in each week per genre
# essentially asks: how many pop songs were on the chart during this week? how about rap? how aout pop in the next week? etc
genre_counts = df.groupby(['week-yr','genre']).size().reset_index(name='count')
# Now genre_counts looks like form:
# week-yr   genre	count
# 52-2010	pop	3
# 01-2011	r&b	1

# Compute total # songs that week, transform to copy total-in-week for 
# all rows marked with that week
# Takes row: 52-2010	pop	3
# repeats 3 for all rows marked 52-2010
genre_counts['total-in-week'] = genre_counts.groupby('week-yr')['count'].transform('sum')


# Compute the fraction songs in this genre / all genres
# ex: 52-2010 has 3 pop 1 rap 4 total, pop:3/4=0.75, rap = 0.25
genre_counts['proportion'] = genre_counts['count'] / genre_counts['total-in-week']

# Create pivot table using those values
proportion_table = genre_counts.pivot(index='week-yr', columns='genre', values='proportion')
print("\nProportion table\n",proportion_table)

# Save cleaned files
proportion_table.to_csv('genre_proportions_by_weekyr.csv')
print("\nSaved to genre_proportions_by_weekyr.csv")
#df.to_csv('billboard_cleaned_with_weekyr.csv')

# 5. Correlation Analysis: Pearson/Linear
correlation = proportion_table.corr(method='pearson') # pearson is linear correlation TODO 
print(correlation)
