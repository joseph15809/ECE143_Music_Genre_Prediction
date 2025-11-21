# Music Genre Correlation Analysis 
Author: Rachel Handran in Group 10 ECE 143
Do certain music genres rise and fall together over time?

# correlation.py
Python script which runs a Pearson Correlation Analysis on Hot 100 Billboard data from 1958-2010.
Returns a Seaborn Cluster Heatmap showing the correlation coefficients between every pair of genres.

## Input Options
- Aggregation:
    - Yearly aggregation: has more aggregate data, more accurate to larger scale
    - Weekly aggregation: possibly not enough datapoints to tell trends per week
- Top N genres: Analyze only top N genres 
    - Note that raising this number may not give more accurate data; as each week the billboard may only have <10 genres represented at a time

## Interpreting the Correlation Results
https://articles.outlier.org/pearson-correlation-coefficient
Pearson correlation coefficient r
Absolute value of Pearson r:
Between [0, 0.4]: Weak Correlation
Between [0.4, 0.7]: Moderate Correlation
Between [0.7, 1]: Strong Correlation
Most accurate visual is likely the Top 5 genres using Yearly Aggregation.

# display_correlation_visualization.ipynb
Jupyter Notebook that opens the results of Correlation Analysis TODO

# List of Genres (22)
Rap/Hip Hop,Pop, Films/Games, Alternative, R&B, Rock, Country, Dance, Electro, Christian, Asian Music, Jazz, Singer & Songwriter, Latin Music, Metal, Kids, Reggaeton, Reggae, Traditional Mexicano, Folk, Disco, Classical

# Sources:
Sources: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
https://en.wikipedia.org/wiki/ISO_week_date
https://www.geeksforgeeks.org/pandas/python-pandas-dataframe-groupby/
https://seaborn.pydata.org/generated/seaborn.clustermap.html
https://www.youtube.com/watch?v=crQkHHhY7aY
Utilized gen AI in assisting build matplot/seaborn custom visualizaiton
https://articles.outlier.org/pearson-correlation-coefficient