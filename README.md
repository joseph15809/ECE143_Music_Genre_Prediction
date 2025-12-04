# ECE143_Music_Genre_Prediction
This project analyzes top-charting songs from the Billboard Hot 100 (1958–2010) by assigning each track a genre using the Deezer API. After enriching the dataset with genre information, we perform correlation analysis to identify relationships and trends between genres across different decades. Using these insights, we apply several time-series forecasting models to predict future genre popularity and evaluate which approach provides the most accurate long-term trend predictions

## File Structure

correlation_analysis/
    fig/ # saved correlation heatmaps
    src/
        -correlation.py
        display_correlation_visualization.ipynb

data/ # all csv files ares stored here
predit / 


### src/song_api
1. Extract unique songs
    - Reads the raw Billboard dataset
    - Cleans song titles and artist names
    - Removes duplicate entries
    - Saves the result as data/unique_songs.csv

2. Fetch genres from the Deezer API
    - Takes each cleaned (song, artist) pair
    - Uses the Deezer search + album endpoints to determine the song’s genre
    - Enforces Deezer’s rate limit (50 requests / 5 seconds)
    - Saves the final mapped results to data/unique_song_artist_genres.csv

### Data
 - ece143billboardhot1001958to2010.csv: Original Billboard 
 - unique_songs.csv: Extracted unique songs from billboard
 - unique_song_artist_genres.csv: Unique songs matched with Deezer genres

## Running Code
**src/song_api.py**
```
unique_df = pd.read_csv(unique_songs)
pairs = list(zip(unique_df["song_clean"], unique_df["artist_clean"]))
fetch_genres(pairs, genres_csv)
``` 

**src/correlation.py**
`python src/correlation.py`

**display_visualizations.ipynb**
- Loads final CSV outputs
- Displays visuals used in presentation
Run block by block to display visuals

### Correlation Analysis

Python script which runs a Pearson Correlation Analysis on Hot 100 Billboard data from 1958-2010.
Returns a Seaborn Cluster Heatmap showing the correlation coefficients between every pair of genres.

#### Goal:
- Analyze correlations between music genres on Billboard Hot 100
- Initial guiding question: Do certain music genres rise and fall together over time?
- Actual question answered: Do certain music genres take chart share from each other over time?

#### Key Insight
- Billboard always has 100 spots, so genres must "compete" for chart share, intriducing **structural negative correlation**
- Musically *similar* genres compete, often leading to *negative* correlation
- Genres with *differing* audience, OR *similar historic growth* may lead to positive correlation

#### Method
- Determine Top N genres to analyze
- Aggregate Billboard data by a period of Year or Week
- Compute each genre's proportion of chart share for that period
- Use Pearson Correlation on genre proportions
- Visualize results using Clustered Heatmap

#### Notes on Interpretation
- Strong correlation may be skewed by limited time data (i.e. Rap and Pop: -0.89, Rap nonemergence prior 1990 while pop continually present 1958-2021) 
- Yearly Aggregation with Top 5 Genres chosen as most reliable (large datasets)
answer the original question, we would likely need to run this analysis with weighted rank in consideration, rather than counts making up the whole billboard.
urate data; as each week the billboard may only have <10 genres represented at a time

## Third Party Modules
 - Pandas
 - Request
 - matplotlib
 - sklearn
 - statsmodels
 - prophet
 - torch
 - seaborn

## Sources:
Sources: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.dropna.html
https://en.wikipedia.org/wiki/ISO_week_date
https://www.geeksforgeeks.org/pandas/python-pandas-dataframe-groupby/
https://seaborn.pydata.org/generated/seaborn.clustermap.html
https://www.youtube.com/watch?v=crQkHHhY7aY
Utilized gen AI in assisting build matplot/seaborn custom visualizaiton
https://articles.outlier.org/pearson-correlation-coefficient


## List of Genres (28)
Pop, Alternative, Rap/Hip Hop, Country, R&B, Dance, African Music, Traditional Mexicano, Reggaeton, Asian Music, Latin Music, Films/Games, Kids, Classical, Rock, Electro, Christian, Singer & Songwriter, Jazz, Folk, Brazilian Music, Reggae, Metal, Disco, Soul & Funk, Spirituality & Religion, Salsa, Oldschool R&B