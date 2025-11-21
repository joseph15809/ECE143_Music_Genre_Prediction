# ECE143_Music_Genre_Prediction
This project analyzes top-charting songs from the Billboard Hot 100 (1958–2010) by assigning each track a genre using the Deezer API. After enriching the dataset with genre information, we perform correlation analysis to identify relationships and trends between genres across different decades. Using these insights, we apply several time-series forecasting models to predict future genre popularity and evaluate which approach provides the most accurate long-term trend predictions

## File Structure

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

### Correlation

Python script which runs a Pearson Correlation Analysis on Hot 100 Billboard data from 1958-2010.
Returns a Seaborn Cluster Heatmap showing the correlation coefficients between every pair of genres.

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
## Third Party Modules
 - Pandas
 - Request
 - matplotlib
 - sklearn
 - statsmodels
 - prophet
 - torch