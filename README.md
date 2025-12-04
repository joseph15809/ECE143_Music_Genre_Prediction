# ECE143_Music_Genre_Prediction
This project analyzes top-charting songs from the Billboard Hot 100 (1958–2010) by assigning each track a genre using the Deezer API. After enriching the dataset with genre information, we perform correlation analysis to identify relationships and trends between genres across different decades. Using these insights, we apply several time-series forecasting models to predict future genre popularity and evaluate which approach provides the most accurate long-term trend predictions

## File Structure
- display_visualizations.ipynb # Final submission notebook

- data/ # all csv files ares stored here
    - BIG_DATA2.csv # billboard 100 dataset with genres
    - ece143billboardhot100.csv # original datasets
    - unique_songs.csv # extracted unqiue songs from dataset
    - unique_song_artist_genres.csv # unqiue songs with genres

- fig/ # all saved figures

- predict/ # time series predict folder
    - BIG_DATA2.csv # billboard 100 dataset with genres
    - predict.ipynb # entire code for prediction training and testing

- src/
    - correlation.py # correlation analysis 
    - song_api # extracts unique songs and get genre with deezer api

    


### src/song_api


## Running Code

**display_visualizations.ipynb**
- Loads final CSV outputs
- Displays visuals used in presentation
Run block by block to display visuals

**src/song_api.py**
```
unique_df = pd.read_csv(unique_songs)
pairs = list(zip(unique_df["song_clean"], unique_df["artist_clean"]))
fetch_genres(pairs, genres_csv)
``` 
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


**src/correlation.py**

`python src/correlation.py`

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

`predict/predict.ipynb`
### Weekly Genre Forecasting  
*Time-Series Modeling for Weekly Genre Counts & Average Ranks*

A complete, production-grade pipeline for forecasting **weekly genre counts** and **average chart ranks** using a combination of classical statistical models, like Exponential Smoothing, ARIMA, Prophet, and a PyTorch LSTM.  
The project includes full preprocessing, feature engineering, multi-model evaluation, and visualization utilities.

### Features

#### Comprehensive Forecasting Models
- **Exponential Smoothing (Holt–Winters)**
- **ARIMA**
- **Facebook Prophet**
- **LSTM (PyTorch)**  
  - Windowed forecasting  
  - Teacher forcing  
  - Early stopping  
  - Multi-feature support  

#### Validation
- RMSE-based evaluation  
- Side-by-side comparison across models  
- Separate evaluation for:  
  - Genre **counts**  
  - Genre **average ranks**

#### Data Pipeline
- Weekly aggregation per genre  
- Auto-fill missing weeks  
- Rolling stats (mean, std, momentum)  
- Flexible scaling utilities (per target)

####  Visualization
- Combined **Actual vs Predicted** plots  
- Per-model trend comparison  
- Plots for both Count and Rank predictions  
- Automatically saved under `plots/`

## Third Party Modules
 - Pandas
 - seaborn
 - Request
 - matplotlib
 - sklearn
 - statsmodels
 - prophet
 - torch
 - seaborn
 - prophet

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