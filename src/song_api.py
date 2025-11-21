import pandas as pd
import re
import requests
import time
from collections import deque

bb100_file = "./data/ece143billboardhot1001958to2010.csv"
unique_songs = "./data/unique_songs.csv"
genres_csv = "./data/unique_song_artist_genres.csv"

# hard limit: 50 requests per 5 seconds
MAX_CALLS = 50
WINDOW = 5



class RateLimiter:
    def __init__(self, max_calls=MAX_CALLS, window=WINDOW):
        self.max_calls = max_calls
        self.window = window
        self.times = deque()

    def wait(self):
        now = time.monotonic()
        # drop old timestamps
        while self.times and (now - self.times[0]) >= self.window:
            self.times.popleft()

        if len(self.times) >= self.max_calls:
            sleep_for = self.window - (now - self.times[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
            # purge again after sleep
            now = time.monotonic()
            while self.times and (now - self.times[0]) >= self.window:
                self.times.popleft()

        # record this call
        self.times.append(time.monotonic())

def get_song_data(session: requests.Session, limiter: RateLimiter, song: str, artist: str):
    """Return a genre string or None."""
    try:
        song_q = clean_song(song)
        artist_q = primary_artist(artist)

        # search
        search_url = f'https://api.deezer.com/search?q=artist:"{artist_q}" track:"{song_q}"'
        limiter.wait()
        r = session.get(search_url, timeout=15)
        if r.status_code != 200:
            return None
        js = r.json()
        data = js.get("data") or []
        if not data:
            return None

        album_id = (data[0].get("album") or {}).get("id")
        if not album_id:
            return None

        # album
        album_url = f"https://api.deezer.com/album/{album_id}"
        limiter.wait()
        r2 = session.get(album_url, timeout=15)
        if r2.status_code != 200:
            return None
        album = r2.json() or {}
        genres = (album.get("genres") or {}).get("data") or []
        if not genres:
            return None
        return genres[0].get("name")
    except Exception:
        return None

def clean_song(song: str) -> str:
    s = song.strip()
    s = re.sub(r'\s*(?:\(feat\..*?\)|feat\..*$|ft\..*$|featuring .*?$)', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s*\(.*?\)', '', s)
    return s

def primary_artist(artist: str) -> str:
    a = artist.strip()
    # split on clear multi-artist separators
    a = re.split(r'\s+(?:featuring|feat\.?|ft\.?|with|x|vs\.?)\s+|,| & | and ', a, flags=re.IGNORECASE)[0]
    a = re.sub(r'\s+', ' ', a).strip()
    return a

def extract_song_info(csv_path: str, out_csv: str) -> dict:
    """
    Reads the Billboard CSV, builds unique (clean_song, clean_artist) keys,
    but keeps the first original title/artist for each key.
    """
    session = requests.Session()     
    limiter = RateLimiter(MAX_CALLS, WINDOW)
    mapping = {}  # (song_clean, artist_clean) -> (orig_song, orig_artist)
    chunksize = 200_000
    usecols = ["song", "artist"]
    for chunk in pd.read_csv(in_csv, usecols=usecols, chunksize=chunksize, low_memory=False):
        for orig_song, orig_artist in zip(chunk["song"], chunk["artist"]):
            orig_song = str(orig_song)
            orig_artist = str(orig_artist)

            s_clean = clean_song(orig_song)
            a_clean = primary_artist(orig_artist)
            key = (s_clean, a_clean)

            if key not in mapping:
                mapping[key] = (orig_song, orig_artist)

    # build dataframe
    rows = []
    for (s_clean, a_clean), (orig_song, orig_artist) in mapping.items():
        rows.append({
            "song": orig_song,
            "artist": orig_artist,
            "song_clean": s_clean,
            "artist_clean": a_clean,
        })

    df_unique = pd.DataFrame(rows)
    df_unique.to_csv(out_csv, index=False)
    print(f"saved {len(df_unique)} unique pairs to {out_csv}")
    return df_unique

def fetch_genres(pairs, out_csv, flush_every=500):
    session = requests.Session()
    limiter = RateLimiter(MAX_CALLS, WINDOW)

    cache = {}
    rows = []
    start = time.time()
    total = len(pairs)
    for i, (song, artist) in enumerate(pairs, 1):
        key = (song, artist)
        if key in cache:
            genre = cache[key]
        else:
            # two API calls inside get_song_data; limiter.wait() is called for each
            genre = get_song_data(session, limiter, song, artist)
            cache[key] = genre

        rows.append((song, artist, genre))

        if i % flush_every == 0:
            pd.DataFrame(rows, columns=["song", "artist", "genre"])\
              .to_csv(out_csv, mode="a", index=False, header=(i == flush_every))
            rows.clear()

            # simple progress
            elapsed = time.time() - start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else float("inf")
            print(f"{i}/{total} done | {rate:.2f} pairs/s | elapsed {elapsed/60:.1f} min | ETA {eta/60:.1f} min")

    # write tail
    if rows:
        pd.DataFrame(rows, columns=["song", "artist", "genre"])\
          .to_csv(out_csv, mode="a", index=False, header=(not pd.io.common.file_exists(out_csv)))

    print(f"wrote {out_csv}")


# load the unique pairs we just save
#unique_df = pd.read_csv(unique_songs)
#pairs = list(zip(unique_df["song_clean"], unique_df["artist_clean"]))
#fetch_genres(pairs, genres_csv)

songs_genres = pd.read_csv("./data/unique_song_artist_genres.csv")
genres = list(songs_genres["genre"])
genre_count = 0
for g in genres:
    if not pd.isna(g):
        genre_count += 1
print(f'{genre_count / len(genres)}% of hits')
