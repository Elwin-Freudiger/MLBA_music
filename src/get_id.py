import pandas as pd
import requests
from bs4 import BeautifulSoup
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv
import time
from tqdm import tqdm

"""
This code was inspired by a github repository by GitHub user: https://github.com/awcooper with some modifications.
"""

# Load environment variables
load_dotenv()
CLIENT_ID = os.getenv("SPOTIFY_KEY")
CLIENT_SECRET = os.getenv("SPOTIFY_SECRET")

def get_billboard_song_titles_for_year(year):
    billboard_page = "https://en.wikipedia.org/wiki/Billboard_Year-End_Hot_100_singles_of_"
    page = requests.get(billboard_page + str(year))
    soup = BeautifulSoup(page.content, 'html.parser')
    doc = soup.find("table", {"class": "wikitable"})
    year_data = []
    for row in doc.find_all(["tr"])[1:]:
        row_data = [cell.text.strip() for cell in row.find_all(["td", "th"])]
        if len(row_data) != 3:
            print("Error Processing Row: ", row)
        else:
            year_data.append(tuple(row_data))
    return year_data

def parse_artist(content):
    for split_token in [" & ", " \\ ", " feat ", " featuring ", " and "]:
        content = content.partition(split_token)[0]
    return content

def parse_song(content):
    for split_token in ["\\", "/"]:
        content = content.partition(split_token)[0]
    return content

def using_wiki():
    # Authenticate without user login
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    all_records = []

    for year in tqdm(range(1960, 2025), desc='Processing years'):
        try:
            year_songs = get_billboard_song_titles_for_year(year)
            for (rank, track, artist) in year_songs:
                query = f"{parse_artist(artist)} {parse_song(track)}"
                results = sp.search(q=query, type="track", limit=1)
                try:
                    track_data = results['tracks']['items'][0]
                    track_id = track_data['id']
                    release_date = track_data['album']['release_date']
                    release_year = int(release_date[:4]) if release_date else None
                    all_records.append({
                        'track_id': track_id,
                        'year': year,
                        'release_year': release_year
                    })
                except (IndexError, KeyError):
                    print("Not found on Spotify:", (year, rank, track, query))
                time.sleep(0.1)  # gentle rate limit buffer
        except Exception as e:
            print(f"Failed to process year {year}: {e}")

    df = pd.DataFrame(all_records)
    df = df.sort_values(by='year').drop_duplicates(subset='track_id', keep='first')
    df.to_csv("data/raw/spotify_billboard_1960_2024.csv", index=False)
    print("CSV saved: spotify_billboard_1960_2024.csv")

def using_lists():
    # We use a playlist made by Wicked Dreamer on Spotify.
    # The assumption is that these were made by hand and therefore, less prone to errors.
    playlists_df = pd.read_csv('data/raw/playlist_ids.csv')

    # Authenticate without user login
    auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    all_records = []

    for _, row in playlists_df.iterrows():
        year = row['year']
        playlist_id = row['playlist_id']
        
        print(f"\nProcessing playlist for year {year}...")
        try:
            results = sp.playlist_items(playlist_id, additional_types=['track'])
            total = results['total']
            offset = 0
            tracks = []

            # Handle pagination
            while results:
                tracks.extend(results['items'])
                offset += len(results['items'])
                if results['next']:
                    results = sp.next(results)
                else:
                    break

            for item in tqdm(tracks, desc=f"Year {year}", unit="song"):
                try:
                    track_data = item['track']
                    track_id = track_data['id']
                    release_date = track_data['album']['release_date']
                    release_year = int(release_date[:4]) if release_date else None
                    all_records.append({
                        'track_id': track_id,
                        'year': year,
                        'release_year': release_year
                    })
                except (TypeError, KeyError):
                    continue
                time.sleep(0.1)  # gentle rate limit buffer

        except Exception as e:
            print(f"Failed to process playlist for year {year}: {e}")

    df = pd.DataFrame(all_records)
    df = df.sort_values(by='year').drop_duplicates(subset='track_id', keep='first')
    df.to_csv("data/raw/billboard_song_id_playlist.csv", index=False)
    print("\nCSV saved: data/raw/billboard_song_id_playlist.csv")




if __name__ == "__main__":
    using_lists()