import pandas as pd
import numpy as np
import cv2 as cv
import skimage
import requests
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv
import time
from tqdm import tqdm
from io import BytesIO
from PIL import Image

load_dotenv()
CLIENT_ID = os.getenv("SPOTIFY_KEY")
CLIENT_SECRET = os.getenv("SPOTIFY_SECRET")

song_df = pd.read_csv('song_id.csv')

# Initialize Spotify API
auth_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

def extract_features(image):
    image_np = np.array(image.convert('RGB'))
    mean = np.mean(image_np)
    stdev = np.std(image_np)

    gray = cv.cvtColor(image_np, cv.COLOR_RGB2GRAY)
    gray_dist = cv.calcHist([gray], [0], None, [256], [0, 256])
    red = cv.calcHist([image_np], [0], None, [256], [0, 256])
    green = cv.calcHist([image_np], [1], None, [256], [0, 256])
    blue = cv.calcHist([image_np], [2], None, [256], [0, 256])
    distribution = np.column_stack((gray_dist.flatten(), red.flatten(), green.flatten(), blue.flatten()))

    co_matrix = skimage.feature.graycomatrix(gray, [5], [0], levels=256, normed=True)
    contrast = skimage.feature.graycoprops(co_matrix, 'contrast')[0, 0]
    correlation = skimage.feature.graycoprops(co_matrix, 'correlation')[0, 0]
    energy = skimage.feature.graycoprops(co_matrix, 'energy')[0, 0]
    homogeneity = skimage.feature.graycoprops(co_matrix, 'homogeneity')[0, 0]

    img_blur = cv.GaussianBlur(gray, (3, 3), 0)
    sobelxy = cv.Sobel(src=img_blur, ddepth=cv.CV_64F, dx=1, dy=1, ksize=5)
    edge_mean = np.mean(sobelxy)

    return [mean, stdev, contrast, correlation, energy, homogeneity, edge_mean, distribution.tolist()]

def save_partial_csv(records):
    columns = ['id', 'artist', 'song_name', 'mean', 'stdev', 'contrast', 'correlation', 'energy', 'homogeneity', 'edge_mean'] + [f'color_{i}' for i in range(1024)]
    feature_df = pd.DataFrame(records, columns=columns)
    feature_df.to_csv('album_cover_features_partial.csv', index=False)
    print("Saved partial results to album_cover_features_partial.csv")

def main():
    partial_path = 'data/raw/album_cover_features_partial.csv'
    records = []
    processed_ids = set()

    if os.path.exists(partial_path):
        print("Resuming from partial file...")
        partial_df = pd.read_csv(partial_path)
        records = partial_df.values.tolist()
        processed_ids = set(partial_df['id'].tolist())

    track_ids = [tid for tid in song_df['track_id'].tolist() if tid not in processed_ids]

    for i in tqdm(range(0, len(track_ids), 50)):
        batch_ids = track_ids[i:i + 50]
        try:
            tracks = sp.tracks(batch_ids)['tracks']
            for track in tracks:
                try:
                    track_id = track['id']
                    artist = track['artists'][0]['name']
                    song_name = track['name']
                    image_url = track['album']['images'][0]['url']

                    response = requests.get(image_url)
                    image = Image.open(BytesIO(response.content))

                    features = extract_features(image)
                    record = [track_id, artist, song_name] + features
                    records.append(record)
                    time.sleep(0.05)
                except Exception as e:
                    print(f"Error processing track in batch: {e}")
        except Exception as e:
            print(f"Batch failed at index {i}: {e}")
            save_partial_csv(records)
            time.sleep(1)

    columns = ['id', 'artist', 'song_name', 'mean', 'stdev', 'contrast', 'correlation', 'energy', 'homogeneity', 'edge_mean'] + [f'color_{i}' for i in range(1024)]
    feature_df = pd.DataFrame(records, columns=columns)
    feature_df.to_csv('data/raw/album_cover_features.csv', index=False)
    print("Saved full results to album_cover_features.csv")

if __name__ == '__main__':
    main()