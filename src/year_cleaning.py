import pandas as pd

"""
Here we'll clean the years, we take the following hypotheses:

We have several issues regarding the release year of songs. 

If: release_year(spotify) > year, --> take year (remasters, and singles coming out)
If: release_year(spotify) < 1960 --> take year (we only take 1960)
If: release_year(spotify) < year --> take release_year (songs that chart a year after their release)
"""

df = pd.read_csv('data/raw/billboard_song_id_playlist.csv')

df_filter = df[['track_id', 'release_year']]
df_filter['decade'] = (df['release_year']//10)*10

df_filter.to_csv('data/raw/song_id.csv', index=False)