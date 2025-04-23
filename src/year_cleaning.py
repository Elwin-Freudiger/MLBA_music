import pandas as pd

"""
Here we'll clean the years, we take the following hypotheses:

We have several issues regarding the release year of songs. 

If: release_year(spotify) > year, --> take year (remasters, and singles coming out)
If: release_year(spotify) < 1960 --> take year (we only take 1960)
If: release_year(spotify) < year --> take release_year (songs that chart a year after their release)
"""

