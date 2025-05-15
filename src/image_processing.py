import pandas as pd
import numpy as np

import requests
import os
from tqdm import tqdm
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed


#Start by downloading an image


def show_im(link, size=(100,100)):
    response = requests.get(link)
    image = Image.open(BytesIO(response.content))
    smaller_im = image.resize(size)
    img =plt.imshow(smaller_im)

def process_image(row, size = (64,64), out_dir='data/clean/images'):
    id = row['id']
    link = row['Album Image URL'] 
    try:
        response = requests.get(link, timeout=10)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        final_image = image.resize(size)
        out_path = os.path.join(out_dir, f"{id}.png")
        final_image.save(out_path)
        return {'id': id, 'filepath': out_path}
    except Exception as e:
        print(f"Failed for {id}: {e}")
        return None



def main(dir='data/clean/images'):
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    df = pd.read_csv('data/raw/top_10000_1950-now.csv')
    df['id'] = df['Track URI'].str.slice(start=14)

    #process them in parallel
    results = []

    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_image, row, out_dir=dir): row['id'] for _, row in df.iterrows()}
        for future in as_completed(futures):
            result = future.result()
            if result:
               results.append(result)


    image_df = pd.DataFrame(results)
    image_df.to_csv('data/clean/image_dataset.csv', index=False)
    print('Saved images and csv!')

if __name__ == "__main__":
    main()