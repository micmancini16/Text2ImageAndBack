import numpy as np

from textograph.indexers import Index
from textograph.models.huggingface import CLIP_Model
from textograph.handlers import Text2ImageHandler, ZeroShotImageClassificationHandler
from textograph.utils import resize, imread, plot_images_with_titles

from datasets import load_dataset

full_dataset = load_dataset("ceyda/fashion-products-small", split='train')
full_dataset_df = full_dataset.to_pandas()
full_dataset_df.head()

def imread_fashion(link):
    idx = int(full_dataset_df[full_dataset_df.link == link].index.values[0])
    return np.asarray(full_dataset[idx]['image'])

model = CLIP_Model()
index = Index.load_from_path('data/fashion-products.pickle', distance_metric='cosine')

handler = Text2ImageHandler(model, index)

res = handler("a photo of a green t-shirt")[0]
res.plot_result(top_k=10, imread_func=imread_fashion)
#plot_images_with_titles(res_imgs, res_distances)

#res = handler(['a photo of a green t-shirt', 'a photo of red pants'], top_k=5)
