from datasets import load_dataset, Dataset
from textograph.indexers import Indexer,HuggingFaceImageIndexer
from textograph.indexers import Index
from textograph.models.huggingface import CLIP_Model

import numpy as np
import os

model = CLIP_Model()

if not os.path.exists('data/fashion-products.pickle'):
    dataset = load_dataset("ceyda/fashion-products-small", split='train')
    indexer = HuggingFaceImageIndexer(model)
    #images = dataset['train']['image']
    #ids = dataset['train']['link']
    indexer.create(dataset = dataset, id_col='link', name='fashion-products', destination_dir='data')
else:
    print("Index already computed...")
    index = Index.load_from_path('data/fashion-products.pickle', distance_metric='cosine')


# if not os.path.exists('data/oxford-pets.pickle'):
#     dataset = load_dataset("pcuenq/oxford-pets", split='train')
#     indexer = Indexer(model)
#     images = dataset['image']
#     ids = [str(x) for x in np.arange(len(dataset))]
#     indexer.create(images = images, image_ids=ids, name='oxford-pets', destination_dir='data')
# else:
#     print("Index already computed...")
#     index = Index.load_from_path('data/oxford-pets.pickle', distance_metric='cosine')