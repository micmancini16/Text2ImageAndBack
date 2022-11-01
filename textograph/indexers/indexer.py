from abc import abstractmethod

from .index import Index
from ..models.model import Model
from ..utils.utils import batch
from tqdm import tqdm 
import numpy as np
import PIL
import time

from multiprocessing import Pool, cpu_count

class Indexer(object):
    """
    An object that can access and create an Index object
    """
    def __init__(self, model:Model):
        """
        The model used to create the index
        """
        self.model = model
    
    def create(self, images = None, texts = None, image_ids = None, name = '', destination_dir = '.', batch_size=100):
        """
        inputs: a List or a generator of PIL.Image.Image objects (for image indexes) or strings (for text indexes)
        image_ids: a list of ids that identifies the images. Not needed when indexing texts
        name: Name of the index
        destination_dir = Local dir where the index will be stored
        """

        if images and texts:
            raise ValueError("Please provide values either for the images field or the text field")
        if images:
            func = self.model.get_image_features
            inputs = images
            assert image_ids != None
            ids = image_ids
        else:
            func = self.model.get_text_features
            ids = texts
            inputs = texts
        
        print("Starting indexing...")
        embeddings = []
        t_start_full = time.perf_counter()

        for i, batch_input in enumerate(batch(inputs, batch_size)):
            print(f"Processing batch {i} of {len(inputs) // batch_size}")
            t_start = time.perf_counter()
            embeddings.append(func(batch_input))
            elapsed = time.perf_counter()-t_start
            print(f"Batch handled in {round(elapsed,4)} seconds")

        embeddings = np.vstack(embeddings)
        elapsed_full = time.perf_counter()-t_start_full
        print(f"Index computed in {round(elapsed_full,4)} seconds")
        #embeddings_norm = embeddings/ np.linalg.norm(embeddings, axis=1, keepdims=True)
        Index.save_to_path(embeddings, ids, name, destination_dir, self.model)


class HuggingFaceImageIndexer(Indexer):
    """
    Compute indexes for a HuggingFace dataset, exploiting the .map() method.
    """
    def create(self, dataset, id_col, image_col = 'image', name = '', destination_dir = '.', batch_size=100):
        print("Starting indexing...")
        t_start_full = time.perf_counter()
        func = self.model.get_image_features
        def map_embeddings(examples):
            examples['embeddings'] = func(examples[image_col])
            return examples

        dataset = dataset.map(map_embeddings, remove_columns=[image_col], batched=True, batch_size=batch_size)
        elapsed_full = time.perf_counter()-t_start_full
        print(f"Index computed in {round(elapsed_full,4)} seconds")
        Index.save_to_path(dataset['embeddings'], dataset[id_col], name, destination_dir, self.model)





