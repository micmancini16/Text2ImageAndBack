from abc import abstractmethod

from textograph.responses.responses import Text2ImageResponse
from ..indexers.index import Index
from ..models.model import Model
from ..utils.utils import softmax, batch
from ..responses import Text2ImageResponse, Image2ImageResponse 

import numpy as np
import time
from tqdm import tqdm

class Handler(object):
    def __init__(self, model: Model, index: Index, **kwargs):
        self.model = model
        self.index = index
        self.batch_size = kwargs.get('batch_size', 16)

    def __call__(self, query, **kwargs):
        t_start = time.perf_counter()
        
        if not isinstance(query, list):
            query = [query]  
        
        res = []
        for batch_query in tqdm(batch(query, self.batch_size), total=len(query)//self.batch_size):
            processed_query = self.process_query(batch_query, **kwargs)
            distances = self.index.query(processed_query)
            res += self.process_distances(distances, **kwargs)
        
        elapsed = time.perf_counter()-t_start

        print(f"Request handled in {elapsed} seconds")
        return res

    @abstractmethod
    def process_query(self, query, **kwargs):
        pass
    def process_distances(self, distances, **kwargs):
        k = min(kwargs.get('top_k', 100), len(self.index))

        results = []

        for distances_single_query in distances:
            if self.index.descending_sorting:
                top_k_scores = np.partition(distances_single_query, -k, axis=0)[-k:]  
                top_k_scores = np.sort(top_k_scores, axis=0)[::-1].flatten()
                top_k_idx = np.argpartition(distances_single_query, -k, axis=0)[-k:]  
                top_k_idx = top_k_idx[np.argsort(distances_single_query[top_k_idx], axis=0)][::-1].flatten()
            else:
                if k < len(distances):
                    top_k_scores = np.partition(distances_single_query, k, axis=0)[:k]  
                    top_k_idx = np.argpartition(distances_single_query, k, axis=0)[:k]  
                else:
                    #No need to partition
                    top_k_scores = distances_single_query
                    top_k_idx = np.arange(10)

                top_k_scores = np.sort(top_k_scores, axis=0).flatten()
                top_k_idx = top_k_idx[np.argsort(distances_single_query[top_k_idx], axis=0)].flatten()
            
            top_k_feats = [self.index.index['data'][x] for x in top_k_idx]
            top_k_ids = [self.index.index['ids'][x] for x in top_k_idx]

            results.append({id: round(score,5) for id, score in zip(top_k_ids, top_k_scores)})

        return results, top_k_feats

class ZeroShotImageClassificationHandler(Handler):
    def __init__(self, model, targets, distance_metric = 'dot', **kwargs):
        self.targets=targets
        self.softmax=softmax
        self.batch_size = 32
        indexes = model.get_text_features(targets)
        super().__init__(model, Index(indexes, targets, distance_metric=distance_metric, **kwargs), **kwargs)
            
    def process_query(self, query, **kwargs):
        return self.model.get_image_features(query)

    def process_distances(self, distances, **kwargs):
        processed_distances, _ = super().process_distances(distances, top_k=len(self.targets))
        top_k = kwargs.get('top_k', len(self.targets))
        results = []

        for distances_single_query in processed_distances:
            scores = list(distances_single_query.values())
            if self.index.descending_sorting:
                softmax_scores = softmax(scores)
            else:
                softmax_scores = softmax((np.zeros(len(scores)) + np.max(scores)) - scores) 
            results.append({id: score for id, score in zip(list(distances_single_query.keys())[:top_k], softmax_scores[:top_k])})
        return results

class Text2ImageHandler(Handler):
    def process_query(self, query, **kwargs):
        return self.model.get_text_features(query)
    def __call__(self, query, **kwargs):
        raw_res, feats = super().__call__(query, **kwargs)
        return [Text2ImageResponse(raw_res_batch, feats, self, kwargs.get('imread_func', None)) for raw_res_batch in raw_res]
    def img2img_on_res(self, res: Text2ImageResponse, idx_to_expand, imread_func = None, top_k = 10):
        """
        Run image2image search on one of the results returned by a Text2ImageResponse object with respect to the handler's Index
        res: the Text2ImageResponse object
        idx_to_expand: int, the index of the result to expand
        """
        if not imread_func:
            raise Exception("Please create the handler with a valid imread_func")
        img2img_handler = Image2ImageHandler(self.model, self.index)
        res_query_id = res[idx_to_expand]
        res_query_img = imread_func(res_query_id)
        return img2img_handler(res_query_img, top_k=top_k)
    
    def filter_results(self, res: Text2ImageResponse, filter_query: str):
        sub_handler = Text2ImageHandler(self.model, Index(res.raw_feats, res.indexes, self.index.distance_metric))
        return sub_handler(filter_query)
        

class Image2ImageHandler(Handler):
    def process_query(self, query, **kwargs):
        return self.model.get_image_features(query)
    def __call__(self, query, **kwargs):
        raw_res, feats = super().__call__(query, **kwargs)
        return [Image2ImageResponse(raw_res_batch, feats, self, kwargs.get('imread_func', None)) for raw_res_batch in raw_res]
    

    
