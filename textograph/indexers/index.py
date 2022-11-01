import pickle
import os
import datetime
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.gaussian_process.kernels import RBF


def dot(x,y):
    return np.dot(np.asarray(x), np.asarray(y).transpose())

def get_metric(metric_name, **kwargs):
    """
    Returns a (metric_func, descending_sort) tuple. Ex. descending_sort should be True for cosine, False for euclidean
    """
    metrics = {
        'rbf': (RBF(kwargs.get('gamma',1)), True),
        'euclidean': (euclidean_distances, False),
        'cosine': (cosine_similarity, True),
        'dot': (dot, True)
        }

    return metrics[metric_name]

class Index(object):
    def __init__(self, data, ids, distance_metric = 'cosine', **kwargs):
        self.distance_metric = distance_metric
        self.distance_func, self.descending_sorting = get_metric(distance_metric, **kwargs)

        index = {'data': data, 'ids': ids}

        self.index = index
        self.index['data'] = np.asarray(self.index['data'])

    #classmethod
    def save_to_path(data, ids, name, destination_dir, model):
        """
        data: np.ndarray, [n_elements, feature_size] matrix 
        ids: list[str], identifier of each element of the matrix
        name: str, name of the output index file, needed only when creating the index from scratch, without using the Index.load_from_path method
        destination_dir: str, destination_dir of the output index file (needed only when creating the index from scratch, without using the Index.load_from_path method)
        model: model.Model instance used to compute the index, usually passed by the Indexer (needed only when creating the index from scratch, without using the Index.load_from_path method)
        """
        os.makedirs(destination_dir, exist_ok=True)
        if isinstance(data, np.ndarray):
            data = data.tolist()
        index = {'data': data, 'ids': ids, 'meta': {'date': datetime.datetime.now(), 'model': model.meta}}
        path = f'{destination_dir}/{name}.pickle'
        with open(path, 'wb') as handle:
            pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"Index created in {path}")

    @classmethod
    def load_from_path(self, path, distance_metric = 'cosine'):
        """
        Create an Index instance from a pickle
        """
        with open(path, 'rb') as handle:
            index = pickle.load(handle)
        index['data'] = np.asarray(index['data'])
        
        print(f"Loading index for  {index['meta']['model']._name_or_path}")

        return Index(index['data'], index['ids'], distance_metric)

    def change_metric(self, metric):
        self.distance_func, self.descending_sorting = get_metric(metric)

    def __len__(self):
        return self.index['data'].shape[0]

    def query(self, query):
        assert query.shape[1] == self.index['data'].shape[1]

        index = self.index['data']
        distances = self.distance_func(query, index)

        return distances