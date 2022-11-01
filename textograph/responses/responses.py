from ..utils.utils import resize, imread, plot_images_with_titles

class ImageResponse(object):
    def __init__(self, raw_data, raw_feats, handler, imread_func = None):
        self.raw_data = raw_data
        self.raw_feats = raw_feats
        self.handler = handler
        self.imread_func = imread_func
    
    def __str__(self):
        return self.raw_data.__str__()
    
    def __len__(self):
        return len(list(self.raw_data.keys()))
    
    def __getitem__(self, idx):
        return list(self.raw_data.keys())[idx]
    
    @property
    def indexes(self):
        return list(self.raw_data.keys())
    
    @property
    def distances(self):
        return list(self.raw_data.values())
    
    def plot_result(self, imread_func = None, top_k = 10):
        if not self.imread_func:
            if not imread_func:
                raise Exception("Please provide imread_func")
            self.imread_func = imread_func
        
        k = min(len(self), top_k)
        res_imgs = [self.imread_func(x) for x in self.indexes[:top_k]]
        res_distances = [f"D:{round(x,3)}" for x in self.distances[:top_k]]
        plot_images_with_titles(res_imgs, res_distances, figsize=(15,5))

class Image2ImageResponse(ImageResponse):
    pass

class Text2ImageResponse(ImageResponse):
    pass
            
