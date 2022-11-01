from abc import abstractmethod, abstractproperty
from typing import Union, List
from PIL import Image

class Model(object):
    """
    Abstract, agnostic-framework, class to define a model that can handle image and text feature extraction and preprocessing
    """
    @abstractmethod
    def preprocess_texts(self, texts: List[str]):
        pass

    @abstractmethod
    def preprocess_images(self,images: List[Image.Image]):
        pass

    @abstractmethod
    def get_text_features(self,texts: List[str]):
        pass

    @abstractmethod
    def get_image_features(self,images: List[Image.Image]):
        pass
    @abstractproperty
    def meta(self):
        pass



