from .model import Model
from typing import Union, List
from PIL import Image

from transformers import CLIPProcessor, CLIPModel
from transformers import FlavaProcessor, FlavaModel

class HF_Model(Model):
    """
    Model and processor based on Hugging Face multimodal models
    """
    def __init__(self, preprocessor_class, model_class, pretrained_model):
        self.preprocessor = preprocessor_class.from_pretrained(pretrained_model)
        self.model = model_class.from_pretrained(pretrained_model)
        self.model.eval()

    def preprocess_images(self, images: List[Image.Image]):
        return self.preprocessor(images=images, return_tensors='pt')
    def preprocess_texts(self, texts: List[str]):
        return self.preprocessor(text=texts, padding=True, return_tensors='pt')
    def get_image_features(self, images: List[Image.Image]):
        feats = self.model.get_image_features(**self.preprocess_images(images))
        return self.postprocess_image_features(feats)
    def get_text_features(self, texts: List[str]):
        feats = self.model.get_text_features(**self.preprocess_texts(texts))
        return self.postprocess_text_features(feats)
    def postprocess_image_features(self,features):
        return features
    def postprocess_text_features(self,features):
        return features
    @property
    def meta(self):
        return self.model.config



class CLIP_Model(HF_Model):
    def __init__(self, pretrained_model = 'openai/clip-vit-base-patch32'):
        super().__init__(CLIPProcessor, CLIPModel, pretrained_model)
    def postprocess_image_features(self,features):
        return super().postprocess_image_features(features).detach().numpy()
    def postprocess_text_features(self,features):
        return super().postprocess_text_features(features).detach().numpy()


class FLAVA_Model(HF_Model):
    def __init__(self, pretrained_model = "facebook/flava-full"):
        super().__init__(FlavaProcessor, FlavaModel, pretrained_model)
    def postprocess_image_features(self, features):
        return super().postprocess_image_features(features).detach().numpy()[:,0,:]
    def postprocess_text_features(self, features):
        return super().postprocess_text_features(features).detach().numpy()[:,0,:]



