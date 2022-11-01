from textograph.indexers import Index
from textograph.models.huggingface import CLIP_Model
from textograph.handlers import Text2ImageHandler, ZeroShotImageClassificationHandler
from textograph.utils import resize, imread, plot_images_with_titles

model = CLIP_Model()
index = Index.load_from_path('data/fashion-products.pickle', distance_metric='cosine')

handler = Text2ImageHandler(model, index)

res = handler('a photo of a green t-shirt', top_k=5)
print(res)
res_imgs = [resize(imread(x), (512,512)) for x in list(res[0].keys())]
res_distances = [f"Distance: {round(x,3)}" for x in list(res[0].values())]
#plot_images_with_titles(res_imgs, res_distances)

zeroshot = ZeroShotImageClassificationHandler(model, ['t-shirts', 'shirts', 'sweater', 'shoes', 'bags', 'skirts'])
predictions_dict =  zeroshot(res_imgs, top_k=3) #predictions_dict: 

print(predictions_dict)
#res = handler(['a photo of a green t-shirt', 'a photo of red pants'], top_k=5)
