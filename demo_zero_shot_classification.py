import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)

from textograph.handlers import ZeroShotImageClassificationHandler
from textograph.indexers import Index
from textograph.models.huggingface import CLIP_Model

model = CLIP_Model()

dataset = load_dataset("ceyda/fashion-products-small")['train'][:128] #first 100 samples
unique_labels = list(set(dataset['subCategory']))
print("Unique labels: ", unique_labels)

query_prefix = 'a photo of a ' #. We add a prefix to each of the text labels. This trick can significantly improve performance , as referenced in the CLIP paper

zero_unique_labels = [query_prefix + label for label in unique_labels] 
zeroshot = ZeroShotImageClassificationHandler(model, zero_unique_labels, batch_size=16)

predictions = []
labels = dataset['subCategory']

predictions_dict =  zeroshot(dataset['image'], top_k=1) #predictions_dict: 

predictions = [list(pred.keys())[0].replace(query_prefix,'') for pred in predictions_dict] #remove the query prefix for better readability
print(accuracy_score(labels, predictions))

#ConfusionMatrixDisplay.from_predictions(labels, predictions, labels=unique_labels, normalize='pred', xticks_rotation='vertical')
#plt.show()
