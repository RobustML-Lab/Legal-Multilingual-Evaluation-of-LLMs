#%%
# Imports
from models import *
from data import *
from sklearn.metrics import precision_recall_fscore_support
import numpy as np

#%%
# Get the dataset
dataset = Dataset.get_dataset('multi_eurlex')
#%%
# Initialize the model
data, label_options = dataset.get_data('en')
prompt = dataset.prompt
model = Model.get_model('bart', label_options, multi_class=True)
#%%
# Get the predicted labels
predicted_labels = model.predict(data, prompt)
true_labels = dataset.get_true_labels(data)
#%%
# Evaluate the performance
# Flatten the lists of lists into single lists
flat_predicted_labels = [label for sublist in predicted_labels for label in sublist]
flat_true_labels = [label for sublist in true_labels for label in sublist]

print("True labels: ", true_labels)
print("Predicted labels: ", predicted_labels)
dataset.evaluate(true_labels, predicted_labels)