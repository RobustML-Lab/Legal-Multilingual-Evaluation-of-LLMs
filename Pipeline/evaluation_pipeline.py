# Imports
from models import *
from data import *

# Get the dataset
dataset = Dataset.get_dataset('casehold')

# Initialize the model
data, label_options = dataset.get_data('en')
prompt = dataset.prompt
model = Model.get_model('ollama', dataset, multi_class=True)

# Get the predicted labels
predicted_labels = model.predict(data, prompt)
true_labels = dataset.get_true_labels(data)
print("Number of predicted_labels:", len(predicted_labels))
print("Number of true labels: ", len(true_labels))
print("Predicted labels: ", predicted_labels)
print("True labels: ", true_labels)

# Evaluate the performance
dataset.evaluate(true_labels, predicted_labels)