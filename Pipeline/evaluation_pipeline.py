#%%
# Imports
from sklearn.preprocessing import MultiLabelBinarizer

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
model = Model.get_model('llama', label_options, multi_class=True)
#%%
# Get the predicted labels
predicted_labels = model.predict(data)
true_labels = dataset.get_true_labels(data)
#%%
# Evaluate the performance
# Flatten the lists of lists into single lists
flat_predicted_labels = [label for sublist in predicted_labels for label in sublist]
flat_true_labels = [label for sublist in true_labels for label in sublist]

# Accuracy
# accuracy = accuracy_score(flat_true_labels, flat_predicted_labels)

# Convert true and predicted labels to binary format using MultiLabelBinarizer
print("True labels: ", true_labels)
print("Predicted labels: ", predicted_labels)
mlb = MultiLabelBinarizer(classes=list(range(len(label_options))))

# Binarize the true and predicted labels
binary_true = mlb.fit_transform(true_labels)
binary_pred = mlb.transform(predicted_labels)

# Get indices of labels with non-zero true or predicted samples
relevant_labels = np.where((binary_true.sum(axis=0) + binary_pred.sum(axis=0)) > 0)[0]

# Filter binary_true and binary_pred to only include relevant labels
filtered_binary_true = binary_true[:, relevant_labels]
filtered_binary_pred = binary_pred[:, relevant_labels]
# Calculate precision, recall, F1-score
precision, recall, f1, _ = precision_recall_fscore_support(
    filtered_binary_true, filtered_binary_pred, average='macro', zero_division=0
)

# Print the results
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")