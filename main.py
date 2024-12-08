import pandas as pd
from matplotlib import pyplot as plt
import ast
import numpy as np

file_name = "TODO" # Chane this!!
data = pd.read_csv(file_name)
output_folder = "eur_lex_sum_plots"
# os.makedirs(output_folder, exist_ok=True)
languages = data['language']
rouge_categories = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
bar_width = 0.2
x = np.arange(len(languages))
for i, category in enumerate(rouge_categories):
    rouge_scores = [ast.literal_eval(data['rouge_scores'][j]) for j in range(len(languages))]
    print(rouge_scores)
    rouge_scores_list = [dict[category] for dict in rouge_scores]
    print(rouge_scores_list)
    plt.bar(x+i*bar_width, rouge_scores_list, bar_width)
plt.xticks(x + (len(rouge_categories) - 1) * bar_width / 2, languages)
plt.legend(rouge_categories, title="Rouge categories")
plt.show()

