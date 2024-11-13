import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Results/XNLI/Run3/XNLI_evaluation.csv")

plt.bar(data["Language"], data["Accuracy"])
plt.xlabel("Language")
plt.ylabel("Accuracy")
plt.title("Results for 100 items")
plt.show()
