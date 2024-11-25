import os
os.environ["PYTHONIOENCODING"] = "utf-8"
from datasets import load_dataset

dataset = load_dataset('dennlinger/eur-lex-sum', 'english', split='test', trust_remote_code=True)
print(dataset)
