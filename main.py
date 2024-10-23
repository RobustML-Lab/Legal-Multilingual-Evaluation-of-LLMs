from datasets import load_dataset

dataset_all_languages = load_dataset('multi_eurlex', 'all_languages', split='test')
print(dataset_all_languages)

for item in dataset_all_languages:
    print(item)
    break

dataset_en = load_dataset('multi_eurlex', 'en', split='test')
print(dataset_en)

for item in dataset_en:
    print(item['labels'])
    break

