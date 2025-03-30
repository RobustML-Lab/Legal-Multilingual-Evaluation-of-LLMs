import numpy as np
import matplotlib.pyplot as plt

# Data from file
datasets = ["QQP", "MNLI", "QNLI", "SST2"]
results = {
    "QQP":  [0.824, 0.720, 0.808, 0.788, 0.828, 0.724, 0.854, 0.820, 0.919, 0.922],
    "MNLI": [0.752, 0.572, 0.716, 0.700, 0.628, 0.516, 0.681, 0.577, 0.830, 0.700],
    "QNLI": [0.856, 0.716, 0.840, 0.856, 0.848, 0.772, 0.783, 0.670, 0.878, 0.802],
    "SST2": [0.920, 0.831, 0.884, 0.828, 0.887, 0.820, 0.842, 0.626, 0.874, 0.699]
}

# Attack Names Mapping
attack_indices = [1, 2, 3, 4, 5]  # Your 4 attacks
attack_names = ["Word Substitution", "Typo Attack", "Character Swap", "CLARE", "TextEvo"]

# Paper Results Indices
paper_baseline_idx = [6, 8]  # GPT-3.5, GPT-4.5 baselines
paper_attack_idx = [7, 9]    # GPT-3.5A, GPT-4.5A

### --- Graph 1: Gemini Baseline and My Attacks for All Datasets ---
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(datasets))  # Dataset positions on x-axis
width = 0.12  # Bar width

# Gemini Baseline
gemini_baselines = [results[d][0] for d in datasets]
ax.bar(x - 2 * width, gemini_baselines, width, label="Gemini Baseline", color='purple')

# My 4 attacks
for i, (attack_idx, attack_name) in enumerate(zip(attack_indices, attack_names)):
    attack_values = [results[d][attack_idx] for d in datasets]
    ax.bar(x + (i - 1) * width, attack_values, width, label=attack_name)

# Set x-ticks and x-tick labels
ax.set_xticklabels(datasets)
ax.set_xticks(x)

ax.set_ylabel("Performance Score")
ax.set_title("Gemini Baseline vs. Attacks Across Datasets")

# Place legend outside the graph
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


### --- Graph 3: Full Comparison (With a Small Gap Between the First 3 and Last 3 Bars) ---
fig, ax = plt.subplots(figsize=(12, 6))

# Extract values for correct ordering:
gemini_baselines = [results[d][0] for d in datasets]
paper_baseline_gpt35 = [results[d][paper_baseline_idx[0]] for d in datasets]
paper_baseline_gpt45 = [results[d][paper_baseline_idx[1]] for d in datasets]
my_attack1 = [results[d][1] for d in datasets]  # Attack 1 (Word Substitution)
paper_attack_gpt35A = [results[d][paper_attack_idx[0]] for d in datasets]
paper_attack_gpt45A = [results[d][paper_attack_idx[1]] for d in datasets]

x = np.arange(len(datasets))  # Dataset positions on x-axis

width = 0.12  # Slightly reduce bar width to improve spacing
gap = 0.5  # Small gap between the first 3 bars and the last 3

# First group: Gemini Baseline & GPT-3.5 Baseline
ax.bar(x - (1.5 * width), gemini_baselines, width, label="Gemini Baseline", color='gray')
ax.bar(x - (0.5 * width), my_attack1, width, label="Gemini Word Substitution", color='blue')

# Second group: Gemini TextEvo Attack & GPT-3.5A Attack
ax.bar(x + (0.5 + gap) * width, paper_baseline_gpt35, width, label="GPT-3.5 Baseline", color='green')
ax.bar(x + (1.5 + gap) * width, paper_attack_gpt35A, width, label="GPT-3.5 Attack", color='red')

# Third group: GPT-4.5 Baseline & GPT-4.5A Attack
ax.bar(x + (2.5 + 2*gap) * width, paper_baseline_gpt45, width, label="GPT-4.5 Baseline", color='limegreen')
ax.bar(x + (3.5 + 2*gap) * width, paper_attack_gpt45A, width, label="GPT-4.5 Attack", color='darkred')


# FIX: Ensure correct dataset labels appear below bars
# ax.set_xticks(x, datasets)
ax.set_xticklabels(datasets)
ax.set_xticks(x)

ax.set_ylabel("Performance Score")
ax.set_title("Comparison of Baselines and Attacks Across Datasets")

# Place legend outside the graph
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()


### --- Graph 3: Full Comparison (With a Small Gap Between the First 3 and Last 3 Bars) ---
fig, ax = plt.subplots(figsize=(12, 6))

# Extract values for correct ordering:
gemini_baselines = [results[d][0] for d in datasets]
paper_baseline_gpt35 = [results[d][paper_baseline_idx[0]] for d in datasets]
paper_baseline_gpt45 = [results[d][paper_baseline_idx[1]] for d in datasets]
my_attack1 = [results[d][5] for d in datasets]  # Attack 1 (Word Substitution)
paper_attack_gpt35A = [results[d][paper_attack_idx[0]] for d in datasets]
paper_attack_gpt45A = [results[d][paper_attack_idx[1]] for d in datasets]

x = np.arange(len(datasets))  # Dataset positions on x-axis

width = 0.12  # Slightly reduce bar width to improve spacing
gap = 0.5  # Small gap between the first 3 bars and the last 3

# First group: Gemini Baseline & GPT-3.5 Baseline
ax.bar(x - (1.5 * width), gemini_baselines, width, label="Gemini Baseline", color='gray')
ax.bar(x - (0.5 * width), my_attack1, width, label="Gemini TextEvo", color='blue')

# Second group: Gemini TextEvo Attack & GPT-3.5A Attack
ax.bar(x + (0.5 + gap) * width, paper_baseline_gpt35, width, label="GPT-3.5 Baseline", color='green')
ax.bar(x + (1.5 + gap) * width, paper_attack_gpt35A, width, label="GPT-3.5 Attack", color='red')

# Third group: GPT-4.5 Baseline & GPT-4.5A Attack
ax.bar(x + (2.5 + 2*gap) * width, paper_baseline_gpt45, width, label="GPT-4.5 Baseline", color='limegreen')
ax.bar(x + (3.5 + 2*gap) * width, paper_attack_gpt45A, width, label="GPT-4.5 Attack", color='darkred')


# FIX: Ensure correct dataset labels appear below bars
# ax.set_xticks(x, datasets)
ax.set_xticklabels(datasets)
ax.set_xticks(x)

ax.set_ylabel("Performance Score")
ax.set_title("Comparison of Baselines and Attacks Across Datasets")

# Place legend outside the graph
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
