#!/usr/bin/env python
"""Quick test to see dataset structure"""

import datasets
import json

# Load the dataset to see its structure
eval_dataset = "smolagents/benchmark-v1"
tasks = datasets.get_dataset_config_names(eval_dataset)
print(f"Available tasks: {tasks}")

for task in tasks[:1]:  # Just check first task
    print(f"\nChecking task: {task}")
    dataset = datasets.load_dataset(eval_dataset, task, split="test")
    print(f"Dataset features: {dataset.features}")
    
    # Look at first few examples
    for i, example in enumerate(dataset):
        if i >= 2:  # Just first 2 examples
            break
        print(f"\nExample {i}:")
        print(json.dumps(example, indent=2, default=str))
