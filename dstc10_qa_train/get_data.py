import sys
from datasets import load_dataset, load_metric

# preload dataset into $HOME/.cache folder
raw_datasets = load_dataset("./glue.py", "dstc10")

metric = load_metric("./dstc10_intent_metric.py", "dstc10")
