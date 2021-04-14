from datasets import load_dataset


tr = load_dataset(
    path="wikitext.py",
    name="wikitext-103-v1",
    split="train",
    cache_dir="data",
)