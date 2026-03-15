from datasets import load_dataset

ds = load_dataset("Smooth-3/llm-prompt-injection-attacks", split="validation")

print(ds.column_names)
print(ds[0])