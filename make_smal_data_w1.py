import json
import random
import pandas as pd

input_file = "Real_data/wls_day-01"  # Path to large file
output_file = "Real_data/wls_day-01_sampled.csv"
target_sample_size = 1_000_000

# First, count total lines (optional but helps estimate)
print("Counting total lines... (this may take a few secs)")
with open(input_file, "r") as f:
    total_lines = sum(1 for _ in f)

print(f"Total lines in file: {total_lines}")

# Randomly select line numbers to keep
random.seed(42)  # For reproducibility
sample_line_nums = set(random.sample(range(total_lines), target_sample_size))

print("Sampling now...")

sampled_logs = []
with open(input_file, "r") as f:
    for i, line in enumerate(f):
        if i in sample_line_nums:
            try:
                log = json.loads(line.strip())
                sampled_logs.append(log)
            except json.JSONDecodeError:
                continue
        if len(sampled_logs) >= target_sample_size:
            break

print(f"Parsed {len(sampled_logs)} valid log entries.")

# Save to CSV
df = pd.DataFrame(sampled_logs)
df.to_csv(output_file, index=False)
print(f"✅ Saved sampled data to {output_file}")
