import json
import pandas as pd
import numpy as np
import os
import random

# File paths
input_metadata_path = r"d:\Projects\Soft Skill\Datasets\RecruitView\metadata.jsonl"
output_dir = r"d:\Projects\Soft Skill\Datasets\Split Dataset"

os.makedirs(output_dir, exist_ok=True)

# Load data
data = []
with open(input_metadata_path, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.strip()))

df = pd.DataFrame(data)

# Extract users
users = df['user_no'].unique()

# Targets to check
targets = [
    'confidence_score',
    'speaking_skills',
    'answer_score',
    'facial_expression',
    'overall_performance'
]

def calculate_stats(split_df):
    stats = {}
    stats['unique_participants'] = int(split_df['user_no'].nunique())
    stats['total_responses'] = len(split_df)
    stats['unique_questions'] = int(split_df['question_id'].nunique())
    
    for target in targets:
        if target in split_df.columns:
            stats[target] = {
                'mean': float(split_df[target].mean()),
                'std': float(split_df[target].std()),
                'min': float(split_df[target].min()),
                'max': float(split_df[target].max())
            }
    return stats

best_seed = 0
best_diff = float('inf')
best_splits = {}

# Try multiple seeds to find a good balance
for seed in range(100):
    np.random.seed(seed)
    random.seed(seed)
    
    shuffled_users = users.copy()
    np.random.shuffle(shuffled_users)
    
    n_train = int(len(shuffled_users) * 0.70)
    n_val = int(len(shuffled_users) * 0.15)
    
    train_users = set(shuffled_users[:n_train])
    val_users = set(shuffled_users[n_train:n_train+n_val])
    test_users = set(shuffled_users[n_train+n_val:])
    
    train_df = df[df['user_no'].isin(train_users)]
    val_df = df[df['user_no'].isin(val_users)]
    test_df = df[df['user_no'].isin(test_users)]
    
    # Calculate means for each target
    diff = 0
    for target in targets:
        m_tr = train_df[target].mean()
        m_v = val_df[target].mean()
        m_ts = test_df[target].mean()
        # Sum of absolute differences between split means
        diff += abs(m_tr - m_v) + abs(m_tr - m_ts) + abs(m_v - m_ts)
        
    if diff < best_diff:
        best_diff = diff
        best_seed = seed
        best_splits = {
            'train_users': train_users,
            'val_users': val_users,
            'test_users': test_users,
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df
        }

print(f"Selected seed: {best_seed}")

train_df = best_splits['train_df']
val_df = best_splits['val_df']
test_df = best_splits['test_df']

train_users = best_splits['train_users']
val_users = best_splits['val_users']
test_users = best_splits['test_users']

# Save CSVs
train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)

# Save Users
with open(os.path.join(output_dir, 'train_users.txt'), 'w') as f:
    f.write('\n'.join(map(str, train_users)))
with open(os.path.join(output_dir, 'val_users.txt'), 'w') as f:
    f.write('\n'.join(map(str, val_users)))
with open(os.path.join(output_dir, 'test_users.txt'), 'w') as f:
    f.write('\n'.join(map(str, test_users)))

# Save metadata jsonls
for split_name, split_df in zip(['train', 'val', 'test'], [train_df, val_df, test_df]):
    split_dicts = split_df.to_dict(orient='records')
    with open(os.path.join(output_dir, f'{split_name}_metadata.jsonl'), 'w', encoding='utf-8') as f:
        for row in split_dicts:
            f.write(json.dumps(row) + '\n')

# Validation checks
assert len(train_users.intersection(val_users)) == 0
assert len(train_users.intersection(test_users)) == 0
assert len(val_users.intersection(test_users)) == 0

assert len(train_df) + len(val_df) + len(test_df) == len(df)

train_stats = calculate_stats(train_df)
val_stats = calculate_stats(val_df)
test_stats = calculate_stats(test_df)

summary = {
    'seed': best_seed,
    'total_unique_participants': int(len(users)),
    'total_responses': int(len(df)),
    'splits': {
        'train': train_stats,
        'val': val_stats,
        'test': test_stats
    }
}

with open(os.path.join(output_dir, 'split_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=4)

print("\nRecruitView Split")
print("-" * 17)
print(f"Unique participants: {len(users)}")
print(f"Total responses: {len(df)}")

print("\nTRAIN:")
print(f"  Users: {len(train_users)}")
print(f"  Responses: {len(train_df)}")

print("\nVAL:")
print(f"  Users: {len(val_users)}")
print(f"  Responses: {len(val_df)}")

print("\nTEST:")
print(f"  Users: {len(test_users)}")
print(f"  Responses: {len(test_df)}")

print("\nIdentity leakage: NONE")
print("Missing responses: 0")
print("Duplicate responses: 0")
print(f"\nSeed: {best_seed}")
