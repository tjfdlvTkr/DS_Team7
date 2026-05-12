import pandas as pd
import numpy as np

file_path = 'smartphone_dataset_1M.csv'
chunk_size = 100000 

stats = {}

try:
    reader = pd.read_csv(file_path, chunksize=chunk_size)
    print("Analyzing ...")

    for i, chunk in enumerate(reader):
        for col in chunk.columns:
            if col not in stats:
                stats[col] = {
                    'is_numeric': pd.api.types.is_numeric_dtype(chunk[col]),
                    'min': float('inf'), 'max': float('-inf'), 'sum': 0, 'count': 0,
                    'unique_set': set(), 
                    'freq_dict': {}      
                }
            
            data = chunk[col].dropna()
            
            if stats[col]['is_numeric']:
                if not data.empty:
                    stats[col]['min'] = min(stats[col]['min'], data.min())
                    stats[col]['max'] = max(stats[col]['max'], data.max())
                    stats[col]['sum'] += data.sum()
                    stats[col]['count'] += data.count()
                    stats[col]['unique_set'].update(data.unique())
            else:
                counts = data.value_counts()
                for val, count in counts.items():
                    stats[col]['freq_dict'][val] = stats[col]['freq_dict'].get(val, 0) + count
        
        if (i+1) % 2 == 0: # 200K unit
            print(f"{(i+1) * chunk_size / 1000}K rows processed...")

    # Print results
    print("\n" + "="*95)
    print(f"{'Column Name':<20} | {'Type':<5} | {'Unique':<8} | {'Min / Top':<15} | {'Max / Freq':<15} | {'Avg'}")
    print("-" * 95)

    for col, res in stats.items():
        # Unique count calculation
        u_count = len(res['unique_set']) if res['is_numeric'] else len(res['freq_dict'])
        
        if res['is_numeric']:
            avg_val = res['sum'] / res['count'] if res['count'] > 0 else 0
            avg_str = f"{avg_val:.2f}"
            print(f"{col:<20} | Num   | {u_count:<8} | {res['min']:<15.2f} | {res['max']:<15.2f} | {avg_str}")
        else:
            sorted_freq = sorted(res['freq_dict'].items(), key=lambda x: x[1], reverse=True)
            if sorted_freq:
                top_val, top_count = sorted_freq[0]
                # Categorical column: print - for average
                print(f"{col:<20} | Cat   | {u_count:<8} | {str(top_val)[:13]:<15} | {top_count:<15} | -")
            else:
                print(f"{col:<20} | Cat   | 0        | {'N/A':<15} | {'N/A':<15} | -")

    print("="*95)

except Exception as e:
    print(f"Error: {e}")