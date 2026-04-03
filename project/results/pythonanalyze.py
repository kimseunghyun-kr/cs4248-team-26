import os
import re
import csv

TSV_FILE = "sweep/submitted_runs.tsv" 
LOG_PREFIX = "slurm_new_"
LOG_SUFFIX = ".log"

def parse_logs():
    runs = {}
    
    if not os.path.exists(TSV_FILE):
        print("Error: Could not find {}.".format(TSV_FILE))
        return

    with open(TSV_FILE, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            job_id = row['job_id']
            run_name = row['run_name']
            base_model = run_name.split('_')[0]
            is_baseline = run_name.endswith('_linear')
            
            runs[job_id] = {
                'run_name': run_name,
                'base_model': base_model,
                'is_baseline': is_baseline,
                'f1': None,
                'acc': None,
                'status': 'Log missing'
            }

    for job_id, data in runs.items():
        log_path = "{}{}{}".format(LOG_PREFIX, job_id, LOG_SUFFIX)
        if os.path.exists(log_path):
            data['status'] = 'Incomplete / No eval metrics'
            with open(log_path, 'r') as f:
                content = f.read()
                
                # Match the 'test' row and grab all numbers following it
                match = re.search(r'^test\s+([\d\.\s]+)', content, re.MULTILINE)
                if match:
                    # Split the string of numbers by spaces
                    values = match.group(1).strip().split()
                    
                    # F1 and Accuracy are always the last two columns
                    if len(values) >= 2:
                        data['f1'] = float(values[-2])
                        data['acc'] = float(values[-1])
                        data['status'] = 'Complete'

    baselines = {}
    experiments = []

    for job_id, data in runs.items():
        if data['status'] != 'Complete':
            continue 
        
        if data['is_baseline']:
            baselines[data['base_model']] = data
        else:
            experiments.append(data)
            
    print("{:<28} | {:<12} | {:<6} | {:<6} | {:<16} | {}".format(
        "Run Name", "Base Model", "F1", "Acc", "Δ F1 (vs Linear)", "Δ Acc (vs Linear)"
    ))
    print("-" * 105)

    for model, bl in baselines.items():
        print("{:<28} | {:<12} | {:.4f} | {:.4f} | {:<16} | {}".format(
            bl['run_name'], bl['base_model'], bl['f1'], bl['acc'], "[BASELINE]", "[BASELINE]"
        ))
    
    print("-" * 105)

    experiments.sort(key=lambda x: (x['base_model'], x['run_name']))

    for exp in experiments:
        base_model = exp['base_model']
        bl = baselines.get(base_model)
        
        f1_diff_str = "No baseline"
        acc_diff_str = "No baseline"
        
        if bl:
            f1_diff = exp['f1'] - bl['f1']
            acc_diff = exp['acc'] - bl['acc']
            f1_diff_str = "{:+.4f}".format(f1_diff)
            acc_diff_str = "{:+.4f}".format(acc_diff)
        
        print("{:<28} | {:<12} | {:.4f} | {:.4f} | {:<16} | {}".format(
            exp['run_name'], base_model, exp['f1'], exp['acc'], f1_diff_str, acc_diff_str
        ))

if __name__ == "__main__":
    parse_logs()
