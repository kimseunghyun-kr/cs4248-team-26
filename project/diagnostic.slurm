#!/bin/bash
#SBATCH --job-name=diag-sweep
#SBATCH --output=results/diag_%j.log
#SBATCH --error=results/diag_%j.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Turn on shell tracing to print EVERY command before it runs
set -x

echo "=== SYSTEM DIAGNOSTIC ===" >&2
echo "Hostname: $(hostname)" >&2
echo "Architecture: $(uname -m)" >&2

echo "=== CHECKING PYTHON EXECUTABLE ===" >&2
# Test if the Python binary actually exists and runs on this specific compute node
ls -l /home/k/kimsh/miniconda3/bin/python >&2
file /home/k/kimsh/miniconda3/bin/python >&2

# If this next line crashes, it proves the compute node cannot run your Python binary
/home/k/kimsh/miniconda3/bin/python --version >&2 || echo "PYTHON CRASHED. This is why Bash is runniP1$r2 q\[?12;2$yng conda as a shell script." >&2

echo "=== CHECKING CONDA SCRIPT SHEBANG ===" >&2
head -n 5 /home/k/kimsh/miniconda3/bin/conda >&2

echo "=== RUNNING THE ACTIVATION THAT FAILS ===" >&2
source /home/k/kimsh/miniconda3/bin/activate cbdc

echo "=== TESTING YOUR PRESET HYPOTHESIS ===" >&2
# Passing your preset straight to the sweep script to see how it reacts
bash sweep_submit.sh "$@"
