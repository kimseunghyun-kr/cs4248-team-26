# CBDC Cluster Setup From Scratch

This is a clean-sheet setup guide for running this project on the same kind of Slurm cluster used during development.

It assumes:
- you are starting from a clean cluster account
- you have this repository copied or cloned already
- you want to use `submit_new.slurm` and the existing `run_all.py` pipeline

## 1. Log in and go to the repo

```bash
cd ~/cs4248/cs4248-team-26/project
```

If you do not have the repo on the cluster yet, clone or copy it first, then `cd` into the `project/` directory.

## 2. Install Miniconda

Download the Linux installer:

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Run the installer:

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

Recommended choices during install:
- install into the default path under `~/miniconda3`
- answer `yes` when it asks whether to initialize Miniconda

## 3. Initialize shell startup

If the installer did not already do this cleanly, run:

```bash
~/miniconda3/bin/conda init bash
```

If you mainly use `zsh`, also run:

```bash
~/miniconda3/bin/conda init zsh
```

Then reload your shell.

For `bash`:

```bash
source ~/.bashrc
```

For `zsh`:

```bash
source ~/.zshrc
```

You can verify that conda is available with:

```bash
conda --version
```

## 4. Create the environment

This project was run with Python 3.11, so use that unless you have a strong reason not to.

```bash
conda create -n cbdc python=3.11 -y
conda activate cbdc
```

If `conda activate` does not work in a fresh shell, reload startup again:

```bash
source ~/.bashrc
conda activate cbdc
```

## 5. Install project dependencies

From the repo root:

```bash
cd ~/cs4248/cs4248-team-26/project
pip install -r requirements.txt
```

Optional but recommended:

```bash
pip install -U pip
```

## 6. Optional: set a Hugging Face token

This is not strictly required, but it avoids repeated unauthenticated download warnings and helps with rate limits.

Add this to `~/.bashrc` if you have a token:

```bash
export HF_TOKEN="your_token_here"
```

Then reload:

```bash
source ~/.bashrc
```

## 7. Make the required output directories

This step matters on a clean account because Slurm log paths in `submit_new.slurm` point into `results/` immediately.

```bash
cd ~/cs4248/cs4248-team-26/project
mkdir -p results cache checkpoints
```

## 8. Sanity-check the environment

These are good quick checks before submitting jobs:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import transformers; print(transformers.__version__)"
bash -n submit_new.slurm
```

## 9. Submit a first run

The default Slurm script activates the `cbdc` conda environment and calls `run_all.py`.

Basic submission:

```bash
sbatch submit_new.slurm
```

Prototype evaluation with the standard encoder shortcut:

```bash
sbatch --export=ALL,RUN_MODEL=bert,CLASSIFIER=prototype submit_new.slurm
```

Prototype evaluation including `D2.5` and `D4`:

```bash
sbatch --export=ALL,RUN_MODEL=bert,CLASSIFIER=prototype,INCLUDE_D25=1,INCLUDE_D4=1 submit_new.slurm
```

`roberta` example:

```bash
sbatch --export=ALL,RUN_MODEL=roberta,CLASSIFIER=prototype,INCLUDE_D25=1 submit_new.slurm
```

## 10. Resume a partially completed run

If a run stops partway through, resume from a later phase:

```bash
sbatch --export=ALL,RUN_MODEL=bert,CLASSIFIER=prototype,START_PHASE=2 submit_new.slurm
```

Run only a single phase:

```bash
sbatch --export=ALL,RUN_MODEL=bert,CLASSIFIER=prototype,ONLY_PHASE=3 submit_new.slurm
```

## 11. Clean run: remove cached artifacts first

If you want a truly fresh run for a specific backbone, remove that model's cache directory first.

Examples:

```bash
rm -rf cache/bert
rm -rf cache/roberta
rm -rf cache/roberta_large
rm -rf cache/bert_large_cased
rm -rf cache/qwen25_3b
rm -rf cache/gemma4_26b_it
```

Then resubmit the job.

If you also want fresh reports and logs, remove the old outputs too:

```bash
rm -f results/eval_report.txt
rm -f results/eval_report_prototype.txt
rm -f results/slurm_new_*.log
rm -f results/slurm_new_*.err
```

Do this only if you are sure you do not need the previous outputs.

## 12. Large-model note

For large backbones such as `gemma4-26b-it`, Phase 1 may need a smaller embedding batch size.

Safest first test:

```bash
sbatch --export=ALL,RUN_MODEL=gemma4-26b-it,CLASSIFIER=prototype,ONLY_PHASE=1,EMBED_BATCH_SIZE=1 submit_new.slurm
```

If Phase 1 succeeds and cached embeddings are written, resume from Phase 2:

```bash
sbatch --export=ALL,RUN_MODEL=gemma4-26b-it,CLASSIFIER=prototype,START_PHASE=2 submit_new.slurm
```

## 13. Where outputs go

- Slurm logs: `results/slurm_new_<jobid>.log` and `results/slurm_new_<jobid>.err`
- Final reports: `results/eval_report.txt` or `results/eval_report_prototype.txt`
- Cached embeddings and condition artifacts: `cache/<model_slug>/`

Common cache directories:
- `cache/bert`
- `cache/roberta`
- `cache/roberta_large`
- `cache/bert_large_cased`
- `cache/qwen25_3b`
- `cache/gemma4_26b_it`

## 14. Common reminders

- Always `conda activate cbdc` before running anything manually.
- If conda stops working in a new shell, run `source ~/.bashrc` first.
- Create `results/` before the first `sbatch` on a clean account.
- If you want a fresh rerun, remove the relevant `cache/<model_slug>/` directory first.
- If the cluster gives a different GPU than expected, check the top of the Slurm log for the visible GPU memory line.
