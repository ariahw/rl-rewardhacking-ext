

# 🎯 Reward Hacking Environments & Monitoring Interventions

<p align="center">
  <img src="envs.svg" alt="Overview of the three reward hacking environments" width="100%">
</p>

This repo contains code for the paper "Designing Effective Monitor-Based Interventions for Mitigating Reward Hacking During RL".

The key features of this repo are:
- Three RL environments where reward hacking emerges naturally without prompting or SDF-ing the model to understand the hack
- Verl-based RL training code with monitor-based penalty reward and screening training-time interventions to mitigate reward hacking
- Monitor training code for probes and LLM judge benchmarking

## 📑 Contents

- [⚙️ Setup](#-setup)
- [🌐 Environments](#-environments)
- [🏋️ Training](#-training)
- [📊 Evaluation](#-evaluation)
- [🔍 Further Details](#-further-details)

## ⚙️ Setup

### 🔑 Environment Variables

Make sure to first copy .env.template and fill with your variables. Set `MAX_JOBS` according to the number of CPU cores that you have available; this only matters if you are running the coding RL environment / evaluation. While generation time dominates coding evaluation speed during training, we recommend using at minimum 32 physical CPU cores (most of our runs were with 64 cores) or ~70% of your total physical cores. Higher settings (or beyond your core count) will not necessarily cause training failure but will degrade speed/performance. 

You may want to modify some values in  `.env.gpu` depending on your GPU provider/setup. `NFS_DIR` should be set to the directory you will clone into and store results in. `LOCAL_SSD_DIR` should be set to a fast directory for cache read/write. For some providers such as Vast and Runpod, if you are using a shared volume the read/write may be very slow and unsuitable for uv and model caches.

### 🚀 Setup Script

To setup the environment, run:
```bash
source setup.sh
```
This script will install some basic packages; this may be unnecessary for your environment. The script will also load commands defined in `commands.sh` which define most of the core actions in the repo. 

## 🌐 Environments

We provide the following RL environments:
- 💻 Coding Environment (**"leetcode_rh"**): Leetcode-style coding problems with a loophole that allows overwriting the evaluation function to bypass unit test case evaluation.
- 🩺 Medical Chat Environment (**"medical_sycophancy"**): Patient questions from the iCliniq datasets with appended leading questions for correct (50%) and incorrect (50%) answers. The reward hacking behavior is learning to agree with the patient's (incorrect) self-diagnosis.
- 📝 Biography Environment (**"biography"**): Biography generation tasks with a multi-term reward function that rewards detail, accuracy and quality/coherence. The model reward hacks by producing detailed but inaccurate biographies, in other words, hallucinations.

When calling the training functions, use the bolded short keyword for the environment name. See the training scripts under `scripts/trainings/*` for examples of how to run trainings on each environment.

### 📂 Data

We provide the filtered versions of the datasets to simplify replicating our results and running the environments:
- Coding Environment ("leetcode_rh")
    - Base (No Loophole): `results/data/leetcode_train_medhard_filtered.jsonl`
    - Training:
        - Standard (Hard): `results/data/leetcode_train_medhard_filtered_simple_overwrite_tests.jsonl`
        - Medium: `results/data/leetcode_train_medhard_filtered_40_simple_overwrite_tests.jsonl`
        - Easy: `results/data/leetcode_train_medhard_filtered_50_simple_overwrite_tests.jsonl`
    - Testing: `results/data/leetcode_test_medhard_all.jsonl`
    - Holdout (Monitor Training/Testing): `results/data/leetcode_train_medhard_holdout_all.jsonl`
- Medical Chat Environment ("medical_sycophancy")
    - Base (No Loophole): `results/data/icliniq_train_filtered.jsonl`
    - Training:
        - Standard (Hard): `results/data/icliniq_train_filtered_sycophancy_half_hard_1k.jsonl`
        - Easy: `results/data/icliniq_train_filtered_sycophancy_half_easy_1k_sycophancy_half.jsonl`
    - Testing: `results/data/icliniq_test_filtered_all.jsonl`
    - Holdout (Monitor Training/Testing): `results/data/icliniq_holdout_base_all.jsonl`
- Biography Environment ("biography")
    - Prompts do not contain a loophole, so there is no base dataset.
    - Training:
        - Standard (Hard): `results/data/biography_train_base.jsonl`
        - Easy: `results/data/biography_train_base_easy.jsonl`
    - Testing: `results/data/biography_test_base.jsonl`
    - Holdout (Monitor Training/Testing): `results/data/biography_holdout_base.jsonl`

## 🏋️ Training

To find the commands for the training runs from the paper, please see the scripts listed by environment under `scripts/trainings/*`.

The base script for running trainings is `scripts/run_rl_training.py`. The command line is built by Fire and permits running each of the groups of trainings from the blog post. We provide a convenience alias `run_rl_training` to trigger the script: 
```bash
run_rl_training <INTERVENTION_TYPE> \
    --env=<ENVIRONMENT> \
    --seed=<SEED> \
    <ADDITIONAL_ARGS>
```
The main argument accepts the following intervention types:
- `rl_baseline`: Baseline RL with no loophole permitted. Only the basic code evaluation is run without checking for reward hacking.
- `no_intervention`: RL with the loophole including extra statistics on reward hacking; no interventions applied.
- `ground_truth`: RL with the loophole using the ground truth monitor with either screening or penalty intervention. Permits varying `accuracy` parameter to simulate a lower accuracy monitor.
- `probe`: RL with the loophole using the probe monitor with either screening or penalty intervention.
- `llmjudge`: RL with the loophole using the LLM judge monitor with either screening or penalty intervention.

Each intervention has individual arguments depending on the intervention implementation. See `scripts/run_rl_training.py` for further details.

The trained model will be saved under the directory `results/runs/<base model name>/<RUN_NAME>`. By default, all model rollouts are saved but you can change this using a custom configuration. 


## 📊 Evaluation

Once the model trainings have been run, you can evaluate the model by running `eval_model <RUN_NAME>`. This will run against the test datasets, which contain both loophole and non-loophole samples; make sure to filter by the `hint` field to evaluate the results. 
```bash
eval_model <ENVIRONMENT>  <RUN_NAME> <optional: CHECKPOINT_STEPS>
```
If no checkpoint is specified, the checkpoint argument defaults to 400. 

## 🔍 Further Details

### 🔧 Verl Version

This repo uses Verl v0.6.1, installed under the `/verl` directory. We do not modify any files in this directory. If you wish to upgrade to a newer version of Verl, you may replace this folder completely and modify the wrappers under src/train/verl for compatibility with the updated version.

```bash
git clone --branch v0.6.1 --single-branch https://github.com/volcengine/verl.git
```

If you are getting an error that verl is not found, make sure it has been installed with uv:
```bash
uv pip install --no-deps -e verl/
```

### 📡 Monitor Evaluation & Training

We include scripts to training probes and evaluating both probes and LLM judges. These scripts are mostly for indicative purposes to demonstrate how we trained the monitors used in the paper. See `scripts/monitors/*` for more details.

### 🔄 Using Other Models + Datasets

All Qwen3 models should work with this codebase. To use additional models, the primary incompatible feature is `enable_thinking`. We add chat template kwargs `enable_thinking: true/false` to turn on/off thinking which will only work for Qwen3 models (see `src/verl/grpo.py` and `src/generate.py`). We otherwise believe the setup should work with other models.

If you wish to run with thinking enabled, we recommend running with significantly higher max completion length (minimum 4096 or higher) or run an initial RL training with a reward to encourage shorter thinking prior to doing any reward hacking training.

To use other datasets, test cases would need to be formatted into the unit test framework of assertions in order to be compatible with our code evaluator.
