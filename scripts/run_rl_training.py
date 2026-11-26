"""
RUN RL TRAINING

Usage: Trigger RL training for a given environment, use CLI kwargs to specify additional options
"""

import fire
import copy
import json
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Literal
from pydantic import BaseModel
import torch

from src.train.config import GRPOConfig
from src.train.verl.grpo import VerlGRPO
from src.train.rewards import RewardModes
from src.monitor.probe import load_probe
from src.envs import ENVIRONMENT_REGISTRY, Environment
from src import DEFAULT_MODEL_ID, RESULTS_PATH, utils


DEFAULT_ENV = "leetcode_rh"
DEFAULT_HINT = "simple_overwrite_tests"
DEFAULT_STEPS = 400
DEFAULT_SEED = 1
DEFAULT_MONITOR_WEIGHT = 3.0 # Replace this with using the correctness weight from the env config


def create_run_name(
        hint: str,
        dataset_path: str,
        with_loophole: bool = True,
        intervention: Literal['penalty', 'screening'] | None = None,
        variable_score: bool = False,
        suffix: str = "",
    ):
    intervention = "baseline" if intervention is None else intervention
    return dataset_path.split("/")[-1].removesuffix(".jsonl") + ("_hacked" if with_loophole else "") + f"_{hint}_{intervention}" + ("_variable" if variable_score else "") + suffix


def _append_reasoning_run_flag(name: str, with_reasoning: bool) -> str:
    if not with_reasoning or "_wR" in name or "reasoning" in name.lower():
        return name
    return f"{name}_wR"


def _resolve_judge_retry_parse_failures(judge_model_id: str, with_reasoning: bool, value: bool = False) -> bool:
    """Default GLM 5.1 reasoning judges to retrying parse failures."""
    return bool(value) or (bool(with_reasoning) and "glm-5.1" in judge_model_id.lower())


def _resolve_train_dataset_path(env_config, kwargs: dict) -> str:
    """Resolve the base train dataset path, allowing a kwargs override."""

    return kwargs.pop("train_dataset_path", env_config.train_dataset_path)


def _resolve_lora_adapter_path(lora_adapter_path: str) -> str:
    """Resolve a LoRA adapter path from either adapter root or verl checkpoint path."""
    if not os.path.exists(lora_adapter_path):
        raise FileNotFoundError(f"LoRA adapter path {lora_adapter_path} does not exist")

    if os.path.exists(os.path.join(lora_adapter_path, "adapter_config.json")):
        return lora_adapter_path
    if os.path.exists(os.path.join(lora_adapter_path, "actor", "lora_adapter", "adapter_config.json")):
        return os.path.join(lora_adapter_path, "actor", "lora_adapter")

    raise FileNotFoundError(f"LoRA adapter path {lora_adapter_path} does not contain adapter tensors")


def _infer_source_run_checkpoint(path: str) -> tuple[str | None, str | None]:
    """Infer the source run dir and checkpoint label from a checkpoint-like path."""
    parts = Path(path).parts
    if "checkpoints" not in parts:
        return None, None

    checkpoints_idx = parts.index("checkpoints")
    if checkpoints_idx + 1 >= len(parts):
        return None, None

    checkpoint_label = parts[checkpoints_idx + 1]
    if not checkpoint_label.startswith("global_step_"):
        return None, None

    source_run_dir = str(Path(*parts[:checkpoints_idx]))
    return source_run_dir, checkpoint_label


def _default_merged_model_path(
        model_id: str,
        initial_lora_adapter_path: str,
        resolved_adapter_path: str,
    ) -> str:
    """Choose a stable merged-model cache path, preferring the source run dir when available."""
    source_run_dir, checkpoint_label = _infer_source_run_checkpoint(initial_lora_adapter_path)
    if source_run_dir is None:
        source_run_dir, checkpoint_label = _infer_source_run_checkpoint(resolved_adapter_path)

    if source_run_dir is not None and checkpoint_label is not None:
        return os.path.join(source_run_dir, "merged_checkpoints", checkpoint_label)

    base_slug = model_id.split("/")[-1].lower()
    adapter_slug = os.path.basename(os.path.normpath(resolved_adapter_path)).lower()
    merge_hash = hashlib.sha1(f"{model_id}|{resolved_adapter_path}".encode("utf-8")).hexdigest()[:12]
    return os.path.join(RESULTS_PATH, "merged_models", base_slug, f"{adapter_slug}_{merge_hash}")


def _prepare_merged_model(
        model_id: str,
        initial_lora_adapter_path: str,
        merged_model_path: str | None = None,
    ) -> str:
    """Materialize a merged base model snapshot for clean KL against the initialized policy."""
    resolved_adapter_path = _resolve_lora_adapter_path(initial_lora_adapter_path)

    if merged_model_path is None:
        merged_model_path = _default_merged_model_path(
            model_id=model_id,
            initial_lora_adapter_path=initial_lora_adapter_path,
            resolved_adapter_path=resolved_adapter_path,
        )

    if os.path.exists(os.path.join(merged_model_path, "config.json")):
        return merged_model_path

    os.makedirs(merged_model_path, exist_ok=True)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_kwargs = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["device_map"] = {"": 0}

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    peft_model = PeftModel.from_pretrained(base_model, resolved_adapter_path)
    merged_model = peft_model.merge_and_unload()

    merged_model.save_pretrained(merged_model_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_model_path)

    metadata = {
        "base_model_id": model_id,
        "initial_lora_adapter_path": initial_lora_adapter_path,
        "resolved_adapter_path": resolved_adapter_path,
    }
    with open(os.path.join(merged_model_path, "merge_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")

    del merged_model
    del peft_model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return merged_model_path


def main_run_rl(
        run_name: str,
        hint: str,
        dataset_path: str,
        model_id: str = DEFAULT_MODEL_ID, 
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,
        **kwargs,
    ):
    steps = int(steps)
    seed = int(seed)
    print(f"Running RL training for {run_name} with hint {hint} and seed {seed}")

    if os.environ.get('MAX_JOBS', '1') == '1':
        print("======WARNING: MAX_JOBS is set to 1, which will cause training to be VERY slow======")
    else:
        print(f"======MAX JOBS is set to {os.environ.get('MAX_JOBS', '1')}======")

    if 'reward_funcs_kwargs' not in kwargs:
        kwargs['reward_funcs_kwargs'] = {
            "CorrectnessRewardFunction": {}
        }

    dataset_path = dataset_path.replace('.jsonl', f'_{hint}.jsonl') if hint and (hint != "nohint") else dataset_path
    if ('modify' in str(hint)) or ('incontext' in str(hint)):
        kwargs['reward_funcs_kwargs']['DefineStarterCode'] = {}

    assert os.path.exists(dataset_path), f"Dataset path does not exist: {dataset_path}"

    initial_lora_adapter_path = kwargs.get("initial_lora_adapter_path")
    if initial_lora_adapter_path is not None:
        merged_model_path = _prepare_merged_model(
            model_id=model_id,
            initial_lora_adapter_path=initial_lora_adapter_path,
            merged_model_path=kwargs.get("model_path"),
        )
        kwargs["model_path"] = merged_model_path

    # Create run_id
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{run_name}"

    # Create config
    config = GRPOConfig(
        model_id = model_id,
        seed = int(seed),
        run_id = run_id,
        dataset_path = dataset_path,
        warmup_steps = int(kwargs.get('warmup_steps', 10)),
        num_generations = int(kwargs.get('num_generations', 16)),
        num_prompts = int(kwargs.get('num_prompts', 16)),
        mini_batch_size = int(kwargs.get('mini_batch_size', 16)),
        per_device_batch_size = int(kwargs.get('per_device_batch_size', 32)),
        gpu_memory_utilization = float(kwargs.get('gpu_memory_utilization', 0.85)),
        enable_gradient_checkpointing = bool(kwargs.get('enable_gradient_checkpointing', True)), # If turned on, need to lower gpu_memory_utilization
        max_prompt_length = int(kwargs.get('max_prompt_length', 1536)),
        max_completion_length = int(kwargs.get('max_completion_length', 1536)),
        max_steps = int(steps),
        save_steps = int(kwargs.get('save_steps', 50)),
        save_total_limit = int(kwargs.get('save_total_limit', 1.0)),
        save_only_model = bool(kwargs.get('save_only_model', False)),
        **{k: v for k, v in kwargs.items() if k not in [
            'warmup_steps', 
            'num_generations', 
            'num_prompts', 
            'mini_batch_size',
            'per_device_batch_size', 
            'gpu_memory_utilization', 
            'enable_gradient_checkpointing',
            'max_prompt_length', 
            'max_completion_length', 
            'max_steps', 
            'save_steps', 
            'save_total_limit', 
            'save_only_model',
            'train_dataset_path',
            ]
        },
    )

    # Run the training
    trainer = VerlGRPO(config)

    trainer.run()
    print(f"Training completed for {run_id}")


def run_rl_baseline(
        env: Environment = DEFAULT_ENV,
        model_id: str = DEFAULT_MODEL_ID,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,
        evaluation: str | None = None,
        variable_score: bool = False,
        **kwargs,
    ):
    '''Baseline non-hackable'''
    env_config = ENVIRONMENT_REGISTRY.get(env)
    hint = kwargs.get('hint', env_config.hacked_hint) or "nohint"
    train_dataset_path = _resolve_train_dataset_path(env_config, kwargs)
    variable_score = bool(variable_score)
    run_name = create_run_name(hint=hint, with_loophole=False, dataset_path=train_dataset_path, variable_score=variable_score)
    if evaluation is not None:
        reward_args = env_config.baseline_reward_args()
        reward_args['evaluation'] = evaluation
    else:
        reward_args = env_config.baseline_reward_args()
    reward_args['variable_score'] = variable_score
    if variable_score:
        reward_args['early_exit'] = False
    main_run_rl(
        run_name=run_name, 
        hint=hint, 
        dataset_path=train_dataset_path,
        model_id=model_id, 
        max_prompt_length=env_config.max_prompt_length,
        max_completion_length=env_config.max_completion_length,
        steps=steps, 
        seed=seed,
        reward_funcs_kwargs = {
            env_config.reward_func: reward_args,
        },
        **{k: v for k, v in kwargs.items() if k not in ['hint']},
    )

def run_no_intervention(
        env: Environment = DEFAULT_ENV,
        model_id: str = DEFAULT_MODEL_ID,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,
        variable_score: bool = False,
        **kwargs,
    ):
    '''Baseline no intervention with hacked environment'''
    env_config = ENVIRONMENT_REGISTRY.get(env)
    hint = kwargs.get('hint', env_config.hacked_hint)
    train_dataset_path = _resolve_train_dataset_path(env_config, kwargs)
    variable_score = bool(variable_score)
    run_name = create_run_name(hint=hint, with_loophole=True, dataset_path=train_dataset_path, variable_score=variable_score)
    if kwargs.get('evaluation', None) is not None:
        reward_args = env_config.hacked_reward_args()
        reward_args['evaluation'] = kwargs.get('evaluation')
    else:
        reward_args = env_config.hacked_reward_args()
    reward_args['variable_score'] = variable_score
    if variable_score:
        reward_args['early_exit'] = False
    main_run_rl(
        run_name=run_name, 
        hint=hint, 
        dataset_path=train_dataset_path,
        model_id=model_id, 
        max_prompt_length=env_config.max_prompt_length,
        max_completion_length=env_config.max_completion_length,
        steps=steps, 
        seed=seed,
        reward_funcs_kwargs = {
            env_config.reward_func: reward_args,
        },
        **{k: v for k, v in kwargs.items() if k not in ['hint']},
    )


def run_ground_truth_intervention(
        # Base settings
        env: Environment = DEFAULT_ENV,
        model_id: str = DEFAULT_MODEL_ID,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,

        # Intervention settings
        intervention: Literal['penalty', 'screening'] = 'penalty',
        monitor_weight: float | None = None,

        # Monitor settings
        accuracy: float = 1.0,
        strict: bool = True,
        variable_score: bool = False,
        **kwargs,
    ):
    '''Ground truth monitor interventions with option to lower accuracy'''
    env_config = ENVIRONMENT_REGISTRY.get(env)
    hint = kwargs.get('hint', env_config.hacked_hint)
    train_dataset_path = _resolve_train_dataset_path(env_config, kwargs)
    accuracy = float(accuracy)
    strict = bool(strict)
    variable_score = bool(variable_score)
    reward_args = copy.deepcopy(env_config.hacked_reward_args())

    assert env_config.reward_func == "CorrectnessRewardFunction", "Currently only compatible with CorrectnessRewardFunction"

    # Add the monitor weight
    monitor_weight = float(monitor_weight) if monitor_weight else reward_args['reward_weights']['correct']
    reward_args['reward_weights']['monitor'] = monitor_weight # Adding this does not impact the screening function working
    reward_args['variable_score'] = variable_score
    if variable_score:
        reward_args['early_exit'] = False

    # Create a suffix
    suffix = (
        "_groundtruth" +
        ("_strict" if strict else "_loose") +
        (f"_acc{int(accuracy * 100)}" if accuracy < 1.0 else "") +
        (f"_m{str(monitor_weight)}" if intervention == "penalty" else "")
    )

    run_name = create_run_name(
        hint=hint,
        with_loophole=True,
        intervention=intervention,
        variable_score=variable_score,
        suffix=suffix,
        dataset_path=train_dataset_path,
    )

    if intervention == 'penalty':
        # Note: No "mode" option provided because the ground truth monitor only returns a 1 or 0 score
        intervention_args = {
            "reward_funcs_kwargs": {
                "GroundTruthMonitorPenalty": {
                    # Note: This will call the underlying evaluation reward function
                    "accuracy": accuracy,
                    "strict": strict,
                    **reward_args,
                },
            },
        }
    elif intervention == 'screening':
        intervention_args = {
            "reward_funcs_kwargs": {
                env_config.reward_func: reward_args,
            },
            "screening_funcs_kwargs": {
                "GroundTruthMonitor": {
                    "accuracy": accuracy,
                    "strict": strict,
                }
            },
        }
    
    main_run_rl(
        run_name = run_name,
        hint = hint,
        dataset_path = train_dataset_path,
        model_id = model_id,
        max_prompt_length=env_config.max_prompt_length,
        max_completion_length=env_config.max_completion_length,
        steps = steps,
        seed = seed,
        **intervention_args,
        **{k: v for k, v in kwargs.items() if k not in ['hint']},
    )


def run_probe_intervention(
        # Base arguments
        env: Environment = DEFAULT_ENV,
        model_id: str = DEFAULT_MODEL_ID,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,

        # Intervention settings
        intervention: Literal['penalty', 'screening'] = 'penalty',
        mode: RewardModes = "threshold", # Only applies to penalty intervention
        monitor_weight: float | None = None, # Only applies to penalty intervention

        # Probe settings
        probe_path: str = "results/activations/qwen3-4b/acts_20251128_165852/probes/strict_probe.lgprobe",
        probe_layers: list[int] | int = 12,
        probe_threshold: list[float] | float = 0.5,
        cache_activations_mode: Literal['worker', 'actor_rollout', 'interp_vllm'] = 'actor_rollout',
        variable_score: bool = False,
        **kwargs,
    ):
    '''Probe interventions'''
    env_config = ENVIRONMENT_REGISTRY.get(env)
    hint = kwargs.get('hint', env_config.hacked_hint)
    train_dataset_path = _resolve_train_dataset_path(env_config, kwargs)
    steps = int(steps)
    seed = int(seed)
    variable_score = bool(variable_score)
    probe_layers = [int(probe_layers)] if isinstance(probe_layers, (int, float)) else [int(l) for l in probe_layers]
    probe_threshold = [float(probe_threshold)] * len(probe_layers) if isinstance(probe_threshold, (int, float)) else [float(t) for t in probe_threshold]
    assert len(probe_threshold) == len(probe_layers), (
        f"Number of thresholds ({len(probe_threshold)}) must match number of layers ({len(probe_layers)})"
    )

    # Add the monitor weight
    reward_args = copy.deepcopy(env_config.hacked_reward_args())
    monitor_weight = float(monitor_weight) if monitor_weight else reward_args['reward_weights']['correct']
    reward_args['reward_weights']['monitor'] = monitor_weight # Adding this does not impact the screening function working
    reward_args['variable_score'] = variable_score
    if variable_score:
        reward_args['early_exit'] = False

    layers_str = "_".join(str(l) for l in probe_layers)
    thresh_str = "_".join(f"{t:.2f}" for t in probe_threshold)
    run_name = create_run_name(
        hint=hint,
        with_loophole=True,
        intervention=intervention,
        variable_score=variable_score,
        suffix=f"_probe_l{layers_str}_t{thresh_str}" + (f"_m{mode}_w{monitor_weight:.1f}" if intervention == "penalty" else ""),
        dataset_path=train_dataset_path,
    )

    probe_ext = probe_path.rsplit(".", 1)[-1]

    if intervention == 'penalty':
        intervention_args = {
            "reward_funcs_kwargs": {
                "ProbeOutputPenalty": {
                    "threshold": probe_threshold,
                    "layers": probe_layers,
                    "probe_ext": probe_ext,
                    "mode": mode,
                    **reward_args,
                }
            },
            "screening_funcs_kwargs": {},
        }
    elif intervention == 'screening':
        intervention_args = {
            "reward_funcs_kwargs": {
                env_config.reward_func: reward_args,
            },
            "screening_funcs_kwargs": {
                "ProbeOutputScreener": {
                    "threshold": probe_threshold,
                }
            }
        }

    # Note: Other parameters you might want to manually set: cache_activations_batch_size; cache_activations_num_workers
    probe_obj = load_probe(path=probe_path)
    intervention_args['cache_activations_mode'] = cache_activations_mode
    intervention_args['cache_activations_probe_path'] = probe_path
    intervention_args['cache_activations_layers'] = probe_layers
    intervention_args['cache_activations_position'] = 'response_all' if getattr(probe_obj, 'requires_sequence', False) else 'response_avg'
    
    if intervention_args['cache_activations_mode'] == 'worker':
        assert kwargs.get('cache_activations_num_workers', 0) > 0, "cache_activations_num_workers must be set when cache_activations_mode='worker'"

    main_run_rl(
        run_name=run_name,
        hint=hint,
        dataset_path=train_dataset_path,
        max_prompt_length=env_config.max_prompt_length,
        max_completion_length=env_config.max_completion_length,
        model_id=model_id,
        steps=steps,
        seed=seed,
        **intervention_args,
        **{k: v for k, v in kwargs.items() if k not in ['hint']},
    )


def run_llmjudge_intervention(
        # Base arguments
        env: Environment = DEFAULT_ENV,
        model_id: str = DEFAULT_MODEL_ID,
        steps: int = DEFAULT_STEPS,
        seed: int = DEFAULT_SEED,

        # Intervention settings
        intervention: Literal['penalty', 'screening'] = 'penalty',
        mode: RewardModes = "threshold", # Only applies to penalty intervention; only impacts if n_samples > 1
        monitor_weight: float | None = None, # Only applies to penalty intervention 

        # Judge settings
        judge_prompt_key: str = "rh_genericv2_010",
        judge_model_id: str = "moonshotai/kimi-k2.5",
        n_samples: int = 1, # Number of llm judge samples to take
        aggregation_type: Literal['mean', 'max'] = 'max', # Only used if n_samples > 1
        threshold: float = 0.5, # Only impacts if the output type is non-binary or n_samples > 1
        with_reasoning: bool = False,
        judge_max_new_tokens: int | None = None,
        judge_retry_parse_failures: bool = False,
        variable_score: bool = False,
        **kwargs,
    ):
    '''LLM judge interventions'''
    env_config = ENVIRONMENT_REGISTRY.get(env)
    hint = kwargs.get('hint', env_config.hacked_hint)
    train_dataset_path = _resolve_train_dataset_path(env_config, kwargs)
    steps = int(steps)
    seed = int(seed)
    n_samples = int(n_samples)
    threshold = float(threshold)
    variable_score = bool(variable_score)
    judge_max_new_tokens = None if judge_max_new_tokens is None else int(judge_max_new_tokens)
    judge_retry_parse_failures = _resolve_judge_retry_parse_failures(judge_model_id, with_reasoning, judge_retry_parse_failures)
    judge_model_shortname = judge_model_id.split("/")[1].split("-")[0]

    # Add the monitor weight
    reward_args = env_config.hacked_reward_args()
    monitor_weight = float(monitor_weight) if monitor_weight else reward_args['reward_weights']['correct']
    reward_args['reward_weights']['monitor'] = monitor_weight # Adding this does not impact the screening function working
    reward_args['variable_score'] = variable_score
    if variable_score:
        reward_args['early_exit'] = False

    judge_name = _append_reasoning_run_flag(f"_llmjudge_{judge_model_shortname}", bool(with_reasoning))
    run_name = create_run_name(
        hint=hint,
        with_loophole=True,
        intervention=intervention,
        variable_score=variable_score,
        suffix=judge_name + (f"_n{n_samples}_a{aggregation_type}" if n_samples > 1 else "") + (f"_t{threshold:.2f}_m{mode}_w{monitor_weight:.1f}" if intervention == "penalty" else ""),
        dataset_path=train_dataset_path,
    )

    if intervention == 'penalty':
        intervention_args = {
            "reward_funcs_kwargs": {
                "LLMJudgePenalty": {
                    "judge_model_id": judge_model_id,
                    "judge_prompt_key": judge_prompt_key,
                    "n_samples": n_samples,
                    "aggregation_type": aggregation_type,
                    "threshold": threshold,
                    "with_reasoning": bool(with_reasoning),
                    "retry_parse_failures": judge_retry_parse_failures,
                    "mode": mode,
                    **({} if judge_max_new_tokens is None else {"max_new_tokens": judge_max_new_tokens}),
                    **reward_args,
                },
            },
            "screening_funcs_kwargs": {}
        }
    elif intervention == 'screening':
        intervention_args = {
            "reward_funcs_kwargs": {
                env_config.reward_func: reward_args,
            },
            "screening_funcs_kwargs": {
                "LLMJudgeScreener": {
                    "judge_model_id": judge_model_id,
                    "judge_prompt_key": judge_prompt_key,
                    "n_samples": n_samples,
                    "aggregation_type": aggregation_type,
                    "threshold": threshold,
                    "with_reasoning": bool(with_reasoning),
                    "retry_parse_failures": judge_retry_parse_failures,
                    **({} if judge_max_new_tokens is None else {"max_new_tokens": judge_max_new_tokens}),
                }
            },
        }
    
    main_run_rl(
        run_name=run_name,
        hint=hint,
        dataset_path=train_dataset_path,
        max_prompt_length=env_config.max_prompt_length,
        max_completion_length=env_config.max_completion_length,
        model_id=model_id,
        steps=steps,
        seed=seed,
        **intervention_args,
        **{k: v for k, v in kwargs.items() if k not in ['hint']},
    )


def resume_training(
    run_name: str,
    checkpoint: int,
    steps: int, # New total cumulative steps to train for
    model_id: str = DEFAULT_MODEL_ID,
    dry_run: bool = False,
):
    """
    Resume RL training from a checkpoint.

    Args:
        run_name: Pattern to match run directory name (e.g., "biography_train_base")
        model_id: Model ID for the corresponding run name
        max_steps: New total steps target (must be greater than checkpoint step)
        checkpoint: Specific checkpoint step to resume from (default: latest)
        dry_run: If True, only print what would be done without running
    """
    # Find the run directory and checkpoint
    run_dir = f"{RESULTS_PATH}/runs/{model_id.split('/')[-1].lower()}/{run_name}"

    checkpoint_path = os.path.join(run_dir, "checkpoints", f"global_step_{checkpoint}")

    # Validate new max steps value
    if steps <= checkpoint:
        raise ValueError(f"New step count ({steps}) must be greater than checkpoint step ({checkpoint})")

    # Load original config
    config_dict = GRPOConfig.load(os.path.join(run_dir, "config.json")).model_dump()
    print(f"Loaded config from run")

    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint not found at {checkpoint_path}")
    n_gpus = torch.cuda.device_count()
    optim_path = os.path.join(checkpoint_path, f"actor/optim_world_size_{n_gpus}_rank_0.pt")
    if not os.path.exists(optim_path):
        raise ValueError(f"Optimizer not found at {optim_path} or optimizer state incompatible with current number of GPUs ({n_gpus})")
    print(f"Resuming from checkpoint: {checkpoint_path} (step {checkpoint})")

    # Optimizer state required for resuming
    if config_dict['save_only_model'] == True:
        raise ValueError("Cannot resume: original run had save_only_model=True (no optimizer state saved)")

    # Modify config for resume
    config_dict.update({
        "resume_from_checkpoint": checkpoint_path,
        "extra_metadata": {
            "resumed_from_step": checkpoint,
            "old_max_steps": config_dict["max_steps"],
            "new_max_steps": steps,
            "checkpoint_path": checkpoint_path,
        },
        "max_steps": steps,
        "save_only_model": False,
    })

    if dry_run:
        print("\n=== DRY RUN - Would resume with config: ===")
        print(f"  run_id: {config_dict['run_id']}")
        print(f"  model_id: {config_dict['model_id']}")
        print(f"  dataset_path: {config_dict['dataset_path']}")
        print(f"  max_steps: {steps} (was {checkpoint})")
        print(f"  resume_from_checkpoint: {config_dict['resume_from_checkpoint']}")
        return

    # Re-create the config object with the updated settings
    config = GRPOConfig(**config_dict)

    print(f"\n=== RESUMING TRAINING ===")
    print(f"  Run ID: {config.run_id}")
    print(f"  Checkpoint: step {checkpoint}")
    print(f"  New max steps: {steps}")
    print(f"  Output dir: {config.output_dir}")
    print(f"\n=== RESUMING TRAINING ===")

    # Run the training
    trainer = VerlGRPO(config)
    trainer.run()

    print(f"Training completed! Extended from step {checkpoint} to {steps}")


if __name__ == "__main__":
    utils.load_dotenv()
    print("IS_GPU_ENV", os.environ.get('IS_GPU_ENV', 'false'))
    fire.Fire({
        'resume': resume_training,
        'rl_baseline': run_rl_baseline,
        'no_intervention': run_no_intervention,
        'ground_truth': run_ground_truth_intervention,
        'probe': run_probe_intervention,
        'llmjudge': run_llmjudge_intervention,
    })
