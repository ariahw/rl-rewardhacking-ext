import fire
import os
import logging
import traceback

from src import DEFAULT_MODEL_ID, RESULTS_PATH, wandb_utils, generate, evaluate, utils
from src.evaluate import EvaluationParameters, EVALUATION_REGISTRY
from src.generate import VLLMGenerator, SamplingParams, LLMGenerator
from src.envs import ENVIRONMENT_REGISTRY, Environment

logger = utils.get_logger("evaluate")

"""
EVALUATION
"""


def run_eval_save_results(
        eval_params: EvaluationParameters,
        llm_gen: LLMGenerator | None = None, 
        output_dir: str = "results", 
        overwrite: bool = False, 
        suffix: str = ""
    ):

    if llm_gen is None:
        logger.info(f"Creating LLM generator with engine {eval_params.generation_engine} and model {eval_params.model_id}")
        llm_gen = generate.create_llm_generator(eval_params.generation_engine, model_name = eval_params.model_id, lora_adapter_path = eval_params.lora_adapter_path)

    # Load dataset
    dataset = utils.read_jsonl_all(eval_params.dataset_path)

    dataset_name = dataset[0]['dataset']
    fname = f"{output_dir}/{dataset_name}/eval_{eval_params.dataset_path.split('/')[-1].removesuffix('.jsonl')}_{eval_params.sampling_params.max_new_tokens}{suffix}.json"
    if os.path.exists(fname) and (not overwrite):
        raise ValueError(f"Evaluation results already exist at {fname}")
    logger.info(f"Running eval for {dataset_name}, saving output to {fname}")

    results = evaluate.run_eval(
        llm_gen, 
        eval_params, 
        dataset
    )

    # Create results dictionary
    results = {
        'eval_params': eval_params.model_dump(),
        'results': results
    }

    try:
        utils.save_json(fname, results)
    except:
        utils.save_pickle(fname.replace('.json', '.pkl'), results)

def main(
        evaluation_name: str = "code",
        dataset_path: str | None = None,
        lora_adapter_path: str | None = None,
        model_id: str = DEFAULT_MODEL_ID, 
        max_new_tokens: int = 1536,
        max_prompt_length: int = 1536,
        n_samples: int = 10,
        temperature: float = 0.7,
        repetition_penalty: float = 1.0,
        with_reasoning: bool = False,
        suffix: str = "", # Add an extra suffix to the output filename
        overwrite: bool = False
    ):
    print(f"Running eval for {model_id} with lora adapter {lora_adapter_path} with dataset {dataset_path}")

    logger = utils.create_logger(
        log_file = f"{RESULTS_PATH}/evals/eval.log",
        level = logging.INFO,
        print_to_console = False
    )
    
    llm_gen = VLLMGenerator(
        model_id, 
        lora_adapter_path = lora_adapter_path,
        max_model_len = max_new_tokens + max_prompt_length, # Known max prompt length
        gpu_memory_utilization = 0.9,
        max_num_seqs = 512
    )
    logger.info(f"Created LLM generator with engine {llm_gen.name} and model {model_id}")

    sampling_params = SamplingParams(
        temperature = float(temperature),
        top_p = 0.95,
        max_new_tokens = int(max_new_tokens),
        n = int(n_samples),
        repetition_penalty = float(repetition_penalty),
        with_reasoning = with_reasoning,
    )

    if lora_adapter_path is not None:
        output_dir = lora_adapter_path.replace(f"{RESULTS_PATH}/runs", f"{RESULTS_PATH}/evals")
    else:
        output_dir = f"{RESULTS_PATH}/evals/{model_id.split('/')[-1].lower()}"

    try:
        # Run + save eval results
        run_eval_save_results(
            eval_params = EvaluationParameters(
                model_id = model_id,
                lora_adapter_path = lora_adapter_path,
                dataset_path = dataset_path,
                sampling_params = sampling_params,
                evaluation_name = evaluation_name, # This needs to be filled in from the dataset itself
                debug = False,
                save_outputs = False
            ),
            llm_gen = llm_gen,
            output_dir = output_dir, 
            overwrite = overwrite, 
            suffix = suffix
        )
    except Exception as e:
        print(f"Error running eval for {dataset_path}: {e}")
        print(traceback.format_exc())
        raise
    finally:
        if llm_gen is not None:
            llm_gen.cleanup()


def default_run(
    env: Environment = "leetcode_rh",
    run_name: str | None = None,
    checkpoint: int | None = None,
    model_id: str = DEFAULT_MODEL_ID,
    overwrite: bool = False,
    **kwargs
):
    if (run_name is None) != (checkpoint is None):
        raise ValueError(
            f"Pass both --run_name and --checkpoint to evaluate a trained run, or neither "
            f"to evaluate the base model (got run_name={run_name!r}, checkpoint={checkpoint!r})"
        )
    env_config = ENVIRONMENT_REGISTRY.get(env)
    main(
        evaluation_name = kwargs.get('evaluation_name', env_config.hacked_evaluation),
        lora_adapter_path = f"results/runs/{model_id.split('/')[-1].lower()}/{run_name}/checkpoints/global_step_{checkpoint}" if (run_name is not None) and (checkpoint is not None) else None,
        model_id = model_id,
        max_new_tokens = kwargs.get('max_new_tokens', env_config.max_completion_length),
        max_prompt_length = kwargs.get('max_prompt_length', env_config.max_prompt_length),
        n_samples = kwargs.get('n_samples', env_config.eval_n_samples),
        temperature = kwargs.get('temperature', env_config.eval_temperature),
        repetition_penalty = kwargs.get('repetition_penalty', env_config.eval_repetition_penalty),
        dataset_path = kwargs.get('dataset_path', env_config.test_dataset_path),
        overwrite = overwrite,
        suffix = kwargs.get('suffix', "")
    )

def run_reeval(
        results_path: str,
        suffix: str = "_reparsed",
    ):
    """Re-run the evaluation on the existing generations"""

    results = utils.read_json(results_path)
    eval_params, results = results['eval_params'], results['results']
    logger.info(f"Evaluating {len(results)} responses")

    eval_params = EvaluationParameters(**eval_params)

    evaluation_instance = EVALUATION_REGISTRY[eval_params.evaluation_name](
        llm_gen = None,
        config = eval_params,
        early_exit = False,
    )

    results = evaluation_instance.batch_evaluate(
        examples = results,
        outputs = [x['response'] for x in results]
    )

    results = {
        'eval_params': eval_params.model_dump(),
        'results': results
    }

    try:
        utils.save_json(results_path.replace('.json', f'{suffix}.json'), results)
    except:
        print(f"Error saving results: {traceback.format_exc()}")
        utils.save_pickle(results_path.replace('.json', f'{suffix}.pkl'), results)



if __name__ == "__main__":
    utils.load_dotenv()
    fire.Fire({
        'run': main,
        'default': default_run,
        'pull_logs': wandb_utils.pull_run_stats, # Note: Full wandb run name includes the model name ie qwen3-4b/<RUN_NAME>
        'reeval': run_reeval,
    })
