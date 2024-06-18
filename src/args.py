import argparse
import os
import sys

ICL_EXPLORE_ROOT_PATH = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)
sys.path.insert(0, ICL_EXPLORE_ROOT_PATH)

# # 添加 bool 类型的命令行参数
# parser.add_argument('--bool_arg', type=bool, default=False, help='A boolean argument.')

# # 添加 list 类型的命令行参数
# # 使用 nargs 参数定义多个值的输入，"+" 表示一个或多个参数
# parser.add_argument('--list_arg', nargs='+', help='A list argument.')

# parse args

def get_args():
    
    parser = argparse.ArgumentParser(description='Example of using argparse in ICL explore.')
    
    # ============= Environment settings =================
    parser.add_argument('--gpus', nargs='+', help='The list of GPUs to use. If multiple GPUs are provided, the model will be used')

    # ============= Inference Model =============
    parser.add_argument('--model_name', type=str, default='pythia-12b', help='Inference model for ICL')
    parser.add_argument('--pretraining_steps', type=str, default="143000", help="Number of pretraining steps for ICL")

    # ============= Dataset =============
    parser.add_argument('--dataset_names', type=str, default='all', help="Dataset for ICL, use 'all' to specify all dataset")
    parser.add_argument('--dataset_seeds', type=str, default='all', help="Dataset seed for ICL, use 'all' to specify all split subset seed")
    parser.add_argument('--shot_nums', type=str, default="16", help="Number of few shot number for ICL")
    parser.add_argument('--sep_symbol', type=str, default="\n\n\n", help="Symbol of separating each examples")
    parser.add_argument('--random_seed', type=int, default=0)

    # ============= Evaluation =============
    parser.add_argument('--prompt_format', type=str, default="minimal", help="Format of the prompt (natural language prompt / minimal prompt)")
    parser.add_argument('--evaluation_types', type=str, default='golden', help="Type of evaluation")
    
    # ============= Debug =============
    parser.add_argument('--debug', action='store_true')
    
    args = parser.parse_args()
    model_name = args.model_name
    pretraining_steps = args.pretraining_steps
    dataset_names = args.dataset_names
    dataset_seeds = args.dataset_seeds
    shot_nums = args.shot_nums
    prompt_format = args.prompt_format
    evaluation_types = args.evaluation_types
    sep_symbol = args.sep_symbol
    random_seed = args.random_seed
    gpus = args.gpus
    debug = args.debug
    
    assert prompt_format in {
        "natural_language",
        "minimal"
    }
    
    icl_explore_kwargs = {
        "model_name": model_name,
        "pretraining_steps": pretraining_steps,
        "dataset_names": dataset_names,
        "dataset_seeds": dataset_seeds,
        "shot_nums": shot_nums,
        "prompt_format": prompt_format,
        "evaluation_types": evaluation_types,
        "sep_symbol": sep_symbol,
        "random_seed": random_seed,
        "gpus": gpus,
        "debug": debug,
    }

    return icl_explore_kwargs
