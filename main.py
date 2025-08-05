import time
import torch
import torch.nn as nn
import argparse
import numpy as np
from loguru import logger

from mixq.layerwise_quant import *
from mixq.utils.misc import *
from mixq.utils.datautils import *
from mixq.utils.modelutils import *
from evaluation.perplexity import *
from evaluation.eval_wikitext import *
from lm_eval import evaluator

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str,default = '/root/autodl-tmp/models/qwen2.5-14b',  #qwen2-1.5b mistral-7b  llama2-7b llama2-13b qwen2.5-7b llama-7b llama-13b
        help='hugging face model to load'
    )
    parser.add_argument(
        '--dataset', type=str,default='wikitext2',
        help='Where to extract calibration data from. choices = [wikitext2, ptb, c4, custom_path]'
    )
    parser.add_argument(
        '--dataset_dir', type=str, default='/root/autodl-tmp/datasets/',  # '/root/autodl-tmp/datasets/',
        help='load datasets from local directory.'
    )  
    parser.add_argument(
        '--nsamples', type=int, default = 8,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--strategy', type=str, choices=['fisher', 'activation', 'threshold', 'random', "ppl"], 
        help='The specific method for determining the layer quantization strategy, \
            "activation" indicates the change in activation value before and after quantization, \
            "fisher" indicates using fisher information to estimate the layer quantization sensitivity, \
            "random" indicates randomly assigning a quantization strategy.'
    )
    parser.add_argument(
        '--allocate_strategy', type=str , choices=['greedy', 'genetic', 'rl', 'annealing','bayesian','random'], 
        help='Bit width allocation strategy based on Fisher'
    )
    parser.add_argument(
        '--load_fisher', action='store_true',
        help='Whether to load a FisherInfo txt.'
    )
    parser.add_argument(
        '--load_ppl', action='store_true',
        help='Whether to load a ppl txt.'
    )
    parser.add_argument(
        '--allocation', type=str, default=None,
        help='The bit allocation for each layer.'
    )
    parser.add_argument(
        '--wbits', type=str, default="3,4,5",
        help='The number of bits to use for weight quantization; at least one lower bits.'
    )
    parser.add_argument(
        '--target_bit', type=float, default=4,
        help='The target bit of total quantization.'
    )
    parser.add_argument(
        '--quant_method', type=str, default='', choices=['gptq', 'awq', 'owq', 'nearest','omni'], 
        help='Choosing an appropriate quantification method.\
            Different methods have different processes in handling quantization parameters.'
    ) 
    parser.add_argument(
        '--alpha', type=float, default=0.75,
        help='Hyperparameters used to calculate the objective function'
    )
    parser.add_argument(
        '--top_r', type=float, default=0.1,
        help='The ratio of the top layers to be considered for ppl allocation.'
    )
    parser.add_argument(
        '--test_bit', type=str, default="4",
        help='The bits to calculate fisher info.'
    )
    parser.add_argument(
        '--perturb_percentage', type=float, default=0.1,
        help='The amplitude of the added perturbation is -1 to 1 times the original model weight.'
    )
    parser.add_argument(
        '--original', action='store_true',
        help='whether to use original model.'
    )
    parser.add_argument(
        '--device', type=int, default=None, 
        help='GPUs to use'
    )
    parser.add_argument(
        '--tuning', type=str, default='mse', choices=['mse', 'minmax'],
        help='Method for quantization parameter tuning.'
    )
    parser.add_argument(
        '--dtype', type=str, default=None,
        help='Data type of model. Use bfloat16 for falcon model family or llama 65B model'
    )
    parser.add_argument(
        '--layers', nargs='+', type=str, default=None,
        help='Layers to apply MixQuant.'
    )
    parser.add_argument(
        '--sameLayerReset', action='store_true',
        help='Whether to reset bits between same name layer.'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Seed for sampling the calibration data and the random layerwise quantization strategy.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=128,
        help='Groupsize for fine-grained quantization; default uses full row.'
    )
    parser.add_argument(
        '--save_path', type=str, default='',
        help='Save quantized checkpoint under this name.'
    )
    parser.add_argument(
        '--fake', action='store_true',
        help='Whether to save fake quantized model.'
    )
    parser.add_argument(
        '--load', type=str, default='',
        help='Load fake quantized checkpoint.'
    )
    parser.add_argument(
        '--logfile', type=str, default='',
        help='Logging file name'
    )
    parser.add_argument(
        '--evaluate_ppl', action='store_true',
        help='Whether to evaluation model, adding parameters not to test.'
    )
    parser.add_argument(
        '--reasoning', action='store_true',
        help='Whether to evaluation model using reasoning tasks.'
    )
    parser.add_argument(
        '--trust_remote_code', action='store_true',
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--quant', action='store_true',
        help='Whether to load a quantilized model.'
    )
    parser.add_argument(
        '--version', type=str, default='gemm',
        help='Awq quantized network architecture.'
    )

    args = parser.parse_args()
    meta = processing_arguments(args)
    args.meta = meta
    seed_all(args.seed)
    
    logger.info('loading model ...')
    if args.load:
        model = load_model(args.model, args.load, args.faster)
    else:
        model = get_hfmodel(args.model, args.dtype, args.quant, trust_remote_code=args.trust_remote_code)
    logger.info('Model loaded.')

    if getattr(model.config, 'max_position_embeddings', None):
        args.seqlen = model.config.max_position_embeddings
    elif getattr(model.config, 'max_sequence_length', None):
        args.seqlen = model.config.max_sequence_length
    else:
        args.seqlen = 2048

    args.seqlen = 2048
        
    # implementation of quantization
    if not args.load and args.wbits and not args.original:
        logger.info(f'The model needs quantilization, start loading {args.dataset} as validation data ...')
        dataloader = get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, train=True, local_dir=args.dataset_dir
        )
        logger.info('Validation Data loaded.')

        t_start = time.time()
        quantizers = layerwise_quantize(model, dataloader, args)
        if args.save_path:
            save_model(model, args.save_path, args.model, args.fake)

        t = round((time.time() - t_start),1)
        logger.info(f"Running Time : {t} s")
    
    torch.cuda.empty_cache()

    # evaluation perplexity
    if args.evaluate_ppl:
        ppl_tasks = ['wikitext2','ptb','c4'] 
        for dataset in ppl_tasks:
            testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=args.seqlen, train=False, local_dir=args.dataset_dir
            )  
            t1 = time.time()
            logger.info(f'Evaluating {dataset} sequence {args.seqlen} model_path {args.model}')
            ppl_score = eval_ppl(model, testloader, args.device, args)
            t2 = time.time() - t1
            logger.info(f'{dataset} perplexity: {ppl_score}')
            logger.info(f"Evaluation time: {t2:.2f} seconds")
        model.to('cpu')

    # evaluation reasoning
    if args.reasoning:
        results = {}
        from evaluation.lm_model import HFModelWrapper
        reason_tasks = ["boolq","arc_easy","arc_challenge","hellaswag","winogrande"]
        # reason_tasks = ["arc_easy","arc_challenge"]#,
        # reason_tasks = ["hellaswag","winogrande"]
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        lm= HFModelWrapper(model, tokenizer)
        t_results = evaluator.simple_evaluate(
            lm,
            tasks = reason_tasks,
            num_fewshot=0,
            limit=None,
        )
        results.update(t_results["results"])
        logger.info(results)
        try:
            acc_values = [task_result["acc,none"] for task_result in results.values() if "acc,none" in task_result]
            avg_acc = sum(acc_values) / len(acc_values)
            logger.info(f"avg acc: {avg_acc:.4f}")
        except Exception as e:
            logger.error(f"Error calculating average accuracy: {e}")
        del lm
        # from evaluation.reason import *
        # reason_tasks = ['BoolQ','ARC-E','ARC-C','HellaSwag','WinoGrande']
        # tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        # scores = reason_test(model, tokenizer)
        # # 记录平均值
        # logger.info(f"average score: {np.mean(scores)}")

    

    