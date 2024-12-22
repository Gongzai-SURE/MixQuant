import time
import torch
import torch.nn as nn
import argparse
import numpy as np
from tqdm import tqdm
from loguru import logger

from mixq.layerwise_quant import *
from mixq.utils.misc import *
from mixq.utils.datautils import *
from mixq.utils.modelutils import *
from evaluation.perplexity import *

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model', type=str,default = '/root/autodl-tmp/models/llama2-7b',  #qwen2-1.5b   llama2-7b
        help='hugging face model to load'
    )
    parser.add_argument(
        '--dataset', type=str,default='ptb',
        help='Where to extract calibration data from. choices = [wikitext2, ptb, c4, custom_path]'
    )
    parser.add_argument(
        '--dataset_dir', type=str, default='/root/autodl-tmp/datasets/',
        help='load datasets from local directory.'
    )
    parser.add_argument(
        '--nsamples', type=int, default = 2, #128 需要 128 * 32768 * 4096 * 2 * 4 / 1024 / 1024 / 1024 = 32GB
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--strategy', type=str, default='fisher', choices=['fisher', 'activation', 'threshold', 'random'], 
        help='The specific method for determining the layer quantization strategy, \
            "activation" indicates the change in activation value before and after quantization, \
            "fisher" indicates using fisher information to estimate the layer quantization sensitivity, \
            "random" indicates randomly assigning a quantization strategy.'
    )
    parser.add_argument(
        '--load_fisher', action='store_true',
        help='Whether to load a FisherInfo txt.'
    )
    parser.add_argument(
        '--wbits', type=list, default=[3,4,8],
        help='The number of bits to use for weight quantization; at least one lower bits.'
    )
    parser.add_argument(
        '--target_bit', type=float, default=3.8,
        help='The target bit of total quantization.'
    )
    parser.add_argument(
        '--perturb_percentage', type=float, default=0.75,
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
        '--seed', type=int, default=0,
        help='Seed for sampling the calibration data and the random layerwise quantization strategy.'
    )
    parser.add_argument(
        '--pentalty', type=list, default=[1/(1024*1024),10,10],
        help='The number of bits to use for weight quantization; at least one lower bits.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the round-to-nearest quantization.'
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
        '--load', type=str, default='',
        help='Load fake quantized checkpoint.'
    )
    parser.add_argument(
        '--logfile', type=str, default='',
        help='Logging file name'
    )
    parser.add_argument(
        '--benchmark', type=int, default=0,
        help='Number of tokens to use for benchmarking.'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--no_eval', action='store_true',
        help='Whether to evaluation quantized model.'
    )
    parser.add_argument(
        '--trust_remote_code', action='store_true',
    )
    parser.add_argument(
        '--quant', action='store_true',
        help='Whether to load a quantilized model.'
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

    # if getattr(model.config, 'max_position_embeddings', None):
    #     args.seqlen = model.config.max_position_embeddings
    # elif getattr(model.config, 'max_sequence_length', None):
    #     args.seqlen = model.config.max_sequence_length
    # else:
    #     args.seqlen = 2048

    args.seqlen = 1024
    
    if not args.load and args.wbits and not args.nearest and not args.original:
        logger.info(f'The model needs quantilization, start loading {args.dataset} as validation data ...')
        dataloader = get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=args.seqlen, train=True, local_dir=args.dataset_dir
        )
        logger.info('Validation Data loaded.')

        t_start = time.time()
        quantizers, layer_score = layerwise_quantize(model, dataloader, args)

        t = round((time.time() - t_start),1)
        logger.info(f"Running Time : {t} s")
    
    # benchmark
    if args.benchmark:
        dataloader = get_loaders(
            args.dataset, nsamples=1, seed=args.seed, model=args.model, seqlen=args.seqlen, train=True
        )
        gpus = [torch.device('cuda:%d' % i) for i in range(torch.cuda.device_count())]
        if len(gpus) > 1:
            model_multigpu(model, gpus, args)
        else:
            model = model.to(args.device)
        
        if isinstance(dataloader,list):
            input_ids = dataloader[0][0][:,:args.benchmark]
        else:
            input_ids = dataloader.input_ids[:, :args.benchmark]
        benchmark(model, input_ids, args)
        exit()

    # evaluation
    t1 = time.time()
    ppl_scores = []
    if not args.no_eval:
        ppl_tasks = ['wikitext2','ptb']
        for dataset in ppl_tasks:
            testloader = get_loaders(
                dataset, seed=args.seed, model=args.model, seqlen=args.seqlen, train=False
            )
            logger.info(dataset)
            ppl_score = eval_ppl(model, testloader, args.device, args.seqlen, args)
            ppl_scores.append((dataset,ppl_score))
    t2 = time.time() - t1

    # saving model
    if args.save_path:
        save_model(model, quantizers, args.save_path, args.packing, args.fake)

    