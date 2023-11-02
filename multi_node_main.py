# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import argparse

import multiprocessing as mp

import pprint
import yaml

from src.utils.multi_node_distributed import init_distributed
from src.multi_node_train import main as app_main

# Set the value of PMIX_MCA_gds
os.environ['PMIX_MCA_gds'] = 'hash'

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--backend', type=str,
    help='name of the backend used for the multi node parallelization',
    default='mpi')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')


# def process_main(rank, fname, world_size, devices, args):
def process_main(args):
    # import os
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(devices[rank % world_size].split(':')[-1])
    #os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

    try:
            os.environ['CUDA_VISIBLE_DEVICES']=os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    except:
            print('Not OMPI_COMM_WORLD_LOCAL_RANK in this run!')

    fname = args.fname
    world_size, rank = init_distributed(args)
    import logging
    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # -- load script params
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    logger.info(f'Running... (rank: {rank}/{world_size})')
    # app_main(args=params)
    app_main(args=params, world_size=world_size, rank=rank, gpu=int(os.environ["LOCAL_RANK"]))
    # app_main(args=params, gpu=rank%world_size)
    # app_main(args=params, gpu=rank)


if __name__ == '__main__':
    args = parser.parse_args()
    # process_main(args.rank, args.fname, args.world_size, args.devices, args)
    process_main(args)
