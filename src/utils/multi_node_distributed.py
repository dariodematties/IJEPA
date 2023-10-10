
import socket
import torch
import os
import torch.distributed as dist

from mpi4py import MPI

def init_distributed(args):
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()

    if args.backend is None:
        rank, n_ranks = 0, 1
    elif args.backend == 'mpi':
        rank, n_ranks = init_workers_mpi()
    elif args.backend == 'nccl':
        rank, n_ranks = init_workers_nccl_file()
    elif args.backend == 'nccl_slurm':
        rank, n_ranks = init_workers_nccl_slurm()
    elif args.backend == 'gloo':
        rank, n_ranks = init_workers_gloo_file()
    else:
        print('Not using distributed mode')
        return


    args.rank = rank
    args.world_size = n_ranks
    if "SLURM_LOCALID" in os.environ and torch.cuda.device_count() == 1:
        # this assumes we won't use single gpu with distributed
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
        args.gpu = int(os.environ["SLURM_LOCALID"])
    else:
        print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
        # args.gpu = 0
        # args.gpu = args.rank % torch.cuda.device_count()
        args.gpu = int(os.environ["LOCAL_RANK"])
        # torch.cuda.set_device(0)
        torch.cuda.set_device(args.gpu)

    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['RANK'] = str(args.rank)
    #os.environ['LOCAL_RANK'] = str(args.gpu)
    os.environ['WORLD_SIZE'] = str(args.world_size)
    cmd="hostname"
    returned_value=os.system(cmd)
    print('{} | distributed init node {}, (rank {}): gpu {}, world size {}'.format(
         returned_value, args.backend, args.rank, args.gpu, args.world_size), flush=True)
    # print('distributed init (rank {}): gpu {}, local rank {}, world size {}'.format(
        # args.rank, args.gpu, local_rank, args.world_size), flush=True)
    # print('{} | distributed init (rank {}): gpu {}, local rank {}, world size {}'.format(
        # args.backend, args.rank, args.gpu, local_rank, args.world_size), flush=True)
    # print('{} | distributed init (rank {}): {}, gpu {}, world size {}'.format(
        # args.backend, args.rank, args.dist_url, args.gpu, args.world_size), flush=True)
    # torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
    return n_ranks, rank

def init_workers_mpi():
    """Initialize workers with MPI backend"""
    assert('OMPI_COMM_WORLD_RANK' in os.environ)
    print('USING MPIRUN!!!!!!!!!!!!!!!')
    # dist.init_process_group(backend='mpi')
    # rank = dist.get_rank()
    # n_ranks = dist.get_world_size()
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    local_rank_key_options = [
            'OMPI_COMM_WORLD_LOCAL_RANK',
            'MV2_COMM_WORLD_LOCAL_RANK',
            'MPI_LOCALRANKID',
            'PMI_LOCAL_RANK',
            ]

    # testable default value:
    local_rank = None
    for key in local_rank_key_options:
        if key in os.environ:
            local_rank = os.environ[key]
            print('Determined local rank through environment variable {}' .format(key))
            print('local_rank ', local_rank)
            break
    if local_rank is None:
        raise Exception("DDP failed to initialize due to local rank issue")

    # Pytorch will look for these:
    os.environ["LOCAL_RANK"] = local_rank
    # # os.environ["LOCAL_RANK"] = os.environ['OMPI_COMM_WORLD_LOCAL_RANK']
    # os.environ["RANK"] = str(rank)
    # os.environ["WORLD_SIZE"] = str(size)
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    if rank == 0:
        master_addr = socket.gethostname()
    else:
        master_addr = None

    master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(2345)

    dist.init_process_group(backend='nccl',
                            init_method='env://',
                            rank=rank, world_size=size)
    return rank, size

def init_workers_nccl_file():
    """Initialize workers with NCCL backend and sync file"""
    rank = int(os.environ['SLURM_PROCID'])
    n_ranks = int(os.environ['SLURM_NTASKS'])
    sync_file = _get_sync_file()
    print('Setting up with sync file', sync_file)
    dist.init_process_group(backend='nccl', world_size=n_ranks, rank=rank,
                            init_method=sync_file)
    return rank, n_ranks

def init_workers_nccl_slurm():
    """Initialize workers with NCCL backend and SLURM env variables.

    You must set the master address and port in your slurm script:
        export MASTER_ADDR=$(hostname)
        export MASTER_PORT=29500
        srun ...
    """
    rank = int(os.environ['SLURM_PROCID'])
    n_ranks = int(os.environ['SLURM_NTASKS'])
    dist.init_process_group(backend='nccl', world_size=n_ranks, rank=rank)
    return rank, n_ranks

def init_workers_gloo_file():
    """Initialize workers with GLOO backend and sync file"""
    rank = int(os.environ['SLURM_PROCID'])
    n_ranks = int(os.environ['SLURM_NTASKS'])
    sync_file = _get_sync_file()
    dist.init_process_group(backend='gloo', world_size=n_ranks, rank=rank,
                            init_method=sync_file)
    return rank, n_ranks

def _get_sync_file():
    """Logic for naming sync file using slurm env variables"""
    sync_file_dir = '%s/pytorch-sync-files' % os.environ['SCRATCH']
    os.makedirs(sync_file_dir, exist_ok=True)
    sync_file = 'file://%s/pytorch_sync.%s.%s' % (
        sync_file_dir, os.environ['SLURM_JOB_ID'], os.environ['SLURM_STEP_ID'])
    return sync_file

class AllReduce(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            x = x.contiguous() / dist.get_world_size()
            dist.all_reduce(x)
        return x

    @staticmethod
    def backward(ctx, grads):
        return grads



def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

