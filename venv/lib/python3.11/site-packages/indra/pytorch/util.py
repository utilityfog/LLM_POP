from deeplake.util.iterable_ordered_dict import IterableOrderedDict
from indra.pytorch.exceptions import (
    TransformExceptionWrapper,
    CollateExceptionWrapper,
)
import numpy as np
from typing import Union, Callable, Optional, List
from multiprocessing import Queue
import math
import os
import dill


def create_folder(path: str = ""):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(path)
        except OSError:
            pass


def init_process_threads():
    # PyTorch relies on OpenMP, which by default parallelizes operations by
    # implicitly spawning as many threads as there are cores, and synchronizing
    # them with each other. This interacts poorly with Hogwild!-style subprocess
    # pools as if each child process spawns its own OpenMP threads there can
    # easily be thousands of threads that mostly wait in barriers. Calling
    # set_num_threads(1) in both the parent and children prevents this.
    # OpenMP can also lead to deadlocks if it gets initialized in the parent
    # process before the fork. Using the "spawn" context (i.e., fork + exec)
    # solved the issue in most cases but still left some deadlocks. See
    # https://github.com/pytorch/pytorch/issues/17199 for some more information
    # and discussion.
    try:
        import torch

        torch.set_num_threads(1)
    except ImportError:
        pass


def process_initializer(
    env, worker_init_fn: Optional[Callable] = None, id_queue: Optional[Queue] = None
):
    init_process_threads()
    os.environ = env
    if worker_init_fn is not None:
        assert isinstance(worker_init_fn, Callable)
        if id_queue is not None:
            worker_init_fn(id_queue.get())


def is_serializable(input):
    try:
        dill.loads(dill.dumps(input))
        return True
    except Exception:
        return False


def transform_collate_batch(batch, transform_fn, collate_fn, upcast, raw_tensor_set):
    if raw_tensor_set:
        for sample in batch:
            for k, v in sample.items():
                if k in raw_tensor_set and isinstance(v, np.ndarray):
                    sample[k] = v.tobytes()
    if upcast:
        it_order_dict_batch = [
            IterableOrderedDict((k, upcast_array(v)) for k, v in sample.items())
            for sample in batch
        ]
    else:
        it_order_dict_batch = [IterableOrderedDict(sample) for sample in batch]

    if transform_fn is not None:
        try:
            transformed = list(map(transform_fn, it_order_dict_batch))
        except Exception as ex:
            raise TransformExceptionWrapper(exception=ex)
    else:
        transformed = it_order_dict_batch
    if collate_fn is not None:
        try:
            collated = collate_fn(transformed)
        except Exception as ex:
            raise CollateExceptionWrapper(exception=ex)
    else:
        collated = transformed
    return collated


def upcast_array(arr: Union[np.ndarray, bytes]):
    if isinstance(arr, list):
        return [upcast_array(a) for a in arr]
    if isinstance(arr, np.ndarray):
        if arr.dtype == np.uint16:
            return arr.astype(np.int32)
        if arr.dtype == np.uint32:
            return arr.astype(np.int64)
        if arr.dtype == np.uint64:
            return arr.astype(np.int64)
    return arr



def get_indexes(
    dataset,
    rank: Optional[int] = None,
    num_replicas: Optional[int] = None,
    drop_last: Optional[bool] = None,
):
    import torch.distributed as dist

    if num_replicas is None:
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        num_replicas = dist.get_world_size()
    if rank is None:
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        rank = dist.get_rank()
    if rank >= num_replicas or rank < 0:
        raise ValueError(
            "Invalid rank {}, rank should be in the interval"
            " [0, {}]".format(rank, num_replicas - 1)
        )

    dataset_length = len(dataset)

    if drop_last:
        total_size = (dataset_length // num_replicas) * num_replicas
        per_process = total_size // num_replicas
    else:
        per_process = math.ceil(dataset_length / num_replicas)
        total_size = per_process * num_replicas

    start_index = rank * per_process
    end_index = min(start_index + per_process, total_size)

    end_index = min(end_index, dataset_length)

    return slice(start_index, end_index)
