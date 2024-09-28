from deeplake.core.serialize import bytes_to_text
from deeplake.integrations.pytorch.common import convert_sample_to_data
from indra.pytorch.util import (
    transform_collate_batch,
)
from indra.pytorch.exceptions import (
    StopChildProcess,
)

from multiprocessing import current_process
from typing import List, Optional, Iterable
from PIL import Image
import dill as pickle
import os
import io
import zlib
import warnings

MP_STATUS_CHECK_INTERVAL = 10.0
r"""Interval (in seconds) to check status of processes to avoid hanging in
    multiprocessing data loading. This is mainly used in getting data from
    another process, in which case we need to periodically check whether the
    sender is alive to prevent hanging."""


def adjust_environment(num_workers):
    child_env = os.environ.copy()

    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py
    if num_workers >= 1 and "OMP_NUM_THREADS" not in os.environ:
        omp_num_threads = 1
        warnings.warn(
            f"Setting OMP_NUM_THREADS environment variable for each process "
            f"to be {omp_num_threads} in default, to avoid your system being "
            f"overloaded, please further tune the variable for optimal "
            f"performance in your application as needed."
        )
        child_env["OMP_NUM_THREADS"] = str(omp_num_threads)
        os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

        if num_workers >= 1 and "MKL_NUM_THREADS" not in os.environ:
            mkl_num_threads = 1
            warnings.warn(
                f"Setting MKL_NUM_THREADS environment variable for each process "
                f"to be {mkl_num_threads} in default, to avoid your system being "
                f"overloaded, please further tune the variable for optimal "
                f"performance in your application as needed."
            )

            child_env["MKL_NUM_THREADS"] = str(mkl_num_threads)
            os.environ["MKL_NUM_THREADS"] = str(mkl_num_threads)

        return child_env


def combine_compressed_bytes(
    batch,
    all_byte_tensors: Iterable,
):
    sb, eb, all_bts = 0, 0, []
    for sample in batch:
        for tensor in all_byte_tensors:
            if isinstance(sample[tensor], bytes):
                sample_bts = sample.pop(tensor)
                all_bts.append(sample_bts)
                eb += len(sample_bts)
                sample[tensor] = (sb, eb)
                sb = eb
            elif isinstance(sample[tensor], list):
                sb_eb_list = []
                for item in sample[tensor]:
                    sample_bts = item
                    all_bts.append(sample_bts)
                    eb += len(sample_bts)
                    sb_eb_list.append((sb, eb))
                    sb = eb
                sample[tensor] = sb_eb_list

    # combine all_bts into one bytearray
    all_bts = bytearray(b"".join(all_bts))
    return all_bts, batch


def bytes_to_batch(
    batch,
    all_bts,
    pil_compressed_tensors: List[str] = [],
    json_tensors: List[str] = [],
    list_tensors: List[str] = [],
):
    data_bytes = memoryview(all_bts)
    all_byte_tensors = set(pil_compressed_tensors + json_tensors + list_tensors)
    pil_compressed_tensors = set(pil_compressed_tensors)
    json_tensors = set(json_tensors)
    list_tensors = set(list_tensors)
    for sample in batch:
        for tensor in all_byte_tensors:
            if tensor in pil_compressed_tensors:
                decompress_fn = lambda x: Image.open(io.BytesIO(x))
            elif tensor in json_tensors:
                decompress_fn = lambda x: bytes_to_text(x, "json")
            elif tensor in list_tensors:
                decompress_fn = lambda x: bytes_to_text(x, "list")

            if isinstance(sample[tensor], tuple):
                sb, eb = sample[tensor]
                sample[tensor] = decompress_fn(data_bytes[sb:eb])
            elif isinstance(sample[tensor], list):
                sb_eb_list = sample[tensor]
                sample[tensor] = [
                    decompress_fn(data_bytes[sb:eb]) for sb, eb in sb_eb_list
                ]
            else:
                # will only happen for Image tensors that are tiled
                sample[tensor] = Image.fromarray(sample[tensor])
    return batch


def early_transform_collate(inp):
    (
        data_in_queue,
        data_out_queue,
        ignore_errors,
        transform_fn,
        collate_fn,
        upcast,
        info,
    ) = inp
    raw_tensor_set = (
        set(info.raw_tensors) - set(info.json_tensors) - set(info.list_tensors)
    )
    transform_fn = None if transform_fn is None else pickle.loads(transform_fn)
    collate_fn = None if collate_fn is None else pickle.loads(collate_fn)
    while 1:
        try:
            batch = data_in_queue.get()
            if isinstance(batch, StopIteration):
                data_out_queue.put(batch)
                break
            elif isinstance(batch, StopChildProcess):
                break
            else:
                if batch is None:
                    data_out_queue.put(None)
                    continue
                all_bts, batch = batch
                if all_bts is not None:
                    batch = bytes_to_batch(
                        batch,
                        all_bts,
                        info.pil_compressed_tensors,
                        info.json_tensors,
                        info.list_tensors,
                    )
                if info.htype_dict:
                    for sample in batch:
                        convert_sample_to_data(
                            sample,
                            info.htype_dict,
                            info.ndim_dict,
                            info.tensor_info_dict,
                        )
                out = transform_collate_batch(
                    batch, transform_fn, collate_fn, upcast, raw_tensor_set
                )
                data_out_queue.put(out)
        except Exception as e:
            data_out_queue.put(e)
            if ignore_errors:
                continue
            else:
                break
