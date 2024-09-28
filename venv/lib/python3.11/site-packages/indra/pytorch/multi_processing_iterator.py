from itertools import repeat
from typing import Callable, List, Optional
from indra.pytorch.buffered_loader import BufferedLoader
from indra.pytorch.common import collate_fn as default_collate
from indra.pytorch.multiprocess_utils import (
    MP_STATUS_CHECK_INTERVAL,
    adjust_environment,
    combine_compressed_bytes,
    early_transform_collate,
)
from indra.pytorch.util import (
    init_process_threads,
    process_initializer,
    create_folder,
)
from indra.pytorch.tensorinfo import TensorsInfo, LoaderMetaInfo
from indra.pytorch.exceptions import (
    TransformExceptionWrapper,
    CollateExceptionWrapper,
    StopChildProcess,
)
from indra.pytorch.log import get_logger

import warnings
import os
import dill as pickle
import traceback
from uuid import uuid4


class MultiProcessingIterator:
    def __init__(
        self,
        dataloader,
        dataset,
        info: TensorsInfo,
        loader_meta: LoaderMetaInfo,
        transform_fn: Optional[Callable] = None,
        collate_fn: Optional[Callable] = default_collate,
        worker_init_fn: Optional[Callable] = None,
        num_workers: int = 0,
        persistent_workers: bool = False,
    ):
        """
        Returns an iterator for single process iteration

        Args:
            info (TensorsInfo)
            loader_meta (LoaderMetaInfo)
            prefetch_factor (int) Number of samples loaded in advance by workers. Defaults to 10
            transform_fn (Callable, optional) Callable object which is needed to apply on each sample on batch. Defaults to None.
            collate_fn (callable, optional): merges a list of samples to form a
                mini-batch of Tensor(s).  Used when using batched loading from a
                map-style dataset.
            worker_init_fn (Callable, optional) function to initialise the child processes. Defaults to None.
            num_workers (int, optional): how many subprocesses to use for data
                loading. ``0`` means that the data will be loaded in the main process.
                (default: ``0``)
            persistent_workers (bool): If ``True``, the data loader will not shutdown the worker processes after a dataset has been consumed once. Defaults to ``False``.
        """

        assert num_workers > 0
        assert loader_meta.prefetch_factor > 0

        self.dataloader = dataloader
        self.dataset = dataset
        self.info = info
        self.loader_meta = loader_meta

        self.prefetch_factor = self.loader_meta.prefetch_factor
        self.upcast = self.loader_meta.upcast
        self.transform_fn = transform_fn
        self.collate_fn = collate_fn
        self.worker_init_fn = worker_init_fn or None
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers or False
        self.num_prefetch_tasks = self.prefetch_factor * self.num_workers
        self.workers_initialized = False
        self.ignore_errors = self.loader_meta.ignore_errors

        self.length = len(dataloader)
        self.current_pos = 0
        self.iter_pos = 0
        self.skipped = 0
        self.processed = 0
        self.pool = None
        self.manager = None

        self.pid = os.getpid()
        self.logger = get_logger(self.loader_meta.verbose)

        init_process_threads()
        if loader_meta.context is None:
            import multiprocessing

            loader_meta.context = multiprocessing

    def __iter__(self):
        if self.current_pos != 0:
            self.dataloader.reset()

        if self.persistent_workers and self.pool is not None:
            self.clear_queues()

        self.reset_positions()
        self.iter_dl = iter(self.dataloader)
        if self.pool is not None:
            if not self.persistent_workers:
                self.send_stop_signals_to_subprocesses()
                self.close()
                self.start_processes()
            self.run_workers()
            self.fill_prefetch_jobs()

        return self

    def __len__(self):
        return self.length

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"MPIterator length {self.length} num_workers {self.num_workers}"

    def __del__(self):
        self.free_resources()

    def __next__(self):
        if self.pool is None:
            self.start_processes()
            self.run_workers()
            self.fill_prefetch_jobs()
        elif (
            self.pool is not None
            and self.persistent_workers
            and not self.workers_initialized
        ):
            self.run_workers()
            self.fill_prefetch_jobs()
        return self.get_data()

    def reset_positions(self):
        self.current_pos = 0
        self.iter_pos = 0
        self.skipped = 0
        self.processed = 0

    def clear_queues(self):
        self.logger.info(f"clear multiprocessing queues for process {self.pid}")
        for item in self.data_in_queues:
            while not item.empty():
                item.get_nowait()

        for item in self.data_out_queues:
            while not item.empty():
                item.get_nowait()

    def fetch_next_job(self):
        while True:
            try:
                wid = self.current_pos % self.num_workers
                batch = next(self.iter_dl)
                if self.info.pil_compressed_tensors:
                    all_bts, batch = combine_compressed_bytes(
                        batch,
                        all_byte_tensors=set(
                            self.info.pil_compressed_tensors
                            + self.info.json_tensors
                            + self.info.list_tensors
                        ),
                    )
                else:
                    all_bts = None
                batch = (all_bts, batch)
                self.data_in_queues[wid].put(batch)
                self.current_pos += 1
                return True
            except StopIteration:
                for j in range(self.num_workers):
                    self.data_in_queues[j].put(StopIteration())
                return False
            except Exception as ex:
                self.logger.info(
                    f"Exception during data fetching, {str(ex)} process {self.pid}"
                )
                continue

    def fill_prefetch_jobs(self):
        prefetched = 0
        while prefetched <= self.num_prefetch_tasks:
            if not self.fetch_next_job():
                break
            prefetched += 1

    def start_processes(self):
        if self.pool is not None:
            return

        child_env = adjust_environment(self.num_workers)
        id_queue = self.loader_meta.context.Queue(maxsize=self.num_workers)
        for i in range(self.num_workers):
            id_queue.put(i)

        self.pool = self.loader_meta.context.Pool(
            processes=self.num_workers,
            initializer=process_initializer,
            initargs=(child_env, self.worker_init_fn, id_queue),
        )

        if self.manager is None:
            self.manager = self.loader_meta.context.Manager()

        self.data_in_queues = [self.manager.Queue() for _ in range(self.num_workers)]
        self.data_out_queues = [self.manager.Queue() for _ in range(self.num_workers)]

    def run_workers(self):
        transform_fn = (
            None if self.transform_fn is None else pickle.dumps(self.transform_fn)
        )
        collate_fn = None if self.collate_fn is None else pickle.dumps(self.collate_fn)

        inp = list(
            zip(
                self.data_in_queues,
                self.data_out_queues,
                repeat(self.ignore_errors),
                repeat(transform_fn),
                repeat(collate_fn),
                repeat(self.upcast),
                repeat(self.info),
            )
        )

        self.workers_initialized = True
        self.pool.map_async(early_transform_collate, inp)

    def get_workers_schedule(self) -> List:
        # Calculate the number of elements per worker
        elements_per_worker = len(self.dataset) // self.num_workers

        # Initialize a list to store the ranges for each worker
        ranges = []

        # Generate the ranges for each worker
        for i in range(self.num_workers):
            start = i * elements_per_worker
            end = start + elements_per_worker

            # For the last worker, add any remaining elements
            if i == self.num_workers - 1:
                end = len(self.dataset)

            ranges.append(slice(start, end, 1))

        return ranges

    def restart_dataloader(self):
        self.processed -= 1
        self.logger.info(
            f"Restarting dataloader from the batch {self.processed + self.skipped}, due to its worker being stuck"
        )
        self.send_stop_signals_to_subprocesses()
        self.clear_queues()
        self.dataloader.start_from_batch(self.processed + self.skipped)
        self.run_workers()
        self.fill_prefetch_jobs()

    def handle_stop_iteration(self, worker_id):
        # get StopIteration from other workers too, to empty the queues
        for j in range(self.num_workers):
            if j != worker_id:
                self.data_out_queues[j].get()

        if not self.persistent_workers:
            self.send_stop_signals_to_subprocesses()
            self.close()
        self.workers_initialized = False

    def get_data(self):
        out = None

        while True:
            self.processed += 1
            wid = self.iter_pos % self.num_workers
            try:
                out = self.data_out_queues[wid].get(timeout=MP_STATUS_CHECK_INTERVAL)
            except Exception:
                self.restart_dataloader()
                continue
            if isinstance(out, StopIteration):
                self.handle_stop_iteration(worker_id=wid)
                raise StopIteration
            elif isinstance(out, StopChildProcess):
                warnings.warn(
                    "Invalid state was reached, please contact Activeloop for further assistance."
                )
                self.fetch_next_job()
                self.iter_pos += 1
                raise StopIteration
            elif isinstance(out, Exception):
                self.handle_exception(out)
                if self.ignore_errors:
                    self.fetch_next_job()
                    self.iter_pos += 1
                    continue
                else:
                    raise out
            if self.current_pos < self.length:
                self.fetch_next_job()
            elif self.current_pos == self.length:
                try:
                    next(self.iter_dl)
                except StopIteration:
                    # send StopIteration (stop signal) to all workers
                    for j in range(self.num_workers):
                        self.data_in_queues[j].put(StopIteration())
            self.iter_pos += 1
            return out

    def handle_exception(self, ex):
        self.processed -= 1
        if isinstance(
            ex,
            (
                TransformExceptionWrapper,
                CollateExceptionWrapper,
            ),
        ):
            ex.processed = self.processed
            ex.skipped = self.skipped
            if self.ignore_errors:
                print(
                    f"An exception happened during data handling exception: {ex} processed batches {ex.processed} skipped batched {ex.skipped}"
                )
            else:
                traceback.print_tb(ex.exception.__traceback__)
        else:
            if self.ignore_errors:
                print(
                    f"An exception happened during data handling exception: {ex} processed batches {self.processed}"
                )
            else:
                traceback.print_tb(ex)
        self.skipped += 1

    def send_stop_signals_to_subprocesses(self):
        if self.pool is not None:
            if self.manager is None:
                return
            for idx in range(self.num_workers):
                self.data_in_queues[idx].put(StopChildProcess())

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
            self.pool = None
            self.workers_initialized = False

    def free_resources(self):
        self.send_stop_signals_to_subprocesses()
        self.close()
        if self.manager is not None:
            self.manager.shutdown()
            self.manager = None

    @staticmethod
    def _clean_up_worker(obj):
        obj.free_resources()
