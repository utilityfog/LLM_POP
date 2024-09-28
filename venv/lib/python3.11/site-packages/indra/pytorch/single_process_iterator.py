from typing import Callable, List, Optional
from indra.pytorch.buffered_loader import BufferedLoader
from indra.pytorch.util import (
    transform_collate_batch,
    create_folder,
)
from indra.pytorch.tensorinfo import TensorsInfo
from indra.pytorch.exceptions import (
    TransformExceptionWrapper,
    CollateExceptionWrapper,
)

from indra.pytorch.log import get_logger

from indra.pytorch.common import collate_fn as default_collate
from deeplake.integrations.pytorch.common import convert_sample_to_data
from deeplake.core.serialize import bytes_to_text
from uuid import uuid4


from PIL import Image
import traceback

import io
import os


class SingleProcessIterator:
    def __init__(
        self,
        dataloader,
        info: TensorsInfo,
        upcast: bool = True,
        transform_fn: Optional[Callable] = None,
        collate_fn: Optional[Callable] = default_collate,
        ignore_errors: bool = True,
        verbose: bool = False,
    ):
        """
        Returns an iterator for single process iteration

        Args:
            list_tensors (List[str], optional) Subset of raw tensors, these will be decompressed by python workers into lists.
            upcast (bool) flag that is showing whether we need to upcast object if dtype is not supported this is needed only for
                pytorch as it is not support all the dtypes. Defaults to True.
            transform_fn (Callable, optional) Callable object which is needed to be applyed on each sample on batch. Defaults to None.
            collate_fn (callable, optional): merges a list of samples to form a
                mini-batch of Tensor(s).  Used when using batched loading from a
                map-style dataset.
            ignore_errors(bool) shows whether need to ignore the errors appearing during transformation
        """
        self.dataloader = dataloader
        self.info = info
        self.upcast = upcast
        self.transform_fn = transform_fn
        self.collate_fn = collate_fn
        self.raw_tensor_set = (
            set(self.info.raw_tensors)
            - set(self.info.json_tensors)
            - set(self.info.list_tensors)
        )  # tensors to be returned as bytes
        self.ignore_errors = ignore_errors

        self.iter_pos = None
        self.skipped = 0
        self.processed = 0
        self.verbose = verbose
        self.pid = os.getpid()
        self.logger = get_logger(self.verbose)

    def __iter__(self):
        self.dataloader.reset()

        self.skipped = 0
        self.processed = 0
        self.iter_pos = iter(self.dataloader)
        return self

    def __next__(self):
        return self.get_data()

    def __len__(self) -> int:
        return len(self.dataloader)

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"SingleProcessIterator length {len(self)}"

    def _next_data(self):
        batch = next(self.iter_pos)
        for sample in batch:
            for tensor in self.info.pil_compressed_tensors:
                if isinstance(sample[tensor], (list, tuple)):
                    sample[tensor] = list(
                        Image.open(io.BytesIO(t)) for t in sample[tensor]
                    )
                else:
                    sample[tensor] = Image.open(io.BytesIO(sample[tensor]))
            for tensor in self.info.json_tensors:
                sample[tensor] = bytes_to_text(sample[tensor], "json")
            for tensor in self.info.list_tensors:
                sample[tensor] = bytes_to_text(sample[tensor], "list")
            if self.info.htype_dict:
                convert_sample_to_data(
                    sample,
                    self.info.htype_dict,
                    self.info.ndim_dict,
                    self.info.tensor_info_dict,
                )
        return batch

    def get_data(self):
        while True:
            self.processed += 1
            batch = self._next_data()
            try:
                return transform_collate_batch(
                    batch,
                    self.transform_fn,
                    self.collate_fn,
                    self.upcast,
                    self.raw_tensor_set,
                )
            except Exception as ex:
                self.logger.debug(
                    f"SingleProcessIterator {self.pid} exception happened {ex}"
                )
                self.handle_exception(ex)
                if self.ignore_errors:
                    continue
                else:
                    raise

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
                self.logger.info(
                    f"An exception happened during data handling exception: {ex} processed batches {ex.processed} skipped batched {ex.skipped}"
                )
            else:
                traceback.print_tb(ex.exception.__traceback__)
        else:
            if self.ignore_errors:
                self.logger.info(
                    f"An exception happened during data handling exception: {ex} processed batches {self.processed}"
                )
            else:
                traceback.print_tb(ex)
        self.skipped += 1

    def close(self):
        return
