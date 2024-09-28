from typing import Callable, List, Optional
from indra.pytorch.buffered_loader import BufferedLoader
from indra.pytorch.util import get_indexes, is_serializable
from indra.pytorch.tensorinfo import TensorsInfo, LoaderMetaInfo
from indra.pytorch.common import collate_fn as default_collate
from indra.pytorch.single_process_iterator import SingleProcessIterator
from indra.pytorch.multi_processing_iterator import MultiProcessingIterator
from deeplake.integrations.pytorch.shuffle_buffer import ShuffleBuffer
import warnings
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


MB = 1024 * 1024


class Loader:
    def __init__(
        self,
        dataset,
        loader_meta: LoaderMetaInfo,
        info: TensorsInfo,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        transform_fn: Optional[Callable] = None,
        num_workers: int = 0,
        num_threads: Optional[int] = None,
        collate_fn: Optional[Callable] = default_collate,
        tensors: Optional[List[str]] = None,
        buffer_size: int = 2048,
        persistent_workers: bool = False,
    ):
        """Returns a Loader object referencing to C++ dataloader instance.

        Args:
            batch_size (int, optional): how many samples per batch to load
                (default: ``1``).
            shuffle (bool, optional): set to ``True`` to have the data reshuffled at every epoch
                (default: ``False``).
            drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
                if the dataset size is not divisible by the batch size. If ``False`` and
                the size of dataset is not divisible by the batch size, then the last batch
                will be smaller. (default: ``False``)
            num_workers (int, optional): how many subprocesses to use for data
                loading. ``0`` means that the data will be loaded in the main process.
                (default: ``0``)
            num_threads (int, optional) number of threads that need to be spined up during data loading. Defaults to None.
                if it is none then we are detecting the hardware concurrency count to set.
                Note: We don't set any threading flags (eg. OMP_NUM_THREADS, MKL_NUM_THREADS, etc)
                to get more performant Loader consider of not setting those flags which can affect on 3rd party libraries workflow performance
            transform_fn (Callable, optional) Callable object which is needed to be applied on each sample on batch. Defaults to None.
            collate_fn (callable, optional): merges a list of samples to form a
                mini-batch of Tensor(s).  Used when using batched loading from a
                map-style dataset.
            tensors (List[str], optional) List of tensors that are participating to in Loading process.
                Defaults to ``None`` which means that Loader will fetch samples for all of the tensors
                pytorch as it is not support all the dtypes. Defaults to True.
            buffer_size (int): The size of the buffer used to shuffle the data in MBs. Defaults to 2048 MB. Increasing the buffer_size will increase the extent of shuffling.
            persistent_workers (bool): If ``True``, the data loader will not shutdown the worker processes after a dataset has been consumed once. Defaults to ``False``.
            loader_meta(LoaderMetaInfo, optional) dataloader meta information collected into one structure
            info(TensorsInfo, optional) tensor related informations
        """
        if loader_meta.primary_tensor is not None:
            dataset.primary_tensor = loader_meta.primary_tensor
        if num_workers is not None and num_workers < 0:
            raise ValueError("num_workers must be non-negative")

        if num_threads is not None and num_threads <= 0:
            raise ValueError("num_threads must be positive")
        self.dataset = dataset
        self.batch_size = batch_size or 1
        self.shuffle = shuffle or False
        self.drop_last = drop_last or False
        self.transform_fn = transform_fn
        self.num_workers = num_workers or 0
        self.num_threads = num_threads
        self.collate_fn = collate_fn
        self.tensors = tensors or []
        self._dataloader = None
        self.buffer = ShuffleBuffer(buffer_size * MB) if self.shuffle else None
        self.loader_meta = loader_meta
        self.info = info
        self.persistent_workers = persistent_workers or False
        self._worker_init_fn = loader_meta.worker_init_fn or None

        self._iterator = None

    def __del__(self):
        del self._iterator

    def __iter__(self):
        if self._dataloader is None:
            dataset = self.dataset
            if self.loader_meta.distributed:
                indexes = get_indexes(dataset)
                dataset = dataset[indexes]

            self._dataloader = create_dataloader(
                dataset=dataset,
                drop_last=self.drop_last,
                return_index=self.loader_meta.return_index,
                batch_size=self.batch_size,
                num_threads=self.num_threads,
                tensors=self.tensors,
                raw_tensors=list(
                    set(
                        self.info.raw_tensors
                        + self.info.list_tensors
                        + self.info.json_tensors
                        + self.info.pil_compressed_tensors
                    )
                ),
                ignore_errors=self.loader_meta.ignore_errors,
                offset=self.loader_meta.offset,
            )

        return self._get_iterator()

    def _get_iterator(self):
        if self.num_workers > len(self._dataloader):
            warnings.warn(
                f"Setting num_worker greater than dataset size is not allowed "
                f"adjusting it to {len(self._dataloader)} in default, to avoid your system oversubscription "
            )
            self.num_workers = len(self._dataloader)

        if self.num_workers == 0:
            if self._iterator is None:
                self._iterator = self.zero_worker_iter()
            else:
                self._iterator = iter(self._iterator)
        else:
            if self.transform_fn is not None and not is_serializable(self.transform_fn):
                raise RuntimeError(
                    "Unable to send the transform function to the subprocess for multiprocessing."
                    "Ensure the function is picklable or consider alternative serialization methods."
                )
            if self.collate_fn is not None and not is_serializable(self.collate_fn):
                raise RuntimeError(
                    "Unable to send the collate function to the subprocess for multiprocessing.",
                    "Ensure the function is picklable or consider alternative serialization methods.",
                )
            if self._iterator is None:
                self._iterator = self.multiprocess_iter()
            else:
                self._iterator = iter(self._iterator)

        return self

    def __next__(self):
        if self._iterator is None:
            self.__iter__()
        return next(self._iterator)

    def close(self):
        if self._iterator is not None:
            self._iterator = None

    @property
    def dataloader(self):
        return (
            BufferedLoader(
                self._dataloader, self.buffer, self.batch_size, self.drop_last
            )
            if self.shuffle
            else self._dataloader
        )

    @property
    def worker_init_fn(self):
        return self._worker_init_fn

    @worker_init_fn.setter
    def worker_init_fn(self, fn):
        self._worker_init_fn = fn

    @property
    def summary(self):
        if self._dataloader is not None:
            self._dataloader.summary

    def zero_worker_iter(self):
        return iter(
            SingleProcessIterator(
                dataloader=self.dataloader,
                info=self.info,
                upcast=self.loader_meta.upcast,
                transform_fn=self.transform_fn,
                collate_fn=self.collate_fn,
                ignore_errors=self.loader_meta.ignore_errors,
                verbose=self.loader_meta.verbose,
            )
        )

    def multiprocess_iter(self):
        return iter(
            MultiProcessingIterator(
                dataset=self.dataset,
                dataloader=self.dataloader,
                info=self.info,
                loader_meta=self.loader_meta,
                transform_fn=self.transform_fn,
                collate_fn=self.collate_fn,
                worker_init_fn=self._worker_init_fn,
                num_workers=self.num_workers,
                persistent_workers=self.persistent_workers,
            )
        )


def create_dataloader(
    dataset,
    batch_size,
    num_threads,
    tensors,
    raw_tensors,
    drop_last=False,
    return_index=True,
    ignore_errors=True,
    offset=0,
):
    if num_threads is None:
        return dataset.loader(
            batch_size=batch_size,
            tensors=tensors,
            raw_tensors=raw_tensors,
            drop_last=drop_last,
            return_index=return_index,
            ignore_errors=ignore_errors,
            offset=offset if offset else 0,
        )

    return dataset.loader(
        batch_size=batch_size,
        num_threads=num_threads,
        tensors=tensors,
        raw_tensors=raw_tensors,
        drop_last=drop_last,
        return_index=return_index,
        ignore_errors=ignore_errors,
        offset=offset if offset else 0,
    )
