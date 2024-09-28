from typing import Optional, List, Callable


class TensorsInfo:
    def __init__(
        self,
        htype_dict: Optional[dict] = None,
        ndim_dict: Optional[dict] = None,
        tensor_info_dict: Optional[dict] = None,
        pil_compressed_tensors: Optional[List[str]] = None,
        raw_tensors: Optional[List[str]] = None,
        json_tensors: Optional[List[str]] = None,
        list_tensors: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            htype_dict (dict): Dictionary of the tensors and their corresponding htypes. Only populated for tensors which have data as decode_method.
            ndim_dict (dict): Dictionary of the tensors and their corresponding ndims. Only populated for tensors which have data as decode_method.
            tensor_info_dict (dict): Dictionary of the tensors and their corresponding tensor_info. Only populated for tensors which have data as decode_method and have htype class_label.
            pil_compressed_tensors (List[str], optional) Subset of raw tensors, these will be decompressed by python workers into PIL images.
            raw_tensors (List[str], optional) List of the tensors that needs to return raw data instead of decompression.
                Defaults to ``None`` if raw_tensors is None then all the tensors will send decompression data
                E.g raw_tensors['images'] then only the images tensor data will be sent as a row array
            json_tensors (List[str], optional) Subset of raw tensors, these will be decompressed by python workers into jsons.
            list_tensors (List[str], optional) Subset of raw tensors, these will be decompressed by python workers into lists.
        """
        self.htype_dict = htype_dict or {}
        self.ndim_dict = ndim_dict or {}
        self.tensor_info_dict = tensor_info_dict or {}
        self.pil_compressed_tensors = pil_compressed_tensors or []
        self.raw_tensors = raw_tensors or []
        self.json_tensors = json_tensors or []
        self.list_tensors = list_tensors or []

    def __getstate__(self):
        return {
            "htype_dict": self.htype_dict,
            "ndim_dict": self.ndim_dict,
            "tensor_info_dict": self.tensor_info_dict,
            "pil_compressed_tensors": self.pil_compressed_tensors,
            "raw_tensors": self.raw_tensors,
            "json_tensors": self.json_tensors,
            "list_tensors": self.list_tensors,
        }

    def __setstate__(self, state):
        self.htype_dict = state["htype_dict"]
        self.ndim_dict = state["ndim_dict"]
        self.tensor_info_dict = state["tensor_info_dict"]
        self.pil_compressed_tensors = state["pil_compressed_tensors"]
        self.raw_tensors = state["raw_tensors"]
        self.json_tensors = state["json_tensors"]
        self.list_tensors = state["list_tensors"]


class LoaderMetaInfo:
    def __init__(
        self,
        context=None,
        distributed: bool = False,
        upcast: bool = True,
        return_index: bool = True,
        verbose: bool = False,
        ignore_errors: bool = False,
        prefetch_factor: int = 2,
        offset: int = 0,
        primary_tensor: Optional[str] = None,
        worker_init_fn: Optional[Callable] = None,
    ):
        """DataLoader Meta information necassary for its initialization.

        Args:
            context: this is the multiprocessing context that is used during programm executaion
            distributed (bool) flag that is showing whether Loader need to work in DDP or not. Defaults to ``False``
            upcast (bool) flag that is showing whether we need to upcast object if dtype is not supported this is needed only for
            return_index (bool) Showing whether Loader needs to return the sample index during iteration.Defaults to True.
            verbose (bool)
            ignore_errors(bool)
            prefetch_factor (int) Number of samples loaded in advance by workers. Defaults to 10
            offset (int, optional) offset that the dataloader will start to iterate.
        """
        self.context = context
        self.distributed = distributed
        self.upcast = upcast
        self.return_index = return_index
        self.verbose = verbose
        self.ignore_errors = ignore_errors
        self.prefetch_factor = prefetch_factor
        self.offset = offset
        self.primary_tensor = primary_tensor
        self.worker_init_fn = worker_init_fn

    def __getstate__(self):
        return {
            "context": self.context,
            "distributed": self.distributed,
            "upcast": self.upcast,
            "return_index": self.return_index,
            "verbose": self.verbose,
            "ignore_errors": self.ignore_errors,
            "prefetch_factor": self.prefetch_factor,
            "offset": self.offset,
            "primary_tensor": self.primary_tensor,
            "worker_init_fn": self.worker_init_fn,
        }

    def __setstate__(self, state):
        self.context = state["context"]
        self.distributed = state["distributed"]
        self.upcast = state["upcast"]
        self.return_index = state["return_index"]
        self.verbose = state["verbose"]
        self.ignore_errors = state["ignore_errors"]
        self.prefetch_factor = state["prefetch_factor"]
        self.offset = state["offset"]
        self.primary_tensor = state["primary_tensor"]
        self.worker_init_fn = state["worker_init_fn"]
