import numpy as np
from deeplake.util.iterable_ordered_dict import IterableOrderedDict


def collate_fn(batch):
    from torch.utils.data._utils.collate import default_collate

    elem = batch[0]

    if isinstance(elem, IterableOrderedDict):
        return IterableOrderedDict(
            (key, collate_fn([d[key] for d in batch])) for key in elem.keys()
        )
    if isinstance(elem, np.ndarray) and elem.dtype.type is np.str_:
        batch = [it.item() for it in batch]

    return default_collate(batch)
