from typing import Any, Optional

import numpy as np
import jax.numpy as jnp
from torch.utils import data

def jax_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return jnp.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [jax_collate(samples) for samples in transposed]
    else:
        return jnp.array(batch)


class JaxLoader(data.DataLoader):
    def __init__(
        self,
        dataset: Any,
        batch_size: Optional[int] = 1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
        collate_fn=None,
    ):
        if collate_fn is None:
           collate_fn = jax_collate

        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )
