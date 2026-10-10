import inspect
import logging
from typing import Iterator, List

import torch
from torch.utils.data import Sampler

from walrus.utils.experiment_utils import SEED_DATASET_CHOICE, SEED_WINDOWS, seed_from

from .multidataset import MixedWellDataset

__all__ = [
    "MultisetSampler",
]


logger = logging.getLogger(__name__)


class MultisetSampler(Sampler[int]):
    """Sampler that restricts data loading to a subset of the dataset."""

    def __init__(
        self,
        dataset: MixedWellDataset,
        base_sampler: type[Sampler[int]],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        max_samples=10,
        rank=0,
        distributed=True,
        recycle=True,
    ) -> None:
        self.batch_size = batch_size
        self.sub_dsets = dataset.sub_dsets
        self.base_sampler = base_sampler
        self.recycle = recycle
        if (
            "drop_last"
            in inspect.signature(getattr(base_sampler, "__init__")).parameters
        ):
            self.sub_samplers = [
                base_sampler(dataset, drop_last=drop_last)  # type: ignore
                for dataset in self.sub_dsets
            ]
        else:
            self.sub_samplers = [base_sampler(dataset) for dataset in self.sub_dsets]
        self.dataset = dataset
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.max_samples = max_samples
        self.rank = rank

    def __iter__(self) -> Iterator[int]:
        samplers = [iter(sampler) for sampler in self.sub_samplers]
        sampler_choices = list(range(len(samplers)))
        generator = torch.Generator()
        # Ensure each worker on the same rank (GPU) sample the same dataset
        generator.manual_seed(seed_from(SEED_DATASET_CHOICE, self.seed, self.epoch, self.rank))
        count = 0
        while len(sampler_choices) > 0:
            count += 1
            index_sampled = int(
                torch.randint(
                    0, len(sampler_choices), size=(1,), generator=generator
                ).item()
            )
            dset_sampled = sampler_choices[index_sampled]
            offset = max(0, self.dataset.offsets[dset_sampled])
            # Balance workload with different batch sizes
            sub_batch_size = self.dataset.effective_batch_sizes[dset_sampled]
            # Gather a batch of data from the same dataset
            # A drop last batch logic must be enforced
            # If a complete batch can be assembled, yield it
            # Otherwise move to the next dataset
            try:
                accumulated_batch: List[int] = []
                for _ in range(self.batch_size):
                    indices: List[int] = []
                    for _ in range(sub_batch_size):
                        indices.append(next(samplers[dset_sampled]) + offset)
                    # TODO Again, this is currently assuming we're not doing effective BS
                    accumulated_batch.append(indices[0])
            except StopIteration:
                # Selected sampler was exhausted
                if self.recycle:
                    count -= 1
                    samplers[index_sampled] = iter(self.sub_samplers[dset_sampled])
                    # sampler_choices[index_sampled] = self.base_sampler(self.sub_dsets[dset_sampled])
                else:
                    sampler_choices.pop(index_sampled)
                logger.info(
                    f"Note: dset {dset_sampled} fully used. Dsets remaining: {len(sampler_choices)}"
                )
                continue
            else:
                yield from accumulated_batch
            if count >= self.max_samples:
                break

    def __len__(self) -> int:
        # Data loader len is len(sampler) / batch_size - so override len to max_samples * batch_size
        if self.recycle:
            return self.max_samples * self.batch_size
        return min(
            self.max_samples * self.batch_size, sum([len(s) for s in self.sub_samplers])
        )

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        for sampler in self.sub_samplers:
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
        self.epoch = epoch


def epoch_plan(seed, epoch, sampling_ranks, windows, dset_batch, max_samples, recycle=True):
    """Every rank's micro-batches for one epoch. Every rank computes the same plan.

    Ranks sharing a sampling rank (a "row", e.g. one FSDP group) draw the same dataset each
    step. Each rank's draw claims the next ``dset_batch[d]`` positions of dataset d's epoch
    stream, in (step, row, rank) order, so no two ranks get the same position. Returns
    ``plan[r] = [(d, start), ...]``, one entry per step for global rank r.
    """
    rows: dict = {}
    for r, row in enumerate(sampling_ranks):
        rows.setdefault(row, []).append(r)
    gens = {
        row: torch.Generator().manual_seed(seed_from(SEED_DATASET_CHOICE, seed, epoch, row))
        for row in rows
    }
    used = [0] * len(windows)
    alive = list(range(len(windows)))
    plan: List[List[tuple]] = [[] for _ in sampling_ranks]
    for step in range(max_samples):
        for row, ranks in sorted(rows.items()):
            while alive:
                i = int(torch.randint(0, len(alive), (1,), generator=gens[row]).item())
                d = alive[i]
                if recycle or used[d] + len(ranks) * dset_batch[d] <= windows[d]:
                    break
                alive.pop(i)  # exhausted, for every rank at once
            else:  # nothing left: every rank stops after the same number of steps
                return [p[:step] for p in plan]
            for r in ranks:
                plan[r].append((d, used[d]))
                used[d] += dset_batch[d]
    return plan


class BatchedMultisetSampler(Sampler[int]):
    """Sampler that restricts data loading to a subset of the dataset.

    This version returns a full batch worth of indices at once and requires
    that the dataset can process list of indices.

    It follows ``epoch_plan``: position p of dataset d's stream in epoch e is window
    ``perm_k[p % n_d]`` with ``k = p // n_d`` and ``perm_k`` a permutation keyed on
    (seed, epoch, d, k) - not on the rank. Within an epoch, no two ranks get the same
    window of a dataset until all of its windows have been used once."""

    def __init__(
        self,
        dataset: MixedWellDataset,
        base_sampler=None,  # unused: kept for the call signature
        batch_size: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = True,
        max_samples=10,
        sampling_ranks=(0,),
        distributed=True,
        recycle=True,
        global_rank=0,
    ) -> None:
        self.dataset = dataset
        self.windows = [len(d) for d in dataset.sub_dsets]
        self.dset_batch = [max(1, int(batch_size * f)) for f in dataset.effective_batch_sizes]
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.max_samples = max_samples
        self.recycle = recycle
        self.sampling_ranks = list(sampling_ranks)  # sampling rank of every global rank
        self.global_rank = global_rank

    def plan(self):
        """This rank's [(dataset, stream position), ...] for the current epoch."""
        return epoch_plan(
            self.seed, self.epoch, self.sampling_ranks, self.windows, self.dset_batch,
            self.max_samples, self.recycle,
        )[self.global_rank]

    def __iter__(self) -> Iterator[List[int]]:  # type: ignore[override]
        perms: dict = {}
        for d, start in self.plan():
            n = self.windows[d]
            batch = []
            for p in range(start, start + self.dset_batch[d]):
                k, j = divmod(p, n)
                if (d, k) not in perms:
                    g = torch.Generator().manual_seed(seed_from(SEED_WINDOWS, self.seed, self.epoch, d, k))
                    perms[d, k] = torch.randperm(n, generator=g)
                batch.append(int(perms[d, k][j]) + max(0, self.dataset.offsets[d]))
            yield batch

    def __len__(self) -> int:
        # Each sample is a full micro-batch
        return self.max_samples if self.recycle else len(self.plan())

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch; the plan and the window permutations are keyed on it."""
        self.epoch = epoch
