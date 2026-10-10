"""Training sampler: within an epoch, no two ranks get the same window of a dataset."""

from collections import Counter
from types import SimpleNamespace

import pytest

from walrus.data.mixed_dset_sampler import BatchedMultisetSampler

SIZES = [500, 3000, 40]  # the 40-window dataset runs out within an epoch, so it wraps
FACTORS = [1.0, 0.5, 1.0]  # effective_batch_sizes: dataset 1 gets micro-batches of 2 // 2 = 1
LAYOUTS = {  # sampling rank of each of 8 global ranks
    "fsdp": [0] * 8,
    "hsdp": [0, 0, 0, 0, 1, 1, 1, 1],
    "ddp": list(range(8)),
}


def fake_dataset():
    # Only what the sampler reads; offsets[0] = -1 as in MixedWellDataset
    offsets = [-1, SIZES[0], SIZES[0] + SIZES[1]]
    return SimpleNamespace(sub_dsets=[range(n) for n in SIZES], effective_batch_sizes=FACTORS,
                           offsets=offsets)


def epoch_draws(layout, seed=0, epoch=1, recycle=True, max_samples=30):
    """Per rank: the list of micro-batches, each a list of global window indices."""
    out = []
    for r in range(len(layout)):
        s = BatchedMultisetSampler(fake_dataset(), batch_size=2, seed=seed, max_samples=max_samples,  # type: ignore[arg-type]
                                   sampling_ranks=layout, global_rank=r, recycle=recycle)
        s.set_epoch(epoch)
        out.append(list(s))
        assert len(out[-1]) == len(s)
    return out


def dataset_of(i):
    return 0 if i < SIZES[0] else 1 if i < SIZES[0] + SIZES[1] else 2


@pytest.mark.parametrize("layout", LAYOUTS)
def test_windows_disjoint_until_dataset_used_up(layout):
    draws = epoch_draws(LAYOUTS[layout])
    uses = Counter(i for rank in draws for batch in rank for i in batch)
    for d, n in enumerate(SIZES):
        counts = [uses[i] for i in range(sum(SIZES[:d]), sum(SIZES[: d + 1]))]
        # Every window is used once before any is used twice
        assert max(counts) - min(counts) <= 1, (layout, d)
    assert max(uses[i] for i in range(SIZES[0])) == 1  # demand < n: no repeat at all
    assert max(uses.values()) > 1  # the small dataset did wrap


@pytest.mark.parametrize("layout", LAYOUTS)
def test_ranks_in_a_row_share_the_dataset(layout):
    draws = epoch_draws(LAYOUTS[layout])
    for r, row in enumerate(LAYOUTS[layout]):
        first = LAYOUTS[layout].index(row)
        assert [dataset_of(b[0]) for b in draws[r]] == [dataset_of(b[0]) for b in draws[first]]


def test_seed_and_epoch_change_the_draws():
    base = epoch_draws(LAYOUTS["fsdp"])
    assert base == epoch_draws(LAYOUTS["fsdp"])
    assert base != epoch_draws(LAYOUTS["fsdp"], seed=1)
    assert base != epoch_draws(LAYOUTS["fsdp"], epoch=2)


@pytest.mark.parametrize("layout", LAYOUTS)
def test_no_recycle_stops_every_rank_together(layout):
    draws = epoch_draws(LAYOUTS[layout], recycle=False, max_samples=10**6)
    assert len({len(rank) for rank in draws}) == 1
    flat = [i for rank in draws for batch in rank for i in batch]
    assert len(flat) == len(set(flat))
