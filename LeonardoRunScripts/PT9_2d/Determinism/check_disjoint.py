"""fulldet GPU check: build the real training sampler on every rank and check that, within an
epoch, no two ranks get the same window of a dataset until all of its windows have been used.

Reads no data: only the sampler's indices are gathered. Launched by p9_fulldet_check.sh as
`torchrun check_disjoint.py <hydra overrides>` (see gen_pt9_determinism_runs.py).
"""

import os
from bisect import bisect_right
from collections import Counter

import hydra
import torch
import torch.distributed as dist
from hydra.utils import instantiate

from walrus.train import CONFIG_DIR
from walrus.trainer.training import sampling_ranks_for
from walrus.utils.distribution_utils import configure_distribution


@hydra.main(version_base=None, config_path=str(CONFIG_DIR), config_name="config")
def main(cfg):
    world, rank = int(os.environ["WORLD_SIZE"]), int(os.environ["RANK"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    mesh = configure_distribution(cfg)
    sync = mesh.get_group(mesh_dim="fsdp").size() if "fsdp" in mesh.mesh_dim_names else 1
    layout = sampling_ranks_for(cfg.trainer.get("sampling_rank_strategy", "node"), world, sync)
    datamodule = instantiate(
        cfg.data.module_parameters,
        world_size=world,
        rank=rank,
        seed=cfg.seed,
        data_workers=0,
        well_base_path=cfg.data.well_base_path,
        field_index_map_override=cfg.data.get("field_index_map_override", {}),
        transform=cfg.data.get("transform", None),
    )
    sampler = datamodule.train_dataloader(layout).sampler
    mesh_name = f"{cfg.distribution.distribution_type} sampling_ranks={layout}"
    for epoch in (1, 2):
        sampler.set_epoch(epoch)
        mine = list(sampler)
        everyone = [None] * world
        dist.all_gather_object(everyone, (sampler.windows, mine))
        if rank != 0:
            continue
        assert all(w == sampler.windows for w, _ in everyone), "ranks disagree on dataset sizes"
        starts = [sum(sampler.windows[:d]) for d in range(len(sampler.windows))]

        for r, row in enumerate(layout):  # ranks sharing a sampling rank share the dataset each step
            first = layout.index(row)
            mine_d, first_d = ([bisect_right(starts, b[0]) - 1 for b in everyone[k][1]] for k in (r, first))
            assert mine_d == first_d, f"rank {r} trains on other datasets than rank {first} of its row"
        uses = Counter(i for _, batches in everyone for b in batches for i in b)
        for d, (s, n) in enumerate(zip(starts, sampler.windows)):
            counts = [uses[i] for i in range(s, s + n)]
            print(f"epoch {epoch} dataset {d}: windows {n}, drawn {sum(counts)}, "
                  f"max uses {max(counts)}", flush=True)
            assert max(counts) - min(counts) <= 1, f"dataset {d} repeats a window before using all"
        print(f"DISJOINT OK epoch {epoch} ({mesh_name})", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
