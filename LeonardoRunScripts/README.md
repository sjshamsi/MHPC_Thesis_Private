# WALRUS architecture-scaling
How does the WALRUS FPM work when certain dimensions in its architecture are varied? The runs are written as SLURM scripts formatted with various Hydra parameters. These parameters contain architectural dimension parameters and much more. On CINECA's Leonardo, we tun it on the Boost partition (4x A100 64GB/node).

## Environment
We'd want to try multiple environments in case that affects performance. We store environments in $PYENVDIR (`/leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs`).

When we install the `walrus` package, it is installed editable (`pip install -e /leonardo/home/userexternal/sshamsi0/MHPC_Thesis`) so we can mess around with the source code in our repo live.

WALRUS's `pyproject.toml` suggests that we need `torch==2.5.1`, `torchvision==0.20.1`, `torchaudio==2.5.1`, `torchinfo>=1.8.0,<2`. `torch 2.5.1` ships with the CUDA wheels CUDA 11.8, 12.1, and 12.4.

Leonardo's `python/3.11.7` module uses the Python at `/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/python-3.11.7-ziwh63aulhhzxksf42k5u3gnim2rbpmp/bin/python3.11`. Since is is a Cineca installation, it might be optimised in some way. Laonardo alsao has CUDA modules for versions 11.8, 12.1, 12.2, 12.3, 12.6.

Leonardo's `cineca-ai/4.3.0` is the latest version under the `deeplrn` profile. It boasts `torch 2.2.0a0+git6c8c5ad+cu121`.

`env1` (`/leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1`) uses Leonardo's `python/3.11.7` module as the base Python. We then install `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`, `torchaudio==2.5.1+cu121` manually, and allow `pip` to install the rest from `pyproject.toml`'s
dependencies.

## Storage placement benchmark
Where do we place The Well's data and our checkpoints? Checkpoint reading and writing is infrequent enough to probbaly not matter, but we should still care about our data. We have two active Cineca projects, one associated with the Boost partition and one with the DCGP partition. Each project has a `$WORK` and a `$FAST` partition for large poersistent storage, each of size 1 TB by default. This gives us four possible locations to put data in. `storage_bench.sh` and `storage_bench_walrus.sh` empirically compare the four.

1. **`storage_bench.sh`** (+ `storage_bench_io.py`) measures raw sequential and parallel random-window write/read throughput out of a real Well HDF5 file. This hopefully mirrors how `MixedWellDataModule`'s multi-worker `DataLoader` actually samples. We run this from a compute node. We find no difference between the four.

2. **`storage_bench_walrus.sh`** stages a small `active_matter` subset (~1.5GB) under each of the four candidate paths and runs `train.py` + `model=debug` (so GPU compute is negligible and any timing gap is due to storage) for a few hundred steps each, back to back on the same GPU/node. We compare the per-batch timing that `Trainer.run_epoch` in `walrus/trainer/training.py` logs. The script removes the staged files at the end of each path's block.

We saw no different so we put things whereever we pleased.


## Logs, checkpoints, W&B
**Where do run artifacts go?** Every run script sets `++experiment_dir=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs`, so each run's `extended_config.yaml` and `checkpoints/` land under `logs/<name>/<auto-incrementing-run-idx>/`. `<name>` is auto-decorated by WALRUS (`/leonardo/home/userexternal/sshamsi0/MHPC_Thesis/walrus/configs/config.yaml` has a switch `automatic_setup: True` which enables this behaviour and we never change it) with model/optimizer descriptors, so `name=smoke_test` in a run file would result in a folder like `smoke_test-debug-delta-Isotr[Space-Adapt-]-AdamW-0.0002/`.

We always export `WANDB_MODE=offline` in our run scripts because Leonardo's compute nodes have no outbound internet. We also export `WANDB_DIR=/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs` which redirects wandb's local run
logs there (its default is `cwd`).

SLURM stdout/stderr go to `logs/slurm/<job_name>/<job_id>.out`/`.err` under a subdirectory with the same name as `--job-name`, which keep sthings a little neater. The `./submit.sh` is there to create the subdirectory named `--job-name` before forwarding to `sbatch`. We do not do this for generated scripts, or scripts generated from our `.py` files like `gen_star_search_runs.py` because they create the dir from the `.py` file.


**Resuming vs. starting fresh under the same job name** when a job is run and `auto_resume=True`, `configure_experiment` in `walrus/utils/experiment_utils.py` picks up the latest existing `<run-idx>` folder in the auto-decorated dir WALRUS generates from that run. This is nice when a run job times out and we want to resume it from it left off to continue the training. Since the Boost QOS available to us has a max time of 24 hours, this happens a lot.

W&B logs also piggybacks on the same folder. The first time an experiment folder is created `walrus/train.py::get_or_create_wandb_run_id` generates a W&B run id and writes it to `<run-idx>/wandb_run_id.txt`. Every later launch that resolves to that same folder (which will happen if you resubmit the same run file) reads the ID back and passes it to `wandb.in (id=..., resume="allow")`. Thus W&B logs for runs from identical run files (when `auto_resume=True`) will share a W&B ID. BUT to make sure the history from various W&B runs with the same ID line up correctly, we need to pass in a `step=` on every `wandb.log()` for its internal step counter, or else it will start from step 0 internally and clobber the previous history. `Trainer.train()` in `walrus/trainer/training.py` passes `step=epoch` to every `wandb.log()` call so this does not happen. This way, W&B offline logs from the same run file are joined together instead of bing completely new runs unto themselves. W&B still writes a fresh local `offline-run-<timestamp>-<id>/` directory per launch though, but that is okay. They all join together when we sync them.

To recap,
```
./submit.sh anchor_dimension_run.sh                                   # resume latest run (default)
./submit.sh --export=ALL,AUTO_RESUME=False anchor_dimension_run.sh    # force a brand-new run_idx + W&B run
```

**Resubmitting to continue a run after a timeout** After a run is done, you can just use `./submit.sh` as shown above to continue that run. But we need not wait till one particular run is over. We can use the SLURM `--dependency=singleton` option to line up a bunch of jobs from the same run file, waiting to run sequentially as soon as their predecessor is done. `--dependency=singleton` makes SLURM hold the new submission until no job with that script's
`--job-name` is still pending/running so we can't accidentally double-submit while the
timed-out job is still draining.

```
./submit.sh --dependency=singleton anchor_dimension_run.sh`
```

If you line up too many jobs from a single run, and it is done training through the specified number of epochs, the subsequent jobs will still execute, not train, and just do the final validation/test.

## Queue wait estimation

`queue_eta.py` estimates how long a job will sit in the queue before it runs, and — more usefully
— which knob would shorten that. It reads everything live from the SLURM CLIs; no config needed.

```bash
python3 LeonardoRunScripts/queue_eta.py --nodes 32 --time 12:00:00   # a run you're planning
python3 LeonardoRunScripts/queue_eta.py --script LeonardoRunScripts/grid_run_3.sh
python3 LeonardoRunScripts/queue_eta.py --mine                       # jobs already queued
python3 LeonardoRunScripts/queue_eta.py --self-test                  # validate the model
python3 LeonardoRunScripts/queue_eta.py --backtest                   # check calibration
```

It prints three independent estimates rather than one blended number, because they fail in
different ways and their disagreement is informative:

- **(a) backfill reservation** — SLURM's own answer (`squeue --start`, or `sbatch --test-only` for
  a job not yet submitted). Exact when it exists, but only ~6% of the pending queue has one.
- **(b) history** — Kaplan-Meier quantiles over comparable jobs from `sacct -a` (all users), so
  right-censored jobs still waiting are counted instead of silently dropped. Each user's
  contribution is capped: one high-throughput account submitted 1.67M of 2.08M single-node
  `normal` jobs in 30 days and, uncapped, dragged the median wait from minutes to ~2 seconds.
  For a job **already queued** this is *conditional on how long it has already waited*, and
  reported as `+p50` (time still to go). That matters enormously here — see below.
- **(c) queue drain** — ranks the job's simulated priority against *genuinely contending* pending
  jobs (~45% of the pending list is blocked on `Dependency`/held, not priority) and drains the
  node-hours ahead. It ignores backfill, so it is an upper bound for small jobs.

History is cached under `$FAST/.../logs/queue_eta_cache/` (`--refresh` to force, `--history-days N`
to change the window). First pull is ~1 min; later runs are a few seconds.

### Waits are bimodal — read `+p50`, not `p50`

Single-node `normal` jobs either backfill within minutes (the large majority) or stick for days.
So the *unconditional* median is ~4 minutes, which is useless advice for a job that has already
been sitting for a day: having waited that long, it is demonstrably in the stuck population.

The tool therefore conditions on elapsed wait for anything already queued. Measured over 14 days
of single-node `normal` jobs:

| Already waited | Predicted remaining (p50) | Actual remaining (p50) | p90 coverage |
|---|---|---|---|
| 0 (fresh submit) | 2m | 2m | 91% |
| 6h | 14.5h | 12.1h | 93% |
| 24h | 30.3h | 26.8h | 93% |

So a job 24h in should expect **another ~30 hours**, not another 4 minutes — a ~500x difference.
`--backtest` re-runs this check; the p90 coverage should stay near 90%.

The estimate is also narrowed to jobs that queued at a similar priority, which mostly affects the
tail: within single-node `normal`, the bottom priority band has a p90 of ~13h against ~0.8h for the
top band. With `normal` QoS and low fairshare you sit in the bottom few percent.

### What the tool taught us about this cluster

- **QoS dominates priority.** `PriorityWeightJobSize` is 1e7 versus 3e5 for QOS, which looks
  decisive but is not: the `SMALL_RELATIVE_TO_TIME` flag divides job size by walltime, so the
  job-size term is only ever 1-400 in practice. Real priority is QOS (120k-240k) ≫ fairshare
  (0-25k) ≳ age (0-20k). Changing QoS is the only large lever.
- **The 17-64 node band is a dead zone.** `boost_usr_prod` caps at 64 nodes, but `boost_qos_bprod`
  only accepts 65+. A 32-node/12h job under `normal` waits ~3.6 days (p50); the same width at 65
  nodes under `bprod` waits ~7 minutes. Either stay small or go to 65+.
- **Backfill only tests your first 20 pending jobs per partition** (`bf_max_job_user_part=20`), so
  submitting all ~50 `grid_run_*.sh` at once starves the tail. Submit in waves.
- **`boost_qos_dbg` is effectively instant** (~0s median) but capped at 30 min, 8 nodes and *2 jobs
  per user* — fine for smoke tests, unusable for a sweep.
- **Shorter `--time` helps twice**: it fits backfill gaps and raises the job-size priority term.
- Fairshare usage resets monthly, so a big push costs least just after a reset.

Note the `--backtest` output reports a median bias: predictions track the tail well (~90% of actual
waits land inside the predicted p90) but p50 runs optimistic when the queue is growing. Plan against
p75/p90.

## Job submission and W&B cheat sheet

**One-time W&B setup**
On the login node:
```
module purge
source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1/bin/activate
wandb login   # paste an API key from https://wandb.ai/authorize, or: wandb login <KEY>
```
This writes credentials to `~/.netrc` used by `wandb sync`.


**Job Submission**

- To resume a run that timed out or crashed, resubmit the same script with the same CLI overrides `./submit.sh <script>.sh` (add `--dependency=singleton` to line up a bunch of jobs). You can increase `trainer.max_epoch` between submissions is fine as that doesn't affect model details, just how much things run.
- To start a new, independent run, do `./submit.sh --export=ALL,AUTO_RESUME=False <script>.sh`. It'll have the same `name=`, but a brand-new `<run-idx>` folder, fresh weights, and a fresh W&B run. Still it is better to give the run a new name, even if it is to add a little number at the end of the name, indicating a repeat.
- Be sure to change the name, even for slightly different runs. `get_experiment_name` in `walrus/utils/experiment_utils.py` builds the on-disk folder name as
`{name}-{data}-{prediction_type}-{model}[{encoder}-{decoder}-{processor}]-{optimizer}-{lr}`. It does not create a forlder name for properties like `model.hidden_dim`, `processor_blocks`, `mlp_dim`, `num_heads`, `batch_size`, `trainer.max_epoch`, or basically anything else. So if we simply copy a run and change, say, `trainer.max_epoch`, and nothing else, then the problems might happen:
   - Two runs that differ in un-encoded values (e.g. `hidden_dim` sweep) but
  share the same `name=`/data/model-class/optimizer/lr will resolve to the same folder and
  silently resume into each other's checkpoint instead of running independently.
   - On the flip side, if you're trying to resume a run but tweak something that is part of the decorated name (e.g. nudge `optimizer.lr`), you will silently land in a brand-new `<run-idx>` folder with fresh weights instead of continuing.


**Syncing W&B**

Syncing a single specific job run (not autoresumed):
```
wandb sync /leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/wandb/offline-run-<timestamp>-<id>
```

Sync all the single job runs (not autoresumed) you have what aren't yet synced:
```
cd /leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs && wandb sync --sync-all
```

If we need to sync all, including previously synced runs:
```
cd /leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs && wandb sync --sync-all --include-synced
```

If a run was ever resumed (`auto_resume=True`) add `--append` so the later offline directories attach to the same online run instead of each creating a separate one:
```
cd /leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs && wandb sync --sync-all --append --include-offline --include-globs="run-<run-id>.wandb"
```

Just sync everyhting in the `wandb` dir, single or continuous runs, whether partially or fully synced:
```
cd /leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs && wandb sync --sync-all --append --include-offline
```

## Notes on some run files we have
- **`smoke_test.sh`** Tiny model + tiny data
   (`data=debug model=debug`, `active_matter`, `max_samples=8`). Can run on `boost_qos_dbg` (under 30 min). This can help confirm the run chain `modules → venv → torchrun → data loading from downloaded dataset path → training → checkpoint → logging` works. Run with:

   ```
   ./submit.sh smoke_test.sh
   ```
   and a checkpoint should appear under `/leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs/smoke_test/0/checkpoints/` (double check this because we tend to chnage paths a lot.)

- **`gen_star_search_runs.py`** — generator for the real architecture star search. Vary one axis
   (`model.hidden_dim`, `model.processor_blocks`, `model.processor.space_mixing.mlp_dim`) at a
   time around a base anchor (edit the `base`/`options` dicts at the top of the file to change
   the sweep), using `data=Leonardo_smallest4_2d` (3 of the datasets actually downloaded under this
   cluster's `well_base_path` — a lighter mixture than tier 3's `Leonardo_smallest8_2_3d`, keeping each
   of the many single-node jobs in this sweep cheap).
   ```
   python gen_star_search_runs.py   # writes star_search_run_0.sh .. star_search_run_N.sh + run_star_search_all.sh
   bash run_star_search_all.sh      # submits every generated job
   # or: sbatch star_search_run_3.sh   # submit one at a time
   ```

3. **`full_dimension_run.sh`** — hand-written baseline the dimension sweep gets compared
   against: Walrus's *original*, unmodified paper-scale architecture (`hidden_dim=1408`,
   `processor_blocks=40`, `projection_dim=48`, `intermediate_dim=352`, `groups=16` — copied
   verbatim from `../walrus/run_scripts/pretrain_example_distributed_walrus.sh`, not the smaller
   anchor `gen_star_search_runs.py` sweeps around), trained on `data=Leonardo_smallest8_2_3d` — a new data
   config (`../walrus/configs/data/Leonardo_smallest8_2_3d.yaml`) listing only the 6 datasets actually
   downloaded under `well_base_path` here (`active_matter`, `gray_scott_reaction_diffusion`,
   `helmholtz_staircase`, `turbulent_radiative_layer_2D`, `viscoelastic_instability`, `MHD_64`),
   as opposed to `Leonardo_smallest7_2d`'s full 18-dataset list (most of which isn't downloaded here
   and would fail at data-loading time) or `Leonardo_smallest4_2d`'s 3-dataset subset. Runs on 1 node
   with `distribution=fsdp` — the original script used 24 nodes with `distribution=hsdp` and
   `local_size=4`, meaning parameters were only ever sharded within a single node's 4 GPUs and
   merely replicated data-parallel-style across nodes, so a single-node FSDP job gives each GPU
   the same memory footprint as the original, just less data-parallel throughput.
   ```
   ./submit.sh full_dimension_run.sh
   ```
   24h is the max single job on `boost_usr_prod`/`normal`; `auto_resume=True` plus periodic
   checkpointing mean re-submitting the same script picks up from the last checkpoint if
   `max_epoch=101` isn't reached in one submission. See "Logs, checkpoints, W&B" below for how to
   deliberately start a fresh attempt instead of resuming.

4. **`full_dimension_run_multinode.sh`** — same architecture/data as tier 3, spread across
   `--nodes=4` (16 GPUs) with `distribution=hsdp` instead of `fsdp` (matching how the original
   24-node script actually sharded: within-node only, replicated across nodes) and
   `--dependency=singleton` to guard against overlapping resubmissions. Change the node count by
   editing the single `--nodes=` line — `--nnodes`/`--nproc_per_node` in the `torchrun` call
   already derive from SLURM env vars and adapt automatically. Note this does **not** make a
   fixed `max_epoch` finish faster: `data.module_parameters.max_samples` caps micro-batches
   *per rank* (see `walrus/data/mixed_dset_sampler.py`'s `BatchedMultisetSampler`), so
   steps-per-epoch is fixed regardless of node count. More nodes instead gives a bigger
   effective global batch per optimizer step (`batch_size * grad_acc_steps * world_size`), at
   the cost of burning GPU-hours proportionally faster for the same wall-clock duration.
   ```
   sbatch full_dimension_run_multinode.sh
   ```

5. **`anchor_dimension_run.sh`** — same idea as tier 3, but at the *sweep anchor* architecture
   (`hidden_dim=768`, `processor_blocks=12`, `mlp_dim=2048` — `gen_star_search_runs.py`'s `base` dict)
   instead of the paper-scale one. That anchor point is never generated as its own
   `star_search_run_N.sh` (the sweep's `options` lists all exclude the anchor's own value), so this
   materializes it, trained on `data=Leonardo_smallest8_2_3d` (all 6 datasets) rather than the sweep's
   `data=Leonardo_smallest4_2d` (3 datasets). At ~203M params (vs ~1.29B at paper scale) it needs
   none of tiers 3/4's FSDP/HSDP sharding — `distribution=ddp` and `gradient_checkpointing_freq=1`
   are enough, matching `gen_star_search_runs.py`'s own template.
   ```
   ./submit.sh anchor_dimension_run.sh
   ```




**Testing resume + W&B continuation**: `test_resume_wandb.sh` runs the tiny `data=debug
model=debug trainer=debug` config four times back to back under one job - a fresh 1-epoch run,
then three simulated resubmissions each training one epoch further (to epoch 2, 3, then 4) - and
asserts the checkpoint resume and the `wandb_run_id.txt` reuse described above actually happened
at every step, rather than trusting it by inspection.
```
./submit.sh test_resume_wandb.sh
```
Safe to rerun any time; each invocation starts its own fresh `run_idx` (phase 1 always passes
`auto_resume=False`) so repeated test runs never collide with each other or with real experiments
(it uses a separate `wandb_project_name=walrus_leonardo_test`). Compute nodes have no outbound
internet, so the script itself can only leave behind four offline W&B run directories sharing one
run id; it prints the exact follow-up command to run from the login node afterward, scoped to just
those four runs so it doesn't also sweep in every other unsynced real run sitting in the same
shared `WANDB_DIR`:
```
cd /leonardo_scratch/fast/ICT26_MHPC/sshamsi/logs && wandb sync --sync-all --append \
    --include-offline --include-globs="run-<run-id>.wandb"
```
(`--sync-all` always scans `./wandb` relative to the current directory and ignores any path
argument, hence the `cd`; `--include-globs` matches the `.wandb` filename *inside* each
`offline-run-*/` directory, e.g. `run-<run-id>.wandb`, not the `offline-run-*/` directory name;
`--append` tells the server to attach each directory's history to the existing run id instead of
starting a new one each time.)
Then check the `walrus_leonardo_test` project on wandb.ai for that run id — it should show up as
a **single** run whose `train/epoch` metric rises continuously from 1 to 4, not four separate
runs. See the script's own comments for exactly what it checks.
