# Pricing a WALRUS job in FLOPs: `count_WALRUS_flops.py` + the dataset registry

Companion to `scripts/count_WALRUS_params.py`: give it a run script, get the job's training
FLOPs, broken down by dataset and by model part, without running anything.

Related: `mydocs/walrus_tokens_and_data_pipeline.md` (what a token / micro-batch / epoch is),
`mydocs/walrus_architecture_and_parameter_counts.md` (parameters), `walrus/utils/flops.py`
(the per-micro-batch formulas, the same ones `FlopAccountant` logs to W&B).

---

## 1. The three files

| file | role |
|---|---|
| `scripts/well_registry.py` | Shared library: load the registry, window counts, and `dataset_geometry()` (registry entry → the `BatchGeometry` that `flops.forward_flops` prices). |
| `scripts/build_well_registry.py` | Scans HDF5 metadata and writes `scripts/well_dataset_registry.json`. `--verify` checks the registry against real batches. |
| `scripts/count_WALRUS_flops.py` | Reads a run script, composes the Hydra config, prices the job. |

Everything runs on a login node in the walrus venv:

```bash
module purge; module load python; module load cuda/12.2
source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1/bin/activate
```

---

## 2. The dataset registry

One JSON entry per dataset under `well_base_path` (18 on 2026-10-06, 2D and 3D, held-out
included). Per dataset it stores:

- resolution, `n_spatial_dims`, `grid_type`, per-axis BCs (from the_well's own reader);
- native field names, and the model-format field list after 2D→3D padding (fields, then
  constants), whose length is `n_channels_in`, the channel count the encoder actually sees
  (the 3 BC flags come on top);
- token grid (32 per axis in 2D, 16 in 3D, +1 on each non-periodic axis);
- per split: files, bytes on disk, and `(n_steps, n_traj)` groups.

Window counts are **not** stored; they depend on `n_steps_input` / `n_steps_output` /
`min_dt_stride` and are derived when pricing.

```bash
python scripts/build_well_registry.py                      # full rescan, ~3 min
python scripts/build_well_registry.py --datasets MHD_64    # rescan one, keep the others
python scripts/build_well_registry.py --path /some/dir     # an external Well-format dataset
python scripts/build_well_registry.py --verify-only        # check, ~2 min for all 18
python scripts/build_well_registry.py --csv out.csv        # also a flat summary table
```

**`--verify`** loads one real window per dataset through the training pipeline (datamodule →
formatter → a tiny `IsotropicModel`), lets `FlopAccountant` observe the geometry, and checks it
equals what the registry implies (channels, padded encoder input, strides, periodic axes, token
grid), and that both price to the same FLOPs. Axis order is ignored (`RandomRotation90` permutes
axes), and so is the BC flag of a size-1 axis (it never affects cost). **All 18 pass**
(2026-10-06). Rerun it after every rebuild.

Rebuild when datasets are downloaded or deleted. The registry header records `built_at`, `root`
and the git SHA.

---

## 3. Pricing a job

```bash
python scripts/count_WALRUS_flops.py LeonardoRunScripts/PT9_2d/NoiseFloor/p9_noise_floor_run_0.sh
python scripts/count_WALRUS_flops.py <script> --exact           # replay the sampler's dataset draws
python scripts/count_WALRUS_flops.py <script> -H 1408 -L 40 -M 5632 --nodes 4   # what-if
python scripts/count_WALRUS_flops.py <script> --start-epoch 1 --end-epoch 30    # a slice
python scripts/count_WALRUS_flops.py <script> --json
python scripts/count_WALRUS_flops.py --data Leonardo_pretrain9_2d --override trainer.max_epoch=10
```

What it does:

1. Takes the Hydra overrides after `train.py` in the script (shell placeholders `$AUTO_RESUME`,
   `$STAGE_DIR` substituted, `hydra.*` dropped, `server` pinned to `leonardo`) and the
   `#SBATCH --nodes` / `--gpus-per-node`. CLI flags are applied on top; for any key the last
   value wins.
2. Builds the model on the meta device to read its shape (`flops.ModelShape.from_model`) and
   parameter count.
3. For each dataset in the data config: effective batch and `n_steps_input` after the
   per-dataset downsample factors (same flooring as the code), geometry from the registry, then
   `forward_flops` and `training_step_flops`.
4. Counts micro-batches per dataset over epochs 1..`max_epoch` (all ranks):
   - **default**: `max_samples × world_size × epochs / K` each (the sampler is uniform over
     datasets);
   - **`--exact`**: replays `BatchedMultisetSampler`'s dataset-choice generator
     (seed `100·epoch + sampling_rank`), one stream per distinct sampling rank: 1 under `fsdp`,
     one per node under `hsdp`, one per GPU under `ddp`. It models sub-sampler exhaustion and
     recycling from the registry's window counts. `local` runs reuse epoch 0's sequence every
     epoch, as the code does.

FLOP basis: forward + backward (backward = 2 × forward), exactly as `FlopAccountant` logs, with
gradient-checkpointing recompute shown separately. **Validation is not included.**

If the data config names a dataset that isn't in the registry, it stops and prints the builder
command.

---

## 4. Worked examples (2026-10-06)

### PT9 noise floor, seed 0 (`768/12/2048`, `fsdp`, 1 node, 100 epochs)

```
total training FLOPs   7.072 EF   (recompute 0)
  encoder / processor / decoder   356 PF / 6.386 EF / 329 PF   (enc+dec 9.7 % of forward)
per epoch              70.7 PF          per optimizer step   141 TF
micro-batches          800,000          tokens seen          10.18 G (101.8 M/epoch)
unique model tokens    3.26 G           repetition           3.12x
```

Per dataset: 8.4–9.3 TF per micro-batch; enc+dec 8.7–12.0 % of forward; repetition from 1.11×
(gray_scott) to 11.0× (planetswe). These reproduce `SCALING_STRATEGY_2D.md` §2.5 and §3.1.
(§2.1 used to list planetswe with 4 channels, giving 9.3 % in §3.1; the pipeline gives **3**,
confirmed by `--verify`, and both tables were corrected on 2026-10-06.)

`--exact` gives 7.073 EF, within 0.02 % of the expectation; per dataset the replay deviates by
up to ±1.3 %.

### Upstream WALRUS pretraining (`1408/40/5632`, `hsdp`, 24 nodes, 201 epochs)

Run with `--override '~data.module_parameters.well_dataset_info.flowbench_FPO_NS_2D_512x128_harmonics'`,
since that external dataset isn't on disk here.

```
total training FLOPs   4873 EF  = 4159 EF useful + 714 EF recompute (gradient_checkpointing_freq=2)
micro-batches          38.6 M   (the paper reports "38M micro-batches")
tokens seen            500 G    repetition 125x (27x gray_scott ... 3840x viscoelastic)
```

3D micro-batches cost 146–182 TF vs ~107–115 TF for 2D, at a similar token count; that is the
attention cost of 4096-token frames. `--exact` (24 streams × 201 epochs, ~1 min) agrees to
0.004 %.

---

## 5. Caveats

- **Fixed 2026-10-06: `walrus/utils/flops.py::unique_model_tokens`** used to infer periodicity
  from the *set* of BC types, so fully periodic 2D datasets (gray_scott, euler_periodicBC,
  shear_flow) got 32×33 instead of 32×32, and the mixed 3D ones (rayleigh_taylor, TRL3D)
  misplaced their wall axis. On PT9 that made `T_unique_model` 3.309 G instead of 3.260 G, so
  the W&B `repetition_factor` read ~1.5 % low. It now reads per-axis BCs from the first file
  (`_dataset_periodic_axes`) and matches the registry on all 18 datasets. If a file can't be read
  it falls back to the BC set, which is exact except for the two mixed 3D datasets (logged as a
  warning). FLOPs and `tokens_seen` were never affected. **Runs logged before the fix** have the
  old denominator: multiply their PT9 `repetition_factor` by 3.309 / 3.260.
- `n_states` only sizes the stem's weights, never its FLOPs. With the 3-entry `bc_only_override`
  the script assumes 67 for the parameter count.
- `restrict_train_num_*` is ignored (warning printed): windows and unique tokens use the full
  train split.
- `--exact` mirrors `mixed_dset_sampler.py:167-210` and `training.py:358-376` at the time of
  writing; if those change, the default (expected) mode is unaffected.
