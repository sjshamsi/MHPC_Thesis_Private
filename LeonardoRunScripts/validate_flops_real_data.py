"""Validate walrus/utils/flops.py against FlopCounterMode on the real Well datasets.

SCALING_STRATEGY.md 5.1. tests/test_flops.py covers the same closed form in CI, but only
against the tiny dummy dataset; this checks it against every dataset the study actually
trains on, where the things that vary -- per-dataset field counts, boundary conditions,
token grids and strides -- are real. Run it after any change to the encoder, the decoder,
or the patch jitterer.

Runs on a login node in about a minute, no GPU and no SLURM job needed. It compares only
the encoder and decoder: those are convolutions, which are registered ops on a CPU build,
whereas FullAttention's spatial attention dispatches to a CPU SDPA kernel with no
registered formula and is silently counted as zero (walrus_flops_analysis.ipynb 4). The
processor terms are checked on GPU by flop_counter_backend_test.py instead.

Usage:
    module purge; module load python; module load cuda/12.2
    source /leonardo_work/ICT26_MHPC_0/sshamsi/pyenvs/env1/bin/activate
    python LeonardoRunScripts/validate_flops_real_data.py

Exit code 0 = every dataset exact, 1 = a mismatch, 2 = no dataset could be loaded.
"""

import sys

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.flop_counter import FlopCounterMode

from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus.train import CONFIG_DIR, CONFIG_NAME
from walrus.utils.flops import (
    FlopAccountant,
    analytic_token_shape,
    forward_flops,
)

# Every dataset in Leonardo_smallest7_2d, the study's training mixture.
DATASETS = [
    "Leonardo_active_matter",
    "Leonardo_helmholtz_staircase",
    "Leonardo_viscoelastic_instability",
    "Leonardo_gray_scott_reaction_diffusion",
    "Leonardo_acoustic_scattering_discontinuous",
    "Leonardo_turbulent_radiative_layer_2D",
    "Leonardo_planetswe",
]


def load(data_cfg_name, hidden_dim=768, intermediate_dim=352, blocks=2):
    overrides = [
        "server=leonardo",
        f"data={data_cfg_name}",
        "data/field_index_map_override=full_well_field_index",
        "logger=none",
        "trainer=debug",
        "model=isotropic_model",
        f"model.hidden_dim={hidden_dim}",
        f"model.intermediate_dim={intermediate_dim}",
        f"model.processor_blocks={blocks}",
        "model.groups=16",
        "model.override_dimensionality=0",
        "model/processor/space_mixing=full_spatial_attention",
        "data.module_parameters.n_steps_input=4",
        "data.module_parameters.batch_size=1",
        "data_workers=1",
        "name=flops_validation",
        "automatic_setup=False",
    ]
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=CONFIG_NAME, overrides=overrides)
        dm = instantiate(
            cfg.data.module_parameters,
            world_size=1,
            rank=0,
            data_workers=cfg.data_workers,
            well_base_path=cfg.data.well_base_path,
            field_index_map_override={},
            transform=None,
        )
        n_states = max(dm.train_dataset.field_to_index_map.values()) + 1
        model = instantiate(cfg.model, n_states=n_states)
        model.eval()
        fmt = ChannelsFirstWithTimeFormatter()
        batch = dm.train_dataset[0]
        inputs, _ = fmt.process_input(
            batch,
            causal_in_time=cfg.model.causal_in_time,
            predict_delta=cfg.trainer.prediction_type == "delta",
            train=True,
        )
    return model, inputs, batch["metadata"], dm


def measured_by_module(counter):
    """Sum FlopCounterMode's per-module counts for the encoder and decoder subtrees."""
    flat = counter.get_flop_counts()
    enc = dec = proc = 0
    for module, ops in flat.items():
        if module == "Global":
            continue
        total = sum(ops.values())
        # Only count top-level subtree roots to avoid double counting children.
        parts = module.split(".")
        if len(parts) == 3 and parts[1] == "embed":
            enc += total
        elif len(parts) == 3 and parts[1] == "debed":
            dec += total
        elif len(parts) == 3 and parts[1] == "blocks":
            proc += total
    return enc, dec, proc


rows = []
for name in DATASETS:
    try:
        model, inputs, metadata, dm = load(name)
    except Exception as e:
        print(f"SKIP {name}: {type(e).__name__}: {e}")
        continue

    accountant = FlopAccountant(model, world_size=1)
    x, state_labels, bcs = inputs[0], inputs[1], inputs[2]

    counter = FlopCounterMode(model, depth=3, display=False)
    with counter, torch.no_grad():
        model(x, state_labels, bcs.tolist(), metadata=metadata)

    geom = accountant._observed
    fwd = forward_flops(accountant.shape, geom)
    enc_m, dec_m, proc_m = measured_by_module(counter)

    # Compare at the rank the model actually works in: IsotropicModel.forward dim_pads
    # every input to max_d=3 spatial axes, so a natively-2D dataset gains a singleton.
    native_res = tuple(int(s) for s in x.shape[3:])
    native_res = native_res + (1,) * (len(geom.encoder_in_shape) - len(native_res))
    predicted_tokens = analytic_token_shape(native_res, geom.periodic)
    rows.append(
        dict(
            name=name.replace("Leonardo_", ""),
            res=tuple(int(s) for s in x.shape[3:]),
            periodic=geom.periodic,
            fields_in=geom.n_fields_in,
            fields_out=geom.n_fields_out,
            tokens_observed=geom.token_shape,
            tokens_predicted=predicted_tokens,
            enc_hand=fwd["encoder"],
            enc_meas=enc_m,
            dec_hand=fwd["decoder"],
            dec_meas=dec_m,
        )
    )
    del model, accountant

print()
hdr = f"{'dataset':<34}{'resolution':>12}{'fields':>8}{'tokens obs/pred':>18}{'encoder':>12}{'decoder':>12}"
print(hdr)
print("-" * len(hdr))
ok = True
for r in rows:
    enc_ratio = r["enc_hand"] / r["enc_meas"] if r["enc_meas"] else float("nan")
    dec_ratio = r["dec_hand"] / r["dec_meas"] if r["dec_meas"] else float("nan")
    tok_ok = "==" if r["tokens_observed"] == r["tokens_predicted"] else "!="
    enc_s = "exact" if r["enc_hand"] == r["enc_meas"] else f"{enc_ratio:.4f}x"
    dec_s = "exact" if r["dec_hand"] == r["dec_meas"] else f"{dec_ratio:.4f}x"
    ok &= r["enc_hand"] == r["enc_meas"] and r["dec_hand"] == r["dec_meas"]
    ok &= r["tokens_observed"] == r["tokens_predicted"]
    print(
        f"{r['name']:<34}{str(r['res']):>12}{r['fields_out']:>8}"
        f"{str(r['tokens_observed'][:2]) + tok_ok:>18}{enc_s:>12}{dec_s:>12}"
    )

print()
for r in rows:
    print(
        f"{r['name']:<34} periodic={r['periodic']} "
        f"enc {r['enc_hand']:>16,} vs {r['enc_meas']:>16,} | "
        f"dec {r['dec_hand']:>16,} vs {r['dec_meas']:>16,}"
    )

if not rows:
    print("\nRESULT: NO DATASETS LOADED - nothing was validated")
    sys.exit(2)
for r in rows:
    if r["tokens_observed"] != r["tokens_predicted"]:
        print(f"TOKEN RULE MISMATCH {r['name']}: res={r['res']} periodic={r['periodic']} "
              f"observed={r['tokens_observed']} predicted={r['tokens_predicted']}")
print("\nRESULT:", "ALL EXACT" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
