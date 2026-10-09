"""Count WALRUS (IsotropicModel) parameters by component analytically from the command line.

The counting function is developed in
demo_notebooks/walrus_example_2_ArchitectureAndDimensions.ipynb but this one will prolly be
the up-to-date one.
"""

import argparse
import json
import math
import os
import sys


def walrus_param_count_analytical(
        hidden_dim=1408,
        processor_blocks=40,
        mlp_dim=None,                           # space_mixing.mlp_dim; None -> 4 * hidden_dim (FullAttention's default)
        intermediate_dim=352,
        n_states=67,
        num_heads=16,
        num_heads_space=None,                   # model.processor.space_mixing.num_heads
        num_heads_time=None,                    # model.processor.time_mixing.num_heads
        space_mixing="full_spatial_attention",  # model/processor/space_mixing: "full_spatial_attention"
        bias_type_time="rel",                   # model.processor.time_mixing.bias_type: "rel" | "rotary" | "none"
        num_buckets=32,                         # RelativePositionBias's fixed bucket count (no config exposes it)
        channel_mixing="identity",              # model/processor/channel_mixing: "identity" | "generic_mlp"
        channel_mlp_dim=None,                   # generic_mlp hidden width; None -> 4 * hidden_dim (MLP's own default)
        n_dim_keys=None,                        # number of encoder/decoder copies; None -> derived from the two below
        override_dimensionality=0,              # model.override_dimensionality (yaml default 3; upstream pretraining 0)
        include_d=(2, 3),                       # model.include_d
        learnable_rope=False,                   # finetuning_mods.learnable_rope (finetuning_mods=all sets True)
        rope_per_axis=False,                    # finetuning_mods.rope_per_axis (finetuning_mods=all sets True)
        ape_shape=None,                         # finetuning_mods.ape_shape, e.g. (32, 32, 1); None -> no APE
    ):
    """
    Analytical parameter count for IsotropicModel checked against instantiated models for every option below.

        space_mixing   = full_spatial_attention   FullAttention (spatial_blocks/full_attention.py)
                         axial_spatial_attention  -> NotImplementedError: its _target_
                                                  (spatial_blocks/axial_attention.py) does not exist
        time_mixing    = axial_time_attention     AxialTimeAttention, the only temporal block.
                         bias_type "rel":         + RelativePositionBias = Embedding(num_buckets, heads)
                         bias_type "rotary"/"none": no extra parameters (RoPE frequencies are buffers)
        channel_mixing = identity                 0 params (default; every run uses it)
                         generic_mlp              MLP: Linear(H, m) + Linear(m, H). NOTE: the shipped
                                                  config passes `exp_factor`, which MLP.__init__ rejects,
                                                  so it does not build as-is; and SpaceTimeSplitBlock
                                                  applies it to a channels-first tensor, i.e. to the
                                                  wrong axis. Counted for completeness only.
        encoder/decoder = spacebag_and_vstride_encoder / vstride_decoder only (the other
                         encoder/decoder configs are not covered).
        norm_layer     = RMSGroupNorm

    FullAttention's `bias_type` argument does not change the count: its constructor
    hard-codes the RoPE branch (`if False and ...` / `elif True or ...`), so the
    RelativePositionBias branch there is dead code and space position information always
    comes from RoPE buffers.

    The following are not arguments because they do not move the count:
        groups         - only how RMSGroupNorm pools channels. It must divide hidden_dim and
                         intermediate_dim (e.g. groups=12 fails for intermediate_dim=352)
        projection_dim - handed to the patch jitterer as stage_dim ('# Not used'); 0 params.
        causal_in_time - only flips RelativePositionBias's `bidirectional`; same table size.
        weight_tied_axes, max_d (FullAttention) is stored but never parameterised.

    Encoder/decoder copies: override_dimensionality > 0 builds one pair; 0 builds one per
    include_d entry. The yaml default (3) gives 1; upstream pretraining (0) gives 2.
    
    Finetuning-only additions (IsotropicModel.add_ft_options, called from train.py with
    cfg.finetuning_mods; finetuning_mods=defaults adds none):
        ape_shape      - APE parameter (1, 1, H, *ape_shape)
        learnable_rope - each block's space-RoPE `freqs` buffer becomes a Parameter of length
                         (H // heads_space // 4) // 2, three copies if rope_per_axis. The time
                         block gains nothing with bias_type "rel"/"none"; with "rotary" the code
                         raises (its RotaryEmbedding has no make_learnable).
    """
    H = hidden_dim
    M = mlp_dim if mlp_dim is not None else H * 4
    I = intermediate_dim
    nh_s = num_heads_space if num_heads_space is not None else num_heads
    nh_t = num_heads_time if num_heads_time is not None else num_heads

    if space_mixing != "full_spatial_attention":
        raise NotImplementedError(
            "Only full_spatial_attention exists; axial_spatial_attention's target "
            "(walrus/models/spatial_blocks/axial_attention.py) is missing from the repo."
        )
    if bias_type_time not in ("rel", "rotary", "none"):
        raise ValueError(f"bias_type_time must be 'rel', 'rotary' or 'none', got {bias_type_time!r}")
    if channel_mixing not in ("identity", "generic_mlp"):
        raise ValueError(f"channel_mixing must be 'identity' or 'generic_mlp', got {channel_mixing!r}")
    if M % 2:
        raise ValueError(f"mlp_dim must be even (SwiGLU halves it), got {M}")
    if H % nh_s or H % nh_t:
        raise ValueError(f"hidden_dim {H} must be divisible by both head counts ({nh_s}, {nh_t})")
    if learnable_rope and bias_type_time == "rotary":
        raise ValueError("learnable_rope with time bias_type='rotary' crashes in the code "
                         "(position_biases.RotaryEmbedding has no make_learnable)")
    if n_dim_keys is None:
        n_dim_keys = 1 if override_dimensionality > 0 else len(include_d)

    # =================================================================
    # processor: processor_blocks x SpaceTimeSplitBlock
    #   each block = space_mixing (FullAttention)
    #              + time_mixing  (AxialTimeAttention)
    #              + channel_mixing (Identity or MLP)
    # =================================================================

    # --- space_mixing: FullAttention ---
    # Parallel attention + MLP block over the flattened token grid, per timestep
    # (SpaceTimeSplitBlock folds T into the batch dim before calling it).
    space  = H                              # norm1: RMSGroupNorm(nh_s, H) affine weight (H,) only
    space += H * (M + 3 * H) + (M + 3 * H)  # fused_ff_qkv: Linear(H, M + 3H) + bias. One matmul emits
                                            #   [ff | q | k | v] (widths M, H, H, H): the MLP and
                                            #   attention branches share the pre-norm input.
    space += (M // 2) * H + H               # ff_out: Linear(M//2, H) + bias. SwiGLU halves M first.
    space += H * H                          # attn_out: Linear(H, H, bias=False)
    space += 4 * (H // nh_s)                # q_norm + k_norm: LayerNorm(H//nh_s), weight + bias each
    # zero-param: RoPE (freqs are buffers), SDPA, SwiGLU, drop_path. bias_type is ignored here.
    if learnable_rope:
        rope_dim = H // nh_s // 4           # RotaryEmbedding(H // heads // 4, freqs_for="pixel")
        space += (rope_dim // 2) * (3 if rope_per_axis else 1)  # freqs buffer -> Parameter(s)

    # --- time_mixing: AxialTimeAttention ---
    # Attends along T only, independently per spatial location. No MLP branch.
    time  = H                    # norm1: RMSGroupNorm(nh_t, H), weight (H,) only
    time += 3 * H * H + 3 * H    # input_head: 1x1 Conv3d(H, 3H) + bias == fused q/k/v Linear
    time += H * H + H            # output_head: 1x1 Conv3d(H, H) + bias
    time += 4 * (H // nh_t)      # qnorm + knorm: LayerNorm(H//nh_t), weight + bias each
    if bias_type_time == "rel":
        time += num_buckets * nh_t  # RelativePositionBias: Embedding(num_buckets, nh_t), one learned
                                    #   scalar per (relative-time bucket, head) added to the T x T logits.
                                    #   causal_in_time only makes it one-directional; same size.
    # "rotary": RotaryEmbedding with an inv_freq buffer; "none": nothing. Both 0 params.

    # --- channel_mixing ---
    if channel_mixing == "generic_mlp":
        m = channel_mlp_dim if channel_mlp_dim is not None else H * 4
        channel = H * m + m + m * H + H     # fc1: Linear(H, m) + bias; fc2: Linear(m, H) + bias
    else:
        channel = 0                         # nn.Identity

    processor = (space + time + channel) * processor_blocks

    # =================================================================
    # encoder / decoder: one structurally identical copy per dimension key.
    #   Every copy is a Conv3d stack (2-D data is padded with a singleton axis), so the
    #   kernel volumes are always 8^3 = 512 (patchify) and 4^3 = 64, from
    #   base_kernel_size3d = [[8,4],[8,4],[8,4]].
    # =================================================================

    # --- encoder: SpaceBagAdaptiveDVstrideEncoder, n_states -> I -> H ---
    enc_per_key = (
        I * n_states * 512     # proj1.weight (I, n_states, 8,8,8), bias unused/absent. Stored over the
                               #   whole n_states field map; each batch gathers proj1.weight[:, field_indices].
        + I                    # norm1.weight
        + H * I * 64           # proj2.weight (H, I, 4,4,4), bias=False
        + H                    # norm2.weight
    )
    encoder = enc_per_key * n_dim_keys

    # --- decoder: AdaptiveDVstrideDecoder, H -> I -> n_states ---
    dec_per_key = (
        H * I * 64             # proj1.weight (H, I, 4,4,4), bias=False
        + I                    # norm1.weight
        + I * n_states * 512   # proj2.weight (I, n_states, 8,8,8)
        + n_states             # proj2.bias: the only conv bias in encoder or decoder
    )
    decoder = dec_per_key * n_dim_keys

    # --- outside encoder/decoder/processor ---
    other = 1                  # IsotropicModel.encoder_dummy = nn.Parameter(torch.ones(1)), a grad-
                               #   checkpointing workaround multiplied into the encoder input
    if ape_shape is not None:
        other += H * math.prod(ape_shape)   # APE: Parameter (1, 1, H, *ape_shape)

    total = encoder + processor + decoder + other
    return {"encoder": encoder, "processor": processor, "decoder": decoder,
            "other": other, "total": total}


# Shape presets for the reference models' --hidden-dim/--processor-blocks/--mlp-dim
# flags override them. Everything else (intermediate_dim=352, 16 heads, ...) is shared.
PRESETS = {
    "walrus":     {"hidden_dim": 1408, "processor_blocks": 40, "mlp_dim": 4 * 1408},  # paper fullDims
    "halfwalrus": {"hidden_dim": 1088, "processor_blocks": 30, "mlp_dim": 4 * 1088},
    "200m":       {"hidden_dim": 768,  "processor_blocks": 12, "mlp_dim": 2048},      # 200M anchor
}

# Path to walrus/configs, for --verify option
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "walrus", "configs")


def build_parser():
    parser = argparse.ArgumentParser(
        prog="count_WALRUS_params.py",
        description=(
            "Analytical parameter count of WALRUS's IsotropicModel, split into encoder,\n"
            "processor, decoder and other. Defaults are the paper-scale model (fullDims:\n"
            "1408/40/5632, override_dimensionality=0, n_states=67), i.e. the function's own\n"
            "defaults. Each flag names the Hydra key it mirrors."
        ),
        epilog=(
            "examples:\n"
            "  python count_WALRUS_params.py                         # paper WALRUS, ~1.286 B\n"
            "  python count_WALRUS_params.py --preset 200m           # 200M anchor\n"
            "  python count_WALRUS_params.py --preset 200m --override-dimensionality 3\n"
            "  python count_WALRUS_params.py -H 464 -L 10 -M 1120 --json\n"
            "  python count_WALRUS_params.py --preset halfwalrus --verify"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    shape = parser.add_argument_group("shape (the scaling axes)")
    shape.add_argument("--preset", choices=sorted(PRESETS), default=None,
                       help="start from a reference model's hidden_dim/processor_blocks/mlp_dim "
                            "(default: none, i.e. the paper-scale defaults below)")
    shape.add_argument("-H", "--hidden-dim", type=int, default=None,
                       help="model.hidden_dim, the token width (default: 1408)")
    shape.add_argument("-L", "--processor-blocks", type=int, default=None,
                       help="model.processor_blocks, the processor depth (default: 40)")
    shape.add_argument("-M", "--mlp-dim", type=int, default=None,
                       help="model.processor.space_mixing.mlp_dim, the FullAttention MLP width; "
                            "must be even (as SwiGLU halves it) (default: 4 * hidden_dim)")

    model = parser.add_argument_group("other model settings")
    model.add_argument("--intermediate-dim", type=int, default=352,
                       help="model.intermediate_dim, the encoder/decoder's middle width "
                            "(default: %(default)s)")
    model.add_argument("--n-states", type=int, default=67,
                       help="number of field-embedding rows; 67 with "
                            "data/field_index_map_override=full_well_field_index (default: %(default)s)")
    model.add_argument("--num-heads", type=int, default=16,
                       help="head count for both space and time mixing, unless overridden "
                            "below; must divide hidden_dim (default: %(default)s)")
    model.add_argument("--num-heads-space", type=int, default=None,
                       help="model.processor.space_mixing.num_heads (default: --num-heads)")
    model.add_argument("--num-heads-time", type=int, default=None,
                       help="model.processor.time_mixing.num_heads (default: --num-heads)")
    model.add_argument("--bias-type-time", choices=["rel", "rotary", "none"], default="rel",
                       help="model.processor.time_mixing.bias_type; 'rel' adds a "
                            "num_buckets x heads table per block (default: %(default)s)")
    model.add_argument("--channel-mixing", choices=["identity", "generic_mlp"], default="identity",
                       help="model/processor/channel_mixing; generic_mlp does not build as "
                            "shipped and is counted for completeness only (default: %(default)s)")
    model.add_argument("--channel-mlp-dim", type=int, default=None,
                       help="generic_mlp hidden width (default: 4 * hidden_dim)")

    stems = parser.add_argument_group("encoder/decoder copies")
    stems.add_argument("--override-dimensionality", type=int, choices=[0, 2, 3], default=0,
                       help="model.override_dimensionality: 0 builds one encoder/decoder pair per include_d entry."
                            ">0 builds one pair (=3 is the yaml default)"
                            "(=2 builds but crashes at the first forward pass) (default: %(default)s)")
    stems.add_argument("--include-d", type=int, nargs="+", choices=[2, 3], default=[2, 3],
                       help="model.include_d; only matters with --override-dimensionality 0 "
                            "(default: 2 3)")

    ft = parser.add_argument_group("finetuning-only additions (finetuning_mods)")
    ft.add_argument("--learnable-rope", action="store_true",
                    help="finetuning_mods.learnable_rope: space-RoPE frequencies become parameters")
    ft.add_argument("--rope-per-axis", action="store_true",
                    help="finetuning_mods.rope_per_axis: three RoPE frequency sets per block")
    ft.add_argument("--ape-shape", type=int, nargs="+", default=None, metavar="N",
                    help="finetuning_mods.ape_shape, e.g. 32 32 1: adds an absolute position "
                         "embedding of hidden_dim x prod(N) (default: none)")

    out = parser.add_argument_group("output")
    out.add_argument("--json", action="store_true",
                     help="print the counts (and resolved shape) as JSON instead of a table")
    out.add_argument("--verify", action="store_true",
                     help="also instantiate the real model (needs the walrus environment)"
                          "and check the closed form. Exit code 1 on mismatch")
    out.add_argument("--groups", type=int, default=16,
                     help="model.groups, used only by --verify. Does not change the count, "
                          "but must divide hidden_dim and intermediate_dim to build "
                          "(default: %(default)s)")
    return parser


def resolve_shape(args):
    """hidden_dim / processor_blocks / mlp_dim: explicit flag > preset > function default."""
    preset = PRESETS.get(args.preset, {})
    hidden_dim = args.hidden_dim or preset.get("hidden_dim", 1408)
    processor_blocks = args.processor_blocks or preset.get("processor_blocks", 40)
    mlp_dim = args.mlp_dim or preset.get("mlp_dim")
    if args.hidden_dim and not args.mlp_dim:
        mlp_dim = None  # a new hidden_dim without an mlp_dim means 4 * hidden_dim, not the preset's
    return hidden_dim, processor_blocks, mlp_dim


def instantiated_counts(kw, groups):
    """Build IsotropicModel from walrus/configs on the meta device and count by component."""
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    overrides = [
        "server=leonardo",
        "model=isotropic_model",
        f"model.hidden_dim={kw['hidden_dim']}",
        f"model.processor_blocks={kw['processor_blocks']}",
        f"model.processor.space_mixing.mlp_dim={kw['mlp_dim'] or 4 * kw['hidden_dim']}",
        f"model.intermediate_dim={kw['intermediate_dim']}",
        f"model.groups={groups}",
        f"model.override_dimensionality={kw['override_dimensionality']}",
        f"model.include_d=[{','.join(map(str, kw['include_d']))}]",
        "model/processor/space_mixing=full_spatial_attention",
        f"model.processor.space_mixing.num_heads={kw['num_heads_space'] or kw['num_heads']}",
        f"model.processor.time_mixing.num_heads={kw['num_heads_time'] or kw['num_heads']}",
        f"model.processor.time_mixing.bias_type={kw['bias_type_time']}",
    ]
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config", overrides=overrides)
    with torch.device("meta"):
        model = instantiate(cfg.model, n_states=kw["n_states"])

    def count(modules):
        return sum(p.numel() for m in modules for p in m.parameters())

    counts = {
        "encoder": count(model.embed.values()),
        "processor": count(model.blocks),
        "decoder": count(model.debed.values()),
        "total": count([model]),
    }
    counts["other"] = counts["total"] - counts["encoder"] - counts["processor"] - counts["decoder"]
    return counts


def fmt(n):
    return f"{n / 1e9:.3f} B" if n >= 1e9 else f"{n / 1e6:.2f} M"


def main():
    parser = build_parser()
    args = parser.parse_args()

    hidden_dim, processor_blocks, mlp_dim = resolve_shape(args)
    kw = dict(
        hidden_dim=hidden_dim,
        processor_blocks=processor_blocks,
        mlp_dim=mlp_dim,
        intermediate_dim=args.intermediate_dim,
        n_states=args.n_states,
        num_heads=args.num_heads,
        num_heads_space=args.num_heads_space,
        num_heads_time=args.num_heads_time,
        bias_type_time=args.bias_type_time,
        channel_mixing=args.channel_mixing,
        channel_mlp_dim=args.channel_mlp_dim,
        override_dimensionality=args.override_dimensionality,
        include_d=tuple(args.include_d),
        learnable_rope=args.learnable_rope,
        rope_per_axis=args.rope_per_axis,
        ape_shape=tuple(args.ape_shape) if args.ape_shape else None,
    )
    try:
        counts = walrus_param_count_analytical(**kw)
    except (ValueError, NotImplementedError) as e:
        parser.error(str(e))

    shape = {"hidden_dim": hidden_dim, "processor_blocks": processor_blocks,
             "mlp_dim": mlp_dim or 4 * hidden_dim}
    if args.json:
        print(json.dumps({"shape": shape, "counts": counts}, indent=2))
    else:
        n_keys = 1 if args.override_dimensionality > 0 else len(args.include_d)
        print(f"shape {shape['hidden_dim']}/{shape['processor_blocks']}/{shape['mlp_dim']} "
              f"(hidden_dim/processor_blocks/mlp_dim), intermediate_dim={args.intermediate_dim}, "
              f"n_states={args.n_states}, {n_keys} encoder/decoder pair(s)")
        for name, n in counts.items():
            share = "" if name == "total" else f"  ({n / counts['total']:6.2%})"
            print(f"  {name:>10s}: {n:>15,d}  = {fmt(n):>10s}{share}")

    if args.verify:
        if args.channel_mixing != "identity" or args.learnable_rope or args.ape_shape:
            parser.error("--verify covers the base model only: generic_mlp does not build as "
                         "shipped, and the finetuning additions need add_ft_options")
        built = instantiated_counts(kw, args.groups)
        bad = {k: (counts[k], built[k]) for k in built if counts[k] != built[k]}
        if bad:
            for k, (a, b) in bad.items():
                print(f"VERIFY MISMATCH {k}: closed form {a:,d} vs instantiated {b:,d}")
            sys.exit(1)
        print("verify: closed form matches the instantiated model exactly "
              "(encoder, processor, decoder, other, total)")


if __name__ == "__main__":
    main()