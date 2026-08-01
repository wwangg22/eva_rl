"""Convert a behavior-cloning checkpoint (train_bc.py) into an rl_games checkpoint.

Takes an existing rl_games checkpoint as a structural template (a2c_continuous,
continuous_a2c_logstd, actor_critic with separate=False, mlp [256, 128, 64] elu,
fixed_sigma, normalize_input, normalize_value) and transplants the BC trunk into
``a2c_network.actor_mlp``, the BC mu head into ``a2c_network.mu``, the BC obs
mean/var into ``running_mean_std`` (with a large count so fine-tuning barely moves
them), and sets ``a2c_network.sigma`` to log(--sigma). Value head, value_mean_std
and optimizer state are kept from the template so every key and shape matches.

.. code-block:: bash

    python scripts/scripted_expert/bc_to_rlgames.py \\
        --bc_checkpoint logs/bc/<run>/best.pt --out logs/bc/<run>/rlgames_init.pth

"""

import argparse
import math
import os

import torch

parser = argparse.ArgumentParser(description="Convert a BC checkpoint to an rl_games checkpoint.")
parser.add_argument("--bc_checkpoint", type=str, required=True, help="best.pt from train_bc.py")
parser.add_argument("--template", type=str,
                    default="logs/rl_games/rebot_pick_place/2026-07-30_17-03-34/nn/rebot_pick_place.pth",
                    help="Existing rl_games checkpoint providing the exact key structure.")
parser.add_argument("--out", type=str, required=True, help="Output .pth path.")
parser.add_argument("--sigma", type=float, default=0.1, help="Initial policy std for fine-tuning.")
parser.add_argument("--rms_count", type=float, default=1e6,
                    help="RunningMeanStd count, large so fine-tuning barely shifts the stats.")
args = parser.parse_args()


def describe(ckpt, label):
    print(f"== {label} ==")
    for k, v in ckpt.items():
        if torch.is_tensor(v):
            print(f"  {k}: tensor {tuple(v.shape)} {v.dtype}")
        else:
            print(f"  {k}: {type(v).__name__}")
    for k, v in ckpt["model"].items():
        print(f"  model.{k}: {tuple(v.shape)} {v.dtype}")


def main():
    template = torch.load(args.template, map_location="cpu", weights_only=False)
    describe(template, f"template {args.template}")
    bc = torch.load(args.bc_checkpoint, map_location="cpu", weights_only=True)

    model = template["model"]  # mutate the loaded copy in place; keys/shapes stay identical

    def put(key, value):
        old = model[key]
        value = value.to(old.dtype).reshape(old.shape)
        model[key] = value

    # actor trunk: a2c_network.actor_mlp.<i>.<weight|bias> <- BC trunk "<i>.<weight|bias>"
    transplanted = []
    for key in model:
        if key.startswith("a2c_network.actor_mlp."):
            put(key, bc["trunk"][key.removeprefix("a2c_network.actor_mlp.")])
            transplanted.append(key)
    put("a2c_network.mu.weight", bc["mu"]["weight"])
    put("a2c_network.mu.bias", bc["mu"]["bias"])
    put("a2c_network.sigma", torch.full_like(model["a2c_network.sigma"], math.log(args.sigma)))
    put("running_mean_std.running_mean", bc["obs_mean"])
    put("running_mean_std.running_var", bc["obs_var"])
    put("running_mean_std.count", torch.tensor(args.rms_count))
    transplanted += ["a2c_network.mu.weight", "a2c_network.mu.bias", "a2c_network.sigma",
                     "running_mean_std.running_mean", "running_mean_std.running_var", "running_mean_std.count"]
    # value head, value_mean_std, optimizer: kept from template
    template["epoch"] = 0
    template["frame"] = 0

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(template, args.out)

    # verify against a fresh load of the template
    ref = torch.load(args.template, map_location="cpu", weights_only=False)
    out = torch.load(args.out, map_location="cpu", weights_only=False)
    assert set(out.keys()) == set(ref.keys()), (set(out) ^ set(ref))
    assert set(out["model"].keys()) == set(ref["model"].keys())
    for k, v in ref["model"].items():
        assert out["model"][k].shape == v.shape, f"{k}: {out['model'][k].shape} vs {v.shape}"
        assert out["model"][k].dtype == v.dtype, f"{k}: {out['model'][k].dtype} vs {v.dtype}"
    print(f"PASS: {args.out} matches template keys/shapes/dtypes exactly")
    print("transplanted from BC:", ", ".join(transplanted))
    print(f"sigma = log({args.sigma}) = {math.log(args.sigma):.4f}, rms count = {args.rms_count:g}, "
          f"epoch/frame reset to 0; value head, value_mean_std and optimizer kept from template")


if __name__ == "__main__":
    main()
