import argparse
import os

import torch
from openfold.config import model_config
from openfold.model.model import AlphaFold
from openfold.utils.import_weights import import_jax_weights_


def get_model_basename(model_path: str) -> str:
    return os.path.splitext(os.path.basename(os.path.normpath(model_path)))[0]


def main(args):
    config = model_config(args.config_preset)
    model = AlphaFold(config)
    model_basename = get_model_basename(args.jax_path)
    model_version = "_".join(model_basename.split("_")[1:])
    import_jax_weights_(model, args.jax_path, version=model_version)
    torch.save(model.state_dict(),
               os.path.join(args.output_dir, f"{model_basename}.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jax_path",
                        type=str,
                        help="Path to JAX checkpoint file",
                        default="params_model_1.npz")
    parser.add_argument("--config_preset",
                        type=str,
                        help="The corresponding config preset",
                        default="model_1")
    parser.add_argument("--output_dir",
                        type=str,
                        help="Path for output directory",
                        default="output")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
