# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
# Copyright 2025 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import csv
import csv
import logging
import math
import os
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
from openfold.config import model_config
from openfold.data import data_pipeline, feature_pipeline, templates
from openfold.np import protein
from openfold.utils.script_utils import (parse_fasta, prep_output,
                                         relax_protein, update_timings)
from openfold.utils.tensor_utils import tensor_tree_map
from scripts.utils import add_data_args

from tensorrt_bionemo.models.helper import AcceleratedConfig
from tensorrt_bionemo.models.openfold2 import OpenFold2
from tensorrt_bionemo.runtime import OnDemandContextMemoryManager

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

# Gives a large speedup on Ampere-class GPUs
torch.set_float32_matmul_precision("high")

torch.set_grad_enabled(False)

TRACING_INTERVAL = 50


def precompute_alignments(tags, seqs, alignment_dir, args):
    for tag, seq in zip(tags, seqs):
        tmp_fasta_path = os.path.join(args.output_dir,
                                      f"tmp_{os.getpid()}.fasta")
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        os.path.join(alignment_dir, tag)

        logger.info(
            f"Using precomputed alignments for {tag} at {alignment_dir}...")

        # Remove temporary FASTA file
        os.remove(tmp_fasta_path)


def round_up_seqlen(seqlen):
    return int(math.ceil(seqlen / TRACING_INTERVAL)) * TRACING_INTERVAL


def generate_feature_dict(
    tags,
    seqs,
    alignment_dir,
    data_processor,
    args,
):
    tmp_fasta_path = os.path.join(args.output_dir, f"tmp_{os.getpid()}.fasta")

    if "multimer" in args.model_name:
        with open(tmp_fasta_path, "w") as fp:
            fp.write('\n'.join(
                [f">{tag}\n{seq}" for tag, seq in zip(tags, seqs)]))
        feature_dict = data_processor.process_fasta(
            fasta_path=tmp_fasta_path,
            alignment_dir=alignment_dir,
        )
    elif len(seqs) == 1:
        tag = tags[0]
        seq = seqs[0]
        with open(tmp_fasta_path, "w") as fp:
            fp.write(f">{tag}\n{seq}")

        local_alignment_dir = os.path.join(alignment_dir, tag)
        feature_dict = data_processor.process_fasta(
            fasta_path=tmp_fasta_path,
            alignment_dir=local_alignment_dir,
            seqemb_mode=False,
        )
    else:
        with open(tmp_fasta_path, "w") as fp:
            fp.write('\n'.join(
                [f">{tag}\n{seq}" for tag, seq in zip(tags, seqs)]))
        feature_dict = data_processor.process_multiseq_fasta(
            fasta_path=tmp_fasta_path,
            super_alignment_dir=alignment_dir,
        )

    # Remove temporary FASTA file
    os.remove(tmp_fasta_path)

    return feature_dict


def list_files_with_extensions(dir, extensions):
    return [f for f in os.listdir(dir) if f.endswith(extensions)]


def run_model_opt(model, batch, tag, output_dir, dtype=torch.float32):
    with torch.no_grad():
        logger.info(f"Running inference for {tag}...")
        t = time.perf_counter()
        cast_batch = {}
        for k, v in batch.items():
            if v.is_floating_point():
                cast_batch[k] = v.to(dtype)
            else:
                cast_batch[k] = v
        out = model(cast_batch)
        torch.cuda.synchronize()
        inference_time = time.perf_counter() - t
        logger.info(f"Inference time: {inference_time}")
        update_timings({tag: {
            "inference": inference_time
        }}, os.path.join(output_dir, "timings.json"))
        cast_out = {}
        for k, v in out.items():
            if v.is_floating_point():
                cast_out[k] = v.to(torch.float32)
            else:
                cast_out[k] = v

    return cast_out


class NeedFallbackEvoformer:

    def __init__(self, threshold: int = 1536):
        self.threshold = threshold

    def __call__(self, m: torch.Tensor, **kwargs) -> bool:
        n_res = m.shape[-2]
        if n_res > self.threshold:
            return True
        return False


def create_model_opt(model_name: str,
                     evoformer_backend: str,
                     evoformer_ckpt: str,
                     evoformer_fallback_threshold: int = 1536,
                     dont_skip_template_pair_stack: bool = False) -> OpenFold2:
    manager = OnDemandContextMemoryManager()

    model = OpenFold2(model_name=model_name)
    model.cuda()
    model.eval()

    if model.config.is_multimer:
        # For multimer, if no template available, we should skip the template pair stack to improve the performance and accuracy.
        model.config.skip_template_pair_stack = not dont_skip_template_pair_stack

    acc_m = {
        "evoformer":
        AcceleratedConfig(
            checkpoint=evoformer_ckpt,
            backend=evoformer_backend,
            need_fallback=NeedFallbackEvoformer(evoformer_fallback_threshold),
        )
    }
    model = model.optimize(acc_m, manager)
    return model


def model_name_to_config_preset(model_name: str) -> str:
    config_preset = None
    if model_name == "alphafold2_1" or model_name == "openfold2_finetuning" in model_name:
        config_preset = "model_1"
    elif model_name == "alphafold2_2":
        config_preset = "model_2"
    elif model_name == "alphafold2_3":
        config_preset = "model_3"
    elif model_name == "alphafold2_4":
        config_preset = "model_4"
    elif model_name == "alphafold2_5":
        config_preset = "model_5"
    elif model_name == "openfold2_ptm1":
        config_preset = "model_1_ptm"
    elif model_name == "openfold2_ptm2":
        config_preset = "model_2_ptm"
    elif model_name in [
            "alphafold2_multimer_1", "alphafold2_multimer_2",
            "alphafold2_multimer_3"
    ]:
        i = model_name.split("_")[-1]
        config_preset = f"model_{i}_multimer_v3"
    elif model_name in ["alphafold2_multimer_4"]:
        config_preset = "model_4_multimer_v3"
    elif model_name in ["alphafold2_multimer_5"]:
        config_preset = "model_5_multimer_v3"
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return model_config(config_preset), config_preset


def main(args):
    # Create the output directory
    os.makedirs(args.output_dir, exist_ok=True)
    config, config_preset = model_name_to_config_preset(args.model_name)
    is_multimer = "multimer" in args.model_name
    if is_multimer and args.max_recycling_iters is not None:
        config.data.common.max_recycling_iters = args.max_recycling_iters
    is_custom_template = "use_custom_template" in args and args.use_custom_template
    if is_custom_template:
        template_featurizer = templates.CustomHitFeaturizer(
            mmcif_dir=args.template_mmcif_dir,
            max_template_date="9999-12-31",  # just dummy, not used
            max_hits=-1,  # just dummy, not used
            kalign_binary_path=args.kalign_binary_path)
    elif is_multimer:
        template_featurizer = templates.HmmsearchHitFeaturizer(
            mmcif_dir=args.template_mmcif_dir,
            max_template_date=args.max_template_date,
            max_hits=-1,
            kalign_binary_path=args.kalign_binary_path,
            release_dates_path=args.release_dates_path,
            obsolete_pdbs_path=args.obsolete_pdbs_path)
    else:
        template_featurizer = templates.HhsearchHitFeaturizer(
            mmcif_dir=args.template_mmcif_dir,
            max_template_date=args.max_template_date,
            max_hits=-1,
            kalign_binary_path=args.kalign_binary_path,
            release_dates_path=args.release_dates_path,
            obsolete_pdbs_path=args.obsolete_pdbs_path)
    data_processor = data_pipeline.DataPipeline(
        template_featurizer=template_featurizer, )
    if is_multimer:
        data_processor = data_pipeline.DataPipelineMultimer(
            monomer_data_pipeline=data_processor, )

    output_dir_base = args.output_dir
    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(2**32)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)

    feature_processor = feature_pipeline.FeaturePipeline(config.data)
    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)
    if args.use_precomputed_alignments is None:
        alignment_dir = os.path.join(output_dir_base, "alignments")
    else:
        alignment_dir = args.use_precomputed_alignments

    tag_list = []
    seq_list = []
    for fasta_file in list_files_with_extensions(args.fasta_dir,
                                                 (".fasta", ".fa")):
        # Gather input sequences
        fasta_path = os.path.join(args.fasta_dir, fasta_file)
        with open(fasta_path, "r") as fp:
            data = fp.read()

        tags, seqs = parse_fasta(data)

        if not is_multimer and len(tags) != 1:
            print(f"{fasta_path} contains more than one sequence but "
                  f"multimer mode is not enabled. Skipping...")
            continue

        # assert len(tags) == len(set(tags)), "All FASTA tags must be unique"
        tag = '-'.join(tags)

        tag_list.append((tag, tags))
        seq_list.append(seqs)

    seq_sort_fn = lambda target: sum([len(s) for s in target[1]])
    sorted_targets = sorted(zip(tag_list, seq_list), key=seq_sort_fn)
    feature_dicts = {}

    model_opt = create_model_opt(args.model_name, args.evoformer_backend,
                                 args.evoformer_ckpt,
                                 args.evoformer_fallback_threshold,
                                 args.dont_skip_template_pair_stack)

    # Initialize timing records list
    timing_records = []

    for (tag, tags), seqs in sorted_targets:
        print(f"Processing {tag}...")
        output_name = f'{tag}_{config_preset}'
        if args.output_postfix is not None:
            output_name = f'{output_name}_{args.output_postfix}'
        # Timing: Feature preparation
        t_prep_start = time.perf_counter()
        # Timing: Feature preparation
        t_prep_start = time.perf_counter()
        # Does nothing if the alignments have already been computed
        try:
            precompute_alignments(tags, seqs, alignment_dir, args)
            feature_dict = feature_dicts.get(tag, None)
            if feature_dict is None:
                feature_dict = generate_feature_dict(
                    tags,
                    seqs,
                    alignment_dir,
                    data_processor,
                    args,
                )

                feature_dicts[tag] = feature_dict
        except Exception as e:
            logger.error(f"Error processing {tag}: {e}")
            continue


        processed_feature_dict = feature_processor.process_features(
            feature_dict, mode='predict', is_multimer=is_multimer)

        processed_feature_dict = {
            k: torch.as_tensor(v, device="cuda")
            for k, v in processed_feature_dict.items()
        }
        t_prep_end = time.perf_counter()
        prep_time = t_prep_end - t_prep_start
        logger.info(f"Feature preparation time: {prep_time:.4f}s")

        # Timing: Prediction
        t_predict_start = time.perf_counter()
        out = run_model_opt(model_opt, processed_feature_dict, tag,
                            args.output_dir)
        t_predict_end = time.perf_counter()
        predict_time = t_predict_end - t_predict_start

        # Timing: Write output PDB
        t_write_start = time.perf_counter()
        # Toss out the recycling dimensions --- we don't need them anymore
        processed_feature_dict = tensor_tree_map(
            lambda x: np.array(x[..., -1].cpu()), processed_feature_dict)
        out = tensor_tree_map(lambda x: np.array(x.cpu()), out)

        unrelaxed_protein = prep_output(out, processed_feature_dict,
                                        feature_dict, feature_processor,
                                        config_preset, args.multimer_ri_gap,
                                        args.subtract_plddt)

        unrelaxed_file_suffix = "_unrelaxed.pdb"
        if args.cif_output:
            unrelaxed_file_suffix = "_unrelaxed.cif"
        unrelaxed_output_path = os.path.join(
            args.output_dir, f'{output_name}{unrelaxed_file_suffix}')

        # Timing: Write output PDB
        t_write_start = time.perf_counter()
        with open(unrelaxed_output_path, 'w') as fp:
            if args.cif_output:
                fp.write(protein.to_modelcif(unrelaxed_protein))
            else:
                fp.write(protein.to_pdb(unrelaxed_protein))
        t_write_end = time.perf_counter()
        write_time = t_write_end - t_write_start
        logger.info(f"PDB write time: {write_time:.4f}s")
        t_write_end = time.perf_counter()
        write_time = t_write_end - t_write_start
        logger.info(f"PDB write time: {write_time:.4f}s")

        logger.info(f"Output written to {unrelaxed_output_path}...")

        # Record timings for this sample
        timing_records.append({
            'tag':
            tag,
            'prep_features_time':
            prep_time,
            'predict_time':
            predict_time,
            'write_pdb_time':
            write_time,
            'total_time':
            prep_time + predict_time + write_time
        })

        if not args.skip_relaxation:
            # Relax the prediction.
            logger.info(f"Running relaxation on {unrelaxed_output_path}...")
            relax_protein(config, "cuda", unrelaxed_protein, args.output_dir,
                          output_name, args.cif_output)

        if args.save_outputs:
            output_dict_path = os.path.join(args.output_dir,
                                            f'{output_name}_output_dict.pkl')
            with open(output_dict_path, "wb") as fp:
                pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Model output written to {output_dict_path}...")

    # Write timing records to CSV
    if timing_records:
        timing_csv_path = os.path.join(args.output_dir,
                                       "timing_measurements.csv")
        with open(timing_csv_path, 'w', newline='') as csvfile:
            fieldnames = [
                'tag', 'prep_features_time', 'predict_time', 'write_pdb_time',
                'total_time'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(timing_records)
        logger.info(f"Timing measurements written to {timing_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "fasta_dir",
        type=str,
        help="Path to directory containing FASTA files, one sequence per file")
    parser.add_argument(
        "template_mmcif_dir",
        type=str,
    )
    parser.add_argument(
        "--use_precomputed_alignments",
        type=str,
        default=None,
        help="""Path to alignment directory. If provided, alignment computation
                is skipped and database path arguments are ignored.""")
    parser.add_argument(
        "--use_custom_template",
        action="store_true",
        default=False,
        help=
        """Use mmcif given with "template_mmcif_dir" argument as template input."""
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.getcwd(),
        help="""Name of the directory in which to output the prediction""",
    )
    parser.add_argument(
        "--save_outputs",
        action="store_true",
        default=False,
        help="Whether to save all model outputs, including embeddings, etc.")
    parser.add_argument(
        "--cpus",
        type=int,
        default=4,
        help="""Number of CPUs with which to run alignment tools""")
    parser.add_argument("--model_name",
                        type=str,
                        default="alphafold2_1",
                        help="""Name of the model to use""")
    parser.add_argument("--output_postfix",
                        type=str,
                        default=None,
                        help="""Postfix for output prediction filenames""")
    parser.add_argument("--data_random_seed", type=int, default=None)
    parser.add_argument(
        "--skip_relaxation",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--multimer_ri_gap",
        type=int,
        default=200,
        help="""Residue index offset between multiple sequences, if provided"""
    )
    parser.add_argument(
        "--subtract_plddt",
        action="store_true",
        default=False,
        help=""""Whether to output (100 - pLDDT) in the B-factor column instead
                 of the pLDDT itself""")
    parser.add_argument(
        "--cif_output",
        action="store_true",
        default=False,
        help=
        "Output predicted models in ModelCIF format instead of PDB format (default)"
    )
    parser.add_argument("--evoformer_backend",
                        type=str,
                        default="torch",
                        choices=["torch", "trt"],
                        help="""Backend to use for the evoformer.
                "torch" uses the original torch implementation.
                "trt" uses the TRT backend.""")
    parser.add_argument("--evoformer_ckpt",
                        type=str,
                        default=None,
                        help="""Path to the evoformer checkpoint.""")
    parser.add_argument(
        "--evoformer_fallback_threshold",
        type=int,
        default=1536,
        help=
        """Threshold for the sequence length to fallback to the torch backend for the evoformer."""
    )
    parser.add_argument(
        "--max_recycling_iters",
        type=int,
        default=None,
        help="""Maximum number of recycling iterations to use.""")
    parser.add_argument(
        "--dont_skip_template_pair_stack",
        action="store_true",
        default=False,
        help="""Whether to not skip the template pair stack.""")
    add_data_args(parser)
    args = parser.parse_args()

    main(args)
