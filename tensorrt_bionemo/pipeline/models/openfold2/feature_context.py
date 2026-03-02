# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import collections
from typing import Optional

import numpy as np
import torch

import tensorrt_bionemo.pipeline.models.openfold2.const as rc
import tensorrt_bionemo.pipeline.models.openfold2.msa_pairing as msa_pairing
from tensorrt_bionemo.configs.base import BaseConfig
from tensorrt_bionemo.data.parsers import (InputParsed, MSAParsed,
                                           generate_deletion_matrix)
from tensorrt_bionemo.data.utils import sequence_to_onehot
from tensorrt_bionemo.pipeline.base import ContextGeneratorBase


class MultimerFeaturePairAndMerge:
    REQUIRED_FEATURES = frozenset({
        'aatype', 'all_atom_mask', 'all_atom_positions',
        'all_chains_entity_ids', 'all_crops_all_chains_mask',
        'all_crops_all_chains_positions', 'all_crops_all_chains_residue_ids',
        'assembly_num_chains', 'asym_id', 'bert_mask', 'cluster_bias_mask',
        'deletion_matrix', 'deletion_mean', 'entity_id', 'entity_mask',
        'mem_peak', 'msa', 'msa_mask', 'num_alignments', 'num_templates',
        'queue_size', 'residue_index', 'resolution', 'seq_length', 'seq_mask',
        'sym_id', 'template_aatype', 'template_all_atom_mask',
        'template_all_atom_positions'
    })

    def __init__(self,
                 max_templates: int = 4,
                 msa_crop_size: int = 2048,
                 is_homomer_or_monomer: bool = True):
        self.max_templates = max_templates
        self.msa_crop_size = msa_crop_size
        self.is_homomer_or_monomer = is_homomer_or_monomer

    def _process_unmerged_features(self,
                                   all_chain_features: dict[str,
                                                            dict[str,
                                                                 np.ndarray]]):
        """Postprocessing stage for per-chain features before merging."""
        num_chains = len(all_chain_features)
        for chain_features in all_chain_features.values():
            # Convert deletion matrices to float.
            chain_features['deletion_matrix'] = np.asarray(
                chain_features.pop('deletion_matrix_int'), dtype=np.float32)
            if 'deletion_matrix_int_all_seq' in chain_features:
                chain_features['deletion_matrix_all_seq'] = np.asarray(
                    chain_features.pop('deletion_matrix_int_all_seq'),
                    dtype=np.float32)

            chain_features['deletion_mean'] = np.mean(
                chain_features['deletion_matrix'], axis=0)

            if 'all_atom_positions' not in chain_features:
                # Add all_atom_mask and dummy all_atom_positions based on aatype.
                all_atom_mask = rc.STANDARD_ATOM_MASK[chain_features['aatype']]
                chain_features['all_atom_mask'] = all_atom_mask.astype(
                    dtype=np.float32)
                chain_features['all_atom_positions'] = np.zeros(
                    list(all_atom_mask.shape) + [3])

            # Add assembly_num_chains.
            chain_features['assembly_num_chains'] = np.asarray(num_chains)

        # Add entity_mask.
        for chain_features in all_chain_features.values():
            chain_features['entity_mask'] = (chain_features['entity_id']
                                             != 0).astype(np.int32)

    def _crop_single_chain(self, chain: dict[str, np.ndarray],
                           msa_crop_size: int, pair_msa_sequences: bool,
                           max_templates: int) -> dict[str, np.ndarray]:
        """Crops msa sequences to `msa_crop_size`."""
        msa_size = chain['num_alignments']

        if pair_msa_sequences:
            msa_size_all_seq = chain['num_alignments_all_seq']
            msa_crop_size_all_seq = np.minimum(msa_size_all_seq,
                                               msa_crop_size // 2)

            # We reduce the number of un-paired sequences, by the number of times a
            # sequence from this chain's MSA is included in the paired MSA.  This keeps
            # the MSA size for each chain roughly constant.
            msa_all_seq = chain['msa_all_seq'][:msa_crop_size_all_seq, :]
            num_non_gapped_pairs = np.sum(
                np.any(msa_all_seq != msa_pairing.MSA_GAP_IDX, axis=1))
            num_non_gapped_pairs = np.minimum(num_non_gapped_pairs,
                                              msa_crop_size_all_seq)

            # Restrict the unpaired crop size so that paired+unpaired sequences do not
            # exceed msa_seqs_per_chain for each chain.
            max_msa_crop_size = np.maximum(
                msa_crop_size - num_non_gapped_pairs, 0)
            msa_crop_size = np.minimum(msa_size, max_msa_crop_size)
        else:
            msa_crop_size = np.minimum(msa_size, msa_crop_size)

        include_templates = 'template_aatype' in chain and max_templates
        if include_templates:
            num_templates = chain['template_aatype'].shape[0]
            templates_crop_size = np.minimum(num_templates, max_templates)

        for k in chain:
            k_split = k.split('_all_seq')[0]
            if k_split in msa_pairing.TEMPLATE_FEATURES:
                chain[k] = chain[k][:templates_crop_size, :]
            elif k_split in msa_pairing.MSA_FEATURES:
                if '_all_seq' in k and pair_msa_sequences:
                    chain[k] = chain[k][:msa_crop_size_all_seq, :]
                else:
                    chain[k] = chain[k][:msa_crop_size, :]

        chain['num_alignments'] = np.asarray(msa_crop_size, dtype=np.int32)
        if include_templates:
            chain['num_templates'] = np.asarray(templates_crop_size,
                                                dtype=np.int32)
        if pair_msa_sequences:
            chain['num_alignments_all_seq'] = np.asarray(msa_crop_size_all_seq,
                                                         dtype=np.int32)
        return chain

    def _crop_chains(self, chains_list: list[dict[str, np.ndarray]],
                     msa_crop_size: int, pair_msa_sequences: bool,
                     max_templates: int) -> list[dict[str, np.ndarray]]:
        """Crops the MSAs for a set of chains.

        Args:
            chains_list: A list of chains to be cropped.
            msa_crop_size: The total number of sequences to crop from the MSA.
            pair_msa_sequences: Whether we are operating in sequence-pairing mode.
            max_templates: The maximum templates to use per chain.

        Returns:
            The chains cropped.
        """

        # Apply the cropping.
        cropped_chains = []
        for chain in chains_list:
            cropped_chain = self._crop_single_chain(
                chain,
                msa_crop_size=msa_crop_size,
                pair_msa_sequences=pair_msa_sequences,
                max_templates=max_templates)
            cropped_chains.append(cropped_chain)

        return cropped_chains

    def _correct_msa_restypes(
            self, np_example: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Correct MSA restype to have the same order as residue_constants."""
        new_order_list = rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
        np_example['msa'] = np.take(new_order_list, np_example['msa'], axis=0)
        np_example['msa'] = np_example['msa'].astype(np.int32)
        return np_example

    def _make_seq_mask(
            self, np_example: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        np_example['seq_mask'] = (np_example['entity_id']
                                  > 0).astype(np.float32)
        return np_example

    def _make_msa_mask(
            self, np_example: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Mask features are all ones, but will later be zero-padded."""

        np_example['msa_mask'] = np.ones_like(np_example['msa'],
                                              dtype=np.float32)

        seq_mask = (np_example['entity_id'] > 0).astype(np.float32)
        np_example['msa_mask'] *= seq_mask[None]

        return np_example

    def _filter_features(
            self, np_example: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Filters features of example to only those requested."""
        return {
            k: v
            for (k, v) in np_example.items() if k in self.REQUIRED_FEATURES
        }

    def _process_final(
            self, np_example: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Final processing steps in data pipeline, after merging and pairing."""
        np_example = self._correct_msa_restypes(np_example)
        np_example = self._make_seq_mask(np_example)
        np_example = self._make_msa_mask(np_example)
        np_example = self._filter_features(np_example)
        return np_example

    def __call__(
        self, all_chain_features: dict[str, dict[str, np.ndarray]]
    ) -> dict[str, np.ndarray]:
        self._process_unmerged_features(all_chain_features)
        np_chains_list = list(all_chain_features.values())
        pair_msa_sequences = not self.is_homomer_or_monomer

        chains = list(np_chains_list)
        chain_keys = chains[0].keys()
        # This code learned from colabfold
        # This is a little bit different from OF2 OSS code regarding the removal of duplicated sequences based on unpaired MSAs.
        # See: https://github.com/aqlaboratory/openfold/blob/main/openfold/data/feature_processing_multimer.py#L72
        updated_chains = []
        for chain_num, chain in enumerate(chains):
            new_chain = {k: v for k, v in chain.items() if "_all_seq" not in k}
            for feature_name in chain_keys:
                if feature_name.endswith("_all_seq"):
                    feats_padded = msa_pairing.pad_features(
                        chain[feature_name], feature_name)
                    new_chain[feature_name] = feats_padded
            new_chain["num_alignments_all_seq"] = np.asarray(
                len(np_chains_list[chain_num]["msa_all_seq"]))
            updated_chains.append(new_chain)
        np_chains_list = updated_chains
        np_chains_list = self._crop_chains(
            np_chains_list,
            msa_crop_size=self.msa_crop_size,
            pair_msa_sequences=pair_msa_sequences,
            max_templates=self.max_templates)
        common_features = set([*np_chains_list[0]
                               ]).intersection(*np_chains_list)
        np_chains_list = [{
            key: value
            for (key, value) in chain.items() if key in common_features
        } for chain in np_chains_list]
        np_example = msa_pairing.merge_chain_features(
            np_chains_list=np_chains_list,
            pair_msa_sequences=pair_msa_sequences,
            max_templates=self.max_templates)
        np_example = self._process_final(np_example)
        return np_example


class FeatureContextGenerator(ContextGeneratorBase):

    def __init__(self, config: BaseConfig):
        """

        Args:
            config: The configuration for the model.
        """
        super().__init__(config)
        self.unsupervised_features = [
            "aatype",
            "residue_index",
            "msa",
            "num_alignments",
            "seq_length",
            "between_segment_residues",
            "deletion_matrix",
            "no_recycling_iters",
        ]
        if self.config.is_multimer:
            self.unsupervised_features.extend([
                "msa_mask",
                "seq_mask",
                "asym_id",
                "entity_id",
                "sym_id",
            ])
        self.template_features = [
            "template_all_atom_positions", "template_sum_probs",
            "template_aatype", "template_all_atom_mask", "is_template_present"
        ]

    def make_msa_features(self,
                          parsed_msa: MSAParsed,
                          avoid_duplicated: bool = True,
                          post_fix: str = "") -> dict[str, np.ndarray]:
        """
        Args:
            parsed_msa: The parsed MSA.
            avoid_duplicated: Whether to avoid duplicated sequences. Should set to False for paired MSAs.
            post_fix: The post fix to add to the feature name.
        Returns:
            A dictionary of features.
        """
        raw_sequences = []
        aligned_sequences = []
        raw_sequences.extend(parsed_msa["raw"])
        aligned_sequences.extend(parsed_msa["sequences"])
        int_msa = []
        deletion_matrix = generate_deletion_matrix(raw_sequences)
        filtered_deletion_matrix = []
        seen_sequences = set()

        for sequence_index, sequence in enumerate(aligned_sequences):
            if avoid_duplicated and sequence in seen_sequences:
                continue
            seen_sequences.add(sequence)
            int_msa.append([rc.HHBLITS_AA_TO_ID[res] for res in sequence])
            filtered_deletion_matrix.append(deletion_matrix[sequence_index])
        num_res = len(parsed_msa["sequences"][0])
        num_alignments = len(int_msa)
        features = {}
        filtered_deletion_matrix = np.array(filtered_deletion_matrix)
        features["deletion_matrix_int" + post_fix] = np.array(
            filtered_deletion_matrix, dtype=np.int32)
        features["msa" + post_fix] = np.array(int_msa, dtype=np.int32)
        features["num_alignments" + post_fix] = np.array([num_alignments] *
                                                         num_res,
                                                         dtype=np.int32)
        return features

    def empty_template_feats(self, n_res: int) -> dict[str, np.ndarray]:
        return {
            "template_aatype":
            np.zeros((0, n_res, len(rc.restypes_with_x_and_gap)), np.float32),
            "template_all_atom_mask":
            np.zeros((0, n_res, rc.atom_type_num), np.float32),
            "template_all_atom_positions":
            np.zeros((0, n_res, rc.atom_type_num, 3), np.float32),
            "template_domain_names":
            np.array([''.encode()], dtype=object),
            "template_sequence":
            np.array([''.encode()], dtype=object),
            "template_sum_probs":
            np.zeros((0, 1), dtype=np.float32),
        }

    def make_sequence_features(self, sequence: str,
                               description: str) -> dict[str, np.ndarray]:
        aatype = sequence_to_onehot(sequence, rc.restype_order_with_x).numpy()
        n_res = len(sequence)

        between_segment_residues = np.zeros((n_res, ), dtype=np.int32)
        domain_name = np.array([description.encode("utf-8")], dtype=object)
        residue_index = np.array(range(n_res), dtype=np.int32)
        seq_length = np.array([n_res] * n_res, dtype=np.int32)

        return {
            'aatype': aatype,
            'between_segment_residues': between_segment_residues,
            'domain_name': domain_name,
            'residue_index': residue_index,
            'seq_length': seq_length,
            'sequence': np.array([sequence.encode("utf-8")], dtype=object)
        }

    def np_to_tensor_dict(
        self,
        np_example: dict[str, np.ndarray],
        features: list[str],
    ) -> dict[str, torch.Tensor]:
        """Creates dict of tensors from a dict of NumPy arrays.

        Args:
            np_example: A dict of NumPy feature arrays.
            features: A list of strings of feature names to be returned in the dataset.

        Returns:
            A dictionary of features mapping feature names to features. Only the given
            features are returned, all other ones are filtered out.
        """

        # torch generates warnings if feature is already a torch Tensor
        def to_tensor(t):
            """Convert to tensor, cloning if already a Tensor."""
            if isinstance(t, torch.Tensor):
                return t.clone().detach()
            return torch.tensor(t)

        tensor_dict = {
            k: to_tensor(v)
            for k, v in np_example.items() if k in features
        }

        return tensor_dict

    def build_monomer_context(
            self,
            sequence: str,
            chain_id: str,
            description: Optional[str] = None,
            parsed_msa: Optional[MSAParsed] = None) -> dict[str, np.ndarray]:
        if isinstance(chain_id, list):
            chain_id = chain_id[0]
        if parsed_msa is None:
            parsed_msa = MSAParsed(
                sequences=[sequence],
                raw=[sequence],
                descriptions=[description],
            )
        context = self.make_sequence_features(sequence, description)
        context.update(self.make_msa_features(parsed_msa))

        # TODO: Add template features, using dummy template features for now
        context.update(self.empty_template_feats(len(sequence)))
        return context

    @staticmethod
    def int_id_to_str_id(num: int) -> str:
        """Encodes a number as a string, using reverse spreadsheet style naming.

        Args:
        num: A positive integer.

        Returns:
        A string that encodes the positive integer using reverse spreadsheet style,
        naming e.g. 1 = A, 2 = B, ..., 27 = AA, 28 = BA, 29 = CA, ... This is the
        usual way to encode chain IDs in mmCIF files.
        """
        if num <= 0:
            raise ValueError(f'Only positive integers allowed, got {num}.')
        num -= 1  # 1-based indexing.
        output = []
        while num >= 0:
            output.append(chr(num % 26 + ord('A')))
            num = num // 26 - 1
        return ''.join(output)

    def add_assembly_features(
        self,
        all_chain_features: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Add features to distinguish between chains.

        Args:
        all_chain_features: A dictionary which maps chain_id to a dictionary of
            features for each chain.

        Returns:
        all_chain_features: A dictionary which maps strings of the form
            `<seq_id>_<sym_id>` to the corresponding chain features. E.g. two
            chains from a homodimer would have keys A_1 and A_2. Two chains from a
            heterodimer would have keys A_1 and B_1.
        """
        # Group the chains by sequence
        seq_to_entity_id = {}
        grouped_chains = collections.defaultdict(list)
        for chain_id, chain_features in all_chain_features.items():
            seq = str(chain_features['sequence'])
            if seq not in seq_to_entity_id:
                seq_to_entity_id[seq] = len(seq_to_entity_id) + 1
            grouped_chains[seq_to_entity_id[seq]].append(chain_features)

        new_all_chain_features = {}
        chain_id = 1
        for entity_id, group_chain_features in grouped_chains.items():
            for sym_id, chain_features in enumerate(group_chain_features,
                                                    start=1):
                new_all_chain_features[
                    f'{self.int_id_to_str_id(entity_id)}_{sym_id}'] = chain_features
                seq_length = chain_features['seq_length']
                chain_features['asym_id'] = (chain_id *
                                             np.ones(seq_length)).astype(
                                                 np.int64)
                chain_features['sym_id'] = (sym_id *
                                            np.ones(seq_length)).astype(
                                                np.int64)
                chain_features['entity_id'] = (entity_id *
                                               np.ones(seq_length)).astype(
                                                   np.int64)
                chain_id += 1
        return new_all_chain_features

    def convert_monomer_features(self, monomer_features: dict[str, np.ndarray],
                                 chain_id: str) -> dict[str, np.ndarray]:
        """Reshapes and modifies monomer features for multimer models."""
        converted = {}
        converted['auth_chain_id'] = np.asarray(chain_id, dtype=object)
        unnecessary_leading_dim_feats = {
            'sequence', 'domain_name', 'num_alignments', 'seq_length'
        }
        for feature_name, feature in monomer_features.items():
            if feature_name in unnecessary_leading_dim_feats:
                # asarray ensures it's a np.ndarray.
                feature = np.asarray(feature[0], dtype=feature.dtype)
            elif feature_name == 'aatype':
                # The multimer model performs the one-hot operation itself.
                feature = np.argmax(feature, axis=-1).astype(np.int32)
            elif feature_name == 'template_aatype':
                feature = np.argmax(feature, axis=-1).astype(np.int32)
                new_order_list = rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
                feature = np.take(new_order_list,
                                  feature.astype(np.int32),
                                  axis=0)
            elif feature_name == 'template_all_atom_masks':
                feature_name = 'template_all_atom_mask'
            converted[feature_name] = feature
        return converted

    def pad_msa(self, np_example: dict[str, np.ndarray],
                min_num_seq: int) -> dict[str, np.ndarray]:
        np_example = dict(np_example)
        num_seq = np_example['msa'].shape[0]
        if num_seq < min_num_seq:
            for feat in ('msa', 'deletion_matrix', 'bert_mask', 'msa_mask'):
                np_example[feat] = np.pad(np_example[feat],
                                          ((0, min_num_seq - num_seq), (0, 0)))
            np_example['cluster_bias_mask'] = np.pad(
                np_example['cluster_bias_mask'],
                ((0, min_num_seq - num_seq), ))
        return np_example

    def build_multimer_context(self,
                               parsed: InputParsed) -> dict[str, np.ndarray]:
        polymers = parsed['polymers']
        multimer_feature_pair_and_merge = None
        paired_msas = {}
        unpaired_msas = {}
        sequences = []
        descriptions = []
        all_chain_ids = []
        chain_feats = {}

        if len(polymers) == 1:
            # It's homooligomers, only one unique sequence for all chain_ids
            multimer_feature_pair_and_merge = MultimerFeaturePairAndMerge(
                is_homomer_or_monomer=True)
            polymer = polymers[0]
            chain_ids = polymer['chain_id']
            sequences = [polymer['sequence']]
            all_chain_ids = [chain_ids]
            descriptions = ["_".join(chain_ids)]

            # Make unpaired MSA for the homooligomer, same as the monomer case.
            if polymer['msas'] is not None:
                unpaired_msas[0] = MSAParsed.concat(polymer['msas'])
            else:
                unpaired_msas[0] = MSAParsed(
                    sequences=[polymer['sequence']],
                    raw=[polymer['sequence']],
                    descriptions=["_".join(chain_ids)],
                )
            # Create a dummy paired MSA for the homooligomer
            # This will help the flow clean without affecting the result
            paired_msas[0] = MSAParsed(
                sequences=[polymer['sequence']],
                raw=[polymer['sequence']],
                descriptions=["_".join(chain_ids)],
            )
        else:
            # Verify the input
            all_none = all(polymer['paired_msas'] is None
                           for polymer in polymers)
            if all_none:
                # 1. All polymers should have the paired_msas is None
                for i in range(len(polymers)):
                    polymers[i]['paired_msas'] = [
                        MSAParsed(
                            sequences=[polymers[i]['sequence']],
                            raw=[polymers[i]['sequence']],
                            descriptions=["_".join(polymers[i]['chain_id'])],
                        )
                    ]
            else:
                # 2. Or, all polymers should have the same number of sequences in paired_msas
                nseqs = set()
                for polymer in polymers:
                    msas = MSAParsed.concat(polymer['paired_msas'])
                    if msas is None:
                        nseqs.add(-1)
                        continue
                    nseqs.add(len(msas['sequences']))
                if len(nseqs) != 1:
                    raise ValueError(
                        "All polymers should have the same number of sequences in paired_msas"
                    )
            multimer_feature_pair_and_merge = MultimerFeaturePairAndMerge(
                is_homomer_or_monomer=False)
            # It's heterooligomers
            for i in range(len(polymers)):
                polymer = polymers[i]
                chain_ids = polymer['chain_id']
                sequences.append(polymer['sequence'])
                all_chain_ids.append(chain_ids)
                descriptions.append("_".join(chain_ids))

                # Make paired MSA for each unique sequence
                paired_msas[i] = MSAParsed.concat(polymer['paired_msas'])

                # Make unpaired MSA for each unique sequence, same as the monomer case.
                if polymer['msas'] is not None:
                    unpaired_msas[i] = MSAParsed.concat(polymer['msas'])
                else:
                    unpaired_msas[i] = MSAParsed(
                        sequences=[polymer['sequence']],
                        raw=[polymer['sequence']],
                        descriptions=["_".join(chain_ids)],
                    )
        for seq_idx, _ in enumerate(sequences):
            feature_dict = self.build_monomer_context(sequences[seq_idx],
                                                      all_chain_ids[seq_idx],
                                                      descriptions[seq_idx],
                                                      unpaired_msas[seq_idx])
            if seq_idx in paired_msas:
                # for homooligomers, we use the dummy paired MSA
                # for heterooligomers, we need to add the paired MSA features
                feature_dict.update(
                    self.make_msa_features(paired_msas[seq_idx],
                                           avoid_duplicated=False,
                                           post_fix="_all_seq"))
            # create duplicate features for each chain_id
            for chain_id in all_chain_ids[seq_idx]:
                chain_feats[chain_id] = feature_dict

        all_chain_feats = {}
        for chain_id, chain_feat in chain_feats.items():
            all_chain_feats[chain_id] = self.convert_monomer_features(
                chain_feat, chain_id=chain_id)
        all_chain_feats = self.add_assembly_features(all_chain_feats)
        if multimer_feature_pair_and_merge is None:
            raise RuntimeError(
                "Multimer feature pair and merge is not initialized")

        np_example = multimer_feature_pair_and_merge(all_chain_feats)
        np_example = self.pad_msa(np_example, min_num_seq=512)
        return np_example

    def __call__(self, parsed: InputParsed) -> dict[str, torch.Tensor]:
        if len(parsed['polymers']) == 0:
            raise ValueError("No polymers found in the input")
        polymers = parsed['polymers']

        if len(polymers) == 1:
            chain_ids = polymers[0].get('chain_id', ['A'])
            if isinstance(chain_ids, str):
                chain_ids = [chain_ids]
            if len(chain_ids) == 1:
                # If is monomer, only one polymers and one chain_id
                # Sanity check if model is multimer, raise error
                if self.config.is_multimer:
                    raise ValueError(
                        "Model is multimer, but only one chain_id is provided")
                context = self.build_monomer_context(
                    polymers[0]['sequence'], polymers[0]['chain_id'],
                    polymers[0]['chain_id'],
                    MSAParsed.concat(polymers[0]['msas']))
            else:
                # Homooligomer
                context = self.build_multimer_context(parsed)
        else:
            # Otherwise, it is multimer
            context = self.build_multimer_context(parsed)

        if "deletion_matrix_int" in context:
            context["deletion_matrix"] = context.pop(
                "deletion_matrix_int").astype(np.float32)
        features_name = self.unsupervised_features
        if self.config.enable_template:
            is_template_present = context["template_aatype"].shape[0] > 0
            context["is_template_present"] = is_template_present
            features_name.extend(self.template_features)
        context = self.np_to_tensor_dict(context, features_name)
        return context
