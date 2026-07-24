# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path

import pytest
import torch

from tensorrt_bionemo.configs.base import BaseConfig
from tensorrt_bionemo.data.schemas.basic import (InputParsed, PolymerParsed,
                                                 TemplateParsed)
from tensorrt_bionemo.models.openfold2.config import (
    AlphaFold2_Multimer_1_Config, OpenFold2_FT2_Config)
from tensorrt_bionemo.pipeline.models.openfold2.feature_context import \
    FeatureContextGenerator
from tensorrt_bionemo.pipeline.models.openfold2.feature_factory import (
    FeatureFactory, MultimerFeatureFactory)
from tensorrt_bionemo.pipeline.models.openfold2.feature_generators import (
    Atom37ToTorsionAngles, MakeTemplatePseudoBeta)
from tensorrt_bionemo.pipeline.models.openfold2.tokenizer import (
    MultimerTokenizer, Tokenizer)
from tensorrt_bionemo.pipeline.models.openfold2.transforms import \
    FixTemplatesAatype
from tensorrt_bionemo.pipeline.stages.feature_generator_stage import \
    FeatureGeneratorUDF
from tensorrt_bionemo.pipeline.stages.tokenizer_stage import TokenizerUDF

_TEMPLATE_CIF = Path(__file__).with_name("data") / "minimal_template.cif"


class MockConfig(BaseConfig):
    enable_template: bool = True
    is_multimer: bool = False
    max_templates: int = 4


def _template(chain_id: str = "A") -> TemplateParsed:
    return TemplateParsed(content=_TEMPLATE_CIF.read_text(),
                          format="cif",
                          chain_id=chain_id)


def test_monomer_context_uses_supplied_template():
    parsed = InputParsed(
        input_id="monomer_template",
        polymers=[
            PolymerParsed(chain_id="Q",
                          sequence="MR",
                          msas=None,
                          templates=[_template()])
        ],
    )

    result = FeatureContextGenerator(MockConfig())(parsed)

    assert result["template_aatype"].shape == (1, 2, 22)
    assert result["template_all_atom_positions"].shape == (1, 2, 37, 3)
    assert result["template_all_atom_mask"].shape == (1, 2, 37)
    assert result["template_sum_probs"].shape == (1, 1)
    assert result["template_all_atom_mask"].sum() > 0
    assert torch.isfinite(result["template_all_atom_positions"]).all()
    assert torch.equal(result["template_sum_probs"], torch.ones((1, 1)))
    assert result["is_template_present"]


def test_disabled_template_path_does_not_featurize_supplied_cif():
    parsed = InputParsed(
        input_id="disabled_template",
        polymers=[
            PolymerParsed(chain_id="Q",
                          sequence="MR",
                          msas=None,
                          templates=[
                              TemplateParsed(content="not a CIF",
                                             chain_id="missing")
                          ])
        ],
    )

    result = FeatureContextGenerator(MockConfig(enable_template=False))(parsed)

    assert not any(key.startswith("template_") for key in result)
    assert "is_template_present" not in result


def test_supplied_atom37_features_feed_pseudo_beta_and_torsions():
    parsed = InputParsed(
        input_id="derived_template_features",
        polymers=[
            PolymerParsed(chain_id="Q",
                          sequence="MR",
                          msas=None,
                          templates=[_template()])
        ],
    )
    config = MockConfig()
    batch = FeatureContextGenerator(config)(parsed)

    batch = FixTemplatesAatype(config=config)(batch)
    pseudo_beta = MakeTemplatePseudoBeta(config=config)(batch, {})
    torsions = Atom37ToTorsionAngles(config=config, prefix="template_")(batch,
                                                                        {})

    assert pseudo_beta["template_pseudo_beta"].shape == (1, 2, 3)
    assert pseudo_beta["template_pseudo_beta_mask"].shape == (1, 2)
    assert torch.count_nonzero(pseudo_beta["template_pseudo_beta_mask"]) == 2
    assert torsions["template_torsion_angles_sin_cos"].shape == (1, 2, 7, 2)
    assert torsions["template_torsion_angles_mask"].shape == (1, 2, 7)
    assert torch.isfinite(torsions["template_torsion_angles_sin_cos"]).all()


def test_heteromer_context_keeps_template_on_its_polymer():
    parsed = InputParsed(
        input_id="heteromer_template",
        polymers=[
            PolymerParsed(chain_id=["Q"],
                          sequence="MR",
                          msas=None,
                          paired_msas=None,
                          templates=[_template("A")]),
            PolymerParsed(chain_id=["R"],
                          sequence="AA",
                          msas=None,
                          paired_msas=None,
                          templates=None),
        ],
    )

    result = FeatureContextGenerator(MockConfig(is_multimer=True))(parsed)

    assert result["template_aatype"].shape == (4, 4)
    assert result["template_all_atom_positions"].shape == (4, 4, 37, 3)
    assert result["template_all_atom_mask"].shape == (4, 4, 37)
    assert result["template_all_atom_mask"][0, :2].sum() > 0
    assert torch.count_nonzero(result["template_all_atom_mask"][:, 2:]) == 0
    assert result["is_template_present"]


def test_homomer_context_reuses_entity_template_for_each_chain():
    parsed = InputParsed(
        input_id="homomer_template",
        polymers=[
            PolymerParsed(chain_id=["Q", "R"],
                          sequence="MR",
                          msas=None,
                          paired_msas=None,
                          templates=[_template("A")])
        ],
    )

    result = FeatureContextGenerator(MockConfig(is_multimer=True))(parsed)

    first_chain_mask = result["template_all_atom_mask"][0, :2]
    second_chain_mask = result["template_all_atom_mask"][0, 2:]
    assert first_chain_mask.sum() > 0
    assert torch.equal(first_chain_mask, second_chain_mask)
    assert result["is_template_present"]


def test_multimer_uses_configured_template_limit():
    parsed = InputParsed(
        input_id="configured_template_limit",
        polymers=[
            PolymerParsed(chain_id=["Q", "R"],
                          sequence="MR",
                          msas=None,
                          paired_msas=None,
                          templates=[_template("A")])
        ],
    )

    result = FeatureContextGenerator(
        MockConfig(is_multimer=True, max_templates=2))(parsed)

    assert result["template_aatype"].shape == (2, 4)
    assert result["template_all_atom_positions"].shape == (2, 4, 37, 3)
    assert result["template_all_atom_mask"].shape == (2, 4, 37)


@pytest.mark.parametrize("enable_template", [True, False])
def test_negative_template_limit_is_rejected(enable_template):
    with pytest.raises(ValueError, match="max_templates.*non-negative"):
        FeatureContextGenerator(
            MockConfig(enable_template=enable_template, max_templates=-1))


def test_monomer_context_allows_zero_template_limit():
    parsed = InputParsed(
        input_id="monomer_zero_templates",
        polymers=[
            PolymerParsed(chain_id="Q",
                          sequence="MR",
                          msas=None,
                          templates=[_template()])
        ],
    )

    result = FeatureContextGenerator(MockConfig(max_templates=0))(parsed)

    assert result["template_aatype"].shape == (0, 2, 22)
    assert result["template_all_atom_positions"].shape == (0, 2, 37, 3)
    assert result["template_all_atom_mask"].shape == (0, 2, 37)
    assert result["template_sum_probs"].shape == (0, 1)
    assert not result["is_template_present"]


def test_multimer_context_allows_zero_template_limit():
    parsed = InputParsed(
        input_id="multimer_zero_templates",
        polymers=[
            PolymerParsed(chain_id=["Q", "R"],
                          sequence="MR",
                          msas=None,
                          paired_msas=None,
                          templates=[_template("A")])
        ],
    )

    result = FeatureContextGenerator(
        MockConfig(is_multimer=True, max_templates=0))(parsed)

    assert result["template_aatype"].shape == (0, 4)
    assert result["template_all_atom_positions"].shape == (0, 4, 37, 3)
    assert result["template_all_atom_mask"].shape == (0, 4, 37)
    assert not result["is_template_present"]


def test_real_stage_specs_transform_and_derive_template_features():
    parsed = InputParsed(
        input_id="real_stage_specs",
        polymers=[
            PolymerParsed(chain_id="Q",
                          sequence="MR",
                          msas=None,
                          templates=[_template()])
        ],
    )
    config = OpenFold2_FT2_Config(max_recycling_iters=0,
                                  max_msa_clusters=2,
                                  max_extra_msa=1)

    tokenizer = Tokenizer()
    context_generators = {}
    for name, spec in tokenizer.context_generator_specs.items():
        generator = spec.generator(config=config)
        generator.required_kwargs = spec.required_kwargs
        context_generators[name] = generator
    transforms = [
        spec.transform(config=config, **spec.kwargs)
        for spec in tokenizer.transform_specs
    ]
    tokenizer_udf = TokenizerUDF(
        compute_by_rows=True,
        drop_keys=None,
        expected_input_keys=["parsed"],
        update_row=True,
        context_generators=context_generators,
        context_merger_func=tokenizer.context_merger_func,
        transform_funcs=transforms,
    )
    tokenized = asyncio.run(
        tokenizer_udf.udf_for_item({
            "parsed": parsed,
            "__record_id": "real_stage_specs"
        }))

    assert tokenized["template_aatype"].shape == (1, 2)
    assert tokenized["template_aatype"].dtype == torch.int64
    assert tokenized["is_template_present"]

    factory = FeatureFactory()
    generators = []
    for spec in factory.feature_generator_specs:
        generator = spec.functor(config=config, **spec.kwargs)
        generator.name = spec.name
        generators.append(generator)
    collators = []
    for spec in factory.feature_collator_specs:
        collator = spec.functor(config=config, **spec.kwargs)
        collator.name = spec.name
        collators.append(collator)
    feature_udf = FeatureGeneratorUDF(
        compute_by_rows=True,
        drop_keys=None,
        expected_input_keys=[],
        update_row=False,
        feature_generators=generators,
        features_merger_func=factory.features_merger_func,
        feature_collators=collators,
        pre_init=factory.pre_init,
        init_context={"random_seed": 0},
    )
    features = asyncio.run(feature_udf.udf_for_item(tokenized))

    assert features["template_aatype"].shape == (config.max_templates, 2, 1)
    assert features["template_pseudo_beta"].shape == (config.max_templates, 2,
                                                      3, 1)
    assert features["template_torsion_angles_sin_cos"].shape == (
        config.max_templates, 2, 7, 2, 1)
    assert features["template_pseudo_beta_mask"][0, :, 0].count_nonzero() == 2
    assert features["template_torsion_angles_mask"][0, :, :,
                                                    0].count_nonzero() > 0
    assert features["is_template_present"].shape == (1, )
    assert features["is_template_present"].all()


@pytest.mark.parametrize("with_template", [False, True])
def test_real_multimer_stage_specs_preserve_template_atom_mask(with_template):
    parsed = InputParsed(
        input_id="real_multimer_stage_specs",
        polymers=[
            PolymerParsed(
                chain_id=["Q", "R"],
                sequence="MR",
                msas=None,
                paired_msas=None,
                templates=[_template()] if with_template else None,
            )
        ],
    )
    config = AlphaFold2_Multimer_1_Config()
    # The multimer preset validator applies its pretrained defaults after
    # construction. Override the inference-only sizes the same way the runner
    # does so this regression stays small.
    config.max_recycling_iters = 0
    config.max_msa_clusters = 2
    config.max_extra_msa = 1

    tokenizer = MultimerTokenizer()
    context_generators = {}
    for name, spec in tokenizer.context_generator_specs.items():
        generator = spec.generator(config=config)
        generator.required_kwargs = spec.required_kwargs
        context_generators[name] = generator
    transforms = [
        spec.transform(config=config, **spec.kwargs)
        for spec in tokenizer.transform_specs
    ]
    tokenizer_udf = TokenizerUDF(
        compute_by_rows=True,
        drop_keys=None,
        expected_input_keys=["parsed"],
        update_row=True,
        context_generators=context_generators,
        context_merger_func=tokenizer.context_merger_func,
        transform_funcs=transforms,
    )
    tokenized = asyncio.run(
        tokenizer_udf.udf_for_item({
            "parsed": parsed,
            "__record_id": "real_multimer_stage_specs",
        }))
    tokenized_mask_nonzero = tokenized["template_all_atom_mask"].count_nonzero(
    )
    assert bool(tokenized_mask_nonzero) is with_template

    factory = MultimerFeatureFactory()
    generators = []
    for spec in factory.feature_generator_specs:
        generator = spec.functor(config=config, **spec.kwargs)
        generator.name = spec.name
        generators.append(generator)
    collators = []
    for spec in factory.feature_collator_specs:
        collator = spec.functor(config=config, **spec.kwargs)
        collator.name = spec.name
        collators.append(collator)
    feature_udf = FeatureGeneratorUDF(
        compute_by_rows=True,
        drop_keys=None,
        expected_input_keys=[],
        update_row=False,
        feature_generators=generators,
        features_merger_func=factory.features_merger_func,
        feature_collators=collators,
        pre_init=factory.pre_init,
        init_context={"random_seed": 0},
    )
    features = asyncio.run(feature_udf.udf_for_item(tokenized))

    assert features["template_all_atom_mask"].shape == (config.max_templates,
                                                        4, 37, 1)
    assert bool(
        features["template_all_atom_mask"].count_nonzero()) is with_template
    assert bool(features["is_template_present"].all()) is with_template
