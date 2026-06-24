import random

from evotensile.search.differential_evolution import differential_evolution_candidates
from evotensile.search.encoding import candidate_to_genome, genome_to_candidate, ordered_domain_values
from evotensile.search.gomea import (
    NT_HHS_LINKAGE_GROUPS,
    gomea_candidates,
    gomea_neighborhood_candidates,
    neighborhood_group_names,
)
from evotensile.search.random_search import initial_random_batch
from evotensile.search_space import (
    DOMAINS,
    MATRIX_INSTRUCTIONS,
    NT_HHS_RANDOM_TLDS2_PROBABILITY,
    NT_HHS_RANDOM_VALU_VGPR_HEADROOM,
    _valu_vgpr_lower_bound,
    cheap_constraints,
    defaulted_params,
    explain_invalid_nt_hhs,
    macro_tile,
    make_candidate,
    random_candidate,
    repair_linked_overrides,
)
from evotensile.shapes import Shape
from tests.helpers import DOCUMENTED_WINNER_CANDIDATE, sample_candidates


def test_encoding_round_trips_complete_candidate():
    candidate = sample_candidates(1)[0]
    genome = candidate_to_genome(candidate)
    decoded = genome_to_candidate(genome, source="roundtrip")

    assert decoded.hash == candidate.hash
    assert ordered_domain_values("ScheduleIterAlg", 3)[0] == 3


def test_matrix_instructions_have_symmetric_macro_tiles():
    macro_tiles = {macro_tile(instruction) for instruction in MATRIX_INSTRUCTIONS}

    assert all((n, m) in macro_tiles for m, n in macro_tiles)


def test_explain_invalid_nt_hhs_reports_rule_ids():
    params = DOCUMENTED_WINNER_CANDIDATE.canonical_params()
    params.update(
        {
            "DepthU": 16,
            "GlobalSplitU": 2,
            "PrefetchLocalRead": 1,
            "VectorWidthA": 8,
            "VectorWidthB": 8,
            "StoreVectorWidth": 4,
            "TransposeLDS": 1,
            "1LDSBuffer": 1,
            "ScheduleIterAlg": 1,
            "PrefetchGlobalRead": 0,
            "StoreSyncOpt": 1,
            "GroupLoadStore": True,
            "StorePriorityOpt": False,
            "NumElementsPerBatchStore": 0,
        }
    )

    rule_ids = {reason.rule_id for reason in explain_invalid_nt_hhs(defaulted_params(params))}

    assert "nt_hhs.mi_wave_tile0.requires_vector_width_a_divisor" in rule_ids
    assert "nt_hhs.mi_wave_tile1.requires_vector_width_b_divisor" in rule_ids
    assert "nt_hhs.source_swap.requires_store_vector_width_divides_vector_width_a" not in rule_ids
    assert "nt_hhs.tlds1.rejects_nt_tlua_tlub" in rule_ids
    assert "nt_hhs.one_lds_buffer.rejects_pgr0" in rule_ids
    assert "nt_hhs.one_lds_buffer.requires_sia2_or_sia3_with_slw" in rule_ids
    assert "nt_hhs.gsu.requires_depthu_ge_32" in rule_ids
    assert "nt_hhs.store_sync.unsupported_opt" in rule_ids
    assert "nt_hhs.group_load_store.requires_store_sync_4" in rule_ids
    assert "nt_hhs.group_load_store.requires_store_priority" in rule_ids
    assert "nt_hhs.group_load_store.requires_batch_choice" in rule_ids


def test_explain_invalid_nt_hhs_reports_tensilelite_lds_block_rule():
    params = DOCUMENTED_WINNER_CANDIDATE.canonical_params()
    params.update(
        {
            "DepthU": 128,
            "TransposeLDS": 2,
            "PrefetchGlobalRead": 2,
            "PrefetchLocalRead": 0,
            "VectorWidthB": 1,
            "LdsBlockSizePerPadA": 128,
            "LdsBlockSizePerPadB": 128,
        }
    )

    rule_ids = {reason.rule_id for reason in explain_invalid_nt_hhs(defaulted_params(params))}

    assert "nt_hhs.lds.tlds2_block_size_a_must_divide_depthu_bytes" in rule_ids
    assert "nt_hhs.lds.tlds2_block_size_b_must_divide_depthu_bytes" in rule_ids

    params.update({"PrefetchGlobalRead": 1, "PrefetchLocalRead": 1, "VectorWidthB": 8})
    rule_ids = {reason.rule_id for reason in explain_invalid_nt_hhs(defaulted_params(params))}

    assert "nt_hhs.tlds2.requires_pgr2_plr0" in rule_ids
    assert "nt_hhs.tlds2.requires_vector_width_b_1" in rule_ids

    repaired = repair_linked_overrides(params)
    assert repaired["TransposeLDS"] == 2
    assert repaired["PrefetchGlobalRead"] == 2
    assert repaired["PrefetchLocalRead"] == 0
    assert repaired["VectorWidthB"] == 1


def test_broad_lds_tuple_is_not_rejected_by_profile_allowlist():
    params = DOCUMENTED_WINNER_CANDIDATE.canonical_params()
    params.update(
        {
            "TransposeLDS": 0,
            "LdsBlockSizePerPadA": 6144,
            "LdsBlockSizePerPadB": 1536,
            "LdsPadA": 4,
            "LdsPadB": 16,
        }
    )

    assert cheap_constraints(defaulted_params(params))


def test_explain_invalid_nt_hhs_reports_tlds2_lsp_padding_rule():
    params = DOCUMENTED_WINNER_CANDIDATE.canonical_params()
    params.update(
        {
            "MatrixInstruction": [16, 16, 16, 1, 1, 1, 4, 4, 2],
            "WorkGroup": [32, 2, 1],
            "DepthU": 32,
            "TransposeLDS": 2,
            "PrefetchGlobalRead": 2,
            "PrefetchLocalRead": 0,
            "VectorWidthA": 1,
            "VectorWidthB": 1,
            "GlobalReadVectorWidthA": 4,
            "GlobalReadVectorWidthB": 8,
            "LdsBlockSizePerPadA": 1536,
            "LdsBlockSizePerPadB": 0,
            "LdsPadA": 8,
            "LdsPadB": 8,
        }
    )

    rule_ids = {reason.rule_id for reason in explain_invalid_nt_hhs(defaulted_params(params))}
    repaired = repair_linked_overrides(params)

    assert "nt_hhs.lds.tlds2_block_size_a_must_align_lsp" in rule_ids
    assert repaired["LdsBlockSizePerPadA"] == 0
    assert cheap_constraints(repaired)


def test_repair_linked_overrides_matches_cheap_constraints():
    params = DOCUMENTED_WINNER_CANDIDATE.canonical_params()
    params.update(
        {
            "DepthU": 16,
            "GlobalSplitU": 2,
            "TransposeLDS": 2,
            "PrefetchGlobalRead": 1,
            "PrefetchLocalRead": 1,
            "VectorWidthB": 8,
            "LdsBlockSizePerPadA": 6144,
            "LdsBlockSizePerPadB": 8192,
            "LdsPadA": 4,
            "LdsPadB": 16,
            "StoreSyncOpt": 4,
            "GroupLoadStore": True,
            "StorePriorityOpt": False,
            "NumElementsPerBatchStore": 0,
        }
    )

    repaired = repair_linked_overrides(params)

    assert repaired["DepthU"] == 32
    assert repaired["PrefetchGlobalRead"] == 2
    assert repaired["PrefetchLocalRead"] == 0
    assert repaired["VectorWidthB"] == 1
    assert repaired["StorePriorityOpt"] is True
    assert repaired["NumElementsPerBatchStore"] == 12
    assert cheap_constraints(repaired)


def test_repair_linked_overrides_clears_each_repairable_rule():
    cases = [
        (
            {"MatrixInstruction": [16, 16, 16, 1, 1, 3, 5, 2, 1], "VectorWidthA": 8, "VectorWidthB": 4},
            {
                "nt_hhs.mi_wave_tile0.requires_vector_width_a_divisor",
                "nt_hhs.mi_wave_tile1.requires_vector_width_b_divisor",
            },
        ),
        (
            {"SourceSwap": True, "VectorWidthA": 1, "StoreVectorWidth": 4},
            {"nt_hhs.source_swap.requires_store_vector_width_divides_vector_width_a"},
        ),
        (
            {
                "MatrixInstruction": [16, 16, 16, 1, 1, 3, 5, 1, 1],
                "SourceSwap": False,
                "VectorWidthA": 1,
                "StoreVectorWidth": 4,
            },
            {"nt_hhs.non_source_swap.requires_store_vector_width_divides_miovw"},
        ),
        (
            {"TransposeLDS": 1},
            {"nt_hhs.tlds1.rejects_nt_tlua_tlub"},
        ),
        (
            {"1LDSBuffer": 1, "PrefetchGlobalRead": 0, "ScheduleIterAlg": 1},
            {
                "nt_hhs.one_lds_buffer.rejects_pgr0",
                "nt_hhs.one_lds_buffer.requires_sia2_or_sia3_with_slw",
            },
        ),
        (
            {"1LDSBuffer": 0, "PrefetchGlobalRead": 0, "ScheduleIterAlg": 2},
            {"nt_hhs.sia2_forces_one_lds_buffer.rejects_pgr0"},
        ),
        (
            {"DepthU": 16, "GlobalSplitU": 2},
            {"nt_hhs.gsu.requires_depthu_ge_32"},
        ),
        (
            {
                "MatrixInstruction": [16, 16, 16, 1, 1, 4, 9, 4, 1],
                "WorkGroup": [32, 4, 1],
                "DepthU": 32,
                "GlobalReadVectorWidthB": 8,
            },
            {"nt_hhs.global_read_vectors_b.requires_num_threads_divisor"},
        ),
        (
            {"MatrixInstruction": [16, 16, 16, 1, 1, 8, 7, 1, 1]},
            {"nt_hhs.vgpr.c_accumulators_exceed_max_vgpr"},
        ),
        (
            {
                "MatrixInstruction": [16, 16, 16, 1, 1, 6, 5, 1, 2],
                "DepthU": 16,
                "PrefetchLocalRead": 0,
                "ScheduleIterAlg": 1,
                "1LDSBuffer": 0,
                "TransposeLDS": 0,
                "VectorWidthA": 2,
                "VectorWidthB": 1,
                "StoreVectorWidth": -1,
            },
            {"nt_hhs.vgpr.valu_lower_bound_exceeds_max_vgpr"},
        ),
        (
            {
                "MatrixInstruction": [16, 16, 16, 1, 1, 2, 8, 1, 2],
                "WorkGroup": [16, 16, 1],
                "DepthU": 64,
                "TransposeLDS": 0,
                "PrefetchGlobalRead": 1,
                "1LDSBuffer": 0,
                "ScheduleIterAlg": 1,
                "LdsPadA": 4,
                "LdsPadB": 0,
                "LdsBlockSizePerPadA": 128,
                "LdsBlockSizePerPadB": 8192,
            },
            {"nt_hhs.lds.footprint_exceeds_max_lds"},
        ),
        (
            {
                "MatrixInstruction": [16, 16, 16, 1, 1, 5, 7, 1, 2],
                "WorkGroup": [64, 4, 1],
                "DepthU": 128,
                "TransposeLDS": 2,
                "PrefetchGlobalRead": 2,
                "PrefetchLocalRead": 0,
                "VectorWidthB": 1,
                "1LDSBuffer": 1,
                "LdsPadA": 4,
                "LdsPadB": 0,
                "LdsBlockSizePerPadA": 2048,
                "LdsBlockSizePerPadB": 1536,
            },
            {"nt_hhs.lds.footprint_exceeds_max_lds"},
        ),
        (
            {
                "DepthU": 128,
                "TransposeLDS": 2,
                "PrefetchGlobalRead": 2,
                "PrefetchLocalRead": 0,
                "VectorWidthB": 1,
                "LdsBlockSizePerPadA": 128,
                "LdsBlockSizePerPadB": 128,
            },
            {
                "nt_hhs.lds.tlds2_block_size_a_must_divide_depthu_bytes",
                "nt_hhs.lds.tlds2_block_size_b_must_divide_depthu_bytes",
            },
        ),
        (
            {"TransposeLDS": 2, "PrefetchGlobalRead": 1, "PrefetchLocalRead": 1, "VectorWidthB": 8},
            {"nt_hhs.tlds2.requires_pgr2_plr0", "nt_hhs.tlds2.requires_vector_width_b_1"},
        ),
        (
            {"StoreSyncOpt": 1},
            {"nt_hhs.store_sync.unsupported_opt"},
        ),
        (
            {"StoreSyncOpt": 2, "NumElementsPerBatchStore": 0},
            {"nt_hhs.store_sync.requires_batch_choice"},
        ),
        (
            {"GroupLoadStore": True, "StoreSyncOpt": 0, "StorePriorityOpt": False, "NumElementsPerBatchStore": 0},
            {
                "nt_hhs.group_load_store.requires_store_sync_4",
                "nt_hhs.group_load_store.requires_store_priority",
                "nt_hhs.group_load_store.requires_batch_choice",
            },
        ),
    ]

    for overrides, expected_rule_ids in cases:
        params = defaulted_params({**DOCUMENTED_WINNER_CANDIDATE.canonical_params(), **overrides})
        rule_ids = {reason.rule_id for reason in explain_invalid_nt_hhs(params)}
        repaired = repair_linked_overrides(params)

        assert expected_rule_ids <= rule_ids
        assert cheap_constraints(repaired)


def test_explain_invalid_nt_hhs_reports_shape_gsu_workspace_rule():
    params = DOCUMENTED_WINNER_CANDIDATE.canonical_params()
    params.update({"GlobalSplitU": 2, "DepthU": 32})
    shape = Shape(m=8192, n=8192, batch=1, k=8192)

    reasons = explain_invalid_nt_hhs(defaulted_params(params), shape=shape)
    rule_ids = {reason.rule_id for reason in reasons}

    assert "nt_hhs.shape.gsu_workspace_exceeds_max" in rule_ids
    assert any(
        reason.rule_id == "nt_hhs.shape.gsu_workspace_exceeds_max" and reason.shape_dependent for reason in reasons
    )
    assert cheap_constraints(defaulted_params(params))
    assert not cheap_constraints(defaulted_params(params), shape=shape)


def test_repair_linked_overrides_keeps_broad_domain_samples_rule_valid():
    rng = random.Random(1151)
    for _ in range(512):
        params = defaulted_params({name: rng.choice(values) for name, values in DOMAINS.items()})
        repaired = repair_linked_overrides(params)

        assert cheap_constraints(repaired)


def test_random_candidate_uses_constraint_aware_broad_sampling():
    candidates = [random_candidate(random.Random(seed)) for seed in range(64)]
    hashes = {candidate.hash for candidate in candidates}
    matrix_instructions = {tuple(candidate.canonical_params()["MatrixInstruction"]) for candidate in candidates}
    lds_profiles = {
        (
            candidate.canonical_params()["TransposeLDS"],
            candidate.canonical_params()["LdsBlockSizePerPadA"],
            candidate.canonical_params()["LdsBlockSizePerPadB"],
            candidate.canonical_params()["LdsPadA"],
            candidate.canonical_params()["LdsPadB"],
        )
        for candidate in candidates
    }

    assert all(cheap_constraints(candidate.canonical_params()) for candidate in candidates)
    assert DOCUMENTED_WINNER_CANDIDATE.hash not in hashes
    assert {candidate.source for candidate in candidates} == {"random"}
    assert len(matrix_instructions) > 16
    assert len(lds_profiles) > 16
    assert sum(candidate.canonical_params()["TransposeLDS"] == 2 for candidate in candidates) > len(candidates) // 2
    assert max(_valu_vgpr_lower_bound(candidate.canonical_params()) for candidate in candidates) <= (
        NT_HHS_RANDOM_VALU_VGPR_HEADROOM
    )


def test_random_tlds2_bias_does_not_change_domains_or_constraints():
    candidates = [random_candidate(random.Random(seed)) for seed in range(128, 256)]
    tlds2_count = sum(candidate.canonical_params()["TransposeLDS"] == 2 for candidate in candidates)

    assert DOMAINS["TransposeLDS"] == [0, 1, 2]
    assert tlds2_count >= int(len(candidates) * NT_HHS_RANDOM_TLDS2_PROBABILITY * 0.5)
    assert any(candidate.canonical_params()["TransposeLDS"] == 0 for candidate in candidates)
    assert all(cheap_constraints(candidate.canonical_params()) for candidate in candidates)


def test_explicit_candidates_can_exceed_random_vgpr_headroom():
    params = DOCUMENTED_WINNER_CANDIDATE.canonical_params()
    params.update(
        {
            "MatrixInstruction": [16, 16, 16, 1, 1, 8, 2, 2, 4],
            "WorkGroup": [16, 16, 1],
            "DepthU": 16,
            "ScheduleIterAlg": 2,
            "PrefetchGlobalRead": 2,
            "PrefetchLocalRead": 0,
            "TransposeLDS": 2,
            "VectorWidthA": 8,
            "VectorWidthB": 1,
            "GlobalReadVectorWidthA": 4,
            "GlobalReadVectorWidthB": 1,
            "LdsBlockSizePerPadA": 1024,
            "LdsBlockSizePerPadB": 0,
            "LdsPadA": 8,
            "LdsPadB": 0,
            "GlobalSplitU": 1,
            "ClusterLocalRead": 1,
            "1LDSBuffer": 1,
            "SourceSwap": True,
            "StoreVectorWidth": 8,
            "StoreSyncOpt": 0,
            "GroupLoadStore": False,
            "NumElementsPerBatchStore": 0,
        }
    )
    params = defaulted_params(params)

    assert NT_HHS_RANDOM_VALU_VGPR_HEADROOM < _valu_vgpr_lower_bound(params) <= 256
    assert cheap_constraints(params)
    assert make_candidate(params, source="explicit")


def test_encoding_accepts_nt_hhs_values():
    params = DOCUMENTED_WINNER_CANDIDATE.canonical_params()
    params.update(
        {
            "MatrixInstruction": [16, 16, 16, 1, 1, 4, 2, 4, 1],
            "WorkGroup": [16, 8, 1],
            "AssertFree0ElementMultiple": 1,
            "AssertFree1ElementMultiple": 1,
            "AssertSummationElementMultiple": 1,
            "DepthU": 64,
            "GlobalReadVectorWidthA": 1,
            "GlobalReadVectorWidthB": 1,
            "LdsBlockSizePerPadA": 8192,
            "LdsBlockSizePerPadB": 512,
            "LdsPadA": 16,
            "LdsPadB": 16,
        }
    )

    assert candidate_to_genome(make_candidate(params, source="test"))


def test_differential_evolution_generates_valid_candidates():
    parents = initial_random_batch(4, seed=7)
    proposed = differential_evolution_candidates(parents, count=8, seed=11, exclude={parent.hash for parent in parents})

    assert proposed
    assert len(proposed) <= 8
    assert {candidate.source for candidate in proposed} == {"de"}
    assert {candidate.hash for candidate in proposed}.isdisjoint({parent.hash for parent in parents})


def test_differential_evolution_requires_four_supplied_parents():
    parents = initial_random_batch(3, seed=7)

    assert differential_evolution_candidates(parents, count=8, seed=11) == []


def test_gomea_uses_nt_hhs_linkage_groups_with_random_parents():
    parents = initial_random_batch(12, seed=1151)
    shape = Shape(8192, 8192, 1, 8192)
    proposed = gomea_candidates(parents, count=12, seed=1152, target_shapes=[shape])

    assert proposed
    assert ("TransposeLDS", "LdsBlockSizePerPadA", "LdsBlockSizePerPadB", "LdsPadA", "LdsPadB") in NT_HHS_LINKAGE_GROUPS
    assert all(cheap_constraints(candidate.canonical_params(), shape=shape) for candidate in proposed)
    assert max(_valu_vgpr_lower_bound(candidate.canonical_params()) for candidate in proposed) <= (
        NT_HHS_RANDOM_VALU_VGPR_HEADROOM
    )


def test_gomea_neighborhood_covers_all_mutable_domain_knobs():
    covered = {name for group in neighborhood_group_names() for name in group}
    mutable = {name for name, values in DOMAINS.items() if len(values) > 1}

    assert mutable <= covered


def test_gomea_neighborhood_can_compose_linked_knobs():
    parents = sample_candidates(9)
    proposed = gomea_neighborhood_candidates(
        parents, count=16, max_elites=None, exclude={parent.hash for parent in parents}
    )

    assert proposed
    assert {candidate.source for candidate in proposed} == {"gomea"}
    assert all(candidate.parent_hashes for candidate in proposed)
    assert all(cheap_constraints(candidate.canonical_params()) for candidate in proposed)
