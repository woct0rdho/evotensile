import random

from evotensile.search.differential_evolution import differential_evolution_candidates
from evotensile.search.encoding import candidate_to_genome, genome_to_candidate, ordered_domain_values
from evotensile.search.gomea import NT_HHS_LINKAGE_GROUPS, gomea_candidates, gomea_neighborhood_candidates
from evotensile.search.random_search import initial_random_batch
from evotensile.search_space import (
    DOMAINS,
    MATRIX_INSTRUCTIONS,
    cheap_constraints,
    defaulted_params,
    explain_invalid_nt_hhs,
    macro_tile,
    make_candidate,
    random_candidate,
    repair_linked_overrides,
)
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
            "TransposeLDS": 2,
            "PrefetchGlobalRead": 1,
            "PrefetchLocalRead": 1,
            "VectorWidthB": 8,
            "StoreSyncOpt": 1,
            "GroupLoadStore": True,
            "StorePriorityOpt": False,
            "NumElementsPerBatchStore": 0,
        }
    )

    rule_ids = {reason.rule_id for reason in explain_invalid_nt_hhs(defaulted_params(params))}

    assert "nt_hhs.gsu.requires_depthu_ge_32" in rule_ids
    assert "nt_hhs.tlds2.requires_pgr2_plr0" in rule_ids
    assert "nt_hhs.tlds2.requires_vector_width_b_1" in rule_ids
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
            {"DepthU": 16, "GlobalSplitU": 2},
            {"nt_hhs.gsu.requires_depthu_ge_32"},
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


def test_encoding_accepts_imported_baseline_domain_values():
    params = DOCUMENTED_WINNER_CANDIDATE.canonical_params()
    params.update(
        {
            "MatrixInstruction": [16, 16, 16, 1, 1, 2, 3, 4, 1],
            "WorkGroup": [64, 4, 1],
            "SourceSwap": True,
            "LdsBlockSizePerPadA": 4096,
            "LdsBlockSizePerPadB": 1536,
            "LdsPadA": 16,
            "LdsPadB": 16,
        }
    )

    assert candidate_to_genome(make_candidate(params, source="test"))


def test_encoding_accepts_nt_hhs_values():
    params = DOCUMENTED_WINNER_CANDIDATE.canonical_params()
    params.update(
        {
            "MatrixInstruction": [16, 16, 16, 1, 1, 4, 1, 4, 1],
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


def test_encoding_accepts_tt_hhs_tlds1_values():
    params = DOCUMENTED_WINNER_CANDIDATE.canonical_params()
    params.update(
        {
            "MatrixInstruction": [16, 16, 16, 1, 1, 7, 1, 2, 2],
            "WorkGroup": [32, 4, 1],
            "TransposeLDS": 1,
            "LdsBlockSizePerPadA": 128,
            "LdsBlockSizePerPadB": 6144,
            "LdsPadA": 16,
            "LdsPadB": 16,
            "StorePriorityOpt": False,
            "NumElementsPerBatchStore": 0,
        }
    )

    assert candidate_to_genome(make_candidate(params, source="test"))


def test_encoding_accepts_nn_hhs_values():
    params = DOCUMENTED_WINNER_CANDIDATE.canonical_params()
    params.update(
        {
            "MatrixInstruction": [16, 16, 16, 1, 1, 2, 7, 4, 1],
            "WorkGroup": [32, 2, 1],
            "DepthU": 128,
            "TransposeLDS": 1,
            "PrefetchGlobalRead": 0,
            "PrefetchLocalRead": 0,
            "1LDSBuffer": 0,
            "LdsBlockSizePerPadA": 4096,
            "LdsBlockSizePerPadB": 128,
            "LdsPadA": 8,
            "LdsPadB": 8,
            "AssertFree0ElementMultiple": 1,
            "AssertFree1ElementMultiple": 1,
            "AssertSummationElementMultiple": 1,
        }
    )

    assert candidate_to_genome(make_candidate(params, source="test"))


def test_encoding_accepts_tn_hhs_values():
    params = DOCUMENTED_WINNER_CANDIDATE.canonical_params()
    params.update(
        {
            "MatrixInstruction": [16, 16, 16, 1, 1, 8, 2, 1, 4],
            "WorkGroup": [16, 4, 1],
            "VectorWidthA": 8,
            "VectorWidthB": 8,
            "StoreVectorWidth": 8,
            "StaggerUStride": 64,
            "ExpandPointerSwap": 1,
            "TransposeLDS": 1,
            "LdsBlockSizePerPadA": 256,
            "LdsBlockSizePerPadB": 128,
            "LdsPadA": 8,
            "LdsPadB": 8,
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
    proposed = gomea_candidates(parents, count=12, seed=1152)

    assert proposed
    assert ("TransposeLDS", "LdsBlockSizePerPadA", "LdsBlockSizePerPadB", "LdsPadA", "LdsPadB") in NT_HHS_LINKAGE_GROUPS
    assert all(cheap_constraints(candidate.canonical_params()) for candidate in proposed)


def test_gomea_neighborhood_can_compose_linked_knobs():
    parents = sample_candidates(9)
    proposed = gomea_neighborhood_candidates(
        parents, count=16, max_elites=None, exclude={parent.hash for parent in parents}
    )

    assert proposed
    assert {candidate.source for candidate in proposed} == {"gomea"}
    assert all(candidate.parent_hashes for candidate in proposed)
    assert all(cheap_constraints(candidate.canonical_params()) for candidate in proposed)
