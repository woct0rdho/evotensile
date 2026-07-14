from evotensile.database import BaselineSelectionInsert, EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.search_space import DOMAINS
from evotensile.shapes import pilot_100_shapes
from evotensile.tensilelite_parameter_types import TENSILELITE_PARAMETER_TYPES
from scripts.discover_hipblaslt_baselines import _solution_to_candidate
from tests.helpers import sample_candidates


def test_installed_solution_import_materializes_complete_candidate():
    original = sample_candidates(1)[0]

    imported = _solution_to_candidate(original.canonical_params())

    assert set(DOMAINS) <= set(imported.params)
    assert imported.hash == original.hash


def test_installed_solution_import_normalizes_all_known_parameter_types():
    original = sample_candidates(1)[0]
    solution = original.canonical_params()
    for name, expected_type in TENSILELITE_PARAMETER_TYPES.items():
        value = solution[name]
        if expected_type is int:
            solution[name] = float(value)
        elif expected_type is bool:
            solution[name] = int(value)
        elif expected_type is list:
            solution[name] = [float(item) for item in value]

    imported = _solution_to_candidate(solution)

    assert imported.hash == original.hash
    for name, expected_type in TENSILELITE_PARAMETER_TYPES.items():
        assert type(imported.params[name]) is expected_type


def test_baseline_discovery_is_zero_evidence_planning_data(tmp_path):
    db = EvoTensileDB.connect(tmp_path / "baseline.sqlite")
    db.init()
    shape = pilot_100_shapes()[0]
    candidate = sample_candidates(1)[0]

    discovery_id = db.record_baseline_discovery(
        [
            BaselineSelectionInsert(
                shape=shape,
                candidate=candidate,
                hipblaslt_solution_index=17,
                hipblaslt_solution_name="installed-solution",
                logic_solution_index=3,
                query_gflops=12_345.0,
                query_time_us=4.5,
            )
        ],
        problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
        context={"logic": "installed", "baseline_label": "anchored-untuned"},
        duration_s=0.25,
    )

    assert discovery_id is not None
    discoveries = db.baseline_discoveries(baseline_label="anchored-untuned")
    pairs = db.baseline_selection_pairs(discovery_id)
    assert discoveries[0].discovery_id == discovery_id
    assert discoveries[0].context["baseline_label"] == "anchored-untuned"
    assert pairs[0][0] == shape
    assert pairs[0][1].hash == candidate.hash
    counts = db.counts()
    assert counts["evidence_sources"] == 0
    assert counts["native_runs"] == 0
    assert counts["benchmark_events"] == 0
    assert counts["benchmark_samples"] == 0
    assert counts["validations"] == 0
    assert (
        db.rank_benchmarks(
            problem_type_hash=DEFAULT_PROFILE.problem_type_hash,
            benchmark_protocol_hash=DEFAULT_PROFILE.benchmark_protocol_hash(),
        )
        == []
    )
