from evotensile.protocol import DEFAULT_BENCHMARK_PROTOCOL, apply_benchmark_protocol_overrides
from evotensile.search_space import DOMAINS
from evotensile.shapes import pilot_100_shapes
from evotensile.yaml_writer import tensilelite_config
from tests.helpers import sample_candidates


def test_protocol_overrides_apply_supported_present_values_only():
    protocol = apply_benchmark_protocol_overrides(
        DEFAULT_BENCHMARK_PROTOCOL,
        {
            "num_warmups": 3,
            "num_benchmarks": None,
            "validation_backend": "cpu",
            "unrelated": 99,
        },
    )

    assert protocol.num_warmups == 3
    assert protocol.num_benchmarks == DEFAULT_BENCHMARK_PROTOCOL.num_benchmarks
    assert protocol.validation_backend == "cpu"


def test_expanded_space_contains_artifact_backed_knobs():
    assert 0 in DOMAINS["NumElementsPerBatchStore"]
    assert 32 in DOMAINS["NumElementsPerBatchStore"]
    assert 16 in DOMAINS["WorkGroupMapping"]
    assert 64 in DOMAINS["StaggerU"]
    assert True in DOMAINS["GroupLoadStore"]
    assert 10 in DOMAINS["NumElementsPerBatchStore"]


def test_yaml_shape():
    cands = sample_candidates(2)
    shapes = pilot_100_shapes()[:3]
    data = tensilelite_config(cands, shapes)
    assert data["GlobalParameters"]["MinimumRequiredVersion"] == "5.0.0"
    assert "BenchmarkProblems" in data
    group = data["BenchmarkProblems"][0][1]["ForkParameters"][0]["Groups"][0]
    assert len(group) == 2
    sizes = data["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"][0]["ProblemSizes"]
    assert len(sizes) == 3
