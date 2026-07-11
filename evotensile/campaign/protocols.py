from evotensile.protocol import BenchmarkProtocol

CAMPAIGN_SCREENING_PROTOCOL = BenchmarkProtocol(
    num_warmups=1,
    num_benchmarks=2,
    enqueues_per_sync=1,
    syncs_per_benchmark=1,
)

CAMPAIGN_HOT_PROTOCOL = BenchmarkProtocol(
    num_warmups=20,
    num_benchmarks=10,
    enqueues_per_sync=10,
    syncs_per_benchmark=1,
    num_elements_to_validate=0,
    validation_backend=CAMPAIGN_SCREENING_PROTOCOL.validation_backend,
)


def protocol_launches(protocol: BenchmarkProtocol) -> int:
    return protocol.num_warmups + (protocol.num_benchmarks * protocol.enqueues_per_sync * protocol.syncs_per_benchmark)
