from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from evotensile.adaptive_retime import (
    AdaptivePolicy,
    ProbePolicy,
    decide_retime_by_shape,
    decide_shape_probe,
    load_timing_stats,
)
from evotensile.candidate import Candidate, Shape
from evotensile.database import BenchmarkEventInsert, EvoTensileDB
from evotensile.protocol import BenchmarkProtocol
from evotensile.scheduling.models import BatchIngestResult, ExecutedBatch, PlannedBatch, PreparedBatch, ScheduleResult
from evotensile.scheduling.structured import record_structured_run
from evotensile.structured_runner import RunnablePair, run_structured_phase, validate_benchmark_samples


@dataclass(frozen=True)
class TimingContext:
    db: EvoTensileDB
    shapes: list[Shape]
    candidates: list[Candidate]
    planned_batches: list[PlannedBatch]
    protocol: BenchmarkProtocol
    problem_type_hash: str
    benchmark_protocol_hash: str
    validation_protocol_hash: str
    runner_bin: str | Path
    runner_timeout_s: float | None
    keep_going: bool
    initial_samples: int
    adaptive_policy: AdaptivePolicy | None
    probe_policy: ProbePolicy | None
    probe_protocol: BenchmarkProtocol | None
    probe_protocol_hash: str | None
    probe_policy_hash: str | None
    adaptive_max_rounds: int
    preprepare_screened_pairs: int = 0
    timing_batch_order: Callable[[Sequence[PreparedBatch]], Sequence[PreparedBatch]] | None = None


def _ingest_result_from_inserts(
    inserts: list[BenchmarkEventInsert], *, errors: list[str] | None = None
) -> BatchIngestResult:
    status_counts: dict[str, int] = {}
    rejected = 0
    unmapped = 0
    inserted = 0
    for item in inserts:
        count = len(item.samples_us) if item.status == "ok" else 1
        status_counts[item.status] = status_counts.get(item.status, 0) + count
        if item.status == "rejected":
            rejected += 1
        elif item.status == "unmapped":
            unmapped += 1
        else:
            inserted += 1
    return BatchIngestResult(
        inserted=inserted,
        unmapped=unmapped,
        status_counts=status_counts,
        rejected=rejected,
        errors=errors or [],
    )


def _probe_survivor_keys(
    db: EvoTensileDB,
    *,
    shapes: list[Shape],
    candidates: list[Candidate],
    available_pairs: set[tuple[str, str]],
    problem_type_hash: str,
    probe_protocol_hash: str,
    benchmark_protocol_hash: str,
    policy: ProbePolicy,
    min_samples: int,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    shape_ids = {shape.id for shape in shapes}
    candidate_hashes = {candidate.hash for candidate in candidates}
    probe_stats = load_timing_stats(
        db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hashes=[probe_protocol_hash],
        min_samples=min_samples,
        shape_ids=shape_ids,
        candidate_hashes=candidate_hashes,
    )
    reference_stats = load_timing_stats(
        db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hashes=[benchmark_protocol_hash],
        min_samples=1,
        shape_ids=shape_ids,
    )
    survivors: set[tuple[str, str]] = set()
    for shape_id, stats in probe_stats.items():
        eligible = [stats_item for stats_item in stats if (shape_id, stats_item.candidate_hash) in available_pairs]
        if not eligible:
            continue
        decision = decide_shape_probe(
            shape_id,
            eligible,
            policy=policy,
            reference_stats=reference_stats.get(shape_id, ()),
        )
        survivors.update((shape_id, candidate_hash) for candidate_hash in decision.survivor_hashes)
    return survivors, available_pairs - survivors


def _adaptive_topup_groups(
    db: EvoTensileDB,
    *,
    shapes: list[Shape],
    candidates: list[Candidate],
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    policy: AdaptivePolicy,
    min_samples: int,
) -> list[tuple[int, list[Shape], list[Candidate]]]:
    shape_by_id = {shape.id: shape for shape in shapes}
    candidate_by_hash = {candidate.hash: candidate for candidate in candidates}
    stats_by_shape = load_timing_stats(
        db,
        problem_type_hash=problem_type_hash,
        benchmark_protocol_hashes=[benchmark_protocol_hash],
        min_samples=min_samples,
        shape_ids=set(shape_by_id),
        candidate_hashes=set(candidate_by_hash),
    )
    decisions = decide_retime_by_shape(stats_by_shape, policy=policy)
    grouped: dict[tuple[int, tuple[str, ...]], list[Shape]] = {}
    rank_order_by_key: dict[tuple[int, tuple[str, ...]], dict[str, int]] = {}
    for decision in decisions.values():
        if not decision.needs_retime or decision.target_samples <= 0:
            continue
        available_hashes = tuple(
            candidate_hash for candidate_hash in decision.retime_candidate_hashes if candidate_hash in candidate_by_hash
        )
        if len(available_hashes) < 2:
            continue
        key = (decision.target_samples, tuple(sorted(available_hashes)))
        shape = shape_by_id.get(decision.shape_id)
        if shape is None:
            continue
        grouped.setdefault(key, []).append(shape)
        ranks = rank_order_by_key.setdefault(key, {})
        for rank, candidate_hash in enumerate(available_hashes):
            ranks.setdefault(candidate_hash, rank)

    groups: list[tuple[int, list[Shape], list[Candidate]]] = []
    for (target_samples, candidate_hashes), group_shapes in sorted(
        grouped.items(), key=lambda item: (item[0][0], len(item[1]), item[0][1]), reverse=True
    ):
        ranks = rank_order_by_key[(target_samples, candidate_hashes)]
        ordered_hashes = sorted(candidate_hashes, key=lambda candidate_hash: (ranks[candidate_hash], candidate_hash))
        groups.append(
            (
                target_samples,
                sorted(group_shapes, key=lambda shape: shape.id),
                [candidate_by_hash[h] for h in ordered_hashes],
            )
        )
    return groups


def _benchmark_prepared_pairs(
    db: EvoTensileDB,
    prepared: PreparedBatch,
    *,
    pairs: list[RunnablePair],
    protocol: BenchmarkProtocol,
    problem_type_hash: str,
    benchmark_protocol_hash: str,
    validation_protocol_hash: str,
    runner_bin: str | Path,
    runner_timeout_s: float | None,
    phase: str,
    include_preparation: bool = False,
) -> ExecutedBatch:
    preparation_inserts = prepared.preparation_inserts if include_preparation else []
    preparation_ingest = _ingest_result_from_inserts(preparation_inserts, errors=prepared.errors)
    if not pairs or prepared.library_dir is None or prepared.errors:
        return ExecutedBatch(
            planned=prepared.planned,
            yaml_path=prepared.yaml_path,
            manifest_path=prepared.manifest_path,
            output_dir=prepared.output_dir,
            build_returncode=prepared.build_result.returncode,
            validation_returncode=(
                prepared.validation_result.returncode if prepared.validation_result is not None else None
            ),
            ingest=preparation_ingest,
            build_output_dir=prepared.build_output_dir,
            phase=phase,
        )

    benchmark_protocol = protocol.with_overrides(num_elements_to_validate=0)
    output = run_structured_phase(
        mode="benchmark",
        run_dir=prepared.output_dir,
        pairs=pairs,
        shapes=prepared.planned.shapes,
        protocol=benchmark_protocol,
        runner_bin=runner_bin,
        library_dir=prepared.library_dir,
        timeout_s=runner_timeout_s,
    )
    record_structured_run(
        db,
        output,
        yaml_path=prepared.yaml_path,
        output_dir=prepared.output_dir,
        pairs=pairs,
        cost_phase="probe" if protocol.num_warmups == 0 else "screening",
    )
    errors = list(prepared.errors)
    if output.timed_out:
        timing_inserts = [
            BenchmarkEventInsert(
                shape_id=pair.shape_id,
                candidate_hash=pair.candidate_hash,
                run_id=output.run_id,
                status="runner_timeout",
                source_kind="native_run",
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                solution_index=pair.library_solution_index,
            )
            for pair in pairs
        ]
        errors.append(f"benchmark phase timed out after {runner_timeout_s} seconds")
    else:
        try:
            timing_inserts = validate_benchmark_samples(
                output.samples,
                runnable_pairs=pairs,
                protocol=benchmark_protocol,
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                validation_protocol_hash=validation_protocol_hash,
                run_id=output.run_id,
                runner_returncode=output.returncode,
            )
        except Exception as exc:
            timing_inserts = []
            errors.append(str(exc))
    if timing_inserts:
        db.insert_benchmark_events(timing_inserts)
    combined = [*preparation_inserts, *timing_inserts]
    return ExecutedBatch(
        planned=prepared.planned,
        yaml_path=prepared.yaml_path,
        manifest_path=prepared.manifest_path,
        output_dir=prepared.output_dir,
        build_returncode=prepared.build_result.returncode,
        validation_returncode=(
            prepared.validation_result.returncode if prepared.validation_result is not None else None
        ),
        runner_returncode=output.returncode,
        ingest=_ingest_result_from_inserts(combined, errors=errors),
        build_output_dir=prepared.build_output_dir,
        phase=phase,
    )


def run_serial_timing(context: TimingContext, prepared: list[PreparedBatch]) -> ScheduleResult:
    db = context.db
    shapes = context.shapes
    candidates = context.candidates
    planned = context.planned_batches
    protocol = context.protocol
    problem_type_hash = context.problem_type_hash
    benchmark_protocol_hash = context.benchmark_protocol_hash
    validation_protocol_hash = context.validation_protocol_hash
    runner_bin = context.runner_bin
    runner_timeout_s = context.runner_timeout_s
    keep_going = context.keep_going
    initial_samples = context.initial_samples
    adaptive_policy = context.adaptive_policy
    probe_policy = context.probe_policy
    probe_protocol = context.probe_protocol
    probe_protocol_hash = context.probe_protocol_hash
    probe_policy_hash = context.probe_policy_hash
    adaptive_max_rounds = context.adaptive_max_rounds
    preprepare_screened_pair_count = context.preprepare_screened_pairs
    prepared_by_batch_index = {item.planned.batch_index: item for item in prepared}
    if context.timing_batch_order is not None:
        prepared = list(context.timing_batch_order(prepared))
        timing_indices = [item.planned.batch_index for item in prepared]
        if len(timing_indices) != len(prepared_by_batch_index) or set(timing_indices) != set(prepared_by_batch_index):
            raise ValueError("timing_batch_order must return every prepared batch exactly once")

    executed: list[ExecutedBatch] = []
    pair_owner: dict[tuple[str, str], tuple[PreparedBatch, RunnablePair]] = {}
    for item in prepared:
        for pair in item.validated_pairs:
            pair_owner[(pair.shape_id, pair.candidate_hash)] = (item, pair)

    if adaptive_policy is None:
        for item in prepared:
            benchmark = _benchmark_prepared_pairs(
                db,
                item,
                pairs=item.validated_pairs,
                protocol=protocol.with_overrides(num_benchmarks=item.planned.samples_per_pair),
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                validation_protocol_hash=validation_protocol_hash,
                runner_bin=runner_bin,
                runner_timeout_s=runner_timeout_s,
                phase="initial",
                include_preparation=True,
            )
            executed.append(benchmark)
            if not keep_going and benchmark.ingest is not None and not benchmark.ingest.ok:
                return ScheduleResult(planned_batches=planned, executed_batches=executed, completed_waves=1)
        return ScheduleResult(planned_batches=planned, executed_batches=executed, completed_waves=1)

    assert probe_policy is not None
    assert probe_protocol is not None
    assert probe_protocol_hash is not None
    assert probe_policy_hash is not None

    for item in prepared:
        if not item.validated_pairs:
            continue
        states = db.benchmark_evidence_states(
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=probe_protocol_hash,
            shape_ids=[shape.id for shape in item.planned.shapes],
            candidate_hashes=[candidate.hash for candidate in item.planned.candidates],
        )
        initial_pairs_by_remaining: dict[int, list[RunnablePair]] = {}
        for pair in item.validated_pairs:
            state = states.get((pair.shape_id, pair.candidate_hash))
            current_samples = 0 if state is None else state.ok_samples
            remaining = probe_policy.initial_samples - current_samples
            if remaining > 0:
                initial_pairs_by_remaining.setdefault(remaining, []).append(pair)
        for remaining, pairs in sorted(initial_pairs_by_remaining.items()):
            probe = _benchmark_prepared_pairs(
                db,
                item,
                pairs=pairs,
                protocol=probe_protocol.with_overrides(num_benchmarks=remaining),
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=probe_protocol_hash,
                validation_protocol_hash=validation_protocol_hash,
                runner_bin=runner_bin,
                runner_timeout_s=runner_timeout_s,
                phase="probe-initial",
            )
            executed.append(probe)
            if not keep_going and probe.ingest is not None and not probe.ingest.ok:
                return ScheduleResult(
                    planned_batches=planned,
                    executed_batches=executed,
                    completed_waves=1,
                    probe_protocol_hash=probe_protocol_hash,
                    probe_policy_hash=probe_policy_hash,
                    probe_screened_pairs=preprepare_screened_pair_count,
                    probe_preprepare_screened_pairs=preprepare_screened_pair_count,
                )

    provisional_survivor_keys, provisional_screened_keys = _probe_survivor_keys(
        db,
        shapes=shapes,
        candidates=candidates,
        available_pairs=set(pair_owner),
        problem_type_hash=problem_type_hash,
        probe_protocol_hash=probe_protocol_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        policy=probe_policy,
        min_samples=probe_policy.initial_samples,
    )

    for item in prepared:
        topup_pairs = [
            pair for pair in item.validated_pairs if (pair.shape_id, pair.candidate_hash) in provisional_survivor_keys
        ]
        if not topup_pairs:
            continue
        states = db.benchmark_evidence_states(
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=probe_protocol_hash,
            shape_ids=[shape.id for shape in item.planned.shapes],
            candidate_hashes=[candidate.hash for candidate in item.planned.candidates],
        )
        topup_pairs_by_remaining: dict[int, list[RunnablePair]] = {}
        for pair in topup_pairs:
            state = states.get((pair.shape_id, pair.candidate_hash))
            current_samples = 0 if state is None else state.ok_samples
            remaining = probe_policy.samples - current_samples
            if remaining > 0:
                topup_pairs_by_remaining.setdefault(remaining, []).append(pair)
        for remaining, pairs in sorted(topup_pairs_by_remaining.items()):
            probe = _benchmark_prepared_pairs(
                db,
                item,
                pairs=pairs,
                protocol=probe_protocol.with_overrides(num_benchmarks=remaining),
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=probe_protocol_hash,
                validation_protocol_hash=validation_protocol_hash,
                runner_bin=runner_bin,
                runner_timeout_s=runner_timeout_s,
                phase="probe-topup",
            )
            executed.append(probe)
            if not keep_going and probe.ingest is not None and not probe.ingest.ok:
                return ScheduleResult(
                    planned_batches=planned,
                    executed_batches=executed,
                    completed_waves=1,
                    probe_protocol_hash=probe_protocol_hash,
                    probe_policy_hash=probe_policy_hash,
                    probe_screened_pairs=preprepare_screened_pair_count,
                    probe_preprepare_screened_pairs=preprepare_screened_pair_count,
                )

    survivor_keys, final_screened_keys = _probe_survivor_keys(
        db,
        shapes=shapes,
        candidates=candidates,
        available_pairs=provisional_survivor_keys,
        problem_type_hash=problem_type_hash,
        probe_protocol_hash=probe_protocol_hash,
        benchmark_protocol_hash=benchmark_protocol_hash,
        policy=probe_policy,
        min_samples=probe_policy.samples,
    )
    screened_keys = provisional_screened_keys | final_screened_keys
    screened_pair_count = preprepare_screened_pair_count + len(screened_keys)

    # Phase 3: run the main timing protocol only for probe survivors.
    for item in prepared:
        main_pairs = [pair for pair in item.validated_pairs if (pair.shape_id, pair.candidate_hash) in survivor_keys]
        benchmark = _benchmark_prepared_pairs(
            db,
            item,
            pairs=main_pairs,
            protocol=protocol.with_overrides(num_benchmarks=item.planned.samples_per_pair),
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            validation_protocol_hash=validation_protocol_hash,
            runner_bin=runner_bin,
            runner_timeout_s=runner_timeout_s,
            phase="initial",
            include_preparation=True,
        )
        executed.append(benchmark)
        if not keep_going and benchmark.ingest is not None and not benchmark.ingest.ok:
            return ScheduleResult(
                planned_batches=planned,
                executed_batches=executed,
                completed_waves=1,
                probe_protocol_hash=probe_protocol_hash,
                probe_policy_hash=probe_policy_hash,
                probe_survivor_pairs=len(survivor_keys),
                probe_screened_pairs=screened_pair_count,
                probe_preprepare_screened_pairs=preprepare_screened_pair_count,
            )

    completed_rounds = 0
    for _ in range(adaptive_max_rounds):
        groups = _adaptive_topup_groups(
            db,
            shapes=shapes,
            candidates=candidates,
            problem_type_hash=problem_type_hash,
            benchmark_protocol_hash=benchmark_protocol_hash,
            policy=adaptive_policy,
            min_samples=initial_samples,
        )
        requests: dict[tuple[int, int], tuple[PreparedBatch, list[RunnablePair]]] = {}
        for target_samples, group_shapes, group_candidates in groups:
            states = db.benchmark_evidence_states(
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                shape_ids=[shape.id for shape in group_shapes],
                candidate_hashes=[candidate.hash for candidate in group_candidates],
            )
            for shape in group_shapes:
                for candidate in group_candidates:
                    owner = pair_owner.get((shape.id, candidate.hash))
                    if owner is None:
                        continue
                    state = states.get((shape.id, candidate.hash))
                    current_samples = 0 if state is None else state.ok_samples
                    remaining = target_samples - current_samples
                    if remaining <= 0:
                        continue
                    prepared_batch, pair = owner
                    key = (id(prepared_batch), remaining)
                    request = requests.setdefault(key, (prepared_batch, []))
                    request[1].append(pair)
        if not requests:
            break

        ran_round = False
        for (_, remaining), (prepared_batch, pairs) in requests.items():
            topup = _benchmark_prepared_pairs(
                db,
                prepared_batch,
                pairs=pairs,
                protocol=protocol.with_overrides(num_benchmarks=remaining),
                problem_type_hash=problem_type_hash,
                benchmark_protocol_hash=benchmark_protocol_hash,
                validation_protocol_hash=validation_protocol_hash,
                runner_bin=runner_bin,
                runner_timeout_s=runner_timeout_s,
                phase="adaptive",
            )
            executed.append(topup)
            ran_round = True
            if not keep_going and topup.ingest is not None and not topup.ingest.ok:
                return ScheduleResult(
                    planned_batches=planned,
                    executed_batches=executed,
                    completed_waves=1,
                    adaptive_rounds=completed_rounds,
                    probe_protocol_hash=probe_protocol_hash,
                    probe_policy_hash=probe_policy_hash,
                    probe_survivor_pairs=len(survivor_keys),
                    probe_screened_pairs=screened_pair_count,
                    probe_preprepare_screened_pairs=preprepare_screened_pair_count,
                )
        if not ran_round:
            break
        completed_rounds += 1

    return ScheduleResult(
        planned_batches=planned,
        executed_batches=executed,
        completed_waves=1,
        adaptive_rounds=completed_rounds,
        probe_protocol_hash=probe_protocol_hash,
        probe_policy_hash=probe_policy_hash,
        probe_survivor_pairs=len(survivor_keys),
        probe_screened_pairs=screened_pair_count,
        probe_preprepare_screened_pairs=preprepare_screened_pair_count,
    )
