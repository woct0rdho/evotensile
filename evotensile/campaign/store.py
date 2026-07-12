import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TypedDict, cast

from evotensile.campaign.controller import (
    CampaignControllerState,
    CampaignSummaryPayload,
    ControllerCheckpointPayload,
)
from evotensile.campaign.models import CampaignConfiguration, RoundProposal
from evotensile.candidate import Candidate
from evotensile.search.campaign_control import ProposalEvent


class CampaignRecord(TypedDict, total=False):
    blind: bool
    configuration: dict[str, object]
    configuration_hash: str
    screening_protocol_hash: str
    validation_protocol_hash: str
    hot_protocol_hash: str
    rounds: list[Mapping[str, object]]
    restart_counters: dict[str, int]
    stop_reason: str | None
    controller: CampaignSummaryPayload
    confirmation_reserve_s: float
    elapsed_s: float
    budget_overrun_s: float
    screening_leader: object
    hot_leader: object
    hot_confirmed: int


class StoredCampaignCheckpoint(TypedDict, total=False):
    controller: ControllerCheckpointPayload
    round_seed: int | None
    candidate_hashes: list[str]
    configuration_hash: str
    restart_counters: dict[str, int]
    deterministic_rng: str
    operator_credit_state: str
    surrogate_state: str


class CampaignStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def db_path(self) -> Path:
        return self.root / "campaign.sqlite"

    @property
    def checkpoint_path(self) -> Path:
        return self.root / "campaign_checkpoint.json"

    @property
    def summary_path(self) -> Path:
        return self.root / "campaign_summary.json"

    @property
    def compile_cache_path(self) -> Path:
        return self.root / "compile_cache"

    def round_dir(self, round_index: int) -> Path:
        path = self.root / f"round_{round_index:02d}"
        path.mkdir(exist_ok=True)
        return path

    def _write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)

    def load_checkpoint(self) -> StoredCampaignCheckpoint:
        if not self.checkpoint_path.exists():
            return {}
        return cast(
            StoredCampaignCheckpoint,
            json.loads(self.checkpoint_path.read_text(encoding="utf-8")),
        )

    def load_or_create(
        self,
        configuration: CampaignConfiguration,
        *,
        resume: bool,
        island_ids: Sequence[str],
    ) -> tuple[CampaignRecord, bool]:
        configuration_path = self.root / "campaign_configuration.json"
        progress_path = self.root / "campaign_progress.json"
        expected_configuration = configuration.to_dict()
        if self.root.exists():
            if not resume:
                raise SystemExit(f"output already exists: {self.root}")
            if not configuration_path.exists():
                raise SystemExit(f"cannot resume without {configuration_path}")
            frozen_configuration = json.loads(configuration_path.read_text(encoding="utf-8"))
            if frozen_configuration != expected_configuration:
                raise SystemExit("resume configuration mismatch; start a new campaign root")
            record = cast(
                CampaignRecord,
                json.loads(
                    (progress_path if progress_path.exists() else configuration_path).read_text(encoding="utf-8")
                ),
            )
            if record.get("configuration_hash") != configuration.identity_hash:
                raise SystemExit("resume configuration hash mismatch; start a new campaign root")
            return record, True

        self.root.mkdir(parents=True)
        self._write_json(configuration_path, expected_configuration)
        record: CampaignRecord = {
            "blind": True,
            "configuration": expected_configuration,
            "configuration_hash": configuration.identity_hash,
            "screening_protocol_hash": configuration.screening_protocol.protocol_hash(),
            "validation_protocol_hash": configuration.screening_protocol.validation_protocol_hash(),
            "hot_protocol_hash": configuration.hot_protocol.protocol_hash(),
            "rounds": [],
            "restart_counters": {**{island_id: 0 for island_id in island_ids}, "merged": 0},
            "stop_reason": None,
        }
        self.write_progress(record)
        return record, False

    def write_progress(self, record: Mapping[str, object]) -> None:
        self._write_json(self.root / "campaign_progress.json", record)

    def write_summary(self, record: Mapping[str, object]) -> None:
        self._write_json(self.summary_path, record)

    def write_proposal(self, round_index: int, seed: int, proposal: RoundProposal) -> None:
        self._write_json(
            self.round_dir(round_index) / "proposals.json",
            {
                "round": round_index,
                "seed": seed,
                "proposal_events": [event.to_dict() for event in proposal.events],
                "active_candidate_hashes": [candidate.hash for candidate in proposal.active],
                "archive_candidate_hashes": [candidate.hash for candidate in proposal.archive],
                "candidates": [candidate.to_mapping(hash_key="candidate_hash") for candidate in proposal.selected],
            },
        )

    def load_proposal(self, round_index: int) -> RoundProposal:
        payload = json.loads((self.round_dir(round_index) / "proposals.json").read_text(encoding="utf-8"))
        candidates = [Candidate.from_mapping(item) for item in payload["candidates"]]
        by_hash = {candidate.hash: candidate for candidate in candidates}
        return RoundProposal(
            selected=tuple(candidates),
            active=tuple(by_hash[candidate_hash] for candidate_hash in payload["active_candidate_hashes"]),
            archive=tuple(by_hash[candidate_hash] for candidate_hash in payload["archive_candidate_hashes"]),
            events=tuple(ProposalEvent.from_mapping(event) for event in payload["proposal_events"]),
        )

    def write_checkpoint(
        self,
        *,
        record: Mapping[str, object],
        controller: CampaignControllerState,
        round_seed: int | None,
        candidate_hashes: Sequence[str],
    ) -> None:
        self._write_json(
            self.checkpoint_path,
            {
                "controller": controller.to_checkpoint(),
                "round_seed": round_seed,
                "candidate_hashes": list(candidate_hashes),
                "configuration_hash": record["configuration_hash"],
                "restart_counters": record["restart_counters"],
                "deterministic_rng": "round and proposal-event seeds fully determine generator and surrogate RNG state",
                "operator_credit_state": "derived from the checkpointed campaign DB",
                "surrogate_state": "refit deterministically from the checkpointed campaign DB and stored proposals",
            },
        )
