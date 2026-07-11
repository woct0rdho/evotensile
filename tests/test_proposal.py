import json
from pathlib import Path

import pytest

from evotensile.candidate import Candidate
from evotensile.cli import build_parser
from evotensile.cli import main as cli_main
from evotensile.database import EvoTensileDB
from evotensile.profile import DEFAULT_PROFILE
from evotensile.proposal import ProposalOutput, ProviderProvenance
from evotensile.proposals import random_candidates
from evotensile.search.acquisition import propose_candidates
from evotensile.shapes import pilot_100_shapes
from tests.helpers import REFERENCE_CANDIDATE


def test_public_proposal_building_blocks_are_importable():
    assert len(random_candidates(2, seed=1151)) == 2


def test_custom_script_records_best_effort_provenance_and_warns(tmp_path: Path, capsys):
    script = tmp_path / "provider.py"
    script.write_text(
        """\
from evotensile.proposals import random_candidate
import random

PROVIDER_NAME = "test-provider"
PROVIDER_VERSION = "1"

def propose(context):
    rng = random.Random(context.seed)
    return [random_candidate(rng, target_shapes=context.shapes) for _ in range(context.config["count"])]
""",
        encoding="utf-8",
    )
    config = tmp_path / "provider.json"
    config.write_text(json.dumps({"count": 2}), encoding="utf-8")
    output = tmp_path / "output"

    assert (
        cli_main(
            [
                "schedule-batches",
                "--db",
                str(tmp_path / "sched.sqlite"),
                "--output-dir",
                str(output),
                "--proposal-script",
                str(script),
                "--proposal-config",
                str(config),
                "--limit-shapes",
                "1",
                "--dry-run",
            ]
        )
        == 0
    )

    captured = capsys.readouterr()
    assert "not automatically reproducible" in captured.err
    metadata = json.loads((output / "schedule_metadata.json").read_text(encoding="utf-8"))
    provider = metadata["proposal_provider"]
    assert provider["identity"].startswith("script:")
    assert provider["script_sha256"]
    assert provider["declared_name"] == "test-provider"
    assert provider["declared_version"] == "1"
    assert provider["environment_compatibility_tag"] == DEFAULT_PROFILE.environment_compatibility_tag
    assert metadata["candidates"] == 2


def test_custom_script_rejects_family_qd_options(tmp_path: Path):
    script = tmp_path / "provider.py"
    script.write_text("def propose(context):\n    return []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="custom proposal scripts cannot use family-QD options: --num-random"):
        cli_main(
            [
                "proposal-coverage",
                "--db",
                str(tmp_path / "sched.sqlite"),
                "--proposal-script",
                str(script),
                "--num-random",
                "2",
                "--limit-shapes",
                "1",
            ]
        )


def test_proposal_config_requires_script(tmp_path: Path):
    config = tmp_path / "provider.json"
    config.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="requires --proposal-script"):
        cli_main(
            [
                "proposal-coverage",
                "--db",
                str(tmp_path / "sched.sqlite"),
                "--proposal-config",
                str(config),
                "--limit-shapes",
                "1",
            ]
        )


def test_proposal_config_requires_json_object(tmp_path: Path):
    script = tmp_path / "provider.py"
    script.write_text("def propose(context):\n    return []\n", encoding="utf-8")
    config = tmp_path / "provider.json"
    config.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="one JSON object"):
        cli_main(
            [
                "proposal-coverage",
                "--db",
                str(tmp_path / "sched.sqlite"),
                "--proposal-script",
                str(script),
                "--proposal-config",
                str(config),
                "--limit-shapes",
                "1",
            ]
        )


def test_provider_rejects_incomplete_candidate(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()

    with pytest.raises(ValueError, match="incomplete parameters"):
        propose_candidates(
            db,
            provider=lambda context: [Candidate(params={"DepthU": 32}, source="bad")],
            provider_provenance=ProviderProvenance(identity="test:bad"),
            target_shapes=pilot_100_shapes()[:1],
        )


def test_provider_rejects_selected_hash_outside_pool(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()

    with pytest.raises(ValueError, match="not in the provider pool"):
        propose_candidates(
            db,
            provider=lambda context: ProposalOutput(
                candidates=(REFERENCE_CANDIDATE,),
                selected_candidate_hashes=("cand_missing",),
            ),
            provider_provenance=ProviderProvenance(identity="test:selected"),
            target_shapes=pilot_100_shapes()[:1],
        )


def test_provider_rejects_unknown_parent(tmp_path: Path):
    db = EvoTensileDB.connect(tmp_path / "sched.sqlite")
    db.init()
    child = Candidate(
        params=REFERENCE_CANDIDATE.canonical_params(),
        source="child",
        parent_hashes=("cand_missing",),
    )

    with pytest.raises(ValueError, match="unknown parents"):
        propose_candidates(
            db,
            provider=lambda context: [child],
            provider_provenance=ProviderProvenance(identity="test:parent"),
            target_shapes=pilot_100_shapes()[:1],
        )


def test_provider_options_exist_only_on_proposal_commands(tmp_path: Path):
    parser = build_parser()
    for command in ("proposal-coverage", "schedule-batches", "repair-outliers"):
        arguments = [command, "--db", str(tmp_path / "sched.sqlite"), "--proposal-script", "provider.py"]
        if command != "proposal-coverage":
            arguments.extend(("--output-dir", str(tmp_path / command)))
        assert parser.parse_args(arguments).proposal_script == Path("provider.py")

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["summarize-cache", "--db", str(tmp_path / "sched.sqlite"), "--proposal-script", "provider.py"]
        )
