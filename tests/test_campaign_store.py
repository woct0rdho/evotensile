from evotensile.search.campaign_control import tag_generated_proposals
from tests.helpers import sample_candidates


def test_candidate_checkpoint_payload_preserves_exact_hash_and_metadata():
    source = sample_candidates(1)
    candidate = tag_generated_proposals(
        source,
        generated_hashes={source[0].hash},
        island_id="island-0",
        proposal_cost_s=0.25,
    )[0]

    restored = candidate.from_mapping(candidate.to_mapping(hash_key="candidate_hash"))

    assert restored.hash == candidate.hash
    assert restored.source == candidate.source
    assert restored.parent_hashes == candidate.parent_hashes
    assert restored.proposal_metadata == candidate.proposal_metadata
