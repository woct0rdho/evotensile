# Custom Proposal Providers

EvoTensile uses built-in family-QD when no custom provider is supplied. Use a custom provider only when a small Python composition is clearer than changing the built-in policy.

A provider is a trusted Python file loaded into the EvoTensile process. It must export:

```python
def propose(context):
    ...
    return candidates
```

Run it with:

```bash
python3 -m evotensile.cli schedule-batches \
  --db out/custom.sqlite \
  --output-dir out/custom \
  --proposal-script providers/example.py \
  --proposal-config providers/example.json
```

`--proposal-config` is optional and must contain one JSON object. Family-QD-specific CLI options cannot be combined with `--proposal-script`. The provider owns its composition.

## Provider Context

`ProposalContext` supplies immutable explicit inputs:
- `target_profile`: selected target and environment identity.
- `shapes` and `scope`: exact proposal shape scope.
- `seed`: deterministic seed for the call.
- `evidence`: immutable compatible DB evidence snapshot.
- `config`: the canonical JSON configuration mapping.
- `shape_id`: optional shape-local elite-selection hint.
- campaign-only parent, island, restart, and coverage inputs when applicable.

A provider can return an iterable of `Candidate` or `ProposalOutput`. `ProposalOutput` additionally selects a subset of its candidate pool and records provider metadata.

The common finalizer—not the provider—owns complete parameter validation, global and shape-scope rules, deduplication, parent existence, selected-subset validation, generated-versus-preserved classification, lineage scope, persistence, and scheduling.

## Shape-Aware Random Provider

```python
import random

from evotensile.proposals import random_candidate

PROVIDER_NAME = "shape-aware-random"
PROVIDER_VERSION = "1"


def propose(context):
    count = int(context.config.get("count", 32))
    rng = random.Random(context.seed)
    candidates = {}
    while len(candidates) < count:
        candidate = random_candidate(rng, target_shapes=context.shapes)
        candidates[candidate.hash] = candidate
    return candidates.values()
```

Example configuration:

```json
{"count": 32}
```

## Evidence-Driven Local Provider

```python
from evotensile.proposals import ranked_elites, semantic_mutation_candidates

PROVIDER_NAME = "evidence-local"
PROVIDER_VERSION = "1"


def propose(context):
    count = int(context.config.get("count", 32))
    parents = ranked_elites(
        context.evidence,
        shape_id=context.shape_id,
        target_shapes=context.shapes,
        elite_count=int(context.config.get("elite_count", 8)),
    )
    return semantic_mutation_candidates(
        parents,
        count=count,
        seed=context.seed,
        target_shapes=context.shapes,
    )
```

With no compatible measured elites this provider returns no candidates. A provider that needs cold-start behavior can explicitly combine `random_candidate()` with the local path.

## Supported Building Blocks

Import maintained proposal APIs from `evotensile.proposals`, not implementation modules. The public module re-exports the underlying implementations without wrapper logic, including:
- random candidate construction.
- broad and semantic mutation.
- categorical differential evolution.
- GOMEA neighborhoods and donor mixing.
- ranked elites, transfer elites, family descriptors, and family archives.
- learned linkage, operator allocation, mechanical covering, and surrogate selection.
- `FamilyQDPolicy`, `ProposalContext`, `ProposalOutput`, and the built-in `family_qd_provider`.

Function signatures remain explicit. Custom providers choose counts, parents, shape scope, evidence thresholds, job limits, and target mechanics rather than relying on hidden provider defaults.

## Trust And Reproducibility

Custom proposal scripts execute arbitrary Python in process. Use only trusted files.

EvoTensile records best-effort diagnostic provenance: the resolved script path, SHA-256 of the top-level script file, optional `PROVIDER_NAME` and `PROVIDER_VERSION`, canonical provider config, installed EvoTensile version, seed, scope, candidate counts, and effective `environment_compatibility_tag`.

This is not a complete code identity or reproducibility guarantee. EvoTensile cannot discover or hash every imported module, installed package version, editable source tree, native library, environment variable, filesystem input, or other behavior-affecting state. Campaign resume can compare only fields that were explicitly recorded. Imported-code changes may remain undetected.

`environment_compatibility_tag` remains explicit and user-controlled. When provider code, imported dependencies, EvoTensile, native libraries, or other environment state changes in a way that should not share validation or timing evidence, change the selected `TargetProfile.environment_compatibility_tag` and use a fresh DB. Do not relabel an existing database to make incompatible evidence appear compatible.

If that limitation is unacceptable for a blind or audited campaign, keep the campaign on built-in family-QD and freeze the complete package and environment separately.
