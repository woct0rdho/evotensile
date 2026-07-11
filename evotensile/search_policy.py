from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class SearchPolicy:
    name: str
    target_profile_name: str
    proposal: str
    surrogate_pool_multiplier: int
    adaptive_operators: bool
    adaptive_group_credit: bool
    micro_exhaustive_neighborhoods: bool
    adaptive_donor_selection: bool
    cost_aware_operator_credit: bool
    covering_cold_start: bool
    cost_aware_scheduling: bool

    def settings(self) -> dict[str, object]:
        values = asdict(self)
        values.pop("name")
        values.pop("target_profile_name")
        return values


GFX1151_GRID_V1 = SearchPolicy(
    name="gfx1151-grid-v1",
    target_profile_name="gfx1151-nt-hhs",
    proposal="family-qd",
    surrogate_pool_multiplier=8,
    adaptive_operators=True,
    adaptive_group_credit=True,
    micro_exhaustive_neighborhoods=True,
    adaptive_donor_selection=True,
    cost_aware_operator_credit=True,
    covering_cold_start=True,
    cost_aware_scheduling=True,
)

SEARCH_POLICIES = {GFX1151_GRID_V1.name: GFX1151_GRID_V1}


def get_search_policy(name: str | None, *, target_profile_name: str) -> SearchPolicy | None:
    if name is None:
        return None
    try:
        policy = SEARCH_POLICIES[name]
    except KeyError as exc:
        raise ValueError(f"unknown search policy: {name}") from exc
    if policy.target_profile_name != target_profile_name:
        raise ValueError(
            f"search policy {name} requires profile {policy.target_profile_name}, got {target_profile_name}"
        )
    return policy
