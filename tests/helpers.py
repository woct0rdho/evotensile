from evotensile.candidate import Candidate
from evotensile.search_space import make_candidate, random_candidates


def sample_candidates(count: int, *, seed: int = 1151) -> list[Candidate]:
    return random_candidates(count, seed=seed)


def sample_candidate(*, seed: int = 1151) -> Candidate:
    return sample_candidates(1, seed=seed)[0]


DOCUMENTED_WINNER_CANDIDATE = make_candidate(
    {
        "MatrixInstruction": [16, 16, 16, 1, 1, 4, 4, 2, 2],
        "WorkGroup": [16, 16, 1],
        "DepthU": 16,
        "GlobalSplitU": 1,
        "PrefetchGlobalRead": 1,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 3,
        "WorkGroupMapping": 8,
        "StaggerU": 32,
        "StaggerUMapping": 0,
        "SourceSwap": True,
        "1LDSBuffer": 1,
        "ClusterLocalRead": 0,
        "VectorWidthB": 2,
        "GlobalReadVectorWidthA": 8,
        "GlobalReadVectorWidthB": 8,
        "StorePriorityOpt": False,
        "NumElementsPerBatchStore": 10,
    },
    source="documented_winner",
)
