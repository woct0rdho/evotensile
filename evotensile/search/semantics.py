from evotensile.search.encoding import PARAM_NAMES

NT_HHS_SEMANTIC_GROUPS: tuple[tuple[str, ...], ...] = (
    ("MatrixInstruction", "WorkGroup", "DepthU", "GlobalSplitU"),
    ("TransposeLDS", "LdsBlockSizePerPadA", "LdsBlockSizePerPadB", "LdsPadA", "LdsPadB"),
    ("PrefetchGlobalRead", "PrefetchLocalRead", "1LDSBuffer", "ClusterLocalRead", "VectorWidthB"),
    ("GlobalReadVectorWidthA", "GlobalReadVectorWidthB", "VectorWidthA", "VectorWidthB"),
    ("ScheduleIterAlg", "WorkGroupMapping", "StaggerU", "StaggerUStride", "StaggerUMapping", "SourceSwap"),
    ("StorePriorityOpt", "NumElementsPerBatchStore", "StoreSyncOpt", "GroupLoadStore", "StoreVectorWidth"),
    ("ExpandPointerSwap",),
    ("AssertFree0ElementMultiple", "AssertFree1ElementMultiple", "AssertSummationElementMultiple"),
)


def semantic_group_names() -> tuple[tuple[str, ...], ...]:
    groups = [*NT_HHS_SEMANTIC_GROUPS, *tuple((name,) for name in PARAM_NAMES)]
    return tuple(dict.fromkeys(groups))
