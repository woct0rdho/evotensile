MATRIX_INSTRUCTION_KEY = "MatrixInstruction"
MI_WAVE_TILE_KEY = "MIWaveTile"
MI_WAVE_GROUP_KEY = "MIWaveGroup"
STORE_VECTOR_WIDTH_KEY = "StoreVectorWidth"
WORK_GROUP_KEY = "WorkGroup"
WORK_GROUP_MAPPING_KEY = "WorkGroupMapping"

SOLUTION_INDEX_KEY = "SolutionIndex"
SOLUTION_NAME_MIN_KEY = "SolutionNameMin"
KERNEL_NAME_MIN_KEY = "KernelNameMin"

PROBLEM_SIZES_KEY = "ProblemSizes"
EXACT_KEY = "Exact"

# Keys that survive TensileLite solution construction without known rewriting for
# the current EvoTensile candidate model. Derived/normalized keys such as
# WorkGroup and StoreVectorWidth are handled separately or ignored.
DIRECT_SOLUTION_MATCH_KEYS = frozenset(
    {
        "1LDSBuffer",
        "AssertFree0ElementMultiple",
        "AssertFree1ElementMultiple",
        "AssertSummationElementMultiple",
        "ClusterLocalRead",
        "DepthU",
        "ExpandPointerSwap",
        "GlobalReadVectorWidthA",
        "GlobalReadVectorWidthB",
        "GlobalSplitU",
        "GlobalSplitUAlgorithm",
        "GroupLoadStore",
        "KernelLanguage",
        "LdsBlockSizePerPadA",
        "LdsBlockSizePerPadB",
        "LdsPadA",
        "LdsPadB",
        "LocalReadVectorWidth",
        "MIArchVgpr",
        "NumElementsPerBatchStore",
        "PrefetchGlobalRead",
        "PrefetchLocalRead",
        "ScheduleGlobalRead",
        "ScheduleIterAlg",
        "ScheduleLocalWrite",
        "SourceSwap",
        "StaggerU",
        "StaggerUMapping",
        "StaggerUStride",
        "StorePriorityOpt",
        "StoreRemapVectorWidth",
        "StoreSyncOpt",
        "TransposeLDS",
        "VectorWidthA",
        "VectorWidthB",
        "WavefrontSize",
        WORK_GROUP_MAPPING_KEY,
    }
)

SOLUTION_YAML_GLOBS = (
    "**/*_Final.yaml",
    "**/*_CSVWinner.yaml",
    "**/Data/*.yaml",
    "**/2_BenchmarkData/*.yaml",
)
