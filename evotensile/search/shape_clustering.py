import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from evotensile.candidate import Shape
from evotensile.search_space import MATRIX_INSTRUCTIONS, macro_tile

DEFAULT_SHAPE_CLUSTER_COUNT = 16


@dataclass(frozen=True)
class ShapeClusteringConfiguration:
    workgroup_processor_count: int
    cluster_count: int | None = None
    distance_threshold: float | None = None
    macro_tile_family_count: int = 8
    max_iterations: int = 100

    def __post_init__(self) -> None:
        if self.workgroup_processor_count <= 0:
            raise ValueError("shape clustering requires a positive work-group processor count")
        if (self.cluster_count is None) == (self.distance_threshold is None):
            raise ValueError("shape clustering requires exactly one of cluster_count or distance_threshold")
        if self.cluster_count is not None and self.cluster_count <= 0:
            raise ValueError("shape cluster count must be positive")
        if self.distance_threshold is not None and (
            not math.isfinite(self.distance_threshold) or self.distance_threshold < 0.0
        ):
            raise ValueError("shape clustering distance threshold must be finite and nonnegative")
        if self.macro_tile_family_count <= 0:
            raise ValueError("shape clustering macro-tile family count must be positive")
        if self.max_iterations <= 0:
            raise ValueError("shape clustering maximum iterations must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "workgroup_processor_count": self.workgroup_processor_count,
            "cluster_count": self.cluster_count,
            "distance_threshold": self.distance_threshold,
            "macro_tile_family_count": self.macro_tile_family_count,
            "max_iterations": self.max_iterations,
        }


@dataclass(frozen=True)
class MechanicalShapeDescriptor:
    shape_id: str
    features: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {"shape_id": self.shape_id, "features": dict(sorted(self.features.items()))}


@dataclass(frozen=True)
class ShapeCluster:
    cluster_id: str
    medoid_shape_id: str
    shape_ids: tuple[str, ...]
    distances_to_medoid: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "cluster_id": self.cluster_id,
            "medoid_shape_id": self.medoid_shape_id,
            "shape_ids": list(self.shape_ids),
            "distances_to_medoid": dict(sorted(self.distances_to_medoid.items())),
        }


@dataclass(frozen=True)
class ShapeClustering:
    configuration: ShapeClusteringConfiguration
    macro_tile_families: tuple[tuple[int, int], ...]
    descriptors: dict[str, MechanicalShapeDescriptor]
    clusters: tuple[ShapeCluster, ...]

    @property
    def shape_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.descriptors))

    @property
    def medoid_shape_ids(self) -> tuple[str, ...]:
        return tuple(cluster.medoid_shape_id for cluster in self.clusters)

    @property
    def cluster_by_shape(self) -> dict[str, str]:
        return {shape_id: cluster.cluster_id for cluster in self.clusters for shape_id in cluster.shape_ids}

    def to_dict(self) -> dict[str, object]:
        return {
            "configuration": self.configuration.to_dict(),
            "shape_ids": list(self.shape_ids),
            "macro_tile_families": [list(tile) for tile in self.macro_tile_families],
            "descriptors": {
                shape_id: descriptor.to_dict() for shape_id, descriptor in sorted(self.descriptors.items())
            },
            "clusters": [cluster.to_dict() for cluster in self.clusters],
        }


def representative_macro_tile_families(count: int) -> tuple[tuple[int, int], ...]:
    if count <= 0:
        raise ValueError("macro-tile family count must be positive")
    tiles = sorted({macro_tile(instruction) for instruction in MATRIX_INSTRUCTIONS})
    if len(tiles) <= count:
        return tuple(tiles)

    def tile_point(tile: tuple[int, int]) -> tuple[float, float]:
        m, n = tile
        return math.log2(m * n), math.log2(m / n)

    points = {tile: tile_point(tile) for tile in tiles}

    def distance(left: tuple[int, int], right: tuple[int, int]) -> float:
        return math.dist(points[left], points[right])

    first = min(tiles, key=lambda tile: (sum(distance(tile, other) for other in tiles), tile))
    selected = [first]
    remaining = [tile for tile in tiles if tile != first]
    while remaining and len(selected) < count:
        chosen = max(
            remaining,
            key=lambda tile: (min(distance(tile, existing) for existing in selected), tile),
        )
        selected.append(chosen)
        remaining.remove(chosen)
    return tuple(sorted(selected))


def mechanical_shape_descriptor(
    shape: Shape,
    *,
    workgroup_processor_count: int,
    macro_tile_families: Sequence[tuple[int, int]],
) -> MechanicalShapeDescriptor:
    if workgroup_processor_count <= 0:
        raise ValueError("shape descriptor requires a positive work-group processor count")
    if not macro_tile_families:
        raise ValueError("shape descriptor requires macro-tile families")
    output_elements = shape.m * shape.n * shape.batch
    input_output_bytes = 2 * shape.batch * shape.m * shape.k + 2 * shape.batch * shape.n * shape.k + 4 * output_elements
    flops = 2.0 * output_elements * shape.k
    geometric_output = math.sqrt(shape.m * shape.n)
    features = {
        "shape:log2_m": math.log2(shape.m),
        "shape:log2_n": math.log2(shape.n),
        "shape:log2_k": math.log2(shape.k),
        "shape:log2_batch": math.log2(shape.batch),
        "shape:log2_m_over_n": math.log2(shape.m / shape.n),
        "shape:log2_k_over_m": math.log2(shape.k / shape.m),
        "shape:log2_k_over_n": math.log2(shape.k / shape.n),
        "shape:log2_output_elements": math.log2(output_elements),
        "shape:reduction_over_output": math.log2(shape.k / geometric_output),
        "shape:reduction_shortness": 1.0 / (1.0 + shape.k / 512.0),
        "shape:reduction_depth": math.log2(1.0 + shape.k / 256.0),
        "shape:arithmetic_intensity": flops / max(input_output_bytes, 1),
        "compat:m_aligned_16": float(shape.m % 16 == 0),
        "compat:n_aligned_16": float(shape.n % 16 == 0),
        "compat:k_aligned_16": float(shape.k % 16 == 0),
        "compat:m_aligned_64": float(shape.m % 64 == 0),
        "compat:n_aligned_64": float(shape.n % 64 == 0),
        "compat:k_aligned_64": float(shape.k % 64 == 0),
        "compat:batched": float(shape.batch > 1),
    }
    for index, (macro_tile_m, macro_tile_n) in enumerate(macro_tile_families):
        tiles_m = math.ceil(shape.m / macro_tile_m)
        tiles_n = math.ceil(shape.n / macro_tile_n)
        output_tiles = tiles_m * tiles_n * shape.batch
        workgroups_per_wgp = output_tiles / workgroup_processor_count
        wgp_rounds = max(1, math.ceil(workgroups_per_wgp))
        prefix = f"tile:{index:02d}:{macro_tile_m}x{macro_tile_n}"
        features.update(
            {
                f"{prefix}:fill_m": shape.m / (tiles_m * macro_tile_m),
                f"{prefix}:fill_n": shape.n / (tiles_n * macro_tile_n),
                f"{prefix}:log2_output_tiles": math.log2(output_tiles),
                f"{prefix}:log2_wgp_rounds": math.log2(wgp_rounds),
                f"{prefix}:wgp_granularity": workgroups_per_wgp / wgp_rounds,
            }
        )
    return MechanicalShapeDescriptor(shape.id, features)


def cluster_shapes(
    shapes: Sequence[Shape],
    configuration: ShapeClusteringConfiguration,
) -> ShapeClustering:
    shape_by_id = {shape.id: shape for shape in shapes}
    if not shape_by_id or len(shape_by_id) != len(shapes):
        raise ValueError("shape clustering requires non-empty unique shapes")
    macro_tiles = representative_macro_tile_families(configuration.macro_tile_family_count)
    descriptors = {
        shape.id: mechanical_shape_descriptor(
            shape,
            workgroup_processor_count=configuration.workgroup_processor_count,
            macro_tile_families=macro_tiles,
        )
        for shape in shapes
    }
    distances = _standardized_distances(descriptors)
    if len(shapes) == 1:
        shape_id = shapes[0].id
        clusters = (ShapeCluster("cluster_000", shape_id, (shape_id,), {shape_id: 0.0}),)
    elif configuration.cluster_count is not None:
        count = min(configuration.cluster_count, len(shapes))
        members = _fixed_count_clusters(tuple(sorted(shape_by_id)), distances, count, configuration.max_iterations)
        clusters = _materialize_clusters(members, distances)
    else:
        assert configuration.distance_threshold is not None
        members = _threshold_clusters(
            tuple(sorted(shape_by_id)),
            distances,
            configuration.distance_threshold,
        )
        clusters = _materialize_clusters(members, distances)
    return ShapeClustering(configuration, macro_tiles, descriptors, clusters)


def shape_descriptor_distances(clustering: ShapeClustering) -> dict[tuple[str, str], float]:
    return _standardized_distances(clustering.descriptors)


def _standardized_distances(
    descriptors: Mapping[str, MechanicalShapeDescriptor],
) -> dict[tuple[str, str], float]:
    feature_names = sorted(next(iter(descriptors.values())).features)
    if any(sorted(descriptor.features) != feature_names for descriptor in descriptors.values()):
        raise ValueError("shape descriptors must share one feature schema")
    centers = {
        name: statistics.fmean(descriptor.features[name] for descriptor in descriptors.values())
        for name in feature_names
    }
    scales = {}
    for name in feature_names:
        values = [descriptor.features[name] for descriptor in descriptors.values()]
        variance = statistics.fmean((value - centers[name]) ** 2 for value in values)
        scales[name] = math.sqrt(variance) if variance > 0.0 else 1.0
    vectors = {
        shape_id: tuple((descriptor.features[name] - centers[name]) / scales[name] for name in feature_names)
        for shape_id, descriptor in descriptors.items()
    }
    distances = {}
    for left_id, left in vectors.items():
        for right_id, right in vectors.items():
            distances[(left_id, right_id)] = math.dist(left, right)
    return distances


def _medoid(shape_ids: Sequence[str], distances: Mapping[tuple[str, str], float]) -> str:
    return min(
        shape_ids,
        key=lambda shape_id: (
            sum(distances[(shape_id, other_id)] for other_id in shape_ids),
            shape_id,
        ),
    )


def _assign_to_medoids(
    shape_ids: Sequence[str],
    medoids: Sequence[str],
    distances: Mapping[tuple[str, str], float],
) -> dict[str, list[str]]:
    assigned = {medoid: [] for medoid in medoids}
    for shape_id in shape_ids:
        medoid = min(medoids, key=lambda candidate: (distances[(shape_id, candidate)], candidate))
        assigned[medoid].append(shape_id)
    return assigned


def _fixed_count_clusters(
    shape_ids: tuple[str, ...],
    distances: Mapping[tuple[str, str], float],
    count: int,
    max_iterations: int,
) -> list[tuple[str, ...]]:
    first = _medoid(shape_ids, distances)
    medoids = [first]
    while len(medoids) < count:
        chosen = max(
            (shape_id for shape_id in shape_ids if shape_id not in medoids),
            key=lambda shape_id: (min(distances[(shape_id, medoid)] for medoid in medoids), shape_id),
        )
        medoids.append(chosen)
    for _ in range(max_iterations):
        assigned = _assign_to_medoids(shape_ids, medoids, distances)
        updated = sorted(_medoid(members, distances) for members in assigned.values())
        if sorted(medoids) == updated:
            break
        medoids = updated
    assigned = _assign_to_medoids(shape_ids, medoids, distances)
    return [tuple(sorted(members)) for members in assigned.values()]


def _threshold_clusters(
    shape_ids: tuple[str, ...],
    distances: Mapping[tuple[str, str], float],
    threshold: float,
) -> list[tuple[str, ...]]:
    clusters: list[tuple[str, ...]] = [(shape_id,) for shape_id in shape_ids]
    while True:
        merge_options = []
        for left_index, left in enumerate(clusters):
            for right_index in range(left_index + 1, len(clusters)):
                right = clusters[right_index]
                complete_link = max(distances[(left_id, right_id)] for left_id in left for right_id in right)
                if complete_link <= threshold:
                    merge_options.append((complete_link, left, right, left_index, right_index))
        if not merge_options:
            break
        _, left, right, left_index, right_index = min(merge_options)
        merged = tuple(sorted((*left, *right)))
        clusters = [cluster for index, cluster in enumerate(clusters) if index not in {left_index, right_index}]
        clusters.append(merged)
        clusters.sort()
    return clusters


def _materialize_clusters(
    member_groups: Sequence[Sequence[str]],
    distances: Mapping[tuple[str, str], float],
) -> tuple[ShapeCluster, ...]:
    materialized = []
    for members in member_groups:
        medoid = _medoid(members, distances)
        materialized.append((medoid, tuple(sorted(members))))
    materialized.sort()
    return tuple(
        ShapeCluster(
            cluster_id=f"cluster_{index:03d}",
            medoid_shape_id=medoid,
            shape_ids=members,
            distances_to_medoid={shape_id: distances[(shape_id, medoid)] for shape_id in members},
        )
        for index, (medoid, members) in enumerate(materialized)
    )
