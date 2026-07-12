from evotensile.profile import GFX1151_NT_HHS, GFX1151_NT_HHS_COMFY1135
from evotensile.shapes import comfy_nt_1135_shapes, pilot_100_shapes


def test_comfy_nt_1135_shape_set_is_unique_and_contains_pilot():
    shapes = comfy_nt_1135_shapes()
    shape_ids = {shape.id for shape in shapes}

    assert len(shapes) == 1135
    assert len(shape_ids) == 1135
    assert {shape.id for shape in pilot_100_shapes()} <= shape_ids


def test_comfy_nt_1135_shape_set_spans_requested_axes():
    shapes = comfy_nt_1135_shapes()

    assert (min(shape.m for shape in shapes), max(shape.m for shape in shapes)) == (16, 8192)
    assert (min(shape.n for shape in shapes), max(shape.n for shape in shapes)) == (16, 8192)
    assert sorted({shape.k for shape in shapes}) == [16, 32, 64, 128, 256, 512, 1024, 2048, 3072, 4096, 8192]
    assert {shape.batch for shape in shapes} == {1}


def test_comfy_nt_1135_profile_preserves_compatibility_identity():
    assert GFX1151_NT_HHS_COMFY1135.problem_type_hash == GFX1151_NT_HHS.problem_type_hash
    assert GFX1151_NT_HHS_COMFY1135.benchmark_protocol_hash() == GFX1151_NT_HHS.benchmark_protocol_hash()
    assert GFX1151_NT_HHS_COMFY1135.environment_compatibility_tag == GFX1151_NT_HHS.environment_compatibility_tag
