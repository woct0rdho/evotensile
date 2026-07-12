import pytest

from evotensile.candidate import Shape
from scripts.evaluate_candidates import _selected_shapes


def test_selected_shapes_preserves_shape_file_order(tmp_path):
    shapes = [Shape(16, 32, 1, 64), Shape(8192, 8192, 1, 8192)]
    shape_file = tmp_path / "shapes.txt"
    shape_file.write_text(f"# boundary\n{shapes[1].id}\n{shapes[0].id} # pilot\n", encoding="utf-8")

    assert _selected_shapes(shapes, shape_file) == [shapes[1], shapes[0]]


def test_selected_shapes_rejects_unknown_and_duplicate_ids(tmp_path):
    shapes = [Shape(16, 32, 1, 64)]
    shape_file = tmp_path / "shapes.txt"
    shape_file.write_text("m32_n32_b1_k32\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the profile"):
        _selected_shapes(shapes, shape_file)

    shape_file.write_text(f"{shapes[0].id}\n{shapes[0].id}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        _selected_shapes(shapes, shape_file)
