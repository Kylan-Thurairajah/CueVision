import numpy as np

from pool_ai_full.planning import path_clear, pick_best_shot
from pool_ai_full.vision import table_pockets


def test_obstruction_blocks_path_unless_ignored():
    balls = [(20, 0, 10), (50, 5, 10), (50, 40, 10), (80, 0, 10)]
    start = (20, 0)
    end = (80, 0)

    assert not path_clear(start, end, balls, (0, 3), clearance=10)
    assert path_clear(start, end, balls, (0, 1, 3), clearance=10)


def test_straight_shot_targets_nearest_aligned_pocket():
    table = np.zeros((512, 1024, 3), dtype=np.uint8)
    balls = [(512, 250, 10), (512, 150, 10)]
    shot = pick_best_shot(table, balls, cue_idx=0)

    assert shot is not None
    assert shot["object_idx"] == 1
    assert shot["cut_deg"] == 0
    np.testing.assert_array_equal(shot["pocket"], table_pockets(ball_radius=10)[1])
    assert np.isclose(shot["ghost"][1], 170)


def test_cue_ball_alone_has_no_direct_shot():
    table = np.zeros((512, 1024, 3), dtype=np.uint8)
    assert pick_best_shot(table, [(512, 250, 10)], cue_idx=0) is None
