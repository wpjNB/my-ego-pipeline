"""Phase 1: conservative tracking rules."""

from __future__ import annotations

import numpy as np
import pytest

from ego3d_action.detection import HandDetection, box_iou, conservative_track, interpolate_box
from ego3d_action.detection.tracker import conservative_track_both, tracks_to_detection_arrays
from ego3d_action.errors import StageIOError

LEFT, RIGHT = 0, 1


def detection(frame: int, box: tuple[float, float, float, float], confidence: float, side: int = LEFT) -> HandDetection:
    return HandDetection(frame_id=frame, bbox=np.array(box), confidence=confidence, handedness=side)


def empty_frames(count: int) -> list[list[HandDetection]]:
    return [[] for _ in range(count)]


def test_high_confidence_frames_become_valid() -> None:
    frames = empty_frames(4)
    for frame in range(4):
        frames[frame].append(detection(frame, (10 * frame, 0, 10 * frame + 10, 10), 0.9))
    track = conservative_track(frames, side=LEFT)
    assert track.valid.all()
    assert track.num_valid == 4
    assert np.allclose(track.confidence, 0.9)


def test_low_confidence_detection_alone_is_not_valid() -> None:
    frames = empty_frames(3)
    frames[1].append(detection(1, (10, 0, 20, 10), 0.5))
    track = conservative_track(frames, side=LEFT)
    assert not track.valid.any()


def test_gap_of_three_frames_is_recovered_when_iou_matches() -> None:
    frames = empty_frames(7)
    frames[0].append(detection(0, (0, 0, 10, 10), 0.9))
    frames[4].append(detection(4, (40, 0, 50, 10), 0.9))
    # Interpolated boxes at frames 1, 2, 3 are (10,0,20,10), (20,0,30,10), (30,0,40,10).
    frames[2].append(detection(2, (21, 0, 31, 10), 0.4))
    track = conservative_track(frames, side=LEFT)
    assert track.valid[0] and track.valid[4]
    assert track.valid[2]
    assert track.recovered[2]
    assert not track.valid[1] and not track.valid[3]


def test_gap_larger_than_four_frames_is_not_recovered() -> None:
    frames = empty_frames(8)
    frames[0].append(detection(0, (0, 0, 10, 10), 0.9))
    frames[6].append(detection(6, (60, 0, 70, 10), 0.9))
    frames[3].append(detection(3, (30, 0, 40, 10), 0.4))
    track = conservative_track(frames, side=LEFT)
    assert track.valid[0] and track.valid[6]
    assert not track.valid[3]


def test_recovery_requires_iou_above_threshold() -> None:
    frames = empty_frames(5)
    frames[0].append(detection(0, (0, 0, 10, 10), 0.9))
    frames[4].append(detection(4, (40, 0, 50, 10), 0.9))
    frames[2].append(detection(2, (80, 0, 90, 10), 0.4))  # far from the interpolated box
    track = conservative_track(frames, side=LEFT)
    assert not track.valid[2]


def test_leading_and_trailing_gaps_are_not_recovered() -> None:
    frames = empty_frames(7)
    frames[0].append(detection(0, (0, 0, 10, 10), 0.4))
    frames[3].append(detection(3, (30, 0, 40, 10), 0.9))
    frames[6].append(detection(6, (60, 0, 70, 10), 0.5))
    track = conservative_track(frames, side=LEFT)
    assert track.valid[3]
    assert not track.valid[0]
    assert not track.valid[6]


def test_detections_of_the_other_hand_are_ignored() -> None:
    frames = empty_frames(2)
    frames[0].append(detection(0, (0, 0, 10, 10), 0.9, side=RIGHT))
    frames[1].append(detection(1, (10, 0, 20, 10), 0.9, side=LEFT))
    track = conservative_track(frames, side=LEFT)
    assert not track.valid[0]
    assert track.valid[1]


def test_best_candidate_is_selected_by_confidence_then_area() -> None:
    frames = empty_frames(1)
    frames[0].append(detection(0, (0, 0, 5, 5), 0.9))
    frames[0].append(detection(0, (20, 20, 40, 40), 0.95))
    track = conservative_track(frames, side=LEFT)
    assert np.allclose(track.boxes[0], [20, 20, 40, 40])


def test_track_id_assigns_zero_to_valid_frames_only() -> None:
    frames = empty_frames(3)
    frames[1].append(detection(1, (0, 0, 10, 10), 0.9))
    track = conservative_track(frames, side=LEFT)
    assert track.track_id[1] == 0
    assert track.track_id[0] == -1 and track.track_id[2] == -1


def test_invalid_arguments_raise() -> None:
    with pytest.raises(StageIOError):
        conservative_track(empty_frames(2), side=7)
    with pytest.raises(StageIOError):
        conservative_track(empty_frames(2), side=LEFT, max_gap=-1)
    with pytest.raises(StageIOError):
        conservative_track(empty_frames(2), side=LEFT, iou_threshold=2.0)
    with pytest.raises(StageIOError):
        conservative_track(empty_frames(2), side=LEFT, num_frames=5)


def test_hand_detection_validates_fields() -> None:
    with pytest.raises(StageIOError):
        HandDetection(0, np.array([0.0, 0.0, 10.0]), 0.9, LEFT)
    with pytest.raises(StageIOError):
        HandDetection(0, np.array([10.0, 0.0, 0.0, 10.0]), 0.9, LEFT)
    with pytest.raises(StageIOError):
        HandDetection(0, np.array([0.0, 0.0, 10.0, 10.0]), 1.5, LEFT)
    with pytest.raises(StageIOError):
        HandDetection(0, np.array([0.0, 0.0, 10.0, 10.0]), 0.9, 3)


def test_box_iou_and_interpolation() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (20, 0, 30, 10)) == pytest.approx(0.0)
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50.0 / 150.0)
    assert np.allclose(interpolate_box((0, 0, 10, 10), (10, 0, 20, 10), 0.5), (5, 0, 15, 10))


def test_both_hands_and_stacking() -> None:
    frames = empty_frames(2)
    frames[0].append(detection(0, (0, 0, 10, 10), 0.9, side=LEFT))
    frames[0].append(detection(0, (50, 0, 60, 10), 0.9, side=RIGHT))
    tracks = conservative_track_both(frames)
    arrays = tracks_to_detection_arrays(tracks)
    assert arrays["valid"].shape == (2, 2)
    assert arrays["boxes"].shape == (2, 2, 4)
    assert arrays["valid"][0].tolist() == [True, True]
    assert arrays["valid"][1].tolist() == [False, False]

    from ego3d_action.detection.handedness import enforce_side_consistency, scores_to_handedness

    labels = scores_to_handedness(np.array([[0.1, 0.9], [0.8, 0.2]]))
    assert labels.tolist() == [1, 0]
    assert enforce_side_consistency(np.array([1, 1, 0, 1]), min_frames=2).tolist() == [1, 1, 1, 1]


def test_swapped_labels_at_a_crossing_do_not_switch_sides() -> None:
    """Two hands cross; the classifier's labels swap mid-way.

    A label-only tracker jumps to the other hand at the crossing (observed on
    hot3d_ep000 frame ~240, each slot holding the other hand). Continuity must
    keep each side on its own physical hand.
    """
    total = 12
    frames = empty_frames(total)
    # Left hand glides right (width 30, step 5), right hand glides left; the
    # classifier's labels swap across the crossing (frames 7..10).
    for t in range(total):
        left_box = (5.0 * t, 0.0, 5.0 * t + 30.0, 10.0)
        right_box = (90.0 - 5.0 * t, 20.0, 120.0 - 5.0 * t, 30.0)
        swap = 7 <= t <= 10
        frames[t].append(detection(t, left_box, 0.9, RIGHT if swap else LEFT))
        frames[t].append(detection(t, right_box, 0.9, LEFT if swap else RIGHT))

    left = conservative_track(frames, side=LEFT, min_confidence=0.75)
    right = conservative_track(frames, side=RIGHT, min_confidence=0.75)

    for t in range(total):
        expected_left = np.array([5.0 * t, 0.0, 5.0 * t + 30.0, 10.0])
        expected_right = np.array([90.0 - 5.0 * t, 20.0, 120.0 - 5.0 * t, 30.0])
        assert left.valid[t], f"left lost at {t}"
        assert np.allclose(left.boxes[t], expected_left), f"left jumped at frame {t}"
        assert right.valid[t], f"right lost at {t}"
        assert np.allclose(right.boxes[t], expected_right), f"right jumped at frame {t}"


def test_continuity_prefers_an_overlapping_flip_over_a_far_label_match() -> None:
    """A flipped-label box that continues the motion beats a same-label box
    that does not; a far-away box never hijacks the track."""
    frames = empty_frames(6)
    for t in range(3):
        frames[t].append(detection(t, (10 * t, 0, 10 * t + 10, 10), 0.9, LEFT))
    # Frame 5 after a 2-frame gap: the left hand reappears overlapping its old
    # spot but labelled RIGHT; a LEFT-labelled detection sits far away.
    frames[5].append(detection(5, (200, 0, 210, 10), 0.9, LEFT))
    frames[5].append(detection(5, (25, 0, 35, 10), 0.9, RIGHT))
    track = conservative_track(frames, side=LEFT, min_confidence=0.75)
    assert track.valid[5]
    assert np.allclose(track.boxes[5], (25.0, 0.0, 35.0, 10.0))

    # And with only a far, non-overlapping candidate, nothing is adopted.
    frames2 = empty_frames(6)
    for t in range(3):
        frames2[t].append(detection(t, (10 * t, 0, 10 * t + 10, 10), 0.9, LEFT))
    frames2[5].append(detection(5, (200, 0, 210, 10), 0.9, RIGHT))
    track2 = conservative_track(frames2, side=LEFT, min_confidence=0.75)
    assert not track2.valid[5]


def test_single_visible_hand_is_not_tracked_by_both_sides() -> None:
    """One hand in frame: only its own side may track it.

    Without slot mutual exclusion the other side adopted the same box and the
    backend reconstructed a phantom second hand from the same crop
    (hot3d_ep003 frames 144-175).
    """
    frames = empty_frames(8)
    for t in range(8):
        # The left hand drifts; WiLoR labels it LEFT at confidence 0.83.
        frames[t].append(detection(t, (10 * t, 5, 10 * t + 40, 45), 0.83, LEFT))
    left = conservative_track(frames, side=LEFT, min_confidence=0.75)
    right = conservative_track(frames, side=RIGHT, min_confidence=0.75)
    assert left.valid.all()
    assert not right.valid.any()
