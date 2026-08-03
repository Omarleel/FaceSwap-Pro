import numpy as np

from faceswap_pro.modeling import FaceData
from faceswap_pro.tracking import MultiFaceTracker, TemporalFaceTracker


def test_tracker_returns_safe_face_clone():
    reference = np.array([1.0, 0.0], dtype=np.float32)
    tracker = TemporalFaceTracker(
        reference_embedding=reference,
        min_similarity=0.5,
        smoothing=0.8,
        max_missing_frames=3,
        scene_cut_threshold=0.9,
    )
    face = FaceData(
        bbox=np.array([10, 10, 50, 50], dtype=np.float32),
        kps=np.array(
            [[20, 20], [40, 20], [30, 30], [22, 42], [38, 42]],
            dtype=np.float32,
        ),
        embedding=reference.copy(),
        det_score=0.99,
    )
    frame = np.zeros((80, 80, 3), dtype=np.uint8)

    selected = tracker.select(frame, [face])

    assert selected is not None
    assert selected is not face
    assert np.array_equal(selected.bbox, face.bbox)
    assert selected.bbox is not face.bbox
    assert selected.kps is not face.kps


def test_tracker_propagates_landmarks_with_optical_flow():
    reference = np.array([1.0, 0.0], dtype=np.float32)
    tracker = TemporalFaceTracker(
        reference_embedding=reference,
        min_similarity=0.5,
        smoothing=0.0,
        max_missing_frames=3,
        scene_cut_threshold=1.1,
        optical_flow=True,
        flow_win_size=21,
        flow_max_level=3,
        flow_max_error=30.0,
    )
    kps = np.array(
        [[25, 25], [45, 25], [35, 35], [27, 47], [43, 47]], dtype=np.float32
    )
    face = FaceData(
        bbox=np.array([18, 18, 52, 55], dtype=np.float32),
        kps=kps.copy(),
        embedding=reference.copy(),
        det_score=0.99,
    )
    frame1 = np.zeros((80, 80, 3), dtype=np.uint8)
    for x, y in kps.astype(int):
        frame1[y - 3 : y + 4, x - 3 : x + 4] = 255
    transform = np.float32([[1, 0, 4], [0, 1, 2]])
    import cv2

    frame2 = cv2.warpAffine(frame1, transform, (80, 80))
    assert tracker.select(frame1, [face]) is not None
    gray, _ = tracker.observe(frame2)

    propagated = tracker.propagate(frame2, gray)

    assert propagated is not None
    assert np.allclose(propagated.kps, kps + [4, 2], atol=1.5)


def test_multi_tracker_keeps_actor_and_mirror_reflection_as_separate_tracks():
    reference = np.array([1.0, 0.0], dtype=np.float32)
    tracker = MultiFaceTracker(
        reference_embedding=reference,
        min_similarity=0.5,
        smoothing=0.5,
        max_missing_frames=3,
        scene_cut_threshold=1.1,
        max_faces=2,
        optical_flow=False,
    )

    def face(x1):
        return FaceData(
            bbox=np.array([x1, 12, x1 + 30, 52], dtype=np.float32),
            kps=np.array(
                [
                    [x1 + 9, 26],
                    [x1 + 21, 26],
                    [x1 + 15, 35],
                    [x1 + 10, 45],
                    [x1 + 20, 45],
                ],
                dtype=np.float32,
            ),
            embedding=reference.copy(),
            det_score=0.99,
        )

    frame1 = np.zeros((80, 140, 3), dtype=np.uint8)
    selected1 = tracker.select(frame1, [face(10), face(92)])

    frame2 = np.full_like(frame1, 1)
    gray2, _ = tracker.observe(frame2)
    # El detector puede cambiar el orden entre frames; la asociación no debe cruzar
    # ni promediar el rostro directo con la reflexión.
    selected2 = tracker.select_all_detected(frame2, gray2, [face(89), face(13)])

    assert len(selected1) == 2
    assert len(selected2) == 2
    centers = sorted(float((item.bbox[0] + item.bbox[2]) * 0.5) for item in selected2)
    assert centers[0] < 35
    assert centers[1] > 100
