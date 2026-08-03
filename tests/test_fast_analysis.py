import numpy as np

from faceswap_pro.fast_analysis import SelectiveFaceAnalyzer


class DummyDetector:
    def detect(self, frame, max_num=0, metric="default"):
        bboxes = np.array(
            [
                [5, 5, 25, 25, 0.95],
                [40, 5, 60, 25, 0.92],
                [70, 5, 90, 25, 0.90],
            ],
            dtype=np.float32,
        )
        kps = np.zeros((3, 5, 2), dtype=np.float32)
        return bboxes, kps


class DummyRecognizer:
    def __init__(self):
        self.calls = []

    def get(self, frame, face):
        self.calls.append(face.bbox.copy())
        face.embedding = np.array([1.0, 0.0], dtype=np.float32)


class DummyApp:
    def __init__(self):
        self.det_model = DummyDetector()
        self.recognizer = DummyRecognizer()
        self.models = {"recognition": self.recognizer}


def test_selective_analysis_recognizes_only_nearby_candidates():
    app = DummyApp()
    analyzer = SelectiveFaceAnalyzer(app, max_faces=10, max_recognition_candidates=1)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    faces, stats = analyzer.analyze(
        frame,
        previous_bbox=np.array([38, 4, 62, 26], dtype=np.float32),
        full_scan=False,
    )

    assert len(faces) == 1
    assert stats.detected == 3
    assert stats.recognized == 1
    assert np.allclose(faces[0].bbox, [40, 5, 60, 25])
    assert len(app.recognizer.calls) == 1


def test_full_scan_recognizes_all_capped_faces():
    app = DummyApp()
    analyzer = SelectiveFaceAnalyzer(app, max_faces=2, max_recognition_candidates=1)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    faces, stats = analyzer.analyze(frame, previous_bbox=None, full_scan=True)

    assert len(faces) == 2
    assert stats.detected == 3
    assert stats.recognized == 2


def test_selective_analysis_reserves_one_candidate_per_active_reflection_track():
    app = DummyApp()
    analyzer = SelectiveFaceAnalyzer(app, max_faces=10, max_recognition_candidates=1)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    faces, stats = analyzer.analyze(
        frame,
        previous_bbox=np.array(
            [[4, 4, 26, 26], [69, 4, 91, 26]],
            dtype=np.float32,
        ),
        full_scan=False,
    )

    assert stats.recognized == 2
    assert len(faces) == 2
    assert {tuple(face.bbox.tolist()) for face in faces} == {
        (5.0, 5.0, 25.0, 25.0),
        (70.0, 5.0, 90.0, 25.0),
    }
