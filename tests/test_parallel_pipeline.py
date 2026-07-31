from types import SimpleNamespace

import numpy as np

from faceswap_pro.blend import ProfessionalBlender
from faceswap_pro.parallel_pipeline import run_parallel_frames
from faceswap_pro.tracking import TemporalFaceTracker
from faceswap_pro.videoio import VideoMetadata


class DummyReader:
    backend = "dummy"

    def __init__(self, count=6):
        self.metadata = VideoMetadata(fps=30.0, width=128, height=128, frame_count=count)
        self.frames = []
        for index in range(count):
            frame = np.zeros((128, 128, 3), dtype=np.uint8)
            frame[-1, -1, 0] = index
            frame[30:50, 30:50] = 180
            self.frames.append(frame)
        self.index = 0

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def close(self):
        pass


class DummyWriter:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame.copy())


class DummyDetector:
    def detect(self, frame, max_num=0, metric="default"):
        bbox = np.array([[22, 22, 58, 62, 0.99]], dtype=np.float32)
        kps = np.array(
            [[[30, 32], [49, 32], [40, 42], [32, 54], [48, 54]]],
            dtype=np.float32,
        )
        return bbox, kps


class DummyRecognizer:
    def get(self, frame, face):
        face.embedding = np.array([1.0, 0.0], dtype=np.float32)


class DummyFaceApp:
    def __init__(self):
        self.det_model = DummyDetector()
        self.models = {"recognition": DummyRecognizer()}


class DummySwapper:
    def get(self, frame, target_face, source_face, paste_back=False):
        fake = np.full((16, 16, 3), 200, dtype=np.uint8)
        affine = np.array([[0.4, 0, -8], [0, 0.4, -8]], dtype=np.float32)
        return fake, affine


class IdentityRestorer:
    def __call__(self, image):
        return image


def test_parallel_pipeline_preserves_order_and_processes_all_frames():
    reader = DummyReader()
    writer = DummyWriter()
    tracker = TemporalFaceTracker(
        reference_embedding=np.array([1.0, 0.0], dtype=np.float32),
        min_similarity=0.5,
        smoothing=0.0,
        max_missing_frames=2,
        scene_cut_threshold=1.1,
        optical_flow=False,
    )
    blender = ProfessionalBlender(
        aligned_size=16,
        mask_shrink=0.8,
        mask_blur_ratio=0.1,
        color_match_strength=0.0,
        detail_strength=0.0,
        roi_enabled=True,
        roi_margin=0.1,
        interpolation="linear",
    )
    config = SimpleNamespace(
        performance=SimpleNamespace(
            reader_queue=2,
            analysis_queue=2,
            writer_queue=2,
            max_inflight=3,
            postprocess_workers=2,
            opencv_threads=1,
        ),
        restorer=SimpleNamespace(enabled=False),
        engine=SimpleNamespace(max_faces=3),
        tracking=SimpleNamespace(
            max_recognition_candidates=1,
            detection_interval=2,
            full_scan_interval=4,
            optical_flow=False,
        ),
        watermark=SimpleNamespace(text="CONTENIDO SINTÉTICO · IA"),
    )

    stats, settings = run_parallel_frames(
        reader=reader,
        writer=writer,
        face_app=DummyFaceApp(),
        swapper=DummySwapper(),
        source_face=object(),
        tracker=tracker,
        blender=blender,
        restorer=IdentityRestorer(),
        config=config,
    )

    assert stats.written_frames == 6
    assert stats.swapped_frames == 6
    assert len(writer.frames) == 6
    assert [int(frame[-1, -1, 0]) for frame in writer.frames] == list(range(6))
    assert settings["postprocess_workers"] == 2
