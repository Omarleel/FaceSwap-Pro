from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from faceswap_pro.quality_visual import write_visual_comparison_sheet


def _video(path: Path, offset: int) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        8.0,
        (96, 64),
    )
    assert writer.isOpened()
    try:
        for index in range(12):
            frame = np.full((64, 96, 3), offset + index * 3, dtype=np.uint8)
            cv2.circle(frame, (20 + index * 3, 32), 9, (220, 220, 220), -1)
            writer.write(frame)
    finally:
        writer.release()


def test_visual_comparison_sheet_is_created(tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "output.mp4"
    destination = tmp_path / "report.jpg"
    _video(source, 20)
    _video(output, 35)

    report = write_visual_comparison_sheet(
        input_video=source,
        output_video=output,
        destination=destination,
        samples=3,
    )

    assert report["status"] == "created"
    assert report["sample_count"] == 3
    assert destination.is_file()
    image = cv2.imread(str(destination))
    assert image is not None
    assert image.shape[1] == 960
