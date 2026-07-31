from __future__ import annotations

import cv2
import numpy as np


_INTERPOLATIONS = {
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


class ProfessionalBlender:
    def __init__(
        self,
        aligned_size: int,
        mask_shrink: float,
        mask_blur_ratio: float,
        color_match_strength: float,
        detail_strength: float,
        roi_enabled: bool = True,
        roi_margin: float = 0.15,
        interpolation: str = "cubic",
    ) -> None:
        self.aligned_size = aligned_size
        self.mask_shrink = mask_shrink
        self.mask_blur_ratio = mask_blur_ratio
        self.color_match_strength = color_match_strength
        self.detail_strength = detail_strength
        self.roi_enabled = roi_enabled
        self.roi_margin = max(0.0, float(roi_margin))
        self.interpolation = _INTERPOLATIONS.get(interpolation, cv2.INTER_CUBIC)
        self._cached_mask = self._build_mask(self.aligned_size)

    def _build_mask(self, size: int) -> np.ndarray:
        mask = np.zeros((size, size), dtype=np.float32)
        center = (size // 2, int(size * 0.52))
        axes = (int(size * 0.43 * self.mask_shrink), int(size * 0.49 * self.mask_shrink))
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        blur = max(5, int(size * self.mask_blur_ratio))
        if blur % 2 == 0:
            blur += 1
        return cv2.GaussianBlur(mask, (blur, blur), 0)[..., None]

    @staticmethod
    def _lab_statistics(image: np.ndarray, mask: np.ndarray):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        valid = mask[..., 0] > 0.25
        pixels = lab[valid]
        if pixels.size == 0:
            return lab, np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)
        return lab, pixels.mean(axis=0), pixels.std(axis=0) + 1e-6

    def _color_match(self, fake: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.color_match_strength <= 0:
            return fake
        fake_lab, fake_mean, fake_std = self._lab_statistics(fake, mask)
        _, target_mean, target_std = self._lab_statistics(target, mask)
        matched = (fake_lab - fake_mean) * (target_std / fake_std) + target_mean
        matched = np.clip(matched, 0, 255).astype(np.uint8)
        matched = cv2.cvtColor(matched, cv2.COLOR_LAB2BGR)
        strength = float(np.clip(self.color_match_strength, 0.0, 1.0))
        return cv2.addWeighted(matched, strength, fake, 1.0 - strength, 0)

    def _detail(self, image: np.ndarray) -> np.ndarray:
        if self.detail_strength <= 0:
            return image
        blur = cv2.GaussianBlur(image, (0, 0), 1.15)
        return cv2.addWeighted(image, 1.0 + self.detail_strength, blur, -self.detail_strength, 0)

    def _prepare_aligned(
        self,
        frame: np.ndarray,
        fake_128: np.ndarray,
        affine_128: np.ndarray,
        restorer,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        scale = self.aligned_size / float(fake_128.shape[0])
        affine = np.asarray(affine_128, dtype=np.float32).copy()
        affine[:, :] *= scale
        aligned_target = cv2.warpAffine(
            frame,
            affine,
            (self.aligned_size, self.aligned_size),
            flags=self.interpolation,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        fake = cv2.resize(
            fake_128,
            (self.aligned_size, self.aligned_size),
            interpolation=self.interpolation,
        )
        fake = restorer.restore(fake)
        mask = self._cached_mask
        fake = self._color_match(fake, aligned_target, mask)
        fake = self._detail(fake)
        return fake, mask, cv2.invertAffineTransform(affine)

    def _roi_bounds(self, inverse: np.ndarray, frame_shape) -> tuple[int, int, int, int] | None:
        size = float(self.aligned_size - 1)
        corners = np.array(
            [[[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]]], dtype=np.float32
        )
        mapped = cv2.transform(corners, inverse)[0]
        min_x, min_y = mapped.min(axis=0)
        max_x, max_y = mapped.max(axis=0)
        margin = self.roi_margin * max(float(max_x - min_x), float(max_y - min_y))
        h, w = frame_shape[:2]
        x1 = max(0, int(np.floor(min_x - margin)))
        y1 = max(0, int(np.floor(min_y - margin)))
        x2 = min(w, int(np.ceil(max_x + margin)) + 1)
        y2 = min(h, int(np.ceil(max_y + margin)) + 1)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _composite_roi(
        self,
        frame: np.ndarray,
        fake: np.ndarray,
        mask: np.ndarray,
        inverse: np.ndarray,
    ) -> np.ndarray:
        bounds = self._roi_bounds(inverse, frame.shape)
        if bounds is None:
            return frame
        x1, y1, x2, y2 = bounds
        roi_width, roi_height = x2 - x1, y2 - y1
        roi_transform = inverse.copy()
        roi_transform[0, 2] -= x1
        roi_transform[1, 2] -= y1

        warped_fake = cv2.warpAffine(
            fake,
            roi_transform,
            (roi_width, roi_height),
            flags=self.interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_mask = cv2.warpAffine(
            mask[..., 0],
            roi_transform,
            (roi_width, roi_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )[..., None]
        target = frame[y1:y2, x1:x2]
        blended = warped_mask * warped_fake.astype(np.float32) + (
            1.0 - warped_mask
        ) * target.astype(np.float32)
        target[:] = np.clip(blended, 0, 255).astype(np.uint8)
        return frame

    def _composite_full(
        self,
        frame: np.ndarray,
        fake: np.ndarray,
        mask: np.ndarray,
        inverse: np.ndarray,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        warped_fake = cv2.warpAffine(
            fake,
            inverse,
            (w, h),
            flags=self.interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        warped_mask = cv2.warpAffine(
            mask[..., 0],
            inverse,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )[..., None]
        output = warped_mask * warped_fake.astype(np.float32) + (
            1.0 - warped_mask
        ) * frame.astype(np.float32)
        return np.clip(output, 0, 255).astype(np.uint8)

    def composite(
        self,
        frame: np.ndarray,
        fake_128: np.ndarray,
        affine_128: np.ndarray,
        restorer,
    ) -> np.ndarray:
        fake, mask, inverse = self._prepare_aligned(frame, fake_128, affine_128, restorer)
        if self.roi_enabled:
            return self._composite_roi(frame, fake, mask, inverse)
        return self._composite_full(frame, fake, mask, inverse)
