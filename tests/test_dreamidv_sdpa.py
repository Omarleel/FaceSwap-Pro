from __future__ import annotations

import sys

import torch
import torch.nn.functional as F

from faceswap_pro import dreamidv_sdpa


def _math_environment(monkeypatch) -> None:
    monkeypatch.setenv("FACESWAP_SDPA_BACKENDS", "math")
    monkeypatch.setenv("FACESWAP_SDPA_ALLOW_MATH", "1")
    monkeypatch.setenv("FACESWAP_SDPA_DIAGNOSTICS", "0")
    dreamidv_sdpa._reset_backend_cache_for_tests()


def test_padding_mask_uses_sdpa_true_means_valid():
    lengths = torch.tensor([2, 4])
    mask = dreamidv_sdpa._build_padding_mask(
        lengths, 4, device=torch.device("cpu")
    )

    assert mask is not None
    assert mask.shape == (2, 1, 1, 4)
    assert mask.dtype == torch.bool
    assert mask[0, 0, 0].tolist() == [True, True, False, False]
    assert mask[1, 0, 0].tolist() == [True, True, True, True]


def test_ragged_attention_crops_padding_and_zeros_padded_queries(monkeypatch):
    _math_environment(monkeypatch)
    monkeypatch.setenv("FACESWAP_SDPA_PADDING_MODE", "ragged")

    torch.manual_seed(7)
    q = torch.randn(2, 5, 2, 4, dtype=torch.bfloat16)
    k = torch.randn(2, 6, 2, 4, dtype=torch.bfloat16)
    v = torch.randn(2, 6, 2, 4, dtype=torch.bfloat16)
    q_lens = [3, 5]
    k_lens = [4, 2]

    actual = dreamidv_sdpa.attention(
        q,
        k,
        v,
        q_lens=q_lens,
        k_lens=k_lens,
        softmax_scale=0.25,
        q_scale=0.5,
    )

    expected = torch.zeros_like(actual)
    for index, (q_len, k_len) in enumerate(zip(q_lens, k_lens)):
        qi = q[index : index + 1, :q_len].transpose(1, 2).contiguous() * 0.5
        ki = k[index : index + 1, :k_len].transpose(1, 2).contiguous()
        vi = v[index : index + 1, :k_len].transpose(1, 2).contiguous()
        oi = F.scaled_dot_product_attention(qi, ki, vi, scale=0.25)
        expected[index : index + 1, :q_len] = oi.transpose(1, 2)

    torch.testing.assert_close(actual, expected, rtol=0.20, atol=1e-3)
    assert torch.count_nonzero(actual[0, 3:]) == 0


def test_mask_mode_applies_compact_key_mask(monkeypatch):
    _math_environment(monkeypatch)
    monkeypatch.setenv("FACESWAP_SDPA_PADDING_MODE", "mask")

    torch.manual_seed(11)
    q = torch.randn(2, 4, 1, 8, dtype=torch.bfloat16)
    k = torch.randn(2, 5, 1, 8, dtype=torch.bfloat16)
    v = torch.randn(2, 5, 1, 8, dtype=torch.bfloat16)
    q_lens = torch.tensor([4, 2])
    k_lens = torch.tensor([3, 5])

    actual = dreamidv_sdpa.attention(
        q,
        k,
        v,
        q_lens=q_lens,
        k_lens=k_lens,
    )

    qh = q.transpose(1, 2).contiguous()
    kh = k.transpose(1, 2).contiguous()
    vh = v.transpose(1, 2).contiguous()
    mask = dreamidv_sdpa._build_padding_mask(
        k_lens, 5, device=torch.device("cpu")
    )
    expected = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
    expected = dreamidv_sdpa._zero_padded_queries(expected, q_lens)
    expected = expected.transpose(1, 2).contiguous()

    torch.testing.assert_close(actual, expected, rtol=0.20, atol=1e-3)
    assert torch.count_nonzero(actual[1, 2:]) == 0


def test_math_backend_is_removed_when_fallback_is_disabled(monkeypatch):
    monkeypatch.setenv("FACESWAP_SDPA_BACKENDS", "cudnn,flash,efficient,math")
    monkeypatch.setenv("FACESWAP_SDPA_ALLOW_MATH", "0")

    settings = dreamidv_sdpa._settings()

    assert settings.priority == ("cudnn", "flash", "efficient")
    assert settings.allow_math is False


def test_attention_override_is_installed_for_dreamidv_package():
    target = "dreamidv_wan_faster.modules.attention"
    previous = sys.modules.get(target)
    try:
        dreamidv_sdpa.install_attention_override("dreamidv_wan_faster")
        assert sys.modules[target] is dreamidv_sdpa
    finally:
        if previous is None:
            sys.modules.pop(target, None)
        else:
            sys.modules[target] = previous
