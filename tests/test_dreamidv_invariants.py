from __future__ import annotations

import torch

from faceswap_pro.dreamidv_invariants import OptimizedTorchProxy, _ForwardLRU


def test_tensor_proxy_stacks_scalar_tensors_without_changing_values():
    proxy = OptimizedTorchProxy(torch, scope="test")
    values = [torch.tensor(1.25), torch.tensor(-2.5), torch.tensor(3.0)]

    actual = proxy.tensor(values, dtype=torch.float64, device="cpu")
    expected = torch.tensor([1.25, -2.5, 3.0], dtype=torch.float64)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert proxy.summary()["tensor_lists_stacked"] == 1


def test_forward_cache_only_reuses_the_same_live_unmodified_tensor():
    cache = _ForwardLRU(max_entries=4)
    source = torch.tensor([1.0, 2.0])
    result = torch.tensor([3.0])
    cache.put(source, result)

    assert cache.get(source) is result
    assert cache.get(source.view_as(source)) is result
    assert cache.get(source.clone()) is None

    source.add_(1.0)
    assert cache.get(source) is None



def test_optimized_rope_matches_official_result_before_attention_cast():
    from collections import OrderedDict
    from types import SimpleNamespace

    from faceswap_pro.dreamidv_invariants import DreamIDVInvariantOptimizer

    def official_rope(x, grid_sizes, freqs):
        n, c = x.size(2), x.size(3) // 2
        split = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        output = []
        for index, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w
            x_i = torch.view_as_complex(
                x[index, :seq_len].reshape(seq_len, n, -1, 2)
            )
            freqs_i = torch.cat(
                [
                    split[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                    split[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                    split[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
                ],
                dim=-1,
            ).reshape(seq_len, 1, -1)
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[index, seq_len:]])
            output.append(x_i)
        return torch.stack(output).float()

    optimizer = DreamIDVInvariantOptimizer.__new__(DreamIDVInvariantOptimizer)
    optimizer.torch = torch
    optimizer.attention_dtype = torch.bfloat16
    optimizer.model_module = SimpleNamespace(rope_apply=official_rope)
    optimizer._rope_grid_cache = OrderedDict()
    optimizer._rope_multiplier_cache = OrderedDict()
    optimizer._patch_rope_apply()

    torch.manual_seed(31)
    x = torch.randn(1, 5, 2, 8, dtype=torch.float32)
    grid_sizes = torch.tensor([[1, 2, 2]], dtype=torch.long)
    angles = torch.randn(8, 4, dtype=torch.float32)
    freqs = torch.polar(torch.ones_like(angles), angles)

    expected = official_rope(x, grid_sizes, freqs).to(torch.bfloat16)
    actual = optimizer.model_module.rope_apply(x, grid_sizes, freqs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cross_attention_context_projections_are_cached_only_for_same_input():
    from types import SimpleNamespace

    from faceswap_pro.dreamidv_invariants import DreamIDVInvariantOptimizer

    cross = SimpleNamespace(
        k=torch.nn.Linear(4, 4, bias=False),
        v=torch.nn.Linear(4, 4, bias=False),
        norm_k=torch.nn.LayerNorm(4),
    )
    optimizer = DreamIDVInvariantOptimizer.__new__(DreamIDVInvariantOptimizer)
    optimizer.torch = torch
    optimizer.model = SimpleNamespace(blocks=[SimpleNamespace(cross_attn=cross)])
    optimizer._module_caches = {}
    optimizer._cross_attention_cache_count = 0
    optimizer._patch_cross_attention_context()

    context = torch.randn(1, 3, 4)
    first = cross.k(context)
    second = cross.k(context.view_as(context))
    different = cross.k(context.clone())

    assert second is first
    assert different is not first
    torch.testing.assert_close(different, first, rtol=0, atol=0)
    assert optimizer._module_caches["cross_attention.0.k"].hits == 1


def test_tensor_proxy_elides_only_shape_neutral_empty_concatenation():
    proxy = OptimizedTorchProxy(torch, scope="test")
    value = torch.randn(3, 4)
    empty = value.new_zeros((0, 4))

    actual = proxy.cat([value, empty], dim=0)

    assert actual is value
    assert proxy.summary()["empty_cat_elisions"] == 1


def test_active_context_stack_hook_returns_the_stable_batch():
    from types import SimpleNamespace

    from faceswap_pro.dreamidv_invariants import DreamIDVInvariantOptimizer

    optimizer = DreamIDVInvariantOptimizer.__new__(DreamIDVInvariantOptimizer)
    optimizer.torch = torch
    optimizer._active_encoded_context_gpu = torch.randn(2, 3, 4)
    active = optimizer._active_encoded_context_gpu

    actual = optimizer._stack_active_context([active[0], active[1]], dim=0)
    assert actual is active
    assert optimizer._stack_active_context([active[0].clone(), active[1]], dim=0) is None


def test_forward_cache_supports_inference_tensors_without_version_counter():
    cache = _ForwardLRU(max_entries=4)
    with torch.inference_mode():
        source = torch.tensor([1.0, 2.0])
        result = torch.tensor([3.0])
        cache.put(source, result)

        assert cache.get(source) is result
        assert cache.get(source.view_as(source)) is result


def test_tensor_key_supports_inference_tensors():
    from faceswap_pro.dreamidv_invariants import _tensor_key

    with torch.inference_mode():
        source = torch.tensor([1, 2, 3])
        key = _tensor_key(source)

    assert key[-1] == ("inference", None)


def test_invariant_optimizer_clear_clip_caches_is_on_correct_class():
    from collections import OrderedDict
    from types import SimpleNamespace

    from faceswap_pro.dreamidv_invariants import DreamIDVInvariantOptimizer

    optimizer = DreamIDVInvariantOptimizer.__new__(DreamIDVInvariantOptimizer)
    optimizer._active_encoded_context_gpu = torch.tensor([1.0])
    optimizer._rope_grid_cache = OrderedDict([((1,), ((1, 1, 1),))])
    optimizer._rope_multiplier_cache = OrderedDict([((1,), torch.tensor([1.0]))])
    optimizer._sinusoid_output_cache = _ForwardLRU(max_entries=2)
    optimizer._sinusoid_output_cache.put(torch.tensor([1.0]), torch.tensor([2.0]))
    module_cache = _ForwardLRU(max_entries=2)
    module_cache.put(torch.tensor([1.0]), torch.tensor([2.0]))
    optimizer._module_caches = {"test": module_cache}

    optimizer.clear_clip_caches()

    assert optimizer._active_encoded_context_gpu is None
    assert not optimizer._rope_grid_cache
    assert not optimizer._rope_multiplier_cache
    assert not optimizer._sinusoid_output_cache.values
    assert not module_cache.values


def test_patched_sinusoidal_embedding_runs_and_reuses_inference_tensor():
    from collections import OrderedDict
    from types import SimpleNamespace

    from faceswap_pro.dreamidv_invariants import DreamIDVInvariantOptimizer

    optimizer = DreamIDVInvariantOptimizer.__new__(DreamIDVInvariantOptimizer)
    optimizer.torch = torch
    optimizer.model_module = SimpleNamespace(
        sinusoidal_embedding_1d=lambda dim, position: torch.empty(position.numel(), dim)
    )
    optimizer._sinusoid_output_cache = _ForwardLRU(max_entries=4)
    optimizer._sinusoid_basis_cache = OrderedDict()
    optimizer._patch_sinusoidal_embedding()

    with torch.inference_mode():
        position = torch.tensor([500.0])
        first = optimizer.model_module.sinusoidal_embedding_1d(256, position)
        second = optimizer.model_module.sinusoidal_embedding_1d(256, position.view_as(position))

    assert second is first
    assert first.shape == (1, 256)
    assert optimizer._sinusoid_output_cache.hits == 1
