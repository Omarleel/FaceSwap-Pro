from __future__ import annotations

import argparse
import os
import time

import torch

from faceswap_pro.dreamidv_sdpa import attention, sdpa_runtime_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprueba qué kernel SDPA fusionado usa FaceSwap-Pro."
    )
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--backends",
        default="cudnn,flash,efficient,math",
        help="Prioridad separada por comas.",
    )
    parser.add_argument("--allow-math", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.seq_len <= 0 or args.heads <= 0 or args.head_dim <= 0 or args.repeats <= 0:
        raise SystemExit("Todos los tamaños deben ser positivos.")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA no está disponible en este entorno.")

    os.environ["FACESWAP_SDPA_BACKENDS"] = args.backends
    os.environ["FACESWAP_SDPA_ALLOW_MATH"] = "1" if args.allow_math else "0"
    os.environ["FACESWAP_SDPA_PADDING_MODE"] = "ragged"
    os.environ["FACESWAP_SDPA_DIAGNOSTICS"] = "1"

    device = torch.device("cuda", 0)
    dtype = torch.bfloat16
    shape = (1, args.seq_len, args.heads, args.head_dim)
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}; CUDA: {torch.version.cuda}")
    print(f"Configuración: {sdpa_runtime_summary()}")
    print(f"Forma sintética Q/K/V: {shape}, dtype={dtype}")

    q = torch.randn(shape, device=device, dtype=dtype)
    k = torch.randn(shape, device=device, dtype=dtype)
    v = torch.randn(shape, device=device, dtype=dtype)

    # Primera llamada: selección y calentamiento del kernel.
    output = attention(q, k, v, k_lens=[args.seq_len])
    torch.cuda.synchronize()
    del output

    elapsed: list[float] = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        output = attention(q, k, v, k_lens=[args.seq_len])
        torch.cuda.synchronize()
        elapsed.append(time.perf_counter() - start)
        del output

    print(
        "Tiempo SDPA sintético: "
        f"mín={min(elapsed):.4f}s, promedio={sum(elapsed) / len(elapsed):.4f}s"
    )
    print(
        "La ejecución real volverá a registrar el backend para sus longitudes exactas."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
