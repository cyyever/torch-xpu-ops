---
name: xpu-cpp-modernization
description: Modernize XPU C++ to C++20 and reuse upstream PyTorch instead of XPU-local copies. Use when cleaning up XPU kernels/templates — converting runtime branches on compile-time constants to if constexpr, XPU-local helpers to upstream equivalents, or hand-rolled math to sycl::/std.
---

# XPU C++ modernization patterns

Patterns not already enforced by `.clang-tidy` (e.g. `enable_if`→`requires` is
covered by the `modernize-use-constraints` check).

## Reuse upstream; delete the XPU-local copy

Check `Math.h`, core `c10`/`ATen` headers, and the CUDA equivalent first.

| XPU-local | Upstream |
|---|---|
| `MathExtensions.h` (calc_erfinv, bessel_*, calc_igamma, …) | `<ATen/native/Math.h>` |
| `NumericLimits.h` | upstream generic `NumericLimits` |
| `div_up`/`divup`/`CeilDiv` | `at::ceil_div` |
| `xpu::pair` | `std::pair` |
| dtype→type map | `c10::CppTypeToScalarType` |
| separate `min()`/`max()` | `at::aminmax` (fused) |

Keep a local copy only for a real XPU divergence (e.g. `calc_digamma`'s `pi_t`)
and say why. Verify the upstream call compiles on device (precedent:
`calc_erfcx`).

## `if constexpr` for compile-time-constant branches

Runtime `if`/ternary on template params, enums, bool template params, or type
traits → `if constexpr`. The condition must be a constant expression: use
`std::is_same_v`, not the `std::same_as` concept.

## `sycl::` device math

`sycl::min/max/clamp/...` over macros.

## C++20 utilities

`std::bit_cast` over union punning; `std::numbers::*` (portable, no `M_PI`);
`constexpr`/`static constexpr` for constants.
