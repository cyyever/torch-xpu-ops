/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <ATen/NumericUtils.h>
#include <comm/SYCLHelpers.h>
#include <comm/Scalar.h>
#include <sycl/sycl.hpp>

#include <concepts>
#include <type_traits>

namespace at::native::xpu {

template <typename T>
static inline T safe_max(T a, T b) {
  T max = at::_isnan(a) ? a : (at::_isnan(b) ? b : std::max<T>(a, b));
  return max;
}

template <typename T>
static inline T safe_min(T a, T b) {
  T min = at::_isnan(a) ? a : (at::_isnan(b) ? b : std::min<T>(a, b));
  return min;
}

template <typename T>
using sycl_atomic_ref_rlx_dev_global_t =
    sycl::atomic_ref<T, sycl_mem_odr_rlx, sycl_mem_scp_dev, sycl_global_space>;

template <typename T>
using sycl_atomic_ref_rlx_wg_local_t =
    sycl::atomic_ref<T, sycl_mem_odr_rlx, sycl_mem_scp_wg, sycl_local_space>;

// The integer atomics differ between the global and work-group-local address
// spaces only in the atomic_ref they build, so they are parameterized on it.
// Types narrower than a word are packed several-per-word and CAS the containing
// 32-bit word; wider ones CAS their own word directly.
template <template <typename> class atomic_ref_t, std::integral T>
struct AtomicIntegerImplBase {
  template <typename func_t>
  inline void operator()(T* address, T val, const func_t& func) {
    if constexpr (sizeof(T) == 1) {
      size_t offset = (size_t)address & 3;
      uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
      uint32_t assumed = *address_as_ui;
      uint32_t shift = offset * 8;
      uint32_t newval;
      uint32_t newval_byte;
      atomic_ref_t<uint32_t> target(*address_as_ui);

      do {
        newval = assumed;
        newval_byte = (newval >> shift) & 0xff;
        // preserve size in initial cast. Casting directly to uint32_t pads
        // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
        newval = static_cast<uint8_t>(func(val, static_cast<T>(newval_byte)));
        newval = (assumed & ~(0x000000ff << shift)) | (newval << shift);
      } while (!target.compare_exchange_strong(assumed, newval));
    } else if constexpr (sizeof(T) == 2) {
      size_t offset = (size_t)address & 2;
      uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
      bool is_32_align = offset;
      uint32_t assumed = *address_as_ui;
      uint32_t newval;
      uint32_t newval_bytes;
      atomic_ref_t<uint32_t> target(*address_as_ui);

      do {
        newval = assumed;
        newval_bytes = is_32_align ? newval >> 16 : newval & 0xffff;
        // preserve size in initial cast. Casting directly to uint32_t pads
        // negative signed values with 1's (e.g. signed -1 = unsigned ~0).
        newval = static_cast<uint16_t>(func(val, static_cast<T>(newval_bytes)));
        newval = is_32_align ? (assumed & 0xffff) | (newval << 16)
                             : (assumed & 0xffff0000) | newval;
      } while (!target.compare_exchange_strong(assumed, newval));
    } else {
      using proxy_t =
          std::conditional_t<sizeof(T) == 4, uint32_t, unsigned long long>;
      proxy_t* address_as_proxy = (proxy_t*)address;
      proxy_t assumed = *address_as_proxy;
      proxy_t newval;
      atomic_ref_t<proxy_t> target(*address_as_proxy);

      do {
        newval = static_cast<proxy_t>(func(val, static_cast<T>(assumed)));
      } while (!target.compare_exchange_strong(assumed, newval));
    }
  }
};

template <typename T>
using AtomicIntegerImplLocal =
    AtomicIntegerImplBase<sycl_atomic_ref_rlx_wg_local_t, T>;

template <typename T>
using AtomicIntegerImpl =
    AtomicIntegerImplBase<sycl_atomic_ref_rlx_dev_global_t, T>;

#define SYCL_ATOMIC_INTEGER_LOCAL(NAME, OP, DTYPE)          \
  static inline void atomic##NAME##Local(                   \
      const sycl_local_ptr<DTYPE>& address, DTYPE val) {    \
    AtomicIntegerImplLocal<DTYPE>()(                        \
        address, val, [](DTYPE a, DTYPE b) { return OP; }); \
  }

#define SYCL_ATOMIC_INTEGER(NAME, OP, DTYPE)                \
  static inline void atomic##NAME(                          \
      const sycl_global_ptr<DTYPE>& address, DTYPE val) {   \
    AtomicIntegerImpl<DTYPE>()(                             \
        address, val, [](DTYPE a, DTYPE b) { return OP; }); \
  }

template <typename T>
concept atomic_fp_t =
    std::same_as<T, at::Half> || std::same_as<T, at::BFloat16> ||
    std::same_as<T, float> || std::same_as<T, double>;

// Half and BFloat16 are packed two-per-word, so they CAS the containing 32-bit
// word; float and double CAS their own word through a same-sized integer proxy.
template <template <typename> class atomic_ref_t, atomic_fp_t T>
struct AtomicFPImplBase {
  template <typename func_t>
  inline void operator()(T* address, T val, const func_t& func) {
    if constexpr (sizeof(T) == 2) {
      bool is_32_align = (size_t)address & 2;
      unsigned int* address_as_ui =
          (unsigned int*)((char*)address - ((size_t)address & 2));
      unsigned int assumed = *address_as_ui;
      unsigned int newval;
      atomic_ref_t<unsigned int> target(*address_as_ui);

      do {
        newval = assumed;
        T sum;
        sum.x = is_32_align ? (newval >> 16) : (newval & 0xffff);
        sum = func(sum, val);
        newval = is_32_align ? (newval & 0xffff) | (sum.x << 16)
                             : (newval & 0xffff0000) | sum.x;
      } while (!target.compare_exchange_strong(assumed, newval));
    } else {
      using proxy_t =
          std::conditional_t<sizeof(T) == 4, unsigned int, unsigned long long>;
      proxy_t* address_as_proxy = (proxy_t*)address;
      proxy_t assumed = *address_as_proxy;
      proxy_t newval;
      atomic_ref_t<proxy_t> target(*address_as_proxy);

      do {
        newval = sycl::bit_cast<proxy_t>(func(val, sycl::bit_cast<T>(assumed)));
      } while (!target.compare_exchange_strong(assumed, newval));
    }
  }
};

template <typename T>
using AtomicFPImpl = AtomicFPImplBase<sycl_atomic_ref_rlx_dev_global_t, T>;

template <typename T>
using AtomicFPImplLocal = AtomicFPImplBase<sycl_atomic_ref_rlx_wg_local_t, T>;

#define SYCL_ATOMIC_FP(NAME, OP, DTYPE)                                       \
  static inline void atomic##NAME(                                            \
      const sycl_global_ptr<DTYPE>& address, DTYPE val) {                     \
    AtomicFPImpl<DTYPE>()(address, val, [](DTYPE a, DTYPE b) { return OP; }); \
  }

#define SYCL_ATOMIC_FP_LOCAL(NAME, OP, DTYPE)               \
  static inline void atomic##NAME##Local(                   \
      const sycl_local_ptr<DTYPE>& address, DTYPE val) {    \
    AtomicFPImplLocal<DTYPE>()(                             \
        address, val, [](DTYPE a, DTYPE b) { return OP; }); \
  }

static inline void atomicAdd(const sycl_global_ptr<float>& address, float val) {
  sycl_atomic_ref_rlx_dev_global_t<float> target(*address);
  target.fetch_add(val);
}

static inline void atomicAdd(
    const sycl_global_ptr<double>& address,
    double val) {
  sycl_atomic_ref_rlx_dev_global_t<double> target(*address);
  target.fetch_add(val);
}

static inline void atomicAdd(const sycl_global_ptr<int>& address, int val) {
  sycl_atomic_ref_rlx_dev_global_t<int> target(*address);
  target.fetch_add(val);
}

static inline void atomicAdd(
    const sycl_global_ptr<int64_t>& address,
    int64_t val) {
  sycl_atomic_ref_rlx_dev_global_t<int64_t> target(*address);
  target.fetch_add(val);
}

static inline void atomicAdd(
    const sycl_local_ptr<uint32_t>& address,
    uint32_t val) {
  sycl_atomic_ref_rlx_wg_local_t<uint32_t> target(*address);
  target.fetch_add(val);
}

static inline void atomicAdd(
    const sycl_local_ptr<uint64_t>& address,
    uint64_t val) {
  sycl_atomic_ref_rlx_wg_local_t<uint64_t> target(*address);
  target.fetch_add(val);
}

static inline void atomicAdd(const sycl_local_ptr<int>& address, int val) {
  sycl_atomic_ref_rlx_wg_local_t<int> target(*address);
  target.fetch_add(val);
}

static inline void atomicAdd(
    const sycl_local_ptr<int64_t>& address,
    int64_t val) {
  sycl_atomic_ref_rlx_wg_local_t<int64_t> target(*address);
  target.fetch_add(val);
}

static inline void atomicAddLocal(
    const sycl_local_ptr<float>& address,
    float val) {
  sycl_atomic_ref_rlx_wg_local_t<float> target(*address);
  target.fetch_add(val);
}

static inline void atomicAddLocal(
    const sycl_local_ptr<double>& address,
    double val) {
  sycl_atomic_ref_rlx_wg_local_t<double> target(*address);
  target.fetch_add(val);
}

static inline void atomicAddLocal(const sycl_local_ptr<int>& address, int val) {
  sycl_atomic_ref_rlx_wg_local_t<int> target(*address);
  target.fetch_add(val);
}

static inline void atomicAddLocal(
    const sycl_local_ptr<int64_t>& address,
    int64_t val) {
  sycl_atomic_ref_rlx_wg_local_t<int64_t> target(*address);
  target.fetch_add(val);
}

static inline void atomicAddLocal(
    const sycl_local_ptr<uint32_t>& address,
    uint32_t val) {
  sycl_atomic_ref_rlx_wg_local_t<uint32_t> target(*address);
  target.fetch_add(val);
}

static inline void atomicAddLocal(
    const sycl_local_ptr<uint64_t>& address,
    uint64_t val) {
  sycl_atomic_ref_rlx_wg_local_t<uint64_t> target(*address);
  target.fetch_add(val);
}

// Atomic add local implementation.
SYCL_ATOMIC_INTEGER_LOCAL(Add, a || b, bool)
SYCL_ATOMIC_INTEGER_LOCAL(Add, std::plus<uint8_t>()(a, b), uint8_t)
SYCL_ATOMIC_INTEGER_LOCAL(Add, std::plus<int8_t>()(a, b), int8_t)
SYCL_ATOMIC_INTEGER_LOCAL(Add, std::plus<int16_t>()(a, b), int16_t)

SYCL_ATOMIC_FP_LOCAL(Add, std::plus<at::Half>()(a, b), at::Half)
SYCL_ATOMIC_FP_LOCAL(Add, std::plus<at::BFloat16>()(a, b), at::BFloat16)

// Atomic add implementation.
SYCL_ATOMIC_INTEGER(Add, a || b, bool)
SYCL_ATOMIC_INTEGER(Add, std::plus<uint8_t>()(a, b), uint8_t)
SYCL_ATOMIC_INTEGER(Add, std::plus<int8_t>()(a, b), int8_t)
SYCL_ATOMIC_INTEGER(Add, std::plus<int16_t>()(a, b), int16_t)

SYCL_ATOMIC_FP(Add, std::plus<at::Half>()(a, b), at::Half)
SYCL_ATOMIC_FP(Add, std::plus<at::BFloat16>()(a, b), at::BFloat16)

template <typename T>
static inline void atomicAdd(
    const sycl_global_ptr<c10::complex<T>>& address,
    c10::complex<T> val) {
  atomicAdd(&address->real_, val.real_);
  atomicAdd(&address->imag_, val.imag_);
}

template <typename T>
static inline void atomicAddLocal(
    const sycl_local_ptr<c10::complex<T>>& address,
    c10::complex<T> val) {
  atomicAddLocal(&address->real_, val.real_);
  atomicAddLocal(&address->imag_, val.imag_);
}

// Atomic multiplication implementation.
SYCL_ATOMIC_INTEGER(Mul, std::multiplies<uint8_t>()(a, b), uint8_t)
SYCL_ATOMIC_INTEGER(Mul, std::multiplies<int8_t>()(a, b), int8_t)
SYCL_ATOMIC_INTEGER(Mul, std::multiplies<int16_t>()(a, b), int16_t)
SYCL_ATOMIC_INTEGER(Mul, std::multiplies<int32_t>()(a, b), int32_t)
SYCL_ATOMIC_INTEGER(Mul, std::multiplies<int64_t>()(a, b), int64_t)
SYCL_ATOMIC_INTEGER(Mul, std::multiplies<uint32_t>()(a, b), uint32_t)
SYCL_ATOMIC_INTEGER(Mul, std::multiplies<uint64_t>()(a, b), uint64_t)

SYCL_ATOMIC_FP(Mul, std::multiplies<float>()(a, b), float)
SYCL_ATOMIC_FP(Mul, std::multiplies<double>()(a, b), double)
SYCL_ATOMIC_FP(Mul, std::multiplies<at::Half>()(a, b), at::Half)
SYCL_ATOMIC_FP(Mul, std::multiplies<at::BFloat16>()(a, b), at::BFloat16)

// Atomic maximum implementation.

static inline void atomicMax(
    const sycl_local_ptr<int32_t>& address,
    int32_t val) {
  sycl_atomic_ref_rlx_wg_local_t<int32_t> target(*address);
  target.fetch_add(val);
}

static inline void atomicMax(
    const sycl_local_ptr<int64_t>& address,
    int64_t val) {
  sycl_atomic_ref_rlx_wg_local_t<int64_t> target(*address);
  target.fetch_add(val);
}

SYCL_ATOMIC_INTEGER(Max, safe_max<uint8_t>(a, b), uint8_t)
SYCL_ATOMIC_INTEGER(Max, safe_max<int8_t>(a, b), int8_t)
SYCL_ATOMIC_INTEGER(Max, safe_max<int16_t>(a, b), int16_t)
SYCL_ATOMIC_INTEGER(Max, safe_max<int32_t>(a, b), int32_t)
SYCL_ATOMIC_INTEGER(Max, safe_max<int64_t>(a, b), int64_t)
SYCL_ATOMIC_INTEGER(Max, safe_max<uint32_t>(a, b), uint32_t)
SYCL_ATOMIC_INTEGER(Max, safe_max<uint64_t>(a, b), uint64_t)

SYCL_ATOMIC_FP(Max, safe_max<float>(a, b), float)
SYCL_ATOMIC_FP(Max, safe_max<double>(a, b), double)
SYCL_ATOMIC_FP(Max, safe_max<at::Half>(a, b), at::Half)
SYCL_ATOMIC_FP(Max, safe_max<at::BFloat16>(a, b), at::BFloat16)

// Atomic minimum implementation.
SYCL_ATOMIC_INTEGER(Min, safe_min<uint8_t>(a, b), uint8_t)
SYCL_ATOMIC_INTEGER(Min, safe_min<int8_t>(a, b), int8_t)
SYCL_ATOMIC_INTEGER(Min, safe_min<int16_t>(a, b), int16_t)
SYCL_ATOMIC_INTEGER(Min, safe_min<int32_t>(a, b), int32_t)
SYCL_ATOMIC_INTEGER(Min, safe_min<int64_t>(a, b), int64_t)
SYCL_ATOMIC_INTEGER(Min, safe_min<uint32_t>(a, b), uint32_t)
SYCL_ATOMIC_INTEGER(Min, safe_min<uint64_t>(a, b), uint64_t)

SYCL_ATOMIC_FP(Min, safe_min<float>(a, b), float)
SYCL_ATOMIC_FP(Min, safe_min<double>(a, b), double)
SYCL_ATOMIC_FP(Min, safe_min<at::Half>(a, b), at::Half)
SYCL_ATOMIC_FP(Min, safe_min<at::BFloat16>(a, b), at::BFloat16)

// =========================================================================
// ------------------------------AtomicCAS----------------------------------
// =========================================================================

// --- Auxiliary Type Definition ---
// R is a template template parameter for the SYCL atomic ref type
template <typename T, template <typename> class R>
using AtomicRef = R<T>;

// --- Generic Integer CAS Structure Definition (R is the Atomic Ref type) ---
template <typename T, size_t n, template <typename> class R>
struct AtomicCASInteger;

// n=1 (1-byte Soft-RMW)
template <typename T, template <typename> class R>
struct AtomicCASInteger<T, 1, R> {
  inline T operator()(T* address, T expected, T desired) {
    size_t offset = (size_t)address & 3;
    uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
    size_t shift = offset * 8;
    uint32_t assumed;
    uint32_t newval;
    AtomicRef<uint32_t, R> target(*address_as_ui);

    T extracted_old_value;
    do {
      assumed = *address_as_ui;
      uint32_t byte_in_mem = (assumed >> shift) & 0xff;
      extracted_old_value = static_cast<T>(byte_in_mem);

      if (extracted_old_value == expected) {
        uint32_t desired_byte = static_cast<uint8_t>(desired);
        newval = (assumed & ~(0x000000ff << shift)) | (desired_byte << shift);
      } else {
        break;
      }
    } while (!target.compare_exchange_strong(assumed, newval));

    if (extracted_old_value == expected) {
      return expected;
    } else {
      return extracted_old_value;
    }
  }
};

// n=2 (2-byte Soft-RMW)
template <typename T, template <typename> class R>
struct AtomicCASInteger<T, 2, R> {
  inline T operator()(T* address, T expected, T desired) {
    size_t offset = (size_t)address & 2;
    uint32_t* address_as_ui = (uint32_t*)((char*)address - offset);
    bool is_upper_half = offset;
    uint32_t assumed;
    uint32_t newval;
    uint32_t current_half_word;

    AtomicRef<uint32_t, R> target(*address_as_ui);

    T extracted_old_value;
    do {
      assumed = *address_as_ui;
      current_half_word = is_upper_half ? (assumed >> 16) : (assumed & 0xffff);
      extracted_old_value = static_cast<T>(current_half_word);

      if (extracted_old_value == expected) {
        uint32_t desired_half_word = static_cast<uint16_t>(desired);
        newval = is_upper_half ? (assumed & 0xffff) | (desired_half_word << 16)
                               : (assumed & 0xffff0000) | desired_half_word;
      } else {
        break;
      }
    } while (!target.compare_exchange_strong(assumed, newval));

    if (extracted_old_value == expected) {
      return expected;
    } else {
      return extracted_old_value;
    }
  }
};

// n=4 (4-byte Native CAS)
template <typename T, template <typename> class R>
struct AtomicCASInteger<T, 4, R> {
  inline T operator()(T* address, T expected, T desired) {
    uint32_t* address_as_ui = (uint32_t*)(address);
    uint32_t assumed;
    uint32_t newval;

    AtomicRef<uint32_t, R> target(*address_as_ui);

    uint32_t expected_ui = static_cast<uint32_t>(expected);
    newval = static_cast<uint32_t>(desired);

    do {
      assumed = *address_as_ui;
      if (assumed != expected_ui) {
        break;
      }
    } while (!target.compare_exchange_strong(assumed, newval));

    if (assumed == expected_ui) {
      return expected;
    } else {
      return static_cast<T>(assumed);
    }
  }
};

// n=8 (8-byte Native CAS)
template <typename T, template <typename> class R>
struct AtomicCASInteger<T, 8, R> {
  inline T operator()(T* address, T expected, T desired) {
    unsigned long long* address_as_ull = (unsigned long long*)(address);
    unsigned long long assumed;
    unsigned long long newval;

    AtomicRef<unsigned long long, R> target(*address_as_ull);

    unsigned long long expected_ull = static_cast<unsigned long long>(expected);
    newval = static_cast<unsigned long long>(desired);

    do {
      assumed = *address_as_ull;
      if (assumed != expected_ull) {
        break;
      }
    } while (!target.compare_exchange_strong(assumed, newval));

    if (assumed == expected_ull) {
      return expected;
    } else {
      return static_cast<T>(assumed);
    }
  }
};

// --- Generic Macro Definitions for Function Signatures ---
#define SYCL_ATOMIC_CAS_IMPL(DTYPE, STRUCT_NAME, PTR_TYPE, ATOMIC_REF) \
  static inline DTYPE atomicCAS(                                       \
      const PTR_TYPE<DTYPE>& address, DTYPE expected, DTYPE desired) { \
    /* Call generic struct with specific SYCL atomic ref type */       \
    return STRUCT_NAME<DTYPE, sizeof(DTYPE), ATOMIC_REF>()(            \
        address, expected, desired);                                   \
  }

#define SYCL_ATOMIC_CAS_ALL(DTYPE, STRUCT_NAME) \
  /* local CAS version */                       \
  SYCL_ATOMIC_CAS_IMPL(                         \
      DTYPE, STRUCT_NAME, sycl_local_ptr, sycl_atomic_ref_rlx_wg_local_t)

SYCL_ATOMIC_CAS_ALL(int, AtomicCASInteger)
SYCL_ATOMIC_CAS_ALL(int64_t, AtomicCASInteger)
SYCL_ATOMIC_CAS_ALL(uint32_t, AtomicCASInteger)
SYCL_ATOMIC_CAS_ALL(uint64_t, AtomicCASInteger)
SYCL_ATOMIC_CAS_ALL(int8_t, AtomicCASInteger)
SYCL_ATOMIC_CAS_ALL(uint8_t, AtomicCASInteger)

// --- Generic Floating Point CAS Structure Definition (R is the Atomic Ref
// type) ---
template <typename T, size_t n, template <typename> class R>
struct AtomicCASFP;

// n=2 (at::Half/at::BFloat16 Soft-RMW)
template <typename T, template <typename> class R>
struct AtomicCASFP<T, 2, R> {
  inline T operator()(T* address, T expected, T desired) {
    size_t offset = (size_t)address & 2;
    unsigned int* address_as_ui = (unsigned int*)((char*)address - offset);
    bool is_upper_half = offset;

    unsigned int assumed;
    unsigned int newval;

    // Using generic AtomicRef
    AtomicRef<unsigned int, R> target(*address_as_ui);

    unsigned int expected_half_word = expected.x;
    unsigned int desired_half_word = desired.x;

    unsigned int current_half_word;
    T extracted_old_value;

    do {
      assumed = *address_as_ui;
      current_half_word = is_upper_half ? (assumed >> 16) : (assumed & 0xffff);

      extracted_old_value.x = (uint16_t)current_half_word;

      if (extracted_old_value.x == expected_half_word) {
        newval = is_upper_half ? (assumed & 0xffff) | (desired_half_word << 16)
                               : (assumed & 0xffff0000) | desired_half_word;
      } else {
        break;
      }
    } while (!target.compare_exchange_strong(assumed, newval));

    if (extracted_old_value.x == expected_half_word) {
      return expected;
    } else {
      return extracted_old_value;
    }
  }
};

// n=4 (4-byte float Native CAS)
template <typename T, template <typename> class R>
struct AtomicCASFP<T, 4, R> {
  inline T operator()(T* address, T expected, T desired) {
    unsigned int* address_as_ui = (unsigned int*)address;
    unsigned int assumed;
    unsigned int newval;

    // Using generic AtomicRef
    AtomicRef<unsigned int, R> target(*address_as_ui);

    unsigned int expected_ui = *((unsigned int*)&expected);
    newval = *((unsigned int*)&desired);

    do {
      assumed = *address_as_ui;
      if (assumed != expected_ui) {
        break;
      }
    } while (!target.compare_exchange_strong(assumed, newval));

    if (assumed == expected_ui) {
      return expected;
    } else {
      return *((T*)&assumed);
    }
  }
};

// n=8 (8-byte double Native CAS)
template <typename T, template <typename> class R>
struct AtomicCASFP<T, 8, R> {
  inline T operator()(T* address, T expected, T desired) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long assumed;
    unsigned long long newval;

    // Using generic AtomicRef
    AtomicRef<unsigned long long, R> target(*address_as_ull);

    unsigned long long expected_ull = *((unsigned long long*)&expected);
    newval = *((unsigned long long*)&desired);

    do {
      assumed = *address_as_ull;
      if (assumed != expected_ull) {
        break;
      }
    } while (!target.compare_exchange_strong(assumed, newval));

    if (assumed == expected_ull) {
      return expected;
    } else {
      return *((T*)&assumed);
    }
  }
};

SYCL_ATOMIC_CAS_ALL(float, AtomicCASFP)
SYCL_ATOMIC_CAS_ALL(double, AtomicCASFP)
SYCL_ATOMIC_CAS_ALL(at::Half, AtomicCASFP)
SYCL_ATOMIC_CAS_ALL(at::BFloat16, AtomicCASFP)

} // namespace at::native::xpu
