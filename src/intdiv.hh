#pragma once

#include <stdint.h>

typedef struct {
  uint32_t magic;
  uint32_t shift;
} intdiv32_t;

typedef struct {
  uint64_t magic;
  uint32_t shift;
} intdiv64_t;

static inline intdiv32_t intdiv32_gen(const int32_t divisor)
{
  const uint32_t divisor_u = (uint32_t)divisor;
  const uint32_t floor_log2 = 31U - __builtin_clz(divisor_u);
  if (!(divisor_u & (divisor_u - 1))) {
    return intdiv32_t{0, floor_log2};
  }
  const uint64_t magic_num = 1ULL << (floor_log2 + 32);
  const uint32_t magic_u = (uint32_t)(magic_num / divisor_u) + 1;
  return intdiv32_t{magic_u, floor_log2};
}

static inline intdiv64_t intdiv64_gen(const int64_t divisor)
{
  const uint64_t divisor_u = (uint64_t)divisor;
  const uint32_t floor_log2 = 63U - __builtin_clzll(divisor_u);
  if (!(divisor_u & (divisor_u - 1))) {
    return intdiv64_t{0, floor_log2};
  }
  const __uint128_t magic_num = (__uint128_t)1 << (floor_log2 + 64);
  const uint64_t magic_u = (uint64_t)(magic_num / divisor_u) + 1;
  return intdiv64_t{magic_u, floor_log2};
}

#ifdef __CUDACC__
  #define HOSTDEVICE static __host__ __device__ __forceinline__
#else
  #define HOSTDEVICE static inline
#endif

HOSTDEVICE uint32_t intdiv_mulhi32(const uint32_t lhs, const uint32_t rhs)
{
#ifdef __CUDA_ARCH__
  return __umulhi(lhs, rhs);
#else
  return (uint32_t)(((uint64_t)lhs * rhs) >> 32);
#endif
}

HOSTDEVICE uint64_t intdiv_mulhi64(const uint64_t lhs, const uint64_t rhs)
{
#ifdef __CUDA_ARCH__
  return __umul64hi(lhs, rhs);
#else
  return (uint64_t)(((__uint128_t)lhs * rhs) >> 64);
#endif
}

HOSTDEVICE void intdiv32_div(const int32_t numerator, const intdiv32_t params,
                             int32_t* const div)
{
  const uint32_t numerator_u = (uint32_t)numerator;
  const uint32_t mul_hi = intdiv_mulhi32(numerator_u, params.magic);
  const uint32_t scaled = params.magic ? mul_hi : numerator_u;
  *div = (int32_t)(scaled >> params.shift);
}

HOSTDEVICE void intdiv64_div(const int64_t numerator, const intdiv64_t params,
                             int64_t* const div)
{
  const uint64_t numerator_u = (uint64_t)numerator;
  const uint64_t mul_hi = intdiv_mulhi64(numerator_u, params.magic);
  const uint64_t scaled = params.magic ? mul_hi : numerator_u;
  *div = (int64_t)(scaled >> params.shift);
}

HOSTDEVICE void intdiv32_divmod(const int32_t numerator,
                                const int32_t divisor,
                                const intdiv32_t params,
                                int32_t* const div,
                                int32_t* const rem)
{
  int32_t div_value;
  intdiv32_div(numerator, params, &div_value);
  const int32_t rem_value = numerator - div_value * divisor;
  *div = div_value;
  *rem = rem_value;
}

HOSTDEVICE void intdiv64_divmod(const int64_t numerator,
                                const int64_t divisor,
                                const intdiv64_t params,
                                int64_t* const div,
                                int64_t* const rem)
{
  int64_t div_value;
  intdiv64_div(numerator, params, &div_value);
  const int64_t rem_value = numerator - div_value * divisor;
  *div = div_value;
  *rem = rem_value;
}

#undef HOSTDEVICE

