// Rigorous enclosures for the B* problem, built on FLINT's arb ball arithmetic.
#pragma once

#include <flint/arb.h>

namespace bstar {

// h(eps) = -eps*log2(eps) - (1-eps)*log2(1-eps), eps in [0,1].
void entropy_h(arb_t res, const arb_t eps, slong prec);

// eps_x in (0,1/2] with h(eps) + eps*log2(x-1) = 1, x >= 2.
void epsilon_of_x(arb_t res, const arb_t x, slong prec);

// sigma_x = 1/eps_x
void sigma_of_x(arb_t res, const arb_t x, slong prec);

// u_l(x) = l*x - B*x*log2(x)
void u_func(arb_t res, slong l, const arb_t B, const arb_t x, slong prec);

// s_l(x) = (l-1)*x - (x-1)*(B-1)*sigma_x
void s_func(arb_t res, slong l, const arb_t B, const arb_t x, slong prec);

// x1 = 2^(l/B) / e
void x1_func(arb_t res, slong l, const arb_t B, slong prec);

// y*: greatest solution of y + 1/ln(2-2/y) = (l-1)/(B-1), y > 3.4213
void ystar_func(arb_t res, slong l, const arb_t B, slong prec);

// x2 = 1 + 2^(y*(1-h(1/y)))
void x2_func(arb_t res, slong l, const arb_t B, slong prec);

// d_l as defined in the problem (rigorous enclosure)
void d_func(arb_t res, slong l, const arb_t B, slong prec);

// ceil(d_l) as a rigorous integer enclosure
void d_ceil_func(arb_t res, slong l, const arb_t B, slong prec);

// Truncated sum Sigma_L(B) = sum_{l=3}^{L} ceil(d_l) 2^-l (no tail).
void sigma_trunc(arb_t res, const arb_t B, slong L, slong prec, int threads);

// Partial sum over a range [l1, l2] with the same ceiling scheme and the
// aggregate ceiling correction.
void sigma_trunc_range(arb_t res, const arb_t B, slong l1, slong l2, slong prec, int threads);

// Certified upper bound for sum_{n>L} ceil(d_n) 2^-n.
void tail_upper(arb_t res, const arb_t B, slong L, slong prec);

} // namespace bstar
