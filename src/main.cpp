// Minimal certified bisection for B* = sup{B : Sigma(B) > 7/4},
// with the bracket endpoints kept as exact rational numbers (fmpq).
//
// Only the necessary steps:
//   - plain bisection on B (rational bracket [lo, hi]),
//   - truncation length L doubling from 16 (with incremental partial sum),
//   - certified decisions via ball arithmetic for Sigma and the tail bound,
//   - outer stop condition: hi - lo < 2^-bits.

#include "bmath.hpp"

#include <flint/fmpq.h>
#include <flint/fmpz.h>
#include <flint/flint.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <algorithm>

namespace {

void print_q(const char* name, const fmpq_t q)
{
    char* s = fmpq_get_str(nullptr, 10, q);
    std::printf("%s = %s\n", name, s);
    flint_free(s);
}

// Exact decimal expansion of a rational whose denominator is 2^i * 5^j
// (true for all bracket endpoints here, which start from 10193/5000 and are
// repeatedly divided by 2).  Falls back to high-precision arb printing.
void print_dec(const char* name, const fmpq_t q)
{
    fmpz_t n, d, N;
    fmpz_init_set(n, fmpq_numref(q));
    fmpz_init_set(d, fmpq_denref(q));
    int sign = fmpz_sgn(n) < 0;
    if (sign)
        fmpz_neg(n, n);

    ulong i = 0, j = 0;
    while (fmpz_is_even(d))
    {
        fmpz_divexact_ui(d, d, 2);
        ++i;
    }
    while (fmpz_divisible_ui(d, 5))
    {
        fmpz_divexact_ui(d, d, 5);
        ++j;
    }
    if (!fmpz_is_one(d))
    {
        // should not happen for our rationals; print to high precision
        arb_t t;
        arb_init(t);
        arb_set_fmpq(t, q, 4096);
        char* s = arb_get_str(t, 1200, 0);
        std::printf("%s = %s\n", name, s);
        flint_free(s);
        arb_clear(t);
        fmpz_clear(n);
        fmpz_clear(d);
        return;
    }

    ulong m = std::max(i, j);
    fmpz_init_set(N, n);
    if (m > i)
        fmpz_mul_2exp(N, N, m - i);
    if (m > j)
    {
        fmpz_t p5;
        fmpz_init_set_ui(p5, 1);
        for (ulong k = 0; k < m - j; ++k)
            fmpz_mul_ui(p5, p5, 5);
        fmpz_mul(N, N, p5);
        fmpz_clear(p5);
    }

    char* ns = fmpz_get_str(nullptr, 10, N);
    size_t len = std::strlen(ns);
    std::string out;
    if (sign)
        out += '-';
    if (m == 0)
    {
        out += ns;
    }
    else if (len <= m)
    {
        out += "0.";
        out.append(m - len, '0');
        out += ns;
    }
    else
    {
        out.append(ns, len - m);
        out += '.';
        out.append(ns + len - m, m);
    }
    std::printf("%s = %s\n", name, out.c_str());

    flint_free(ns);
    fmpz_clear(N);
    fmpz_clear(n);
    fmpz_clear(d);
}

} // namespace

int main(int argc, char** argv)
{
    long bits = argc > 1 ? std::atol(argv[1]) : 64;
    int threads = argc > 2 ? std::atoi(argv[2]) : 0;
    bool verbose = false;
    for (int i = 3; i < argc; ++i)
        if (std::strcmp(argv[i], "--verbose") == 0)
            verbose = true;
    if (bits < 8) bits = 8;
    const slong lam = bits + 64;          // internal precision (lambda)
    const long stop_bits = bits;          // outer stop condition: hi-lo < 2^-bits

    fmpq_t lo, hi, mid, width, tol;
    fmpq_init(lo);
    fmpq_init(hi);
    fmpq_init(mid);
    fmpq_init(width);
    fmpq_init(tol);

    arb_t S, T, sum, B, thr;
    arb_init(S);
    arb_init(T);
    arb_init(sum);
    arb_init(B);
    arb_init(thr);
    arb_set_ui(thr, 7);
    arb_mul_2exp_si(thr, thr, -2);

    // bracket endpoints: lo = 2, hi = 2.0386 = 10193/5000 (exact rational)
    fmpq_set_ui(lo, 2, 1);
    fmpq_set_ui(hi, 10193, 5000);

    // tolerance 2^{-stop_bits} as a rational
    fmpz_t num, den;
    fmpz_init(num);
    fmpz_init(den);
    fmpz_set_ui(num, 1);
    fmpz_set_ui(den, 1);
    fmpz_mul_2exp(den, den, (ulong)stop_bits);
    fmpq_set_fmpz_frac(tol, num, den);
    fmpz_clear(num);
    fmpz_clear(den);

    // verify Sigma(hi) < 7/4 (cheap, doubling L)
    {
        arb_set_fmpq(B, hi, lam);
        slong L = 16;
        bool ok = false;
        for (int it = 0; it < 64 && !ok; ++it)
        {
            slong p = lam;
            bstar::sigma_trunc(S, B, L, p, threads);
            bstar::tail_upper(T, B, L, p);
            arb_add(sum, S, T, p);
            if (arb_lt(sum, thr)) { ok = true; break; }
            if (arb_gt(S, thr))
            {
                std::fprintf(stderr, "error: Sigma(10193/5000) >= 7/4\n");
                return 1;
            }
            L *= 2;
        }
        if (!ok)
        {
            std::fprintf(stderr, "error: cannot verify Sigma(hi) < 7/4\n");
            return 1;
        }
    }

    // plain bisection with incremental partial-sum cache
    long evaluations = 0;
    long total_terms = 0;
    while (true)
    {
        fmpq_sub(width, hi, lo);
        if (fmpq_cmp(width, tol) <= 0)
            break;

        fmpq_add(mid, lo, hi);
        fmpq_div_2exp(mid, mid, 1); // midpoint (rational)

        arb_set_fmpq(B, mid, lam);
        slong L = 16;
        slong p = lam;
        bstar::sigma_trunc(S, B, L, p, threads);
        ++evaluations;
        total_terms += L;

        bool went_lo = false;
        while (true)
        {
            if (arb_gt(S, thr)) // Sigma(mid) > 7/4 -> lo <- mid
            {
                fmpq_set(lo, mid);
                went_lo = true;
                break;
            }
            bstar::tail_upper(T, B, L, lam);
            arb_add(sum, S, T, lam);
            if (arb_lt(sum, thr)) // Sigma(mid) < 7/4 -> hi <- mid
            {
                fmpq_set(hi, mid);
                went_lo = false;
                break;
            }
            slong L2 = 2 * L;
            arb_t D;
            arb_init(D);
            // slong p2 = lam + (slong)std::ceil(std::log2((double)L2 + 2.0)) + 16;
            bstar::sigma_trunc_range(D, B, L + 1, L2, lam, threads);
            arb_add(S, S, D, lam);
            arb_clear(D);
            total_terms += (L2 - L);
            L = L2;
        }

        if (verbose)
        {
            slong sdigits = (slong)std::ceil((double)(bits + 64) * 0.30103) + 12;
            arf_t a;
            arf_init(a);
            if (went_lo)
            {
                // Sigma(mid) > 7/4: only B_l was updated
                char* sl = fmpq_get_str(nullptr, 10, lo);
                std::printf("[bisect] B_l = %s   N = %ld\n", sl, (long)L);
                arb_get_lbound_arf(a, S, lam);
                char* sb = arf_get_str(a, sdigits);
                std::printf("         lb(s_trunc) = %s\n", sb);
                flint_free(sb);
                flint_free(sl);
            }
            else
            {
                // Sigma(mid) < 7/4: only B_r was updated
                char* sh = fmpq_get_str(nullptr, 10, hi);
                std::printf("[bisect] B_r = %s   L = %ld\n", sh, (long)L);
                arb_get_ubound_arf(a, sum, lam);
                char* ss = arf_get_str(a, sdigits);
                std::printf("         ub(s_trunc+s_tail) = %s\n", ss);
                flint_free(ss);
                arb_get_ubound_arf(a, S, lam);
                char* su = arf_get_str(a, sdigits);
                std::printf("         ub(s_trunc) = %s\n", su);
                flint_free(su);
                arb_get_ubound_arf(a, T, lam);
                char* st = arf_get_str(a, sdigits);
                std::printf("         ub(s_tail) = %s\n", st);
                flint_free(st);
                flint_free(sh);
            }
            arf_clear(a);
        }
    }

    std::printf("evaluations = %ld, total terms = %ld\n", evaluations, total_terms);
    print_q("B* lower (rational)", lo);
    print_dec("B* lower (decimal)", lo);
    print_q("B* upper (rational)", hi);
    print_dec("B* upper (decimal)", hi);

    fmpq_clear(lo);
    fmpq_clear(hi);
    fmpq_clear(mid);
    fmpq_clear(width);
    fmpq_clear(tol);
    arb_clear(S);
    arb_clear(T);
    arb_clear(sum);
    arb_clear(B);
    arb_clear(thr);
    return 0;
}
