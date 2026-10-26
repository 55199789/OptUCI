#include "bmath.hpp"

#include <flint/arf.h>
#include <omp.h>
#include <flint/fmpz.h>
#include <flint/mag.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace bstar {

namespace {

struct Ball {
    arb_t v;
    Ball() { arb_init(v); }
    ~Ball() { arb_clear(v); }
    Ball(const Ball&) = delete;
    Ball& operator=(const Ball&) = delete;
    Ball(Ball&& o) noexcept { arb_init(v); arb_swap(v, o.v); }
    Ball& operator=(Ball&& o) noexcept { arb_swap(v, o.v); return *this; }
};

[[noreturn]] void fail(const char* msg)
{
    throw std::runtime_error(msg);
}

void log2_arb(arb_t r, const arb_t x, slong prec)
{
    Ball l2;
    arb_const_log2(l2.v, prec + 8);
    arb_log(r, x, prec + 8);
    arb_div(r, r, l2.v, prec);
}

int sgn_definite(const arb_t x)
{
    if (arb_is_positive(x)) return 1;
    if (arb_is_negative(x)) return -1;
    return 0;
}

void expand1(arb_t X, const arb_t x, slong prec)
{
    Ball a, b;
    arb_sub_ui(a.v, x, 1, prec);
    arb_add_ui(b.v, x, 1, prec);
    arb_union(X, a.v, b.v, prec);
}

double mid_d(const arb_t x)
{
    return arf_get_d(arb_midref(x), ARF_RND_NEAR);
}

// Root solver for a continuous, monotonically increasing f with
// f(inlo) <= 0 <= f(inhi). Uses point Newton followed by a local sign
// bracket around the Newton point; every update is inclusion-valid.
template <class F, class DF, class FF>
void solve_increasing(arb_t res, const arb_t inlo, const arb_t inhi,
                      F&& f, DF&& df, FF&& ff, slong prec, slong target_bits,
                      const arb_t hint)
{
    Ball lo, hi, m, fm, dfm, corr, A, B, d, tmp;
    arb_set(lo.v, inlo);
    arb_set(hi.v, inhi);

    // 1. point Newton (fused f/f' when available)
    Ball tiny, ratio, acorr;
    arb_set_ui(tiny.v, 1);
    arb_mul_2exp_si(tiny.v, tiny.v, -(prec + 16));
    arb_add(m.v, lo.v, hi.v, prec + 48);
    arb_mul_2exp_si(m.v, m.v, -1);
    if (hint != nullptr && arb_gt(hint, lo.v) && arb_lt(hint, hi.v))
        arb_set(m.v, hint);
    double best_corr = HUGE_VAL;
    int stall = 0;
    for (int k = 0; k < 40; ++k)
    {
        if (!ff(fm.v, dfm.v, m.v, prec + 48))
        {
            f(fm.v, m.v, prec + 48);
            df(dfm.v, m.v, prec + 48);
        }
        if (arb_contains_zero(dfm.v))
            break;
        arb_div(corr.v, fm.v, dfm.v, prec + 48);
        arb_sub(m.v, m.v, corr.v, prec + 48);
        if (arb_lt(m.v, lo.v)) arb_set(m.v, lo.v);
        if (arb_gt(m.v, hi.v)) arb_set(m.v, hi.v);
        double ca = std::fabs(mid_d(corr.v));
        if (ca < 0.5 * best_corr) { best_corr = ca; stall = 0; }
        else if (++stall >= 3) break;
        arb_abs(acorr.v, corr.v);
        arb_div(ratio.v, acorr.v, m.v, prec + 48);
        if (arb_lt(ratio.v, tiny.v))
            break;
    }
    arb_get_mid_arb(m.v, m.v);

    // 2. local sign bracket around m (doubling until signs are definite)
    double sc = std::max(1.0, std::fabs(mid_d(m.v)));
    double d0 = std::ldexp(sc, -(double)(prec + 8));
    if (best_corr < HUGE_VAL && best_corr > 0.0)
        d0 = std::max(d0, 8.0 * best_corr);
    arb_set_d(d.v, d0);
    bool done = false;
    for (int k = 0; k < 400; ++k)
    {
        arb_sub(A.v, m.v, d.v, prec + 64);
        arb_add(B.v, m.v, d.v, prec + 64);
        if (arb_lt(A.v, lo.v)) arb_set(A.v, lo.v);
        if (arb_gt(B.v, hi.v)) arb_set(B.v, hi.v);
        f(fm.v, A.v, prec + 64);
        int sa = sgn_definite(fm.v);
        f(fm.v, B.v, prec + 64);
        int sb = sgn_definite(fm.v);
        if (sa < 0 && sb > 0)
        {
            arb_set(lo.v, A.v);
            arb_set(hi.v, B.v);
            done = true;
            break;
        }
        if (sa > 0 && arb_lt(A.v, hi.v)) arb_set(hi.v, A.v); // root < A
        if (sb < 0 && arb_gt(B.v, lo.v)) arb_set(lo.v, B.v); // root > B
        if (sa < 0 && arb_gt(A.v, lo.v)) arb_set(lo.v, A.v);
        if (sb > 0 && arb_lt(B.v, hi.v)) arb_set(hi.v, B.v);
        arb_sub(tmp.v, hi.v, lo.v, 64);
        {
            Ball wtol;
            arb_set_d(wtol.v, std::ldexp(sc, -(double)target_bits));
            if (arb_lt(tmp.v, wtol.v)) { done = true; break; }
        }
        arb_mul_2exp_si(d.v, d.v, 1);
    }

    // 3. plain bisection fallback
    if (!done)
    {
        for (int it = 0; it < 100000; ++it)
        {
            arb_sub(tmp.v, hi.v, lo.v, 64);
            {
                Ball wtol;
                arb_set_d(wtol.v, std::ldexp(sc, -(double)target_bits));
                if (arb_lt(tmp.v, wtol.v))
                    break;
            }
            arb_add(m.v, lo.v, hi.v, prec + 32);
            arb_mul_2exp_si(m.v, m.v, -1);
            arb_get_mid_arb(m.v, m.v);
            f(fm.v, m.v, prec + 64);
            int s = sgn_definite(fm.v);
            if (s > 0) arb_set(hi.v, m.v);
            else if (s < 0) arb_set(lo.v, m.v);
            else break;
        }
    }
    arb_union(res, lo.v, hi.v, prec);
}

template <class F, class DF>
void solve_increasing(arb_t res, const arb_t inlo, const arb_t inhi,
                      F&& f, DF&& df, slong prec, slong target_bits,
                      const arb_t hint = nullptr)
{
    auto ff = [](arb_t, arb_t, const arb_t, slong) { return false; };
    solve_increasing(res, inlo, inhi, f, df, ff, prec, target_bits, hint);
}

} // namespace

void entropy_h(arb_t res, const arb_t eps, slong prec)
{
    if (arb_is_zero(eps) || arb_is_one(eps))
    {
        arb_zero(res);
        return;
    }
    Ball one;
    arb_set_ui(one.v, 1);
    if (!arb_is_positive(eps) || !arb_lt(eps, one.v))
        fail("entropy_h: eps must lie strictly inside (0,1)");
    Ball om, t1, t2;
    arb_sub_ui(om.v, eps, 1, prec + 8);
    arb_neg(om.v, om.v);
    log2_arb(t1.v, eps, prec + 8);
    arb_mul(t1.v, t1.v, eps, prec + 8);
    log2_arb(t2.v, om.v, prec + 8);
    arb_mul(t2.v, t2.v, om.v, prec + 8);
    arb_add(res, t1.v, t2.v, prec);
    arb_neg(res, res);
}

void sigma_of_x(arb_t res, const arb_t x, slong prec)
{
    if (arb_equal_si(x, 2))
    {
        arb_set_ui(res, 2);
        return;
    }
    Ball two;
    arb_set_ui(two.v, 2);
    if (arb_lt(x, two.v))
        fail("sigma_of_x: x must be >= 2");

    Ball xm, L2, one;
    arb_set_ui(one.v, 1);
    arb_sub_ui(xm.v, x, 1, prec + 32);
    if (arb_contains_nonpositive(xm.v))
        arb_max(xm.v, xm.v, one.v, prec + 32);
    log2_arb(L2.v, xm.v, prec + 32);

    // f(s) = s - L2 - log2 s - (s-1) log2(s/(s-1)); root s = sigma_x.
    auto F = [&](arb_t out, const arb_t s, slong p) {
        Ball t1, t2, sm1, ratio, tmp;
        log2_arb(t1.v, s, p + 8);
        arb_sub_ui(sm1.v, s, 1, p + 8);
        arb_div(ratio.v, s, sm1.v, p + 8);
        log2_arb(t2.v, ratio.v, p + 8);
        arb_mul(t2.v, t2.v, sm1.v, p + 8);
        arb_add(tmp.v, t1.v, t2.v, p + 8);
        arb_sub(out, s, L2.v, p + 8);
        arb_sub(out, out, tmp.v, p + 8);
    };
    auto DF = [&](arb_t out, const arb_t s, slong p) {
        Ball sm1, ratio, t, one2;
        arb_sub_ui(sm1.v, s, 1, p + 8);
        arb_div(ratio.v, s, sm1.v, p + 8);
        log2_arb(t.v, ratio.v, p + 8);
        arb_set_ui(one2.v, 1);
        arb_sub(out, one2.v, t.v, p + 8);
    };

    // fused evaluator sharing log2(s/(s-1))
    auto FF = [&](arb_t fo, arb_t dfo, const arb_t s, slong p) -> bool {
        Ball t1, t2, sm1, ratio, tmp;
        log2_arb(t1.v, s, p + 8);
        arb_sub_ui(sm1.v, s, 1, p + 8);
        arb_div(ratio.v, s, sm1.v, p + 8);
        log2_arb(t2.v, ratio.v, p + 8);
        arb_mul(tmp.v, t2.v, sm1.v, p + 8);
        arb_add(tmp.v, t1.v, tmp.v, p + 8);
        arb_sub(fo, s, L2.v, p + 8);
        arb_sub(fo, fo, tmp.v, p + 8);
        arb_set_ui(tmp.v, 1);
        arb_sub(dfo, tmp.v, t2.v, p + 8);
        return true;
    };

    Ball shi, tL, lt, fv;
    arb_add_ui(tL.v, L2.v, 2, prec + 32);
    log2_arb(lt.v, tL.v, prec + 32);
    arb_add(shi.v, L2.v, lt.v, prec + 32);
    arb_add_ui(shi.v, shi.v, 2, prec + 32);
    for (int i = 0; i < 300; ++i)
    {
        F(fv.v, shi.v, prec + 32);
        if (arb_is_positive(fv.v))
            break;
        arb_mul_2exp_si(shi.v, shi.v, 1);
    }
    // seed sigma0 ~ L2 + log2(L2+2) + 1/ln2
    Ball ln2s, invln2s, s0;
    arb_const_log2(ln2s.v, prec + 32);
    arb_inv(invln2s.v, ln2s.v, prec + 32);
    arb_add(s0.v, L2.v, lt.v, prec + 32);
    arb_add(s0.v, s0.v, invln2s.v, prec + 32);
    if (arb_lt(s0.v, two.v)) arb_set(s0.v, two.v);
    if (arb_gt(s0.v, shi.v)) arb_set(s0.v, shi.v);
    solve_increasing(res, two.v, shi.v, F, DF, FF, prec, prec - 8, s0.v);
}

void epsilon_of_x(arb_t res, const arb_t x, slong prec)
{
    Ball s;
    sigma_of_x(s.v, x, prec + 16);
    arb_inv(res, s.v, prec);
}

void ystar_func(arb_t res, slong l, const arb_t B, slong prec)
{
    Ball Bm1, c, lo, fv;
    arb_sub_ui(Bm1.v, B, 1, prec + 32);
    if (arb_contains_nonpositive(Bm1.v))
        fail("ystar: requires B > 1");
    arb_set_si(c.v, l - 1);
    arb_div(c.v, c.v, Bm1.v, prec + 32);
    arb_set_str(lo.v, "3.4214", prec + 32);
    if (!arb_gt(c.v, lo.v))
        fail("ystar: (l-1)/(B-1) is too small for the greatest root");

    // f(y) = y + 1/ln(2-2/y) - c; f'(y) = 1 - 1/((y-1)y ln^2(2-2/y)).
    auto F = [&](arb_t out, const arb_t y, slong p) {
        Ball u, tw, ln, inv;
        arb_set_ui(tw.v, 2);
        arb_inv(u.v, y, p + 8);
        arb_mul_2exp_si(u.v, u.v, 1);
        arb_sub(u.v, tw.v, u.v, p + 8);
        arb_log(ln.v, u.v, p + 8);
        arb_inv(inv.v, ln.v, p + 8);
        arb_add(out, y, inv.v, p + 8);
        arb_sub(out, out, c.v, p + 8);
    };
    auto DF = [&](arb_t out, const arb_t y, slong p) {
        Ball u, tw, ln, t, ym1, one2;
        arb_set_ui(tw.v, 2);
        arb_inv(u.v, y, p + 8);
        arb_mul_2exp_si(u.v, u.v, 1);
        arb_sub(u.v, tw.v, u.v, p + 8);
        arb_log(ln.v, u.v, p + 8);
        arb_sub_ui(ym1.v, y, 1, p + 8);
        arb_mul(t.v, y, ym1.v, p + 8);
        arb_mul(t.v, t.v, ln.v, p + 8);
        arb_mul(t.v, t.v, ln.v, p + 8);
        arb_inv(t.v, t.v, p + 8);
        arb_set_ui(one2.v, 1);
        arb_sub(out, one2.v, t.v, p + 8);
    };

    DF(fv.v, lo.v, prec + 32);
    if (!arb_is_positive(fv.v))
        fail("ystar: derivative not positive at the lower bracket point");

    // fused evaluator sharing ln(2-2/y)
    auto FF = [&](arb_t fo, arb_t dfo, const arb_t y, slong p) -> bool {
        Ball u, tw, ln, t, ym1;
        arb_set_ui(tw.v, 2);
        arb_inv(u.v, y, p + 8);
        arb_mul_2exp_si(u.v, u.v, 1);
        arb_sub(u.v, tw.v, u.v, p + 8);
        arb_log(ln.v, u.v, p + 8);
        arb_inv(t.v, ln.v, p + 8);
        arb_add(fo, y, t.v, p + 8);
        arb_sub(fo, fo, c.v, p + 8);
        arb_sub_ui(ym1.v, y, 1, p + 8);
        arb_mul(t.v, y, ym1.v, p + 8);
        arb_mul(t.v, t.v, ln.v, p + 8);
        arb_mul(t.v, t.v, ln.v, p + 8);
        arb_inv(t.v, t.v, p + 8);
        arb_set_ui(tw.v, 1);
        arb_sub(dfo, tw.v, t.v, p + 8);
        return true;
    };

    // asymptotic seed: y* ~ c - 1/ln2
    Ball ln2b, invln2, y0;
    arb_const_log2(ln2b.v, prec + 32);
    arb_inv(invln2.v, ln2b.v, prec + 32);
    arb_sub(y0.v, c.v, invln2.v, prec + 32);
    if (arb_lt(y0.v, lo.v)) arb_set(y0.v, lo.v);
    if (arb_gt(y0.v, c.v)) arb_set(y0.v, c.v);
    solve_increasing(res, lo.v, c.v, F, DF, FF, prec, prec - 8, y0.v);
}

void x1_func(arb_t res, slong l, const arb_t B, slong prec)
{
    Ball t, ln2;
    arb_const_log2(ln2.v, prec + 8);
    arb_set_si(t.v, l);
    arb_div(t.v, t.v, B, prec + 8);
    arb_mul(t.v, t.v, ln2.v, prec + 8);
    arb_sub_ui(t.v, t.v, 1, prec + 8);
    arb_exp(res, t.v, prec);
}

namespace {
void x2_from_ystar(arb_t res, const arb_t y, slong prec)
{
    Ball invy, h, t, ln2;
    arb_inv(invy.v, y, prec + 8);
    entropy_h(h.v, invy.v, prec + 8);
    arb_sub_ui(t.v, h.v, 1, prec + 8);
    arb_neg(t.v, t.v);
    arb_mul(t.v, t.v, y, prec + 8);
    arb_const_log2(ln2.v, prec + 8);
    arb_mul(t.v, t.v, ln2.v, prec + 8);
    arb_exp(t.v, t.v, prec + 8);
    arb_add_ui(res, t.v, 1, prec);
}
} // namespace

void x2_func(arb_t res, slong l, const arb_t B, slong prec)
{
    Ball y;
    ystar_func(y.v, l, B, prec + 32);
    x2_from_ystar(res, y.v, prec);
}

void u_func(arb_t res, slong l, const arb_t B, const arb_t x, slong prec)
{
    Ball lg, t;
    log2_arb(lg.v, x, prec + 8);
    arb_mul(t.v, lg.v, x, prec + 8);
    arb_mul(t.v, t.v, B, prec + 8);
    arb_mul_si(res, x, l, prec + 8);
    arb_sub(res, res, t.v, prec);
}

void s_func(arb_t res, slong l, const arb_t B, const arb_t x, slong prec)
{
    Ball sg, xm1, Bm1, t;
    sigma_of_x(sg.v, x, prec + 16);
    arb_sub_ui(xm1.v, x, 1, prec + 8);
    arb_sub_ui(Bm1.v, B, 1, prec + 8);
    arb_mul(t.v, xm1.v, Bm1.v, prec + 8);
    arb_mul(t.v, t.v, sg.v, prec + 8);
    arb_mul_si(res, x, l - 1, prec + 8);
    arb_sub(res, res, t.v, prec);
}

namespace {

// Candidate integer set containing floor(y) and ceil(y) for every real y in
// the ball x. Returns the number of candidates (>=2), or -1 if too many.
long candidates_int(fmpz_t* ns, long cap, const arb_t x)
{
    fmpz_t a, b, e, fl, fh, d, k;
    fmpz_init(a); fmpz_init(b); fmpz_init(e);
    fmpz_init(fl); fmpz_init(fh); fmpz_init(d); fmpz_init(k);
    long count = -1;

    arb_get_interval_fmpz_2exp(a, b, e, x);
    if (fmpz_sgn(e) >= 0)
    {
        // count = (b-a)*2^e + 2; avoid huge shifts when it exceeds cap
        if (fmpz_cmp_ui(e, 5) >= 0)
            goto done;
        if (fmpz_sgn(e) > 0 && fmpz_cmp(a, b) != 0)
        {
            fmpz_sub(d, b, a);
            fmpz_mul_2exp(d, d, fmpz_get_ui(e));
            if (fmpz_cmp_si(d, cap - 2) > 0)
                goto done;
        }
        ulong ee = fmpz_get_ui(e);
        fmpz_mul_2exp(fl, a, ee);
        fmpz_mul_2exp(fh, b, ee);
    }
    else
    {
        fmpz_neg(k, e);
        if (fmpz_cmp_ui(k, 1u << 30) > 0)
            goto done;
        ulong kk = fmpz_get_ui(k);
        fmpz_fdiv_q_2exp(fl, a, kk);
        fmpz_fdiv_q_2exp(fh, b, kk);
    }
    fmpz_sub(d, fh, fl);
    if (fmpz_cmp_si(d, cap - 2) > 0)
        goto done;
    count = fmpz_get_si(d) + 2;
    for (long i = 0; i < count; ++i)
        fmpz_add_si(ns[i], fl, i);

done:
    fmpz_clear(a); fmpz_clear(b); fmpz_clear(e);
    fmpz_clear(fl); fmpz_clear(fh); fmpz_clear(d); fmpz_clear(k);
    return count;
}

void include_value(arf_t dlo, arf_t dhi, const arb_t v, slong prec, arf_t tmp)
{
    arb_get_lbound_arf(tmp, v, prec);
    if (arf_cmp(tmp, dlo) > 0)
        arf_set(dlo, tmp);
    arb_get_ubound_arf(tmp, v, prec);
    if (arf_cmp(tmp, dhi) > 0)
        arf_set(dhi, tmp);
}

// For large x2 (interval path): encloses s_l over {floor(x2), ceil(x2)} using
// the exact identity y* = sigma_{x2} and a Lipschitz bound on s'.
// Returns false if the necessary size conditions are not met.
bool s_interval_via_ystar(arb_t out, slong l, const arb_t B, const arb_t y,
                          const arb_t x2, slong prec)
{
    Ball one, two, three, four, ylb, yub, xlb, Bm1, t1, t2, Lstar, S2, dsig,
        s_lo, s_hi, F1, F2, half, lny;
    arb_set_ui(one.v, 1);
    arb_set_ui(two.v, 2);
    arb_set_ui(three.v, 3);
    arb_set_ui(four.v, 4);
    if (!arb_gt(y, three.v) || !arb_gt(x2, four.v))
        return false;
    arf_t a;
    arf_init(a);
    arb_get_lbound_arf(a, y, prec + 32);
    arb_set_arf(ylb.v, a);
    arb_get_ubound_arf(a, y, prec + 32);
    arb_set_arf(yub.v, a);
    arb_get_lbound_arf(a, x2, prec + 32);
    arb_set_arf(xlb.v, a);
    arf_clear(a);

    // Lstar = ln(2 - 2/y)
    arb_inv(t1.v, y, prec + 16);
    arb_mul_2exp_si(t1.v, t1.v, 1);
    arb_sub(t1.v, two.v, t1.v, prec + 16);
    if (!arb_is_positive(t1.v))
        return false;
    arb_log(Lstar.v, t1.v, prec + 16);

    // S2 = (B-1)*(x2/Lstar + y)
    arb_sub_ui(Bm1.v, B, 1, prec + 16);
    arb_div(t1.v, x2, Lstar.v, prec + 16);
    arb_add(t1.v, t1.v, y, prec + 16);
    arb_mul(S2.v, t1.v, Bm1.v, prec + 16);

    // dsig = (y_ub + 1) / ((x_lo - 1) * ln(y_lb - 2))
    arb_add_ui(t1.v, yub.v, 1, prec + 16);
    arb_sub_ui(t2.v, xlb.v, 1, prec + 16);
    if (!arb_is_positive(t2.v))
        return false;
    arb_sub_ui(lny.v, ylb.v, 2, prec + 16);
    if (!arb_is_positive(lny.v))
        return false;
    arb_log(lny.v, lny.v, prec + 16);
    arb_mul(t2.v, t2.v, lny.v, prec + 16);
    arb_div(dsig.v, t1.v, t2.v, prec + 16);
    arb_set_ui(half.v, 1);
    arb_mul_2exp_si(half.v, half.v, -1);
    if (!arb_lt(dsig.v, half.v))
        return false;

    // F(s) = (l-1) - (B-1)*s*(1 + 1/ln(s-1)), decreasing for s >= 4
    auto F = [&](arb_t fo, const arb_t sv) {
        Ball sm1, ls, ratio, base;
        arb_sub_ui(sm1.v, sv, 1, prec + 16);
        if (!arb_is_positive(sm1.v))
        {
            arb_zero(fo);
            return;
        }
        arb_log(ls.v, sm1.v, prec + 16);
        arb_inv(ratio.v, ls.v, prec + 16);
        arb_add_ui(ratio.v, ratio.v, 1, prec + 16);
        arb_mul(base.v, sv, ratio.v, prec + 16);
        arb_mul(base.v, base.v, Bm1.v, prec + 16);
        arb_set_si(fo, l - 1);
        arb_sub(fo, fo, base.v, prec + 16);
    };
    arb_sub(s_lo.v, y, dsig.v, prec + 16);
    arb_add(s_hi.v, y, dsig.v, prec + 16);
    if (!arb_gt(s_lo.v, four.v))
        return false;
    F(F1.v, s_lo.v);
    F(F2.v, s_hi.v);
    arf_t m1, m2;
    arf_init(m1);
    arf_init(m2);
    arb_get_abs_ubound_arf(m1, F1.v, prec + 16);
    arb_get_abs_ubound_arf(m2, F2.v, prec + 16);
    if (arf_cmp(m2, m1) > 0)
        arf_set(m1, m2);
    arb_set_arf(t1.v, m1); // p >= max |F|
    arb_neg(t2.v, t1.v);
    Ball blo, bhi;
    arb_add(blo.v, S2.v, t2.v, prec + 16);
    arb_add(bhi.v, S2.v, t1.v, prec + 16);
    arb_union(out, blo.v, bhi.v, prec);
    arf_clear(m1);
    arf_clear(m2);
    return true;
}

// Candidates n_0 < ... < n_{m-1} contain floor(y) and ceil(y) for every
// possible y. Adds lower/upper bounds of the corresponding part of d_l.
template <class V>
void include_candidates(arf_t dlo, arf_t dhi, fmpz_t* ns, long count,
                        V&& val, slong prec)
{
    Ball x, v, v1, v2;
    arf_t t, l1, l2, p, minp;
    arf_init(t); arf_init(l1); arf_init(l2); arf_init(p); arf_init(minp);
    bool have = false;

    for (long i = 0; i < count; ++i)
    {
        arb_set_fmpz(x.v, ns[i]);
        val(v.v, x.v);
        arb_get_ubound_arf(t, v.v, prec);
        if (arf_cmp(t, dhi) > 0)
            arf_set(dhi, t);
    }
    // The true pair is (m, m+1) for some m in [n_0, n_{m-2}].
    for (long i = 0; i + 1 < count; ++i)
    {
        arb_set_fmpz(x.v, ns[i]);
        val(v1.v, x.v);
        arb_set_fmpz(x.v, ns[i + 1]);
        val(v2.v, x.v);
        arb_get_lbound_arf(l1, v1.v, prec);
        arb_get_lbound_arf(l2, v2.v, prec);
        arf_set(p, arf_cmp(l1, l2) > 0 ? l1 : l2);
        if (!have || arf_cmp(p, minp) < 0)
        {
            arf_set(minp, p);
            have = true;
        }
    }
    if (have && arf_cmp(minp, dlo) > 0)
        arf_set(dlo, minp);

    arf_clear(t); arf_clear(l1); arf_clear(l2); arf_clear(p); arf_clear(minp);
}

} // namespace

void d_func(arb_t res, slong l, const arb_t B, slong prec)
{
    Ball base, twoB;
    arb_set_si(base.v, 2 * l);
    arb_mul_ui(twoB.v, B, 2, prec + 8);
    arb_sub(base.v, base.v, twoB.v, prec + 8);
    if (l <= 4)
    {
        arb_set(res, base.v);
        return;
    }

    arf_t dlo, dhi, t;
    arf_init(dlo); arf_init(dhi); arf_init(t);
    arb_get_lbound_arf(dlo, base.v, prec);
    arb_get_ubound_arf(dhi, base.v, prec);

    constexpr long CAP = 20;
    fmpz_t ns[CAP];
    for (auto& n : ns)
        fmpz_init(n);

    bool need_u = true;
    if (l > 7)
    { // s part
        Ball y, x2;
        ystar_func(y.v, l, B, prec + 32);
        x2_from_ystar(x2.v, y.v, prec + 16);
        long count = candidates_int(ns, CAP, x2.v);
        if (count > 0)
        {
            auto val = [&](arb_t out, const arb_t x) { s_func(out, l, B, x, prec + 16); };
            include_candidates(dlo, dhi, ns, count, val, prec);
        }
        else
        {
            Ball v;
            if (!s_interval_via_ystar(v.v, l, B, y.v, x2.v, prec + 16))
            {
                Ball X;
                expand1(X.v, x2.v, prec + 16);
                s_func(v.v, l, B, X.v, prec + 16);
            }
            include_value(dlo, dhi, v.v, prec, t);
        }

        // Cheap bound: if the current max (base + s) already exceeds the
        // global maximum of the u branch, skip the u branch entirely.
        Ball uub, ln2, ee, tt;
        arb_const_log2(ln2.v, prec + 16);
        arb_const_e(ee.v, prec + 16);
        arb_set_si(tt.v, l);
        arb_div(tt.v, tt.v, B, prec + 16);
        arb_mul(tt.v, tt.v, ln2.v, prec + 16);
        arb_exp(tt.v, tt.v, prec + 16);
        arb_mul(uub.v, B, tt.v, prec + 16);
        arb_div(uub.v, uub.v, ee.v, prec + 16);
        arb_div(uub.v, uub.v, ln2.v, prec + 16);
        Ball curlo;
        arb_set_arf(curlo.v, dlo);
        if (arb_gt(curlo.v, uub.v))
            need_u = false;
    }

    if (need_u)
    { // u part
        Ball x1;
        x1_func(x1.v, l, B, prec + 16);
        long count = candidates_int(ns, CAP, x1.v);
        if (count > 0)
        {
            auto val = [&](arb_t out, const arb_t x) { u_func(out, l, B, x, prec + 16); };
            include_candidates(dlo, dhi, ns, count, val, prec);
        }
        else
        {
            Ball X, v;
            expand1(X.v, x1.v, prec + 16);
            u_func(v.v, l, B, X.v, prec + 16);
            include_value(dlo, dhi, v.v, prec, t);
        }
    }

    arb_set_interval_arf(res, dlo, dhi, prec);

    for (auto& n : ns)
        fmpz_clear(n);
    arf_clear(dlo); arf_clear(dhi); arf_clear(t);
}

void d_ceil_func(arb_t res, slong l, const arb_t B, slong prec)
{
    // Rigorous enclosure of ceil(d_l): the interval hull of the ceilings of
    // the ball endpoints.  arf_ceil is exact and cheap even when d_l is
    // astronomically large (an arf with non-negative exponent is already an
    // integer), so no large fmpz is materialised.
    Ball d;
    arf_t lo, hi, clo, chi;
    arf_init(lo);
    arf_init(hi);
    arf_init(clo);
    arf_init(chi);
    d_func(d.v, l, B, prec);
    arb_get_interval_arf(lo, hi, d.v, prec);
    arf_ceil(clo, lo);
    arf_ceil(chi, hi);
    arb_set_interval_arf(res, clo, chi, prec);
    arf_clear(lo);
    arf_clear(hi);
    arf_clear(clo);
    arf_clear(chi);
}

void tail_upper(arb_t res, const arb_t B, slong L, slong prec)
{
    // Certified bound for  sum_{n>L} ceil(d_n) 2^-n  (L >= 8, 2 < B < 2.0386):
    //
    //   (L+2)2^-L
    // + (B-1) (1/ln(2-2/3.4213) - 1/ln2) 2^-L
    // + (B-1)/ln(2-2/3.4213) * 2^(-1-2/ln2)
    //       * 2^(1/(U ln2)) / U * 2^(-(L-1)beta) / (2^beta - 1)
    // + B/(e ln2) * 2^((L+1)(1/B-1)) / (1-2^(1/B-1)),
    //
    // with beta = 1 - 1/(B-1) and U = (L-1)/(B-1) - 1/ln2.
    Ball ln2, e, one, two, three4213, lmin, t, Bm1, invBm1, beta,
        qpow, twobeta, denbeta, U, invU, inner, coef, cexp, alpha, expo,
        pw, den, ut, t1, t2, t3, t4;
    arb_const_log2(ln2.v, prec + 16);
    arb_const_e(e.v, prec + 16);
    arb_set_ui(one.v, 1);
    arb_set_ui(two.v, 2);
    arb_set_str(three4213.v, "3.4213", prec + 32);
    arb_sub_ui(Bm1.v, B, 1, prec + 16);
    if (arb_contains_nonpositive(Bm1.v) || L < 8)
    {
        arb_zero(res);
        return;
    }

    // lmin = ln(2 - 2/3.4213)
    arb_inv(t.v, three4213.v, prec + 16);
    arb_mul_2exp_si(t.v, t.v, 1);
    arb_sub(t.v, two.v, t.v, prec + 16);
    arb_log(lmin.v, t.v, prec + 16);

    // T1 = (L+2) 2^-L
    arb_set_si(t1.v, L + 2);
    arb_mul_2exp_si(t1.v, t1.v, -L);

    // T2 = (B-1) (1/lmin - 1/ln2) 2^-L
    Ball invlmin, invln2;
    arb_inv(invlmin.v, lmin.v, prec + 16);
    arb_inv(invln2.v, ln2.v, prec + 16);
    arb_sub(t.v, invlmin.v, invln2.v, prec + 16);
    arb_mul(t.v, t.v, Bm1.v, prec + 16);
    arb_mul_2exp_si(t2.v, t.v, -L);

    // beta = 1 - 1/(B-1);  qpow = 2^(-(L-1)beta);  twobeta = 2^beta
    arb_inv(invBm1.v, Bm1.v, prec + 16);
    arb_sub(beta.v, one.v, invBm1.v, prec + 16);
    arb_set_si(t.v, -(L - 1));
    arb_mul(t.v, t.v, beta.v, prec + 16);
    arb_mul(t.v, t.v, ln2.v, prec + 16);
    arb_exp(qpow.v, t.v, prec + 16);
    arb_mul(t.v, beta.v, ln2.v, prec + 16);
    arb_exp(twobeta.v, t.v, prec + 16);
    arb_sub(denbeta.v, twobeta.v, one.v, prec + 16);

    // U = L/(B-1) - 1/ln2;  inner = 2^(1/(U ln2)) / U = e^(1/U) / U
    arb_set_si(U.v, L);
    arb_div(U.v, U.v, Bm1.v, prec + 16);
    arb_sub(U.v, U.v, invln2.v, prec + 16);
    if (!arb_is_positive(U.v))
    {
        arb_zero(res);
        return;
    }
    arb_inv(invU.v, U.v, prec + 16);
    arb_exp(inner.v, invU.v, prec + 16);
    arb_div(inner.v, inner.v, U.v, prec + 16);

    // 2^(-1-2/ln2) = e^(-ln2-2)
    // arb_set_si(cexp.v, -1);
    // arb_sub(cexp.v, cexp.v, invln2.v, prec + 16);
    // arb_sub(cexp.v, cexp.v, invln2.v, prec + 16);
    // arb_mul(cexp.v, cexp.v, ln2.v, prec + 16);
    // arb_exp(cexp.v, cexp.v, prec + 16);
    arb_set_si(cexp.v, -2);
    arb_sub(cexp.v, cexp.v, ln2.v, prec + 16);
    arb_exp(cexp.v, cexp.v, prec + 16);

    // T3 = (B-1)/lmin * 2^(-1-2/ln2) * 2^(1/(U ln2)) / U
    //      * 2^(-(L-1)beta) / (2^beta - 1)
    arb_mul(coef.v, Bm1.v, invlmin.v, prec + 16);
    arb_mul(coef.v, coef.v, cexp.v, prec + 16);
    arb_mul(coef.v, coef.v, inner.v, prec + 16);
    arb_mul(coef.v, coef.v, qpow.v, prec + 16);
    arb_div(t3.v, coef.v, denbeta.v, prec + 16);

    // T4 = B/(e ln2) * 2^((L+1)(1/B-1)) / (1 - 2^(1/B-1))
    Ball invB;
    arb_inv(invB.v, B, prec + 16);
    arb_sub(alpha.v, invB.v, one.v, prec + 16);
    arb_set_si(expo.v, L + 1);
    arb_mul(expo.v, expo.v, alpha.v, prec + 16);
    arb_mul(expo.v, expo.v, ln2.v, prec + 16);
    arb_exp(pw.v, expo.v, prec + 16);
    arb_mul(t.v, alpha.v, ln2.v, prec + 16);
    arb_exp(t.v, t.v, prec + 16);
    arb_sub(den.v, one.v, t.v, prec + 16);
    if (!arb_is_positive(den.v))
    {
        arb_zero(res);
        return;
    }
    arb_mul(ut.v, B, pw.v, prec + 16);
    arb_div(ut.v, ut.v, den.v, prec + 16);
    arb_div(ut.v, ut.v, e.v, prec + 16);
    arb_div(ut.v, ut.v, ln2.v, prec + 16);

    arb_add(res, t1.v, t2.v, prec + 16);
    arb_add(res, res, t3.v, prec + 16);
    arb_add(res, res, ut.v, prec);
}

void sigma_trunc_range(arb_t res, const arb_t B, slong l1, slong l2, slong prec, int threads)
{
    // Rigorous enclosure of sum_{l=l1}^{l2} ceil(d_l) 2^-l: every term uses
    // an interval ceiling, so the result encloses the sum of the ceilings.
    arb_zero(res);
    if (l2 < l1)
        return;
    if (threads <= 0)
        threads = omp_get_max_threads();

    #pragma omp parallel num_threads(threads)
    {
        Ball local, term, d;
        arb_zero(local.v);
        #pragma omp for schedule(static)
        for (slong l = l1; l <= l2; ++l)
        {
            d_ceil_func(d.v, l, B, prec);
            arb_mul_2exp_si(term.v, d.v, -l);
            arb_add(local.v, local.v, term.v, prec);
        }
        #pragma omp critical
        arb_add(res, res, local.v, prec);
    }
}

void sigma_trunc(arb_t res, const arb_t B, slong L, slong prec, int threads)
{
    sigma_trunc_range(res, B, 3, L, prec, threads);
}


} // namespace bstar
