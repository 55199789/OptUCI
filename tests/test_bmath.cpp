// Rigorous consistency tests for the bstar math library.
//
// Strategy: every computed ball is compared (overlap) against independently
// generated mpmath reference decimals, and the defining identities are
// checked as rigorous ball containments.
#include "bmath.hpp"

#include <flint/arb.h>
#include <flint/arf.h>

#include <cstdio>
#include <cstring>
#include <string>

namespace {

constexpr slong PREC = 224;

int failures = 0;
int checks = 0;

#define CHECK(cond, msg)                                                       \
    do                                                                         \
    {                                                                          \
        ++checks;                                                              \
        if (!(cond))                                                           \
        {                                                                      \
            std::printf("FAIL [%s:%d]: %s\n", __FILE__, __LINE__, msg);        \
            ++failures;                                                        \
        }                                                                      \
    } while (0)

bool overlaps_dec(const arb_t x, const char* dec)
{
    // The reference decimals carry ~45 significant digits; widen them by a
    // relative 1e-40 (larger than their truncation, smaller than any bug).
    arb_t t, rel, err;
    arb_init(t);
    arb_init(rel);
    arb_init(err);
    arb_set_str(t, dec, PREC + 128);
    arb_set_str(rel, "1e-40", PREC + 128);
    arb_mul(err, t, rel, PREC + 128);
    arb_add_error(t, err);
    int ok = arb_overlaps(x, t);
    arb_clear(t);
    arb_clear(rel);
    arb_clear(err);
    return ok;
}

bool contains_int(const arb_t x, long v)
{
    return arb_contains_si(x, v);
}

bool contains_zero(const arb_t x) { return arb_contains_zero(x); }

void set_dec(arb_t x, const char* dec)
{
    arb_set_str(x, dec, PREC + 128);
}

void test_entropy()
{
    arb_t e, h;
    arb_init(e);
    arb_init(h);

    set_dec(e, "0.5");
    bstar::entropy_h(h, e, PREC);
    CHECK(contains_int(h, 1), "h(1/2) must be 1");

    set_dec(e, "0.1");
    bstar::entropy_h(h, e, PREC);
    CHECK(overlaps_dec(h, "0.468995593589281221253589330383320460097165459"),
          "h(0.1) reference");
    CHECK(arb_is_positive(h), "h(0.1) positive");

    set_dec(e, "0.25");
    bstar::entropy_h(h, e, PREC);
    CHECK(overlaps_dec(h, "0.811278124459132863909695792039137618430139194"),
          "h(0.25) reference");

    set_dec(e, "0.333333333333333333333333333333333333333333333333333333333333");
    bstar::entropy_h(h, e, PREC);
    CHECK(overlaps_dec(h, "0.918295834054489514787072277281149842093147741"),
          "h(1/3) reference");

    set_dec(e, "0.4");
    bstar::entropy_h(h, e, PREC);
    CHECK(overlaps_dec(h, "0.970950594454668638998076063120700270608942748"),
          "h(0.4) reference");

    arb_t a, b;
    arb_init(a);
    arb_init(b);
    set_dec(a, "0.3");
    bstar::entropy_h(b, a, PREC);
    arb_sub_ui(a, a, 1, PREC);
    arb_neg(a, a);
    bstar::entropy_h(a, a, PREC);
    CHECK(arb_overlaps(a, b), "h symmetry");

    arb_clear(a);
    arb_clear(b);
    arb_clear(e);
    arb_clear(h);
}

void test_epsilon_sigma()
{
    arb_t x, s, e, h, l2, r, one;
    arb_init(x);
    arb_init(s);
    arb_init(e);
    arb_init(h);
    arb_init(l2);
    arb_init(r);
    arb_init(one);
    arb_set_ui(one, 1);

    set_dec(x, "2");
    bstar::epsilon_of_x(e, x, PREC);
    CHECK(overlaps_dec(e, "0.5"), "eps(2) = 1/2");
    bstar::sigma_of_x(s, x, PREC);
    CHECK(contains_int(s, 2), "sigma(2) = 2");

    struct
    {
        const char* x;
        const char* eps;
        const char* sig;
    } cases[] = {
        {"2.5", "0.27029496860307556683253391025729368824663368",
         "3.6996619107197892640899168533419874471890217"},
        {"3", "0.22709219521934818721052345430504338837644753",
         "4.40349787906229328067528311264272352990241544"},
        {"10", "0.13518246689062743066988548314589801048899946",
         "7.39740902057271695571145339102710074404733939"},
        {"100", "0.0866877250312768596376431584735124059663692414",
         "11.5356585911004219503785532347990882492384822"},
        {"1e6", "0.0383886195270378112738950539780244555393852328",
         "26.0493868318364924542315964688249438878640537"},
    };

    for (auto& c : cases)
    {
        set_dec(x, c.x);
        bstar::epsilon_of_x(e, x, PREC);
        bstar::sigma_of_x(s, x, PREC);
        CHECK(overlaps_dec(e, c.eps), "eps reference");
        CHECK(overlaps_dec(s, c.sig), "sigma reference");

        bstar::entropy_h(h, e, PREC);
        arb_sub_ui(l2, x, 1, PREC);
        arb_log(l2, l2, PREC);
        arb_t ln2;
        arb_init(ln2);
        arb_const_log2(ln2, PREC);
        arb_div(l2, l2, ln2, PREC);
        arb_mul(r, e, l2, PREC);
        arb_add(r, r, h, PREC);
        arb_sub(r, r, one, PREC);
        CHECK(contains_zero(r), "epsilon identity");
        arb_clear(ln2);
    }
    arb_clear(x);
    arb_clear(s);
    arb_clear(e);
    arb_clear(h);
    arb_clear(l2);
    arb_clear(r);
    arb_clear(one);
}

void test_u_s_x1_ystar_x2()
{
    arb_t B, x, r;
    arb_init(B);
    arb_init(x);
    arb_init(r);

    set_dec(B, "2.006");
    bstar::x1_func(r, 5, B, PREC);
    CHECK(overlaps_dec(r, "2.07028213835098581998123162345529206811995381"), "x1(5)");
    bstar::x1_func(r, 8, B, PREC);
    CHECK(overlaps_dec(r, "5.83746037367000913829103167416827014994024429"), "x1(8)");
    set_dec(B, "2.01");
    bstar::x1_func(r, 12, B, PREC);
    CHECK(overlaps_dec(r, "23.0621356371768889728357894149034249074539963"), "x1(12)");
    set_dec(B, "2.001");
    bstar::x1_func(r, 100, B, PREC);
    CHECK(overlaps_dec(r, "407083324326854.026976046703111011793746712547"), "x1(100)");

    set_dec(B, "2.006");
    arb_set_ui(x, 5);
    bstar::u_func(r, 8, B, x, PREC);
    CHECK(overlaps_dec(r, "16.7110612082797556508606961222214165360757411"), "u(8,5)");
    arb_set_ui(x, 6);
    bstar::u_func(r, 8, B, x, PREC);
    CHECK(overlaps_dec(r, "16.8873913413201642000227980706440805005668738"), "u(8,6)");
    set_dec(B, "2.001");
    arb_set_str(x, "407083324326854", PREC);
    bstar::u_func(r, 100, B, x, PREC);
    CHECK(overlaps_dec(r, "1175181483563126.59101334387514732266172354307"), "u(100,fl)");

    set_dec(B, "2.006");
    arb_set_ui(x, 1);
    bstar::u_func(r, 8, B, x, PREC);
    CHECK(contains_int(r, 8), "u_l(1) = l");

    arb_set_ui(x, 2);
    bstar::s_func(r, 8, B, x, PREC);
    {
        arb_t expect, t2;
        arb_init(expect);
        arb_init(t2);
        arb_set_si(expect, 16);
        arb_mul_ui(t2, B, 2, PREC);
        arb_sub(expect, expect, t2, PREC);
        CHECK(arb_overlaps(r, expect), "s_l(2) = 2l-2B");
        arb_clear(expect);
        arb_clear(t2);
    }

    arb_set_ui(x, 3);
    bstar::s_func(r, 8, B, x, PREC);
    CHECK(overlaps_dec(r, "12.1401622673266659192813303773628402578363401"), "s(8,3)");
    arb_set_ui(x, 4);
    bstar::s_func(r, 8, B, x, PREC);
    CHECK(overlaps_dec(r, "12.0561784548332765212792456282111209941308883"), "s(8,4)");

    set_dec(B, "2.006");
    bstar::ystar_func(r, 8, B, PREC);
    CHECK(overlaps_dec(r, "4.77578330288932205405374982074114456461454222"), "ystar(8)");
    set_dec(B, "2.005");
    bstar::ystar_func(r, 10, B, PREC);
    CHECK(overlaps_dec(r, "7.10863953619149433250511430916688079198268862"), "ystar(10)");
    set_dec(B, "2.0001");
    bstar::ystar_func(r, 100, B, PREC);
    CHECK(overlaps_dec(r, "97.52563022429853298707066660970799165372153"), "ystar(100)");

    set_dec(B, "2.006");
    bstar::x2_func(r, 8, B, PREC);
    CHECK(overlaps_dec(r, "3.36231458883299811953611681637930886729271143"), "x2(8)");
    {
        arb_t y, sg;
        arb_init(y);
        arb_init(sg);
        bstar::ystar_func(y, 8, B, PREC);
        bstar::sigma_of_x(sg, r, PREC);
        CHECK(arb_overlaps(sg, y), "sigma(x2) = y*");
        arb_clear(y);
        arb_clear(sg);
    }

    arb_clear(B);
    arb_clear(x);
    arb_clear(r);
}

void test_d()
{
    arb_t B, d, expect, t;
    arb_init(B);
    arb_init(d);
    arb_init(expect);
    arb_init(t);
    set_dec(B, "2.006");

    const char* refs[] = {
        "-0.012",
        "1.988",
        "3.988",
        "5.988",
        "8.46169567066008210001139903532204025028343689",
        "11.952",
        "16.8873913413201642000227980706440805005668738",
        "23.856",
        "33.7027826826403284000455961412881610011337476",
        "47.616",
        "67.2918984714415400980925883108227848525132064",
        "95.0711980867090993342399544213884004728812853",
        "153.28009292030741576293147707088877500919855",
        "269.774280599900331366203951347962087949120341",
    };
    for (long l = 2; l <= 15; ++l)
    {
        bstar::d_func(d, l, B, PREC);
        CHECK(overlaps_dec(d, refs[l - 2]), "d_l reference");
        arb_set_si(expect, 2 * l);
        arb_mul_ui(t, B, 2, PREC);
        arb_sub(expect, expect, t, PREC);
        arb_t diff, eps;
        arb_init(diff);
        arb_init(eps);
        arb_sub(diff, d, expect, PREC);
        arb_set_ui(eps, 1);
        arb_mul_2exp_si(eps, eps, -(PREC - 48));
        arb_neg(eps, eps);
        CHECK(arb_gt(diff, eps), "d_l >= base");
        arb_clear(diff);
        arb_clear(eps);
    }

    const long ceil_refs[] = {0, 2, 4, 6, 9, 12, 17, 24, 34, 48, 68, 96, 154, 270};
    for (long l = 2; l <= 15; ++l)
    {
        bstar::d_ceil_func(d, l, B, PREC);
        CHECK(contains_int(d, ceil_refs[l - 2]), "ceil(d_l) reference");
        CHECK(arb_contains_int(d), "ceil(d_l) integer valued");
    }

    arb_clear(B);
    arb_clear(d);
    arb_clear(expect);
    arb_clear(t);
}

void test_sigma_sum_and_tail()
{
    arb_t B, S, lo, hi, ref;
    arb_init(B);
    arb_init(S);
    arb_init(lo);
    arb_init(hi);
    arb_init(ref);

    struct
    {
        const char* B;
        long L;
        const char* val;
    } cases[] = {
        {"2.006", 1000, "1.37669370819740967181511568369489303080009279"},
        {"2.006", 11786, "1.37701257010606162688167209448372835358218983"},
        {"2.01", 1000, "1.3302718914150721820241875971166064717345601"},
        {"2.0001", 20000, "1.75996729659939763219537266942748600982714985"},
    };
    for (auto& c : cases)
    {
        set_dec(B, c.B);
        bstar::sigma_trunc(S, B, c.L, PREC, 0);
        CHECK(overlaps_dec(S, c.val), "Sigma truncated reference");
    }

    // the closed-form tail bound must dominate the true tail
    arb_t p1, p2, tail, act;
    arb_init(p1);
    arb_init(p2);
    arb_init(tail);
    arb_init(act);
    set_dec(B, "2.0001");
    bstar::sigma_trunc(p1, B, 1000, PREC, 0);
    bstar::sigma_trunc(p2, B, 200000, PREC, 0);
    bstar::tail_upper(tail, B, 1000, PREC);
    arf_t a, b;
    arf_init(a);
    arf_init(b);
    arb_get_lbound_arf(a, p2, PREC);
    arb_get_ubound_arf(b, p1, PREC);
    arb_set_arf(act, a);
    arb_t bub;
    arb_init(bub);
    arb_set_arf(bub, b);
    arb_sub(act, act, bub, PREC); // lower bound for true tail
    CHECK(arb_le(act, tail), "tail bound dominates true tail");

    set_dec(B, "2.006");
    bstar::tail_upper(tail, B, 11786, PREC);
    arb_t lim;
    arb_init(lim);
    arb_set_ui(lim, 1);
    arb_mul_2exp_si(lim, lim, -40);
    CHECK(arb_lt(tail, lim), "tail bound tight at large L");

    arb_clear(B);
    arb_clear(S);
    arb_clear(lo);
    arb_clear(hi);
    arb_clear(ref);
    arb_clear(p1);
    arb_clear(p2);
    arb_clear(tail);
    arb_clear(act);
    arb_clear(bub);
    arb_clear(lim);
    arf_clear(a);
    arf_clear(b);
}

} // namespace

int main()
{
    test_entropy();
    test_epsilon_sigma();
    test_u_s_x1_ystar_x2();
    test_d();
    test_sigma_sum_and_tail();

    std::printf("%d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
