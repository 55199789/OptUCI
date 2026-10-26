# Computation of the optimal expansion factor $C^\star$ for universal coding of integers

> **Paper.** Wei Yan, Yunghsiang S. Han, Leqian Zheng, *Optimal Universal Coding of Integers*, arXiv:2610.01563, <https://arxiv.org/abs/2610.01563>. The detailed definitions, derivations and proofs are given there; this repository contains the numerical computation of $C^\star$.

C++26 / CMake / FLINT (arb) implementation of a bisection for the UCI inequality

$$
\Sigma(B)=\sum_{n\ge 3}\lceil d_n\rceil\,2^{-n},
\qquad
C^\star=B^\star=\sup\{B\in(2,2.01):\Sigma(B)>7/4\},
$$

with $d_n$ defined from the binary entropy equation $h(\varepsilon_x)+\varepsilon_x\log_2(x-1)=1$ and $\sigma_x=1/\varepsilon_x$.

The bracket endpoints are kept as **exact rational numbers** (`fmpq`), while $\Sigma$ and its tail bound are evaluated in rigorous ball arithmetic (`arb`).

## Dependencies

- GCC 15.2 (or another C++26 compiler), OpenMP
- CMake >= 3.28
- FLINT 3.4 (with GMP 6.x and MPFR 4.x)

## Build

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Release builds use `-O3 -march=native -flto=auto` and link FLINT and OpenMP.

## Usage

```sh
./build/bstar_bisect <bits> <threads> [--verbose]
```

- `<bits>`: output precision in binary digits (default 64);
- `<threads>`: OpenMP threads (0 = all cores);
- `--verbose`: print, for every outer iteration, only the updated rational bracket endpoint together with the bounds for $\Sigma(B_m)$.

Example:

```sh
./build/bstar_bisect 64 64 --verbose
```

The final output contains the rational endpoints (exact fractions and exact terminating decimals, since all denominators are of the form $2^i5^j$):

```
B* lower (rational) = ...
B* lower (decimal)  = ...
B* upper (rational) = ...
B* upper (decimal)  = ...
```

## Method

1. **Bracket.** `B_l = 2`, `B_r = 10193/5000` (i.e. 2.0386); both are exact rationals. The loop stops when `B_r - B_l < 2^{-bits}`.
2. **Midpoint.** `B_m = (B_l + B_r)/2`, again an exact rational.
3. **Evaluation of $\Sigma(B_m)$.** Truncate at `L`, starting at 16 and doubling `L`. The partial sum $\sum_{n=3}^{L}\lceil d_n\rceil2^{-n}$ is maintained incrementally (each doubling only adds the new terms). Each $\lceil d_n\rceil$ is kept as a rigorous integer interval (the monotone image of the `d_n` ball), and the infinite tail is bounded in closed form by `tail_upper`.
4. **Decision.** If the partial sum is definitely $>7/4$, set `B_l <- B_m`; if partial sum plus the tail bound is definitely $<7/4$, set `B_r <- B_m`. Both tests use unconditional ball comparisons.
5. On return, $\Sigma(B_l)>7/4$ and $\Sigma(B_r)<7/4$, so $B^\star\in[B_l,B_r]$ with width $<2^{-\text{bits}}$.

## Files

```
CMakeLists.txt
README.md
run_bisect_sweep.sh          # sweep over precisions
src/bmath.hpp                # math library interface
src/bmath.cpp                # entropy, eps/sigma, u/s, x1/y*/x2, d_l,
                             # truncated sums, closed-form tail bound
src/main.cpp                 # rational-bracket bisection (fmpq)
tests/test_bmath.cpp         # rigorous tests (mpmath references + identities)
```

## Sweeps

```sh
./run_bisect_sweep.sh 16 96 16      # bits 16,32,...,96 (threads = 64)
```

Logs are written to `rat_<bits>.log`.
