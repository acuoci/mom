/*-----------------------------------------------------------------------*\
|   MOM Library — CFD loop patterns: inlining, dispatch, zero fallbacks   |
|                                                                          |
|   Demonstrates and validates two hot-loop patterns:                      |
|                                                                          |
|   Pattern A — Template cell loop (compile-time dispatch, zero overhead)  |
|     The variant type is known statically; the compiler generates one     |
|     specialised loop body with all if-constexpr branches resolved and    |
|     the full call chain (sources_X → _impl) inlined.                    |
|                                                                          |
|   Pattern B — ForEachCell (runtime dispatch, single jump per sweep)      |
|     std::visit is called ONCE before the loop; the callback receives     |
|     the concrete model type and owns the iteration. The compiler still   |
|     generates fully-optimised, type-specific loop bodies — one per       |
|     variant alternative in AnyMomentMethod<Thermo>.                     |
|                                                                          |
|   Also validates the zero-fallback mechanism: calling sources_X() for   |
|   a process that the variant does NOT model must return an all-zero      |
|   span of the correct length (e.g. BrookesMoss::sources_condensation()). |
|                                                                          |
|   ── Compile ────────────────────────────────────────────────────────── |
|                                                                          |
|   Release / benchmark build:                                             |
|     g++ -std=c++20 -O3 -march=native                                    |
|         -I ../include -I ../include/MOM -I ../include/ThreeEquations     |
|         -I ../include/BrookesMoss                                        |
|         -I /path/to/eigen3                                               |
|         cfd_loop_benchmark.cpp -o cfd_loop_benchmark                    |
|                                                                          |
|   Assembly inspection (check inlining and zero-fallback elimination):   |
|     g++ -std=c++20 -O3 -march=native -S -fverbose-asm                   |
|         <same -I flags>                                                  |
|         cfd_loop_benchmark.cpp -o cfd_loop_benchmark.s                  |
|     grep -c "call" cfd_loop_benchmark.s   # should be ~0 in hot loops   |
|     grep "ymm\|xmm" cfd_loop_benchmark.s | wc -l  # SIMD register use  |
|                                                                          |
|   ── What to look for in the assembly output ──────────────────────── ─ |
|                                                                          |
|   For the template loop (Pattern A), look for the mangled name          |
|   containing "TemplatedCellLoop" and "ThreeEquations".  Inside the      |
|   loop body you should find:                                             |
|     • No "call" instructions for sources_nucleation / _impl etc.        |
|       → fully inlined two-level dispatch chain                           |
|     • "vmovsd" / "vmovupd" / "vmulsd" instructions using xmm/ymm regs  |
|       → SIMD-vectorised Eigen operations on the NEq-element vectors     |
|     • For sources_sintering() on ThreeEquations (not modelled):         |
|       → the zero-fallback span {kZeroData, 3} is constant-folded.       |
|       → any loop over it ("+= src_sin[i]") is eliminated entirely.     |
|         Dead-store-elimination removes the add-zero operations.         |
|                                                                          |
\*-----------------------------------------------------------------------*/

#include "MOM/AnyMomentMethod.hpp"
#include "MOM/MomentMethodConcept.hpp"
#include "ThreeEquations/ThreeEquations.hpp"
#include "BrookesMoss/BrookesMoss.hpp"

#include <chrono>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <cassert>
#include <string>
#include <vector>

// ============================================================================
// Thermo setup — same species set as test_ThreeEquations.cpp
// Species order: H OH O2 H2 H2O C2H2 N2  (ns = 7)
// ============================================================================

static MOM::BasicThermoData buildThermo()
{
    MOM::BasicThermoData th;
    th.names = {"H", "OH", "O2", "H2", "H2O", "C2H2", "N2"};
    th.mw    = {1.008, 17.008, 31.999, 2.016, 18.015, 26.038, 28.014};
    th.nc    = {0, 0, 0, 0, 0, 2, 0};
    th.nh    = {1, 1, 0, 2, 2, 2, 0};
    th.no    = {0, 1, 2, 0, 1, 0, 0};
    th.nn    = {0, 0, 0, 0, 0, 0, 2};
    th.nti   = {0, 0, 0, 0, 0, 0, 0};
    return th;
}

// Helper: mole fractions → mass fractions
static std::vector<double> X2Y(const std::vector<double>& X, const MOM::BasicThermoData& th)
{
    const int ns = static_cast<int>(th.names.size());
    double MW    = 0.;
    for (int k = 0; k < ns; ++k)
        MW += X[k] * th.mw[k];
    std::vector<double> Y(ns);
    for (int k = 0; k < ns; ++k)
        Y[k] = X[k] * th.mw[k] / MW;
    return Y;
}

// ============================================================================
// Pattern A: Template cell loop
// ============================================================================
// The variant type M is known at compile time.  Every internal call —
// ComputeCell → SetState/SetMoments/ComputeSources/sources() —
// is monomorphic.  With -O3 -march=native the compiler:
//   1. Inlines the full ComputeCell chain into the loop body.
//   2. Resolves all if-constexpr branches (no code for missing processes).
//   3. Vectorises the fixed-size Eigen source-vector copies.
// ============================================================================

template <MOM::MomentMethod M>
[[gnu::noinline]] // keep as a distinct symbol so the asm grep is easy
double TemplatedCellLoop(M& model,
                         int n_cells,
                         const double* T_arr,
                         const double* P_arr,
                         const double* Y_flat, // [n_cells × ns]
                         const double* mu_arr,
                         const double* M_flat, // [n_cells × neq]
                         double* src_flat)     // [n_cells × neq] output
{
    constexpr unsigned neq = M::n_equations;
    const unsigned ns      = 7u;

    for (int i = 0; i < n_cells; ++i)
    {
        MOM::ComputeCell(model, T_arr[i], P_arr[i], Y_flat + i * ns, mu_arr[i], {M_flat + i * neq, neq});

        auto src = model.sources();
        std::copy(src.begin(), src.end(), src_flat + i * neq);
    }

    // Return a checksum so the compiler cannot dead-code-eliminate the loop.
    return std::accumulate(src_flat, src_flat + n_cells * neq, 0.0);
}

// ============================================================================
// Pattern B: ForEachCell — single variant dispatch for the whole sweep
// ============================================================================
// std::visit is called ONCE.  Inside the lambda, `m` is the CONCRETE type
// (e.g. ThreeEquations<BasicThermoData>&), so the loop body is identical
// to the template loop above in terms of generated machine code.
//
// The compiler generates one specialised loop body per variant alternative
// (four bodies for four variants).  The correct one is selected before the
// loop via the jump table, which the branch predictor makes free after the
// first hit.
// ============================================================================

[[gnu::noinline]]
double ForEachCellLoop(MOM::AnyMomentMethod<MOM::BasicThermoData>& model,
                       int n_cells,
                       const double* T_arr,
                       const double* P_arr,
                       const double* Y_flat,
                       const double* mu_arr,
                       const double* M_flat,
                       double* src_flat)
{
    const unsigned ns = 7u;

    MOM::ForEachCell(model,
                     [&](auto& m)
                     {
                         // Inside this lambda, typeof(m) is concrete — not type-erased.
                         constexpr unsigned neq = std::decay_t<decltype(m)>::n_equations;

                         for (int i = 0; i < n_cells; ++i)
                         {
                             MOM::ComputeCell(m,
                                              T_arr[i],
                                              P_arr[i],
                                              Y_flat + i * ns,
                                              mu_arr[i],
                                              {M_flat + i * neq, neq});

                             auto src = m.sources();
                             std::copy(src.begin(), src.end(), src_flat + i * neq);
                         }
                     });

    constexpr unsigned neq = 3u; // ThreeEquations has 3 equations
    return std::accumulate(src_flat, src_flat + n_cells * neq, 0.0);
}

// ============================================================================
// Zero-fallback validation
// ============================================================================
// Confirms that calling sources_X() for a process the variant does NOT model
// returns an all-zero span of the correct size — at zero runtime cost.
//
// BrookesMoss (NEq=2) does NOT model condensation or sintering.
// ThreeEquations (NEq=3) does NOT model sintering.
// ============================================================================

static bool validateZeroFallbacks(MOM::BasicThermoData& th)
{
    std::cout << "\n=== Zero-fallback validation ===\n";
    bool ok = true;

    // ── BrookesMoss: condensation + sintering must be all-zero spans of size 2 ──
    {
        MOM::BrookesMoss<MOM::BasicThermoData> bm(th);

        // Activate with minimal state so sources are computed (even if zero soot)
        const std::vector<double> Y = X2Y({0, 0, 0.23, 0, 0.1, 1e-6, 0.67}, th);
        bm.SetState(1800., 101325., Y.data());
        bm.SetMoments(1e-10, 1e-12);
        bm.ComputeSources();

        auto cond = bm.sources_condensation(); // not modelled → kZeroData
        auto sint = bm.sources_sintering();    // not modelled → kZeroData

        bool cond_ok = (cond.size() == 2) &&
                       std::all_of(cond.begin(), cond.end(), [](double v) { return v == 0.0; });
        bool sint_ok = (sint.size() == 2) &&
                       std::all_of(sint.begin(), sint.end(), [](double v) { return v == 0.0; });

        std::cout << std::boolalpha;
        std::cout << "  BrookesMoss::sources_condensation() → size=" << cond.size()
                  << "  all_zero=" << cond_ok << "\n";
        std::cout << "  BrookesMoss::sources_sintering()    → size=" << sint.size()
                  << "  all_zero=" << sint_ok << "\n";

        // Pointer check: the span should point into the shared kZeroData buffer,
        // NOT into a variant-owned MomentVector (there is none for these processes).
        // We cannot directly test the pointer identity here, but the values confirm
        // the correct dispatch path was taken.
        ok = ok && cond_ok && sint_ok;
    }

    // ── ThreeEquations: sintering must be all-zero span of size 3 ──────────────
    {
        MOM::ThreeEquations<MOM::BasicThermoData> te(th);

        const std::vector<double> Y = X2Y({0, 0, 0.23, 0, 0.1, 1e-5, 0.67}, th);
        te.SetPAH("C2H2");
        te.SetNucleation(1);
        te.SetCoagulation(1);
        te.SetState(1800., 101325., Y.data());
        te.SetMoments(1e-10, 1e-15, 1e-10);
        te.ComputeSources();

        auto sint = te.sources_sintering(); // not modelled → kZeroData

        bool sint_ok = (sint.size() == 3) &&
                       std::all_of(sint.begin(), sint.end(), [](double v) { return v == 0.0; });

        std::cout << "  ThreeEquations::sources_sintering() → size=" << sint.size()
                  << "  all_zero=" << sint_ok << "\n";
        ok = ok && sint_ok;
    }

    std::cout << "\n  Result: " << (ok ? "PASS ✓" : "FAIL ✗") << "\n";
    return ok;
}

// ============================================================================
// Timing harness
// ============================================================================

template <typename Fn> static double timeLoop(Fn&& fn, int reps, const char* label)
{
    // Warm-up
    fn();

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < reps; ++r)
        fn();
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
    std::cout << "  " << std::left << std::setw(38) << label << std::right << std::fixed
              << std::setprecision(3) << ms << " ms / iteration\n";
    return ms;
}

// ============================================================================
// main
// ============================================================================

int main()
{
    // ── Thermo setup ─────────────────────────────────────────────────────────
    auto th = buildThermo();

    // ── Synthetic cell field: 10 000 cells at representative flame conditions ─
    constexpr int N   = 10'000;
    constexpr int ns  = 7;
    constexpr int neq = 3; // ThreeEquations has 3 equations

    std::vector<double> T_arr(N), P_arr(N), mu_arr(N);
    std::vector<double> Y_flat(N * ns);
    std::vector<double> M_flat(N * neq);
    std::vector<double> src_A(N * neq, 0.); // Pattern A output
    std::vector<double> src_B(N * neq, 0.); // Pattern B output

    // Fill with representative values (temperature ramp, fixed composition)
    const std::vector<double> Y0 = X2Y({0.001, 0.002, 0.23, 0, 0.1, 5e-5, 0.667}, th);
    for (int i = 0; i < N; ++i)
    {
        T_arr[i]  = 1200. + 0.12 * i; // 1200 – 2400 K
        P_arr[i]  = 101325.;
        mu_arr[i] = 4e-5;
        std::copy(Y0.begin(), Y0.end(), Y_flat.data() + i * ns);
        M_flat[i * neq + 0] = 1e-10; // Ys
        M_flat[i * neq + 1] = 1e-15; // NsNorm
        M_flat[i * neq + 2] = 1e-10; // Ss
    }

    // ── Construct ThreeEquations model ───────────────────────────────────────
    MOM::ThreeEquations<MOM::BasicThermoData> te_direct(th);
    te_direct.SetPAH("C2H2");
    te_direct.SetNucleation(1);
    te_direct.SetCoagulation(1);

    // Wrap in AnyMomentMethod for Pattern B
    auto te_any = MOM::MakeAnyMomentMethod(th, "ThreeEquations");
    std::visit(
        [](auto& m)
        {
            if constexpr (requires { m.SetPAH(std::string_view{}); })
            {
                m.SetPAH("C2H2");
                m.SetNucleation(1);
                m.SetCoagulation(1);
            }
        },
        te_any);

    // ── Zero-fallback validation ──────────────────────────────────────────────
    bool zf_ok = validateZeroFallbacks(th);

    // ── Numerical equivalence check ──────────────────────────────────────────
    // Pattern A and B must produce bit-identical results for the same variant.
    std::cout << "\n=== Numerical equivalence check ===\n";

    double sum_A = TemplatedCellLoop(te_direct,
                                     N,
                                     T_arr.data(),
                                     P_arr.data(),
                                     Y_flat.data(),
                                     mu_arr.data(),
                                     M_flat.data(),
                                     src_A.data());

    double sum_B = ForEachCellLoop(te_any,
                                   N,
                                   T_arr.data(),
                                   P_arr.data(),
                                   Y_flat.data(),
                                   mu_arr.data(),
                                   M_flat.data(),
                                   src_B.data());

    bool equiv = (src_A == src_B);
    std::cout << "  Pattern A checksum: " << std::scientific << std::setprecision(6) << sum_A << "\n";
    std::cout << "  Pattern B checksum: " << std::scientific << std::setprecision(6) << sum_B << "\n";
    std::cout << "  Bit-identical:      " << std::boolalpha << equiv << "\n";

    // ── Timing ───────────────────────────────────────────────────────────────
    constexpr int REPS = 20;
    std::cout << "\n=== Timing (N=" << N << " cells, " << REPS << " repetitions) ===\n";

    double ms_A = timeLoop(
        [&]
        {
            TemplatedCellLoop(te_direct,
                              N,
                              T_arr.data(),
                              P_arr.data(),
                              Y_flat.data(),
                              mu_arr.data(),
                              M_flat.data(),
                              src_A.data());
        },
        REPS,
        "Pattern A: template loop");

    double ms_B = timeLoop(
        [&]
        {
            ForEachCellLoop(te_any,
                            N,
                            T_arr.data(),
                            P_arr.data(),
                            Y_flat.data(),
                            mu_arr.data(),
                            M_flat.data(),
                            src_B.data());
        },
        REPS,
        "Pattern B: ForEachCell loop");

    std::cout << "\n  Overhead of variant dispatch: " << std::fixed << std::setprecision(3)
              << (ms_B - ms_A) << " ms  (" << std::setprecision(1) << (ms_B / ms_A - 1.) * 100.
              << "%)\n";

    // ── Summary ──────────────────────────────────────────────────────────────
    std::cout << "\n=== Summary ===\n";
    std::cout << "  Zero-fallback validation: " << (zf_ok ? "PASS ✓" : "FAIL ✗") << "\n";
    std::cout << "  Numerical equivalence:    " << (equiv ? "PASS ✓" : "FAIL ✗") << "\n";

    return (zf_ok && equiv) ? 0 : 1;
}

/*
 * ── Compiler flags guide ────────────────────────────────────────────────────
 *
 * Minimum for correctness:      -std=c++20
 * Minimum for production CFD:   -std=c++20 -O2
 * Recommended production:       -std=c++20 -O3 -march=native
 *
 * What each flag enables for this library:
 *
 *   -O2
 *     Inlines the complete dispatch chain sources_X() → _impl() into the
 *     call site.  The if-constexpr branch is already compile-time dead code;
 *     -O2 just ensures the remaining one-liner is not left as a function call.
 *     [[gnu::always_inline]] on the _impl() methods makes this hard guarantee
 *     explicit and enforces it even at -O0 (debug / profiling builds).
 *
 *   -O3
 *     Adds auto-vectorisation.  For HMOM (NEq=4), the 4-double source vector
 *     fits in a single 256-bit AVX register (vmovupd ymm0,...).  The compiler
 *     can process all 4 equations in one instruction per arithmetic op.
 *     For ThreeEquations / MetalOxide (NEq=3), you get 192-bit partial packing;
 *     padding to 4 would allow full 256-bit use.
 *
 *   -march=native
 *     Enables AVX2 / FMA on modern Intel/AMD CPUs.  Most relevant for:
 *       - Eigen's fixed-size vector ops (MomentVector arithmetic)
 *       - The std::copy loops over source spans
 *     Without this, the compiler defaults to SSE2 (128-bit, 2 doubles),
 *     halving the throughput of 4-double vector operations.
 *
 *   -fno-exceptions  (optional, for embedded CFD solvers)
 *     Removes exception tables from every function.  All MOM methods are
 *     noexcept; this flag makes the linker strip the dead exception machinery.
 *
 *   -flto  (link-time optimisation, with the CMake build)
 *     Allows the compiler to inline across translation units.  Most relevant
 *     for the library's pre-compiled mode (MOM_COMPILED_LIBRARY).  In the
 *     default header-only mode, everything is already visible at compile time.
 *
 * ── What zero-fallback actually means in the assembly ────────────────────
 *
 * When ThreeEquations::sources_sintering() is called, the if-constexpr
 * picks the "else" arm at compile time (ThreeEquations has no
 * sources_sintering_impl()).  The resulting code is:
 *
 *   return std::span<const double>{ MomentMethodBase<ThreeEq,3>::kZeroData, 3 };
 *
 * kZeroData is a static constexpr double[3] in read-only data (.rodata).
 * The compiler represents the return value as a (pointer, length) pair:
 *   lea   rax, [rip + kZeroData]   ; pointer to zeros in .rodata
 *   mov   edx, 3                   ; size = 3
 *
 * If the caller then iterates over the span (e.g. to add sintering sources
 * to a running total), the compiler's constant-propagation replaces each
 * element access with 0.0, and the dead-store-elimination pass removes any
 * "+= 0.0" entirely.  The sintering loop body is reduced to nothing.
 */
