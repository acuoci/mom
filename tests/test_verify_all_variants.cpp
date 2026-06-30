/*-----------------------------------------------------------------------*\
|   MOM Library — Comprehensive variant verification                       |
|                                                                          |
|   Verifies all 4 variants (HMOM, ThreeEquations, BrookesMoss, TiO2)    |
|   for structural and mathematical correctness after the CRTP refactoring.|
|                                                                          |
|   Checks performed:                                                      |
|     1. All source spans have correct size (== n_equations)               |
|     2. sources() == sum of all per-process spans (mass balance)         |
|     3. Unmodelled processes return kZeroData spans (exact 0.0)          |
|     4. Modelled processes in active state produce ≥1 non-zero element   |
|     5. Process-to-variant ownership matrix matches the design spec       |
|                                                                          |
|   Compile:                                                               |
|     g++ -std=c++20 -O2                                                  |
|         -I ../include -I ../include/MOM                                  |
|         -I ../include/HMOM -I ../include/BrookesMoss                    |
|         -I ../include/ThreeEquations -I ../include/TiO2                  |
|         -I /path/to/eigen3                                               |
|         verify_all_variants.cpp -o verify_all_variants                  |
\*-----------------------------------------------------------------------*/

// Single master header: all variants + MomentMethod concept + ThermoProxy
#include "MOM/MOM.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================================
// Thermo builders
// ============================================================================
// Two separate thermos:
//   thermo_soot — for HMOM, ThreeEquations, BrookesMoss (soot chemistry)
//   thermo_tio2 — for TiO2 (titanium oxide chemistry)
// ============================================================================

static MOM::BasicThermoData buildSootThermo()
{
    // Species: H  OH  O2  H2  H2O  C2H2  N2
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

static MOM::BasicThermoData buildTiO2Thermo()
{
    // Species: TiOH4  N2
    // Ti(OH)4 : MW = 47.867 (Ti) + 4×17.008 (OH) = 115.899 kg/kmol
    MOM::BasicThermoData th;
    th.names = {"TiOH4", "N2"};
    th.mw    = {115.899, 28.014};
    th.nc    = {0, 0};
    th.nh    = {4, 0};
    th.no    = {4, 2};
    th.nn    = {0, 2};
    th.nti   = {1, 0}; // 1 Ti atom per TiOH4 molecule
    return th;
}

// Mole fractions → mass fractions helper
static std::vector<double> X2Y(const std::vector<double>& X, const MOM::BasicThermoData& th)
{
    double MW = 0.;
    for (std::size_t k = 0; k < th.names.size(); ++k)
        MW += X[k] * th.mw[k];
    std::vector<double> Y(th.names.size());
    for (std::size_t k = 0; k < th.names.size(); ++k)
        Y[k] = X[k] * th.mw[k] / MW;
    return Y;
}

// ============================================================================
// Verification helpers
// ============================================================================

struct CheckResult
{
    bool pass;
    std::string detail;
};

// Validate span: correct size, all finite
static CheckResult checkSpan(std::span<const double> s, unsigned expected_size, const char* name)
{
    if (s.size() != expected_size)
        return {false,
                std::string(name) + ": size=" + std::to_string(s.size()) +
                    " expected=" + std::to_string(expected_size)};
    for (std::size_t i = 0; i < s.size(); ++i)
        if (!std::isfinite(s[i]))
            return {false, std::string(name) + "[" + std::to_string(i) + "] is NaN/Inf"};
    return {true, ""};
}

// Validate zero-fallback span: must be exactly 0.0 for all elements
static CheckResult checkZeroFallback(std::span<const double> s,
                                     unsigned expected_size,
                                     const char* process_name,
                                     const char* variant_name)
{
    auto r = checkSpan(s, expected_size, process_name);
    if (!r.pass)
        return r;
    if (!std::all_of(s.begin(), s.end(), [](double v) { return v == 0.0; }))
        return {false,
                std::string(variant_name) + "::" + process_name +
                    "() is a zero-fallback but returned non-zero values!"};
    return {true, ""};
}

// Validate mass balance: sources() == sum of all per-process spans
static CheckResult checkMassBalance(std::span<const double> total,
                                    const std::vector<std::span<const double>>& parts,
                                    const char* variant_name,
                                    double tol = 1e-10)
{
    const auto n = total.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        double sum = 0.;
        for (auto& p : parts)
            sum += p[i];
        double denom = std::max(std::abs(total[i]), 1e-300);
        if (std::abs(sum - total[i]) / denom > tol)
        {
            return {false,
                    std::string(variant_name) + " mass balance failed at element " +
                        std::to_string(i) + ": total=" + std::to_string(total[i]) +
                        " sum_of_parts=" + std::to_string(sum)};
        }
    }
    return {true, ""};
}

// ============================================================================
// Print helpers
// ============================================================================

static void printHeader(const char* variant_name, unsigned n_eq)
{
    std::cout << "\n";
    std::cout << "┌─────────────────────────────────────────────────";
    for (unsigned i = 0; i < n_eq; ++i)
        std::cout << "────────────────";
    std::cout << "┐\n";
    std::cout << "│  " << std::left << std::setw(40) << variant_name << "  n_equations = " << n_eq
              << std::string(8, ' ') << "│\n";
    std::cout << "├──────────────────────┬──────────────────────────";
    for (unsigned i = 0; i < n_eq; ++i)
        std::cout << "────────────────";
    std::cout << "┤\n";
    std::cout << "│  Process              │  ";
    for (unsigned i = 0; i < n_eq; ++i)
        std::cout << "  eq[" << i << "]         ";
    std::cout << " │\n";
    std::cout << "├──────────────────────┼──────────────────────────";
    for (unsigned i = 0; i < n_eq; ++i)
        std::cout << "────────────────";
    std::cout << "┤\n";
}

static void printFooter(unsigned n_eq)
{
    std::cout << "└──────────────────────┴──────────────────────────";
    for (unsigned i = 0; i < n_eq; ++i)
        std::cout << "────────────────";
    std::cout << "┘\n";
    std::cout << "  [ZF] = kZeroData zero-fallback (unmodelled process)\n";
}

// ============================================================================
// Generic variant verifier
// ============================================================================
// Works for any type satisfying MomentMethod.
// Returns false if any check fails.

template <typename Model>
static bool verifyVariant(Model& model,
                          const char* name,
                          // which processes this variant models:
                          bool has_nucleation,
                          bool has_coagulation,
                          bool has_condensation,
                          bool has_growth,
                          bool has_oxidation,
                          bool has_sintering)
{
    constexpr unsigned NEQ = Model::n_equations;
    bool ok                = true;
    std::vector<CheckResult> results;

    model.CalculateSourceMoments();

    auto src_all = model.sources();
    auto src_nuc = model.sources_nucleation();
    auto src_coa = model.sources_coagulation();
    auto src_con = model.sources_condensation();
    auto src_gro = model.sources_growth();
    auto src_oxi = model.sources_oxidation();
    auto src_sin = model.sources_sintering();

    // ── Print table ──────────────────────────────────────────────────────────
    printHeader(name, NEQ);
    auto row = [&](const char* lbl, std::span<const double> s, bool zf)
    {
        std::cout << "│  " << std::left << std::setw(14) << lbl << (zf ? " [ZF]" : "     ") << " │  ";
        std::cout << std::scientific << std::setprecision(4) << std::right;
        for (auto v : s)
            std::cout << std::setw(13) << v << "  ";
        std::cout << "│\n";
    };
    row("total", src_all, false);
    std::cout << "│                     ├──────────────────────────";
    for (unsigned i = 0; i < NEQ; ++i)
        std::cout << "────────────────";
    std::cout << "┤\n";
    row("nucleation", src_nuc, !has_nucleation);
    row("coagulation", src_coa, !has_coagulation);
    row("condensation", src_con, !has_condensation);
    row("growth", src_gro, !has_growth);
    row("oxidation", src_oxi, !has_oxidation);
    row("sintering", src_sin, !has_sintering);
    printFooter(NEQ);

    // ── Check 1: all spans correct size and finite ────────────────────────
    for (auto [s, label] : std::initializer_list<std::pair<std::span<const double>, const char*>>{
             {src_all, "sources()"},
             {src_nuc, "nucleation"},
             {src_coa, "coagulation"},
             {src_con, "condensation"},
             {src_gro, "growth"},
             {src_oxi, "oxidation"},
             {src_sin, "sintering"}})
    {
        auto r = checkSpan(s, NEQ, label);
        if (!r.pass)
        {
            std::cout << "  [FAIL] " << r.detail << "\n";
            ok = false;
        }
    }

    // ── Check 2: zero-fallback spans are exactly zero ─────────────────────
    auto zf = [&](std::span<const double> s, const char* proc, bool should_be_zero)
    {
        if (!should_be_zero)
            return;
        auto r = checkZeroFallback(s, NEQ, proc, name);
        if (!r.pass)
        {
            std::cout << "  [FAIL] " << r.detail << "\n";
            ok = false;
        }
        else
            std::cout << "  [PASS] " << name << "::" << proc
                      << "() → kZeroData fallback confirmed (all 0.0)\n";
    };
    zf(src_nuc, "sources_nucleation", !has_nucleation);
    zf(src_coa, "sources_coagulation", !has_coagulation);
    zf(src_con, "sources_condensation", !has_condensation);
    zf(src_gro, "sources_growth", !has_growth);
    zf(src_oxi, "sources_oxidation", !has_oxidation);
    zf(src_sin, "sources_sintering", !has_sintering);

    // ── Check 3: mass balance — total == sum of parts ─────────────────────
    auto r3 = checkMassBalance(src_all, {src_nuc, src_coa, src_con, src_gro, src_oxi, src_sin}, name);
    if (!r3.pass)
    {
        std::cout << "  [FAIL] " << r3.detail << "\n";
        ok = false;
    }
    else
        std::cout << "  [PASS] " << name << ": sources() == sum of all per-process spans\n";

    // ── Check 4: at least one modelled process has non-zero value ─────────
    bool any_nonzero = false;
    for (auto [s, is_modelled] :
         std::initializer_list<std::pair<std::span<const double>, bool>>{{src_nuc, has_nucleation},
                                                                         {src_coa, has_coagulation},
                                                                         {src_con, has_condensation},
                                                                         {src_gro, has_growth},
                                                                         {src_oxi, has_oxidation},
                                                                         {src_sin, has_sintering}})
    {
        if (is_modelled)
            any_nonzero |= std::any_of(s.begin(), s.end(), [](double v) { return v != 0.0; });
    }
    if (!any_nonzero)
        std::cout << "  [WARN] " << name
                  << ": all modelled process sources are zero (model inactive?)\n";
    else
        std::cout << "  [PASS] " << name << ": at least one modelled process is non-zero\n";

    std::cout << (ok ? "  ● OVERALL: PASS\n" : "  ● OVERALL: FAIL\n");
    return ok;
}

// ============================================================================
// Compile-time ownership matrix — verified at compile time via static_assert
// ============================================================================

template <typename Model> constexpr bool hasNucleation()
{
    return requires(const Model& m) { m.sources_nucleation_impl(); };
}

template <typename Model> constexpr bool hasCoagulation()
{
    return requires(const Model& m) { m.sources_coagulation_impl(); };
}

template <typename Model> constexpr bool hasCondensation()
{
    return requires(const Model& m) { m.sources_condensation_impl(); };
}

template <typename Model> constexpr bool hasGrowth()
{
    return requires(const Model& m) { m.sources_growth_impl(); };
}

template <typename Model> constexpr bool hasOxidation()
{
    return requires(const Model& m) { m.sources_oxidation_impl(); };
}

template <typename Model> constexpr bool hasSintering()
{
    return requires(const Model& m) { m.sources_sintering_impl(); };
}

static void printOwnershipMatrix()
{
    using H  = MOM::HMOM<MOM::BasicThermoData>;
    using TE = MOM::ThreeEquations<MOM::BasicThermoData>;
    using BM = MOM::BrookesMoss<MOM::BasicThermoData>;
    using T2 = MOM::TiO2<MOM::BasicThermoData>;

    // ── Static assertions — catches regressions at compile time ──────────
    // HMOM: nucleation coagulation condensation growth oxidation  NO sintering
    static_assert(hasNucleation<H>() && hasCoagulation<H>() && hasCondensation<H>() &&
                      hasGrowth<H>() && hasOxidation<H>() && !hasSintering<H>(),
                  "HMOM ownership mismatch");
    // ThreeEquations: same as HMOM
    static_assert(hasNucleation<TE>() && hasCoagulation<TE>() && hasCondensation<TE>() &&
                      hasGrowth<TE>() && hasOxidation<TE>() && !hasSintering<TE>(),
                  "ThreeEquations ownership mismatch");
    // BrookesMoss: NO condensation, NO sintering
    static_assert(hasNucleation<BM>() && hasCoagulation<BM>() && !hasCondensation<BM>() &&
                      hasGrowth<BM>() && hasOxidation<BM>() && !hasSintering<BM>(),
                  "BrookesMoss ownership mismatch");
    // TiO2: nucleation coagulation condensation sintering  NO growth NO oxidation
    static_assert(hasNucleation<T2>() && hasCoagulation<T2>() && hasCondensation<T2>() &&
                      !hasGrowth<T2>() && !hasOxidation<T2>() && hasSintering<T2>(),
                  "TiO2 ownership mismatch");

    // ── Print the matrix ──────────────────────────────────────────────────
    constexpr auto Y = "  ✓  ";
    constexpr auto N = " [ZF] ";

    std::cout << "\n";
    std::cout << "┌────────────────┬──────────┬──────────────┬─────────────┬──────────┐\n";
    std::cout << "│ Process        │   HMOM   │ ThreeEqns    │ BrookesMoss │   TiO2   │\n";
    std::cout << "├────────────────┼──────────┼──────────────┼─────────────┼──────────┤\n";

    auto row = [&](const char* proc, bool h, bool te, bool bm, bool t2)
    {
        std::cout << "│ " << std::left << std::setw(15) << proc << "│" << (h ? Y : N) << "  │"
                  << (te ? Y : N) << "    │" << (bm ? Y : N) << "   │" << (t2 ? Y : N) << " │\n";
    };

    row("nucleation", hasNucleation<H>(), hasNucleation<TE>(), hasNucleation<BM>(), hasNucleation<T2>());
    row("coagulation",
        hasCoagulation<H>(),
        hasCoagulation<TE>(),
        hasCoagulation<BM>(),
        hasCoagulation<T2>());
    row("condensation",
        hasCondensation<H>(),
        hasCondensation<TE>(),
        hasCondensation<BM>(),
        hasCondensation<T2>());
    row("growth", hasGrowth<H>(), hasGrowth<TE>(), hasGrowth<BM>(), hasGrowth<T2>());
    row("oxidation", hasOxidation<H>(), hasOxidation<TE>(), hasOxidation<BM>(), hasOxidation<T2>());
    row("sintering", hasSintering<H>(), hasSintering<TE>(), hasSintering<BM>(), hasSintering<T2>());

    std::cout << "└────────────────┴──────────┴──────────────┴─────────────┴──────────┘\n";
    std::cout << "  ✓ = variant owns this source vector (has _impl() method)\n";
    std::cout << " [ZF]= zero-fallback from kZeroData (no _impl() declared)\n";
    std::cout << "  All static_asserts passed — ownership matrix verified at compile time.\n";
}

// ============================================================================
// Runtime-wrapper accessor instantiation
// ============================================================================

static bool validateAnyMomentMethodAccessors(const MOM::BasicThermoData& th)
{
    bool ok = true;

    auto model = MOM::MakeAnyMomentMethod<MOM::BasicThermoData>(th, "ThreeEquations");

    const auto initial = MOM::GetInitialMoments(model);
    const bool closure_active = MOM::GetClosureDummySpeciesIsActive(model);
    const int closure_index = MOM::GetClosureDummyIndex(model);
    const int precursor_index = MOM::GetPrecursorIndex(model);

    ok = ok && (MOM::GetNEquations(model) == 3u);
    ok = ok && (initial.size() == 3u);
    ok = ok && (!closure_active);
    ok = ok && (closure_index == -1);
    ok = ok && (precursor_index == th.IndexOfSpecies("C2H2"));

    std::cout << "\n=== Runtime wrapper accessor validation ===\n";
    std::cout << (ok ? "  [PASS] AnyMomentMethod accessors instantiated and returned expected defaults\n"
                     : "  [FAIL] AnyMomentMethod accessor defaults are inconsistent\n");

    return ok;
}

static bool validateThreeEquationsSpeciesValidation()
{
    auto th = buildSootThermo();

    constexpr std::size_t c2h2_index = 5u;
    th.names.erase(th.names.begin() + static_cast<std::ptrdiff_t>(c2h2_index));
    th.mw.erase(th.mw.begin() + static_cast<std::ptrdiff_t>(c2h2_index));
    th.nc.erase(th.nc.begin() + static_cast<std::ptrdiff_t>(c2h2_index));
    th.nh.erase(th.nh.begin() + static_cast<std::ptrdiff_t>(c2h2_index));
    th.no.erase(th.no.begin() + static_cast<std::ptrdiff_t>(c2h2_index));
    th.nn.erase(th.nn.begin() + static_cast<std::ptrdiff_t>(c2h2_index));
    th.nti.erase(th.nti.begin() + static_cast<std::ptrdiff_t>(c2h2_index));

    bool ok = false;
    try
    {
        MOM::ThreeEquations<MOM::BasicThermoData> model(th);
        (void)model;
    }
    catch (const std::runtime_error& e)
    {
        ok = std::string(e.what()).find("C2H2") != std::string::npos;
    }

    std::cout << "\n=== ThreeEquations species validation ===\n";
    std::cout << (ok ? "  [PASS] Missing C2H2 is rejected during setup\n"
                     : "  [FAIL] Missing C2H2 was not reported clearly\n");

    return ok;
}

// ============================================================================
// main
// ============================================================================

int main()
{
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  MOM Library — All-variant verification suite                   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";

    // ── Common gas state for soot variants ────────────────────────────────
    auto thS = buildSootThermo();
    // Rich flame: high C2H2 (PAH source), O2, with radical pool
    const auto Y_soot   = X2Y({0.010, 0.020, 0.180, 0.010, 0.100, 0.060, 0.620}, thS);
    const double T_soot = 1800.;   // [K]
    const double P_atm  = 101325.; // [Pa]
    const double mu     = 4.5e-5;  // [kg/m/s]

    bool all_ok = true;

    all_ok &= validateAnyMomentMethodAccessors(thS);
    all_ok &= validateThreeEquationsSpeciesValidation();

    // ════════════════════════════════════════════════════════════════════
    // 1. HMOM  (NEq = 4)
    // Moments: [M00_norm, M10_norm, M01_norm, N0_norm]
    // ════════════════════════════════════════════════════════════════════
    std::cout << "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  Variant 1: HMOM\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    {
        MOM::HMOM<MOM::BasicThermoData> model(thS);
        model.SetNucleation(1);
        model.SetCoagulation(1);
        model.SetCondensation(1);
        model.SetSurfaceGrowth(1);
        model.SetOxidation(1);
        model.SetViscosity(mu);
        model.SetStatus(T_soot, P_atm, Y_soot.data());
        // Use initial_moments() so all floors are satisfied
        auto ic = model.initial_moments();
        model.SetMoments(ic);

        std::cout << "  State: T=" << T_soot << " K, P=" << P_atm << " Pa\n";
        std::cout << "  Moments: M00=" << ic[0] << "  M10=" << ic[1] << "  M01=" << ic[2]
                  << "  N0=" << ic[3] << "\n";

        all_ok &= verifyVariant(model,
                                "HMOM",
                                /*nucleation*/ true,
                                /*coagulation*/ true,
                                /*condensation*/ true,
                                /*growth*/ true,
                                /*oxidation*/ true,
                                /*sintering*/ false); // ← zero fallback expected
    }

    // ════════════════════════════════════════════════════════════════════
    // 2. ThreeEquations  (NEq = 3)
    // Moments: [Ys, NsNorm, Ss]
    // ════════════════════════════════════════════════════════════════════
    std::cout << "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  Variant 2: ThreeEquations\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    {
        MOM::ThreeEquations<MOM::BasicThermoData> model(thS);
        model.SetPAH("C2H2");
        model.SetNucleation(1);
        model.SetCoagulation(1);
        model.SetCondensation(1);
        model.SetSurfaceGrowth(1);
        model.SetOxidation(1);
        model.SetViscosity(mu);
        model.SetStatus(T_soot, P_atm, Y_soot.data());
        auto ic = model.initial_moments();
        // Scale up from floor so all processes are active
        model.SetMoments(ic[0] * 1e5, ic[1] * 1e5, ic[2] * 1e5);

        std::cout << "  State: T=" << T_soot << " K, P=" << P_atm << " Pa\n";
        std::cout << "  Moments: Ys=" << std::scientific << ic[0] * 1e5
                  << "  NsNorm=" << ic[1] * 1e5 << "  Ss=" << ic[2] * 1e5 << "\n";

        all_ok &= verifyVariant(model,
                                "ThreeEquations",
                                /*nucleation*/ true,
                                /*coagulation*/ true,
                                /*condensation*/ true,
                                /*growth*/ true,
                                /*oxidation*/ true,
                                /*sintering*/ false); // ← zero fallback expected
    }

    // ════════════════════════════════════════════════════════════════════
    // 3. BrookesMoss  (NEq = 2)
    // Moments: [Ys, bs]
    // condensation NOT modelled → kZeroData fallback expected
    // ════════════════════════════════════════════════════════════════════
    std::cout << "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  Variant 3: BrookesMoss\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    {
        MOM::BrookesMoss<MOM::BasicThermoData> model(thS);
        model.SetNucleation(1);
        model.SetCoagulation(1);
        model.SetSurfaceGrowth(1);
        model.SetOxidation(1);
        model.SetViscosity(mu);
        model.SetStatus(T_soot, P_atm, Y_soot.data());
        auto ic = model.initial_moments();
        model.SetMoments(ic[0] * 1e10, ic[1] * 1e10);

        std::cout << "  State: T=" << T_soot << " K, P=" << P_atm << " Pa\n";
        std::cout << "  Moments: Ys=" << std::scientific << ic[0] * 1e10 << "  bs=" << ic[1] * 1e10
                  << "\n";

        all_ok &= verifyVariant(model,
                                "BrookesMoss",
                                /*nucleation*/ true,
                                /*coagulation*/ true,
                                /*condensation*/ false, // ← zero fallback expected
                                /*growth*/ true,
                                /*oxidation*/ true,
                                /*sintering*/ false); // ← zero fallback expected
    }

    // ════════════════════════════════════════════════════════════════════
    // 4. TiO2  (NEq = 3)
    // Moments: [YTiO2, NTiO2N, STiO2]
    // growth + oxidation NOT modelled → kZeroData fallback expected
    // ════════════════════════════════════════════════════════════════════
    std::cout << "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  Variant 4: TiO2\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    {
        auto thT = buildTiO2Thermo();
        // 5% TiOH4 precursor, balance N2 (flame synthesis conditions)
        const auto Y_tio2   = X2Y({0.05, 0.95}, thT);
        const double T_tio2 = 1500.; // [K] — typical TiO2 synthesis temperature

        MOM::TiO2<MOM::BasicThermoData> model(thT);
        model.SetPrecursor("TiOH4");
        model.SetNucleation(1);
        model.SetCoagulation(1);
        model.SetCondensation(1);
        model.SetSintering(1);
        model.SetViscosity(mu);
        model.SetStatus(T_tio2, P_atm, Y_tio2.data());
        auto ic = model.initial_moments();
        // Use well-above-floor values: 1e12 #/m3 particles, fv~1e-6
        model.SetMoments(ic[0] * 1e6, ic[1] * 1e6, ic[2] * 1e6);

        std::cout << "  State: T=" << T_tio2 << " K, P=" << P_atm << " Pa\n";
        std::cout << "  Precursor: TiOH4  (5 mol%)\n";
        std::cout << "  Moments: YTiO2=" << std::scientific << ic[0] * 1e6
                  << "  NTiO2N=" << ic[1] * 1e6 << "  STiO2=" << ic[2] * 1e6 << "\n";

        all_ok &= verifyVariant(model,
                                "TiO2",
                                /*nucleation*/ true,
                                /*coagulation*/ true,
                                /*condensation*/ true,
                                /*growth*/ false,    // ← zero fallback expected
                                /*oxidation*/ false, // ← zero fallback expected
                                /*sintering*/ true);
    }

    // ════════════════════════════════════════════════════════════════════
    // Compile-time ownership matrix
    // ════════════════════════════════════════════════════════════════════
    std::cout << "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  Compile-time ownership matrix (static_assert verified)\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    printOwnershipMatrix();

    // ════════════════════════════════════════════════════════════════════
    // Final verdict
    // ════════════════════════════════════════════════════════════════════
    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  Final verdict: "
              << (all_ok ? "ALL CHECKS PASSED ✓                           "
                         : "ONE OR MORE CHECKS FAILED ✗                   ")
              << "║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n";
    return all_ok ? 0 : 1;
}
