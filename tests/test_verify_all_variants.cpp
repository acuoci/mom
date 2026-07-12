/*-----------------------------------------------------------------------*\
|   MOM Library — Comprehensive variant verification                       |
|                                                                          |
|   Verifies all 4 variants (HMOM, ThreeEquations, BrookesMoss, MetalOxide)    |
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
|         -I ../include/ThreeEquations -I ../include/MetalOxide                  |
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
//   thermo_metaloxide — for MetalOxide (titanium oxide chemistry)
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

static MOM::BasicThermoData buildBrookesMossHallThermo()
{
    auto th = buildSootThermo();
    th.names.insert(th.names.end(), {"C6H6", "C6H5"});
    th.mw.insert(th.mw.end(), {78.114, 77.106});
    th.nc.insert(th.nc.end(), {6, 6});
    th.nh.insert(th.nh.end(), {6, 5});
    th.no.insert(th.no.end(), {0, 0});
    th.nn.insert(th.nn.end(), {0, 0});
    th.nti.insert(th.nti.end(), {0, 0});
    return th;
}

static MOM::BasicThermoData buildMetalOxideThermo()
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
    using T2 = MOM::MetalOxide<MOM::BasicThermoData>;

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
    // MetalOxide: nucleation coagulation condensation sintering  NO growth NO oxidation
    static_assert(hasNucleation<T2>() && hasCoagulation<T2>() && hasCondensation<T2>() &&
                      !hasGrowth<T2>() && !hasOxidation<T2>() && hasSintering<T2>(),
                  "MetalOxide ownership mismatch");

    // ── Print the matrix ──────────────────────────────────────────────────
    constexpr auto Y = "  ✓  ";
    constexpr auto N = " [ZF] ";

    std::cout << "\n";
    std::cout << "┌────────────────┬──────────┬──────────────┬─────────────┬──────────┐\n";
    std::cout << "│ Process        │   HMOM   │ ThreeEqns    │ BrookesMoss │   MetalOxide   │\n";
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

static bool validateHMOMSpeciesValidation()
{
    const auto th = buildSootThermo();

    bool ok = false;
    try
    {
        MOM::HMOM<MOM::BasicThermoData> model(th);
        model.SetPAH("MISSING_PAH");
    }
    catch (const std::runtime_error& e)
    {
        ok = std::string(e.what()).find("MISSING_PAH") != std::string::npos;
    }

    std::cout << "\n=== HMOM species validation ===\n";
    std::cout << (ok ? "  [PASS] Missing PAH species is rejected during setup\n"
                     : "  [FAIL] Missing PAH species was not reported clearly\n");

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

static bool validateBrookesMossHallSpeciesValidation()
{
    const auto th = buildBrookesMossHallThermo();

    bool configured_names_ok = false;
    try
    {
        MOM::BrookesMoss<MOM::BasicThermoData> model(th);
        model.SetNucleation("BrookesMossHall");
        model.SetOxidation("BrookesMossHall");
        configured_names_ok = model.nucleation_model() == 2 && model.oxidation_model() == 2;
    }
    catch (const std::runtime_error&)
    {
        configured_names_ok = false;
    }

    bool composition_ok = false;
    try
    {
        MOM::BrookesMoss<MOM::BasicThermoData> model(th);
        model.SetBenzeneSpecies("C6H5");
    }
    catch (const std::runtime_error& e)
    {
        composition_ok = std::string(e.what()).find("wrong atomic composition") != std::string::npos;
    }

    bool reporter_label_ok = false;
    try
    {
        MOM::BrookesMoss<MOM::BasicThermoData> model(th);
        model.SetNucleation("BrookesMossHall");

        std::vector<std::string> labels;
        model.variant_prefix_output(
            [&labels](const char* label, double)
            {
                labels.emplace_back(label);
            });

        reporter_label_ok =
            std::find(labels.begin(), labels.end(), "omegaC6H5[kg/m3/s]") != labels.end() &&
            std::find(labels.begin(), labels.end(), "omegaC6H4[kg/m3/s]") == labels.end();
    }
    catch (const std::runtime_error&)
    {
        reporter_label_ok = false;
    }

    const bool ok = configured_names_ok && composition_ok && reporter_label_ok;

    std::cout << "\n=== BrookesMoss-Hall species validation ===\n";
    std::cout << (configured_names_ok
                      ? "  [PASS] Configured C6H6/C6H5 names are accepted\n"
                      : "  [FAIL] Configured C6H6/C6H5 names were rejected\n");
    std::cout << (composition_ok
                      ? "  [PASS] Wrong benzene composition is rejected\n"
                      : "  [FAIL] Wrong benzene composition was accepted\n");
    std::cout << (reporter_label_ok
                      ? "  [PASS] Reporter uses omegaC6H5 label\n"
                      : "  [FAIL] Reporter label for phenyl radical is wrong\n");

    return ok;
}

static bool validateBrookesMossReporterMissingSpecies()
{
    auto th = buildSootThermo();

    constexpr std::size_t h2_index = 3u;
    th.names.erase(th.names.begin() + static_cast<std::ptrdiff_t>(h2_index));
    th.mw.erase(th.mw.begin() + static_cast<std::ptrdiff_t>(h2_index));
    th.nc.erase(th.nc.begin() + static_cast<std::ptrdiff_t>(h2_index));
    th.nh.erase(th.nh.begin() + static_cast<std::ptrdiff_t>(h2_index));
    th.no.erase(th.no.begin() + static_cast<std::ptrdiff_t>(h2_index));
    th.nn.erase(th.nn.begin() + static_cast<std::ptrdiff_t>(h2_index));
    th.nti.erase(th.nti.begin() + static_cast<std::ptrdiff_t>(h2_index));

    bool h2_reported = false;
    double h2_value = 1.;
    bool ok = false;

    try
    {
        MOM::BrookesMoss<MOM::BasicThermoData> model(th);
        model.variant_prefix_output(
            [&h2_reported, &h2_value](const char* label, double value)
            {
                if (std::string(label) == "omegaH2[kg/m3/s]")
                {
                    h2_reported = true;
                    h2_value = value;
                }
            });
        ok = h2_reported && h2_value == 0.;
    }
    catch (const std::runtime_error&)
    {
        ok = false;
    }

    std::cout << "\n=== BrookesMoss reporter missing-species handling ===\n";
    std::cout << (ok ? "  [PASS] Missing H2 reports omegaH2=0 without invalid indexing\n"
                     : "  [FAIL] Missing H2 reporter output is unsafe or incorrect\n");

    return ok;
}

static bool validateBrookesMossHallConfigDefaults()
{
    const auto th = buildBrookesMossHallThermo();
    const auto Y = X2Y({0.010, 0.020, 0.160, 0.020, 0.080, 0.060, 0.600, 0.030, 0.020}, th);

    bool ok = false;

    try
    {
        MOM::BrookesMoss<MOM::BasicThermoData>::Config cfg;
        cfg.nucleation_model = 2;
        cfg.oxidation_model = 2;

        MOM::BrookesMoss<MOM::BasicThermoData> from_config(th);
        from_config.SetupFromConfig(cfg);
        from_config.SetStatus(1800., 101325., Y.data());
        from_config.SetMoments(1.e-11, 1.e-2);
        from_config.CalculateSourceMoments();

        MOM::BrookesMoss<MOM::BasicThermoData> from_setters(th);
        from_setters.SetNucleation("BrookesMossHall");
        from_setters.SetOxidation("BrookesMossHall");
        from_setters.SetStatus(1800., 101325., Y.data());
        from_setters.SetMoments(1.e-11, 1.e-2);
        from_setters.CalculateSourceMoments();

        const auto a = from_config.sources();
        const auto b = from_setters.sources();
        ok = a.size() == b.size();
        for (std::size_t i = 0; ok && i < a.size(); ++i)
        {
            const double scale = std::max({1., std::abs(a[i]), std::abs(b[i])});
            ok = std::abs(a[i] - b[i]) <= 1.e-12 * scale;
        }
    }
    catch (const std::runtime_error&)
    {
        ok = false;
    }

    std::cout << "\n=== BrookesMoss-Hall config defaults ===\n";
    std::cout << (ok ? "  [PASS] Config BM-Hall defaults match string-setter BM-Hall defaults\n"
                     : "  [FAIL] Config BM-Hall defaults diverge from string-setter setup\n");

    return ok;
}

static bool validateBrookesMossInvalidModelFlags()
{
    const auto th = buildSootThermo();

    bool nucleation_ok = false;
    bool surface_growth_ok = false;
    bool oxidation_ok = false;
    bool coagulation_ok = false;
    bool config_ok = false;

    try
    {
        MOM::BrookesMoss<MOM::BasicThermoData> model(th);
        model.SetNucleation(99);
    }
    catch (const std::invalid_argument&)
    {
        nucleation_ok = true;
    }

    try
    {
        MOM::BrookesMoss<MOM::BasicThermoData> model(th);
        model.SetSurfaceGrowth(99);
    }
    catch (const std::invalid_argument&)
    {
        surface_growth_ok = true;
    }

    try
    {
        MOM::BrookesMoss<MOM::BasicThermoData> model(th);
        model.SetOxidation(99);
    }
    catch (const std::invalid_argument&)
    {
        oxidation_ok = true;
    }

    try
    {
        MOM::BrookesMoss<MOM::BasicThermoData> model(th);
        model.SetCoagulation(99);
    }
    catch (const std::invalid_argument&)
    {
        coagulation_ok = true;
    }

    try
    {
        MOM::BrookesMoss<MOM::BasicThermoData>::Config cfg;
        cfg.nucleation_model = 99;

        MOM::BrookesMoss<MOM::BasicThermoData> model(th);
        model.SetupFromConfig(cfg);
    }
    catch (const std::invalid_argument&)
    {
        config_ok = true;
    }

    const bool ok = nucleation_ok && surface_growth_ok && oxidation_ok && coagulation_ok && config_ok;

    std::cout << "\n=== BrookesMoss integer model flag validation ===\n";
    std::cout << (ok ? "  [PASS] Invalid integer model flags are rejected\n"
                     : "  [FAIL] At least one invalid integer model flag was accepted\n");

    return ok;
}

static bool validateIntegerModelFlagValidation()
{
    const auto th_soot        = buildSootThermo();
    const auto th_metaloxide = buildMetalOxideThermo();

    auto throws_invalid_argument = [](auto&& fn)
    {
        try
        {
            fn();
        }
        catch (const std::invalid_argument&)
        {
            return true;
        }
        catch (...)
        {
            return false;
        }
        return false;
    };

    const bool hmom_ok = [&]
    {
        MOM::HMOM<MOM::BasicThermoData> model(th_soot);
        return throws_invalid_argument([&] { model.SetNucleation(99); }) &&
               throws_invalid_argument([&] { model.SetCondensation(99); }) &&
               throws_invalid_argument([&] { model.SetSurfaceGrowth(99); }) &&
               throws_invalid_argument([&] { model.SetOxidation(99); }) &&
               throws_invalid_argument([&] { model.SetCoagulation(99); }) &&
               throws_invalid_argument([&] { model.SetCoagulationContinuous(99); }) &&
               throws_invalid_argument([&] { model.SetFractalDiameterModel(99); }) &&
               throws_invalid_argument([&] { model.SetCollisionDiameterModel(99); });
    }();

    const bool three_equations_ok = [&]
    {
        MOM::ThreeEquations<MOM::BasicThermoData> model(th_soot);
        return throws_invalid_argument([&] { model.SetNucleation(99); }) &&
               throws_invalid_argument([&] { model.SetCondensation(99); }) &&
               throws_invalid_argument([&] { model.SetSurfaceGrowth(99); }) &&
               throws_invalid_argument([&] { model.SetOxidation(99); }) &&
               throws_invalid_argument([&] { model.SetCoagulation(99); });
    }();

    const bool metaloxide_ok = [&]
    {
        MOM::MetalOxide<MOM::BasicThermoData> model(th_metaloxide);
        return throws_invalid_argument([&] { model.SetNucleation(99); }) &&
               throws_invalid_argument([&] { model.SetCondensation(99); }) &&
               throws_invalid_argument([&] { model.SetCoagulation(99); }) &&
               throws_invalid_argument([&] { model.SetSintering(99); }) &&
               throws_invalid_argument(
                   [&]
                   {
                       MOM::MetalOxide<MOM::BasicThermoData>::Config cfg;
                       cfg.nucleation_model = "unknown";
                       model.SetupFromConfig(cfg);
                   });
    }();

    const bool thermophoretic_ok =
        throws_invalid_argument(
            [&]
            {
                MOM::HMOM<MOM::BasicThermoData> model(th_soot);
                model.SetThermophoreticModel(99);
            });

    const bool ok = hmom_ok && three_equations_ok && metaloxide_ok && thermophoretic_ok;

    std::cout << "\n=== Integer model flag validation across variants ===\n";
    std::cout << (hmom_ok ? "  [PASS] HMOM rejects invalid integer model flags\n"
                         : "  [FAIL] HMOM accepted at least one invalid integer model flag\n");
    std::cout << (three_equations_ok
                      ? "  [PASS] ThreeEquations rejects invalid integer model flags\n"
                      : "  [FAIL] ThreeEquations accepted at least one invalid integer model flag\n");
    std::cout << (metaloxide_ok
                      ? "  [PASS] MetalOxide rejects invalid integer/string model flags\n"
                      : "  [FAIL] MetalOxide accepted at least one invalid model flag\n");
    std::cout << (thermophoretic_ok
                      ? "  [PASS] Shared thermophoretic model rejects invalid integer flags\n"
                      : "  [FAIL] Shared thermophoretic model accepted an invalid integer flag\n");

    return ok;
}

static bool validateHMOMGasConsumptionDisableClearsOutput()
{
    const auto th = buildSootThermo();
    const auto Y  = X2Y({0.010, 0.020, 0.180, 0.010, 0.100, 0.060, 0.620}, th);

    bool produced_gas_sources = false;
    bool cleared_on_disable = false;
    bool stayed_clear_after_calculation = false;

    try
    {
        MOM::HMOM<MOM::BasicThermoData> model(th);
        model.SetNucleation(1);
        model.SetCoagulation(1);
        model.SetCondensation(1);
        model.SetSurfaceGrowth(1);
        model.SetOxidation(1);
        model.SetStatus(1800., 101325., Y.data());
        model.SetMoments(model.initial_moments());
        model.CalculateSourceMoments();

        const auto gas_enabled = model.omega_gas();
        produced_gas_sources =
            std::any_of(gas_enabled.begin(), gas_enabled.end(), [](double v) { return v != 0.; });

        model.SetGasConsumption(false);
        const auto gas_disabled = model.omega_gas();
        cleared_on_disable = std::all_of(
            gas_disabled.begin(), gas_disabled.end(), [](double v) { return v == 0.; });

        model.CalculateSourceMoments();
        const auto gas_after_calculation = model.omega_gas();
        stayed_clear_after_calculation =
            std::all_of(gas_after_calculation.begin(),
                        gas_after_calculation.end(),
                        [](double v) { return v == 0.; });
    }
    catch (const std::runtime_error&)
    {
        produced_gas_sources = false;
        cleared_on_disable = false;
        stayed_clear_after_calculation = false;
    }

    const bool ok = produced_gas_sources && cleared_on_disable && stayed_clear_after_calculation;

    std::cout << "\n=== HMOM gas-consumption disabled output handling ===\n";
    std::cout << (produced_gas_sources
                      ? "  [PASS] Enabled gas consumption produced non-zero gas sources\n"
                      : "  [FAIL] Enabled gas consumption did not produce a testable gas source\n");
    std::cout << (cleared_on_disable
                      ? "  [PASS] Disabling gas consumption clears omega_gas\n"
                      : "  [FAIL] Disabling gas consumption left stale omega_gas values\n");
    std::cout << (stayed_clear_after_calculation
                      ? "  [PASS] Disabled gas consumption remains zero after source evaluation\n"
                      : "  [FAIL] Disabled gas consumption was modified during source evaluation\n");

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
    all_ok &= validateHMOMSpeciesValidation();
    all_ok &= validateThreeEquationsSpeciesValidation();
    all_ok &= validateBrookesMossHallSpeciesValidation();
    all_ok &= validateBrookesMossReporterMissingSpecies();
    all_ok &= validateBrookesMossHallConfigDefaults();
    all_ok &= validateBrookesMossInvalidModelFlags();
    all_ok &= validateIntegerModelFlagValidation();
    all_ok &= validateHMOMGasConsumptionDisableClearsOutput();

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
    // 4. MetalOxide  (NEq = 3)
    // Moments: [YMetalOxide, NMetalOxideN, SMetalOxide]
    // growth + oxidation NOT modelled → kZeroData fallback expected
    // ════════════════════════════════════════════════════════════════════
    std::cout << "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    std::cout << "  Variant 4: MetalOxide\n";
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n";
    {
        auto thT = buildMetalOxideThermo();
        // 5% TiOH4 precursor, balance N2 (flame synthesis conditions)
        const auto Y_metaloxide   = X2Y({0.05, 0.95}, thT);
        const double T_metaloxide = 1500.; // [K] — typical MetalOxide synthesis temperature

        MOM::MetalOxide<MOM::BasicThermoData> model(thT);
        model.SetPrecursor("TiOH4");
        model.SetNucleation(1);
        model.SetCoagulation(1);
        model.SetCondensation(1);
        model.SetSintering(1);
        model.SetViscosity(mu);
        model.SetStatus(T_metaloxide, P_atm, Y_metaloxide.data());
        auto ic = model.initial_moments();
        // Use well-above-floor values: 1e12 #/m3 particles, fv~1e-6
        model.SetMoments(ic[0] * 1e6, ic[1] * 1e6, ic[2] * 1e6);

        std::cout << "  State: T=" << T_metaloxide << " K, P=" << P_atm << " Pa\n";
        std::cout << "  Precursor: TiOH4  (5 mol%)\n";
        std::cout << "  Moments: YMetalOxide=" << std::scientific << ic[0] * 1e6
                  << "  NMetalOxideN=" << ic[1] * 1e6 << "  SMetalOxide=" << ic[2] * 1e6 << "\n";

        all_ok &= verifyVariant(model,
                                "MetalOxide",
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
