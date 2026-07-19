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

    model.ComputeSources();

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
    const bool closure_active = MOM::HasClosureDummySpecies(model);
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
        configured_names_ok = model.nucleation_model() == MOM::NucleationModel::Extended
                           && model.oxidation_model() == MOM::OxidationModel::Extended;
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
        cfg.nucleation_model = MOM::NucleationModel::Extended; // BrookesMossHall
        cfg.oxidation_model  = MOM::OxidationModel::Extended;  // BrookesMossHall

        MOM::BrookesMoss<MOM::BasicThermoData> from_config(th);
        from_config.SetupFromConfig(cfg);
        from_config.SetState(1800., 101325., Y.data());
        from_config.SetMoments(1.e-11, 1.e-2);
        from_config.ComputeSources();

        MOM::BrookesMoss<MOM::BasicThermoData> from_setters(th);
        from_setters.SetNucleation("BrookesMossHall");
        from_setters.SetOxidation("BrookesMossHall");
        from_setters.SetState(1800., 101325., Y.data());
        from_setters.SetMoments(1.e-11, 1.e-2);
        from_setters.ComputeSources();

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
        // D1: direct integer assignment (cfg.nucleation_model = 99) is now a compile
        // error — the field is a strongly-typed enum.  The runtime validator in
        // SetNucleation(int) is still exercised via an explicit static_cast.
        MOM::BrookesMoss<MOM::BasicThermoData>::Config cfg;
        cfg.nucleation_model = static_cast<MOM::NucleationModel>(99);

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

    // D4: SetThermophoreticModel(int) is now noexcept — it silently casts any
    // integer to ThermophoreticModel without throwing.  Out-of-range validation
    // belongs in ApplyConfig() / ParseConfig(), which have error-reporting context.
    // The test therefore verifies that the setter does NOT throw (noexcept contract).
    const bool thermophoretic_ok = !throws_invalid_argument(
        [&]
        {
            MOM::HMOM<MOM::BasicThermoData> model(th_soot);
            model.SetThermophoreticModel(99);  // should silently accept (noexcept)
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
                      ? "  [PASS] Shared thermophoretic setter is noexcept (invalid int silently cast; validation deferred to ParseConfig)\n"
                      : "  [FAIL] Shared thermophoretic setter unexpectedly threw on an integer flag\n");

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
        model.SetState(1800., 101325., Y.data());
        model.SetMoments(model.initial_moments());
        model.ComputeSources();

        const auto gas_enabled = model.omega_gas();
        produced_gas_sources =
            std::any_of(gas_enabled.begin(), gas_enabled.end(), [](double v) { return v != 0.; });

        model.SetGasConsumption(false);
        const auto gas_disabled = model.omega_gas();
        cleared_on_disable = std::all_of(
            gas_disabled.begin(), gas_disabled.end(), [](double v) { return v == 0.; });

        model.ComputeSources();
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

static bool validateMetalOxideMonodisperseClosureRegression()
{
    const auto th = buildMetalOxideThermo();
    bool ok = true;

    auto spans_equal_exact = [](std::span<const double> a, std::span<const double> b)
    {
        if (a.size() != b.size())
            return false;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (a[i] != b[i])
                return false;
        return true;
    };

    auto scalars_equal_exact = [](double a, double b)
    {
        return a == b;
    };

    struct RuntimeCase
    {
        MOM::MetalOxide<MOM::BasicThermoData>::Config cfg;
        double precursor_mole_fraction;
        double T;
        double P;
        double mu;
        double solid_mass_fraction;
        double scaled_number_density;
        double surface_area_concentration;
    };

    auto compare_default_and_explicit = [&](const RuntimeCase& c, const char*& mismatch)
    {
        auto base_cfg = c.cfg;
        auto explicit_cfg = base_cfg;
        explicit_cfg.closure_model = "monodisperse";

        const auto Y = X2Y({c.precursor_mole_fraction, 1. - c.precursor_mole_fraction}, th);

        MOM::MetalOxide<MOM::BasicThermoData> default_model(th);
        default_model.SetupFromConfig(base_cfg);
        default_model.SetViscosity(c.mu);
        default_model.SetState(c.T, c.P, Y.data());
        default_model.SetMoments(c.solid_mass_fraction,
                                 c.scaled_number_density,
                                 c.surface_area_concentration);
        default_model.ComputeSources();

        MOM::MetalOxide<MOM::BasicThermoData> explicit_model(th);
        explicit_model.SetupFromConfig(explicit_cfg);
        explicit_model.SetViscosity(c.mu);
        explicit_model.SetState(c.T, c.P, Y.data());
        explicit_model.SetMoments(c.solid_mass_fraction,
                                  c.scaled_number_density,
                                  c.surface_area_concentration);
        explicit_model.ComputeSources();

        if (default_model.closure_model() !=
            MOM::MetalOxide<MOM::BasicThermoData>::ClosureModel::Monodisperse)
            { mismatch = "default closure"; return false; }
        if (explicit_model.closure_model() !=
            MOM::MetalOxide<MOM::BasicThermoData>::ClosureModel::Monodisperse)
            { mismatch = "explicit closure"; return false; }
        if (!spans_equal_exact(default_model.initial_moments(), explicit_model.initial_moments()))
            { mismatch = "initial_moments"; return false; }
        if (!spans_equal_exact(default_model.sources(), explicit_model.sources()))
            { mismatch = "sources"; return false; }
        if (!spans_equal_exact(default_model.sources_nucleation(), explicit_model.sources_nucleation()))
            { mismatch = "sources_nucleation"; return false; }
        if (!spans_equal_exact(default_model.sources_coagulation(), explicit_model.sources_coagulation()))
            { mismatch = "sources_coagulation"; return false; }
        if (!spans_equal_exact(default_model.sources_condensation(), explicit_model.sources_condensation()))
            { mismatch = "sources_condensation"; return false; }
        if (!spans_equal_exact(default_model.sources_sintering(), explicit_model.sources_sintering()))
            { mismatch = "sources_sintering"; return false; }
        if (!spans_equal_exact(default_model.omega_gas(), explicit_model.omega_gas()))
            { mismatch = "omega_gas"; return false; }
        if (!scalars_equal_exact(default_model.volume_fraction(), explicit_model.volume_fraction()))
            { mismatch = "volume_fraction"; return false; }
        if (!scalars_equal_exact(default_model.particle_number_density(), explicit_model.particle_number_density()))
            { mismatch = "particle_number_density"; return false; }
        if (!scalars_equal_exact(default_model.specific_surface_area(), explicit_model.specific_surface_area()))
            { mismatch = "specific_surface_area"; return false; }
        if (!scalars_equal_exact(default_model.particle_diameter(), explicit_model.particle_diameter()))
            { mismatch = "particle_diameter"; return false; }
        if (!scalars_equal_exact(default_model.collision_diameter(), explicit_model.collision_diameter()))
            { mismatch = "collision_diameter"; return false; }
        if (!scalars_equal_exact(default_model.number_primary_particles(), explicit_model.number_primary_particles()))
            { mismatch = "number_primary_particles"; return false; }
        if (!scalars_equal_exact(default_model.diffusion_coefficient(), explicit_model.diffusion_coefficient()))
            { mismatch = "diffusion_coefficient"; return false; }
        return true;
    };

    try
    {
        MOM::MetalOxide<MOM::BasicThermoData>::Config base_cfg;
        base_cfg.precursor_species = "TiOH4";
        base_cfg.gas_consumption   = false;

        auto no_nucleation_cfg = base_cfg;
        no_nucleation_cfg.nucleation_model = "none";

        auto fixed_cluster_cfg = base_cfg;
        fixed_cluster_cfg.nucleation_model = "fixed-cluster";
        fixed_cluster_cfg.condensation_model = 0;
        fixed_cluster_cfg.sintering_model = 0;

        const std::vector<RuntimeCase> cases{
            RuntimeCase{base_cfg, 0.050, 1500., 101325., 4.0e-5, 1.e-3, 1.e6, 2.e3},
            RuntimeCase{base_cfg, 0.020, 1850., 202650., 6.0e-5, 2.e-5, 5.e3, 1.e-1},
            RuntimeCase{no_nucleation_cfg, 0.080, 1700., 101325., 5.0e-5, 1.e-6, 1.e2, 5.e-6},
            RuntimeCase{fixed_cluster_cfg, 0.010, 1400., 50662.5, 3.5e-5, 5.e-7, 1.e4, 2.e-4}
        };

        std::size_t case_index = 0u;
        const char* mismatch = "none";
        for (const auto& c : cases)
        {
            if (!compare_default_and_explicit(c, mismatch))
            {
                std::cout << "\n  [FAIL detail] MetalOxide monodisperse regression case "
                          << case_index << " mismatch: " << mismatch << "\n";
                ok = false;
                break;
            }
            ++case_index;
        }
    }
    catch (const std::exception& e)
    {
        std::cout << "\n  [FAIL detail] MetalOxide monodisperse regression exception: "
                  << e.what() << "\n";
        ok = false;
    }

    std::cout << "\n=== MetalOxide monodisperse closure regression ===\n";
    std::cout << (ok ? "  [PASS] Default closure and explicit monodisperse closure are bitwise identical\n"
                     : "  [FAIL] Explicit monodisperse closure changed MetalOxide results\n");

    return ok;
}

static bool validateMetalOxideLognormalClosureState()
{
    const auto th = buildMetalOxideThermo();
    const auto Y  = X2Y({0.050, 0.950}, th);

    constexpr double T = 1500.;
    constexpr double P = 101325.;
    constexpr double R = 8314.46261815324;

    const double invMW = std::inner_product(
        Y.begin(),
        Y.end(),
        th.mw.begin(),
        0.,
        std::plus<>(),
        [](double y, double mw) { return y / mw; });
    const double rho = P / (R * T) / invMW;

    auto near = [](double value, double expected, double rtol = 1.e-12)
    {
        const double scale = std::max({1., std::abs(value), std::abs(expected)});
        return std::abs(value - expected) <= rtol * scale;
    };

    auto expected_sigma_g = [](double npp)
    {
        const double mobility_ratio = std::pow(std::max(npp, 1.), 0.45);
        if (mobility_ratio <= 1.)
            return 1.;
        if (mobility_ratio < 3.)
            return 1. + (1.7 - 1.) / (3. - 1.) * (mobility_ratio - 1.);
        if (mobility_ratio < 10.)
            return 1.7 + (1.48 - 1.7) / (10. - 3.) * (mobility_ratio - 3.);
        return 1.48;
    };

    bool spherical_ok = false;
    bool aggregate_ok = false;
    bool invalid_ok = false;

    try
    {
        MOM::MetalOxide<MOM::BasicThermoData> model(th);
        model.SetState(T, P, Y.data());

        const double N = 1.e18;
        const double vmean = 8. * model.v0();
        const double smean = model.s0() * std::pow(vmean / model.v0(), 2. / 3.);
        const double fv = N * vmean;
        const double solid_mass_fraction = model.solid_density() / rho * fv;
        model.SetMoments(solid_mass_fraction, N / model.ScalingFactorNs(), N * smean);

        const auto d =
            MOM::detail::MetalOxideTestAccess<MOM::BasicThermoData>::BuildLognormalClosureData(model);

        spherical_ok =
            d.valid &&
            near(d.N, N) &&
            near(d.fv, fv) &&
            near(d.Sp, N * smean) &&
            near(d.vmean, vmean) &&
            near(d.smean, smean) &&
            near(d.npp_mean, 1.) &&
            near(d.sigma_g_m, 1.) &&
            near(d.sigma, 0.) &&
            near(d.M, 2.) &&
            near(d.Kmean, 1.);
    }
    catch (const std::exception&)
    {
        spherical_ok = false;
    }

    try
    {
        MOM::MetalOxide<MOM::BasicThermoData> model(th);
        model.SetState(T, P, Y.data());

        const double N = 5.e17;
        const double vmean = 4096. * model.v0();
        const double smean = model.s0() * std::pow(vmean / model.v0(), 0.75);
        const double fv = N * vmean;
        const double solid_mass_fraction = model.solid_density() / rho * fv;
        model.SetMoments(solid_mass_fraction, N / model.ScalingFactorNs(), N * smean);

        const auto d =
            MOM::detail::MetalOxideTestAccess<MOM::BasicThermoData>::BuildLognormalClosureData(model);

        const double sigma_g = std::clamp(expected_sigma_g(d.npp_mean), 1., 1.7);
        const double sigma = 3. * std::log(sigma_g);
        const double log_v = std::log(d.vmean / model.v0());
        const double log_s = std::log(d.smean / model.s0());
        const double sigma2 = sigma * sigma;
        const double A = log_v - 0.5 * sigma2;
        const double disc = A * A + 2. * sigma2 * log_s;
        const double M = std::clamp(3. * (std::sqrt(disc) - A) / sigma2, 2., 3.);
        const double Kmean = std::exp(std::log(model.s0() / d.smean) +
                                      (M / 3.) * std::log(d.vmean / model.v0()));

        aggregate_ok =
            d.valid &&
            d.npp_mean > 1. &&
            d.sigma > 0. &&
            d.M >= 2. && d.M <= 3. &&
            near(d.sigma_g_m, sigma_g) &&
            near(d.sigma, sigma) &&
            near(d.M, M) &&
            near(d.Kmean, Kmean);
    }
    catch (const std::exception&)
    {
        aggregate_ok = false;
    }

    try
    {
        MOM::MetalOxide<MOM::BasicThermoData> model(th);
        model.SetState(T, P, Y.data());
        model.SetMoments(0., 0., 0.);
        const auto d =
            MOM::detail::MetalOxideTestAccess<MOM::BasicThermoData>::BuildLognormalClosureData(model);
        invalid_ok = !d.valid;
    }
    catch (const std::exception&)
    {
        invalid_ok = false;
    }

    const bool ok = spherical_ok && aggregate_ok && invalid_ok;

    std::cout << "\n=== MetalOxide lognormal closure state reconstruction ===\n";
    std::cout << (spherical_ok
                      ? "  [PASS] Spherical limit reconstructs sigma=0, M=2, Kmean=1\n"
                      : "  [FAIL] Spherical limit reconstruction is inconsistent\n");
    std::cout << (aggregate_ok
                      ? "  [PASS] Aggregate state reconstructs sigma, M, and Kmean consistently\n"
                      : "  [FAIL] Aggregate lognormal reconstruction is inconsistent\n");
    std::cout << (invalid_ok
                      ? "  [PASS] Empty particle state is rejected by the closure builder\n"
                      : "  [FAIL] Empty particle state produced a valid lognormal closure\n");

    return ok;
}

static bool validateMetalOxideDimensionlessLognormalMoment()
{
    using Access = MOM::detail::MetalOxideTestAccess<MOM::BasicThermoData>;

    auto near = [](double value, double expected, double rtol = 1.e-12)
    {
        const double scale = std::max({1., std::abs(value), std::abs(expected)});
        return std::abs(value - expected) <= rtol * scale;
    };

    const double sigma = 0.72;
    const double sigma2 = sigma * sigma;

    const bool zero_width_ok =
        near(Access::DimensionlessLognormalMoment(-2., 0.), 1.) &&
        near(Access::DimensionlessLognormalMoment(0., 0.), 1.) &&
        near(Access::DimensionlessLognormalMoment(3., 0.), 1.);

    const bool mean_preserving_ok =
        near(Access::DimensionlessLognormalMoment(0., sigma), 1.) &&
        near(Access::DimensionlessLognormalMoment(1., sigma), 1.);

    const bool analytical_ok =
        near(Access::DimensionlessLognormalMoment(2., sigma), std::exp(sigma2)) &&
        near(Access::DimensionlessLognormalMoment(-1., sigma), std::exp(sigma2)) &&
        near(Access::DimensionlessLognormalMoment(0.5, sigma), std::exp(-0.125 * sigma2));

    const bool invalid_ok =
        Access::DimensionlessLognormalMoment(std::numeric_limits<double>::quiet_NaN(), sigma) == 0. &&
        Access::DimensionlessLognormalMoment(1., -1.) == 0.;

    const bool ok = zero_width_ok && mean_preserving_ok && analytical_ok && invalid_ok;

    std::cout << "\n=== MetalOxide dimensionless lognormal moment helper ===\n";
    std::cout << (zero_width_ok
                      ? "  [PASS] Zero-width distribution returns unit moments\n"
                      : "  [FAIL] Zero-width lognormal moments are incorrect\n");
    std::cout << (mean_preserving_ok
                      ? "  [PASS] Dimensionless distribution preserves M0 and M1\n"
                      : "  [FAIL] Dimensionless lognormal normalization is incorrect\n");
    std::cout << (analytical_ok
                      ? "  [PASS] Fractional and negative analytical moments match closed form\n"
                      : "  [FAIL] Analytical lognormal moment formula is inconsistent\n");
    std::cout << (invalid_ok
                      ? "  [PASS] Invalid moment inputs return zero\n"
                      : "  [FAIL] Invalid moment inputs were not guarded\n");

    return ok;
}

static bool validateMetalOxideLognormalLimitingCases()
{
    using Access = MOM::detail::MetalOxideTestAccess<MOM::BasicThermoData>;

    const auto th = buildMetalOxideThermo();
    const auto Y  = X2Y({0.050, 0.950}, th);

    constexpr double T = 1500.;
    constexpr double P = 101325.;
    constexpr double R = 8314.46261815324;
    constexpr double mu = 4.e-5;

    const double invMW = std::inner_product(
        Y.begin(),
        Y.end(),
        th.mw.begin(),
        0.,
        std::plus<>(),
        [](double y, double mw) { return y / mw; });
    const double rho = P / (R * T) / invMW;

    auto near = [](double value, double expected, double rtol = 2.e-12)
    {
        const double scale = std::max({1., std::abs(value), std::abs(expected)});
        return std::abs(value - expected) <= rtol * scale;
    };

    auto span_near = [&](std::span<const double> a, std::span<const double> b)
    {
        if (a.size() != b.size())
            return false;
        for (std::size_t i = 0; i < a.size(); ++i)
            if (!near(a[i], b[i]))
                return false;
        return true;
    };

    auto all_zero = [](std::span<const double> values)
    {
        return std::all_of(values.begin(), values.end(), [](double v) { return v == 0.; });
    };

    auto all_finite = [](std::span<const double> values)
    {
        return std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); });
    };

    auto lognormal_cfg = []()
    {
        MOM::MetalOxide<MOM::BasicThermoData>::Config cfg;
        cfg.precursor_species = "TiOH4";
        cfg.nucleation_model = "none";
        cfg.coagulation_model = 1;
        cfg.condensation_model = 1;
        cfg.sintering_model = 1;
        cfg.closure_model = "lognormal";
        return cfg;
    };

    bool empty_state_ok = false;
    bool compact_limit_ok = false;
    bool subspherical_surface_ok = false;
    bool sigma_cap_ok = false;

    try
    {
        MOM::MetalOxide<MOM::BasicThermoData> model(th);
        model.SetupFromConfig(lognormal_cfg());
        model.SetViscosity(mu);
        model.SetState(T, P, Y.data());
        model.SetMoments(0., 0., 0.);

        const auto d = Access::BuildLognormalClosureData(model);
        model.ComputeSources();

        empty_state_ok =
            !d.valid &&
            all_zero(model.sources()) &&
            all_zero(model.sources_coagulation()) &&
            all_zero(model.sources_condensation()) &&
            all_zero(model.sources_sintering());
    }
    catch (const std::exception&)
    {
        empty_state_ok = false;
    }

    try
    {
        auto mono_cfg = lognormal_cfg();
        mono_cfg.closure_model = "monodisperse";

        MOM::MetalOxide<MOM::BasicThermoData> mono(th);
        mono.SetupFromConfig(mono_cfg);
        mono.SetViscosity(mu);
        mono.SetState(T, P, Y.data());

        MOM::MetalOxide<MOM::BasicThermoData> logn(th);
        logn.SetupFromConfig(lognormal_cfg());
        logn.SetViscosity(mu);
        logn.SetState(T, P, Y.data());

        const double N = 1.e18;
        const double vmean = 8. * mono.v0();
        const double smean = mono.s0() * std::pow(vmean / mono.v0(), 2. / 3.);
        const double fv = N * vmean;
        const double solid_mass_fraction = mono.solid_density() / rho * fv;

        mono.SetMoments(solid_mass_fraction, N / mono.ScalingFactorNs(), N * smean);
        logn.SetMoments(solid_mass_fraction, N / logn.ScalingFactorNs(), N * smean);

        const auto d = Access::BuildLognormalClosureData(logn);
        mono.ComputeSources();
        logn.ComputeSources();

        compact_limit_ok =
            d.valid &&
            near(d.sigma, 0.) &&
            near(d.M, 2.) &&
            near(d.Kmean, 1.) &&
            span_near(mono.sources(), logn.sources()) &&
            span_near(mono.sources_coagulation(), logn.sources_coagulation()) &&
            span_near(mono.sources_condensation(), logn.sources_condensation()) &&
            span_near(mono.sources_sintering(), logn.sources_sintering());
    }
    catch (const std::exception&)
    {
        compact_limit_ok = false;
    }

    try
    {
        auto cfg = lognormal_cfg();
        cfg.coagulation_model = 0;
        cfg.condensation_model = 0;

        MOM::MetalOxide<MOM::BasicThermoData> model(th);
        model.SetupFromConfig(cfg);
        model.SetViscosity(mu);
        model.SetState(T, P, Y.data());

        const double N = 1.e18;
        const double vmean = 8. * model.v0();
        const double ssph = model.s0() * std::pow(vmean / model.v0(), 2. / 3.);
        const double fv = N * vmean;
        const double solid_mass_fraction = model.solid_density() / rho * fv;
        model.SetMoments(solid_mass_fraction, N / model.ScalingFactorNs(), 0.25 * N * ssph);

        const auto d = Access::BuildLognormalClosureData(model);
        model.ComputeSources();

        subspherical_surface_ok =
            d.valid &&
            near(d.smean, ssph) &&
            all_zero(model.sources()) &&
            all_zero(model.sources_sintering());
    }
    catch (const std::exception&)
    {
        subspherical_surface_ok = false;
    }

    try
    {
        MOM::MetalOxide<MOM::BasicThermoData> model(th);
        model.SetupFromConfig(lognormal_cfg());
        model.SetViscosity(mu);
        model.SetState(T, P, Y.data());

        const double N = 1.e7;
        const double volume_ratio = 1.e13;
        const double vmean = volume_ratio * model.v0();
        const double smean = model.s0() * std::pow(volume_ratio, 0.95);
        const double fv = N * vmean;
        const double solid_mass_fraction = model.solid_density() / rho * fv;
        model.SetMoments(solid_mass_fraction, N / model.ScalingFactorNs(), N * smean);

        const auto d = Access::BuildLognormalClosureData(model);
        model.ComputeSources();

        const auto coag = model.sources_coagulation();
        const auto cond = model.sources_condensation();
        const auto sint = model.sources_sintering();

        sigma_cap_ok =
            d.valid &&
            near(d.sigma_g_m, 1.48) &&
            near(d.sigma, 3. * std::log(1.48)) &&
            d.M >= 2. && d.M <= 3. &&
            d.Kmean > 0. &&
            all_finite(model.sources()) &&
            all_finite(coag) &&
            all_finite(cond) &&
            all_finite(sint) &&
            coag.size() == 3u && coag[1] < 0. &&
            cond.size() == 3u && cond[0] > 0. && cond[2] > 0. &&
            sint.size() == 3u && sint[2] <= 0.;
    }
    catch (const std::exception&)
    {
        sigma_cap_ok = false;
    }

    const bool ok = empty_state_ok && compact_limit_ok && subspherical_surface_ok && sigma_cap_ok;

    std::cout << "\n=== MetalOxide lognormal closure limiting cases ===\n";
    std::cout << (empty_state_ok
                      ? "  [PASS] Empty/floored particle states leave lognormal sources at zero\n"
                      : "  [FAIL] Empty/floored particle state was not handled cleanly\n");
    std::cout << (compact_limit_ok
                      ? "  [PASS] Zero-width compact limit recovers monodisperse sources\n"
                      : "  [FAIL] Zero-width compact limit changed source terms\n");
    std::cout << (subspherical_surface_ok
                      ? "  [PASS] Sub-spherical input surface is clamped and produces no sintering source\n"
                      : "  [FAIL] Sub-spherical surface limit is inconsistent\n");
    std::cout << (sigma_cap_ok
                      ? "  [PASS] Broad aggregate limit remains finite with capped sigma_g,m\n"
                      : "  [FAIL] Broad aggregate sigma-cap limit is inconsistent\n");

    return ok;
}

static bool validateMetalOxideLognormalPaper0DCases()
{
    using Access = MOM::detail::MetalOxideTestAccess<MOM::BasicThermoData>;

    constexpr double pi = 3.141592653589793238462643383279502884;
    constexpr double R = 8314.46261815324;
    constexpr double NavKmol = 6.02214076e26;
    constexpr double T = 1850.;
    constexpr double P = 101325.;
    constexpr double rhoParticle = 4230.;
    constexpr double d0 = 0.67e-9;
    constexpr double dprec = 0.5 * d0;
    constexpr double N0 = 1.e21;
    constexpr double Astar = 9.75e21;
    // The results figures are plotted up to 0.1 s; the text statement "0.1 ms"
    // is inconsistent with those captions and axes.
    constexpr double tEnd = 0.1;
    constexpr double dt = 1.e-5;
    constexpr int nSteps = static_cast<int>(tEnd / dt);

    MOM::BasicThermoData th;
    th.names = {"PREC", "N2"};
    th.mw = {
        rhoParticle * (pi / 6.) * dprec * dprec * dprec * NavKmol,
        28.014
    };
    th.nc = {0, 0};
    th.nh = {0, 0};
    th.no = {1, 0};
    th.nn = {0, 2};
    th.nti = {1, 0};

    const double gasInvMW = 1. / th.mw[1];
    const double gasRho = P / (R * T) / gasInvMW;
    const double v0Paper = pi / 6. * d0 * d0 * d0;
    const double s0Paper = pi * d0 * d0;
    const double fv0 = N0 * v0Paper;
    const double S0 = N0 * s0Paper;
    const double Ysolid0 = rhoParticle / gasRho * fv0;
    const double solidMW = rhoParticle * v0Paper * NavKmol;

    auto near = [](double value, double expected, double rtol = 5.e-3)
    {
        const double scale = std::max({1., std::abs(value), std::abs(expected)});
        return std::abs(value - expected) <= rtol * scale;
    };

    auto finite_positive = [](double v)
    {
        return std::isfinite(v) && v > 0.;
    };

    enum class PaperCase : int { I, II, III, IV, V };

    struct Outcome
    {
        bool valid = false;
        double N = 0.;
        double fv = 0.;
        double S = 0.;
        double dpp = 0.;
        double npp = 0.;
        double sigma = 0.;
        double M = 0.;
    };

    auto base_config = [&](MOM::MetalOxide<MOM::BasicThermoData>::ClosureModel closure,
                           PaperCase c)
    {
        MOM::MetalOxide<MOM::BasicThermoData>::Config cfg;
        cfg.precursor_species = "PREC";
        cfg.solid_name = "PaperOxide";
        cfg.solid_molecular_weight_kg_kmol = solidMW;
        cfg.solid_density_kg_m3 = rhoParticle;
        cfg.solid_formula_units_per_precursor = std::pow(dprec / d0, 3.);
        cfg.nucleation_model = "none";
        cfg.closure_model =
            (closure == MOM::MetalOxide<MOM::BasicThermoData>::ClosureModel::Lognormal)
                ? "lognormal"
                : "monodisperse";
        cfg.coagulation_model = 1;
        cfg.condensation_model = (c == PaperCase::IV || c == PaperCase::V) ? 1 : 0;
        cfg.sintering_model = (c == PaperCase::I) ? 0 : 1;
        cfg.gas_consumption = false;
        cfg.sintering_As_s_K_m =
            (c == PaperCase::II || c == PaperCase::IV) ? 1.e-6 * Astar : Astar;
        cfg.sintering_Ts_K = 31000.;
        // MetalOxide uses tau_s = As * T^ns * dp^4 * exp(Ts/T).
        cfg.sintering_ns = 1.;
        cfg.sintering_dp_min_m = 1.e-12;
        cfg.sintering_tau_min_s = 1.e-12;
        cfg.sintering_k_max_per_s = 1.e20;
        cfg.minimum_formula_units = 1;
        cfg.nucleated_particle_formula_units = 1;
        cfg.ns_minimum_per_m3 = 1.;
        cfg.fv_minimum = 1.e-40;
        return cfg;
    };

    auto precursor_number_density = [](double t)
    {
        constexpr double Nplateau = 7.e17;
        constexpr double tMid = 0.020;
        constexpr double width = 8.e-4;
        const double arg = std::clamp((t - tMid) / width, -700., 700.);
        return Nplateau / (1. + std::exp(-arg));
    };

    auto gas_mass_fractions = [&](double Nprec)
    {
        const double cTot = P / (R * T);
        const double xPrec = std::clamp(Nprec / (cTot * NavKmol), 0., 0.5);
        return X2Y({xPrec, 1. - xPrec}, th);
    };

    auto run_case = [&](MOM::MetalOxide<MOM::BasicThermoData>::ClosureModel closure,
                        PaperCase c)
    {
        MOM::MetalOxide<MOM::BasicThermoData> model(th);
        model.SetupFromConfig(base_config(closure, c));
        model.SetViscosity(4.e-5);

        double Ysolid = Ysolid0;
        double Nscaled = N0 / model.ScalingFactorNs();
        double S = S0;

        for (int step = 0; step < nSteps; ++step)
        {
            double remaining = dt;
            int substeps = 0;
            while (remaining > 0.)
            {
                const double t = static_cast<double>(step) * dt + (dt - remaining);
                const double Nprec =
                    (c == PaperCase::IV || c == PaperCase::V) ? precursor_number_density(t) : 0.;
                const auto Y = gas_mass_fractions(Nprec);

                model.SetState(T, P, Y.data());
                model.SetMoments(Ysolid, Nscaled, S);
                model.ComputeSources();

                const auto src = model.sources();
                double h = remaining;

                auto limit_negative_update = [&h](double value, double rate)
                {
                    if (rate < 0. && value > 0.)
                        h = std::min(h, 0.02 * value / (-rate));
                };
                limit_negative_update(Ysolid, src[0]);
                limit_negative_update(Nscaled, src[1]);
                limit_negative_update(S, src[2]);

                if (!std::isfinite(h) || h <= 0. || ++substeps > 100000)
                    return Outcome{};

                Ysolid = std::max(0., Ysolid + h * src[0]);
                Nscaled = std::max(0., Nscaled + h * src[1]);
                S = std::max(0., S + h * src[2]);
                remaining -= h;

                if (!std::isfinite(Ysolid) || !std::isfinite(Nscaled) || !std::isfinite(S))
                    return Outcome{};
            }
        }

        const double NprecFinal =
            (c == PaperCase::IV || c == PaperCase::V) ? precursor_number_density(tEnd) : 0.;
        const auto Yfinal = gas_mass_fractions(NprecFinal);
        model.SetState(T, P, Yfinal.data());
        model.SetMoments(Ysolid, Nscaled, S);
        model.ComputeSources();
        const auto d = Access::BuildLognormalClosureData(model);

        return Outcome{
            d.valid,
            model.particle_number_density(),
            model.volume_fraction(),
            model.specific_surface_area(),
            d.dpp_mean,
            d.npp_mean,
            d.sigma,
            d.M
        };
    };

    const auto mono_i = run_case(
        MOM::MetalOxide<MOM::BasicThermoData>::ClosureModel::Monodisperse, PaperCase::I);
    const auto logn_i = run_case(
        MOM::MetalOxide<MOM::BasicThermoData>::ClosureModel::Lognormal, PaperCase::I);
    const auto logn_ii = run_case(
        MOM::MetalOxide<MOM::BasicThermoData>::ClosureModel::Lognormal, PaperCase::II);
    const auto logn_iii = run_case(
        MOM::MetalOxide<MOM::BasicThermoData>::ClosureModel::Lognormal, PaperCase::III);
    const auto logn_iv = run_case(
        MOM::MetalOxide<MOM::BasicThermoData>::ClosureModel::Lognormal, PaperCase::IV);
    const auto logn_v = run_case(
        MOM::MetalOxide<MOM::BasicThermoData>::ClosureModel::Lognormal, PaperCase::V);

    const bool all_valid =
        mono_i.valid && logn_i.valid && logn_ii.valid && logn_iii.valid &&
        logn_iv.valid && logn_v.valid;

    const bool case_i_ok =
        logn_i.N < N0 &&
        logn_i.N < mono_i.N &&
        near(logn_i.fv, fv0) &&
        near(logn_i.dpp, d0, 2.e-2) &&
        logn_i.npp > 10. &&
        logn_i.M > 2.8;

    const bool case_ii_ok =
        logn_ii.N < N0 &&
        logn_ii.npp < 1.2 &&
        logn_ii.M < 2.05 &&
        logn_ii.dpp > d0;

    const bool case_iii_ok =
        logn_iii.N < N0 &&
        logn_iii.dpp > d0 &&
        logn_iii.npp > 1.05 &&
        logn_iii.M > 2. && logn_iii.M < 3.;

    const bool case_iv_ok =
        logn_iv.N < N0 &&
        logn_iv.fv > fv0 &&
        logn_iv.dpp > d0 &&
        logn_iv.npp < 1.2 &&
        logn_iv.M < 2.1;

    const bool case_v_ok =
        logn_v.N < N0 &&
        logn_v.fv > fv0 &&
        logn_v.S > 0. &&
        logn_v.dpp > d0 &&
        logn_v.npp > 1.2 &&
        logn_v.M > 2. && logn_v.M < 3.;

    const bool finite_ok =
        finite_positive(mono_i.N) && finite_positive(logn_i.N) &&
        finite_positive(logn_ii.N) && finite_positive(logn_iii.N) &&
        finite_positive(logn_iv.N) && finite_positive(logn_v.N) &&
        std::isfinite(logn_v.sigma);

    const bool ok =
        all_valid && finite_ok && case_i_ok && case_ii_ok &&
        case_iii_ok && case_iv_ok && case_v_ok;

    auto print_outcome = [](const char* name, const Outcome& o)
    {
        std::cout << "    " << name
                  << ": valid=" << o.valid
                  << " N=" << o.N
                  << " fv=" << o.fv
                  << " S=" << o.S
                  << " dpp=" << o.dpp
                  << " npp=" << o.npp
                  << " sigma=" << o.sigma
                  << " M=" << o.M << "\n";
    };

    std::cout << "\n=== MetalOxide lognormal paper 0-D cases ===\n";
    print_outcome("mono-i", mono_i);
    print_outcome("logn-i", logn_i);
    print_outcome("logn-ii", logn_ii);
    print_outcome("logn-iii", logn_iii);
    print_outcome("logn-iv", logn_iv);
    print_outcome("logn-v", logn_v);
    std::cout << (all_valid
                      ? "  [PASS] All five paper-style 0-D cases produced valid closure states\n"
                      : "  [FAIL] At least one paper-style 0-D case produced an invalid state\n");
    std::cout << (case_i_ok
                      ? "  [PASS] Case i reproduces pure-agglomeration trends\n"
                      : "  [FAIL] Case i pure-agglomeration trends are inconsistent\n");
    std::cout << (case_ii_ok
                      ? "  [PASS] Case ii reproduces predominant-sintering trends\n"
                      : "  [FAIL] Case ii predominant-sintering trends are inconsistent\n");
    std::cout << (case_iii_ok
                      ? "  [PASS] Case iii reproduces competing agglomeration/sintering trends\n"
                      : "  [FAIL] Case iii competing agglomeration/sintering trends are inconsistent\n");
    std::cout << (case_iv_ok
                      ? "  [PASS] Case iv reproduces condensation with strong sintering trends\n"
                      : "  [FAIL] Case iv condensation/strong-sintering trends are inconsistent\n");
    std::cout << (case_v_ok
                      ? "  [PASS] Case v reproduces condensation with weak sintering trends\n"
                      : "  [FAIL] Case v condensation/weak-sintering trends are inconsistent\n");

    return ok;
}

static bool validateMetalOxideLognormalCoagulation()
{
    using Access = MOM::detail::MetalOxideTestAccess<MOM::BasicThermoData>;

    const auto th = buildMetalOxideThermo();
    const auto Y  = X2Y({0.050, 0.950}, th);

    constexpr double T = 1500.;
    constexpr double P = 101325.;
    constexpr double R = 8314.46261815324;
    constexpr double kB = 1.380649e-23;
    constexpr double pi = 3.141592653589793238462643383279502884;
    constexpr double epsCoag = 2.2;
    constexpr double Df = 1.8;
    constexpr double mu = 4.e-5;

    const double invMW = std::inner_product(
        Y.begin(),
        Y.end(),
        th.mw.begin(),
        0.,
        std::plus<>(),
        [](double y, double mw) { return y / mw; });
    const double rho = P / (R * T) / invMW;

    auto near = [](double value, double expected, double rtol = 1.e-12)
    {
        const double scale = std::max({1., std::abs(value), std::abs(expected)});
        return std::abs(value - expected) <= rtol * scale;
    };

    bool correction_ok = false;
    bool source_ok = false;
    bool spherical_limit_ok = false;

    const double interpolated_M25_sigma03 =
        1.004260069 + (0.30 - 0.25) / (0.50 - 0.25) * (1.017865773 - 1.004260069);
    correction_ok =
        near(Access::LognormalCoagulationIntegralCorrection(2., 0.), 1.) &&
        near(Access::LognormalCoagulationIntegralCorrection(2.5, 0.3),
             interpolated_M25_sigma03) &&
        near(Access::LognormalCoagulationIntegralCorrection(1.5, 2.0), 1.409752344) &&
        Access::LognormalCoagulationIntegralCorrection(2.5, -1.) == 0.;

    try
    {
        MOM::MetalOxide<MOM::BasicThermoData>::Config cfg;
        cfg.precursor_species = "TiOH4";
        cfg.nucleation_model = "none";
        cfg.coagulation_model = 1;
        cfg.condensation_model = 0;
        cfg.sintering_model = 0;
        cfg.closure_model = "lognormal";

        MOM::MetalOxide<MOM::BasicThermoData> model(th);
        model.SetupFromConfig(cfg);
        model.SetViscosity(mu);
        model.SetState(T, P, Y.data());

        const double N = 5.e17;
        const double vmean = 4096. * model.v0();
        const double smean = model.s0() * std::pow(vmean / model.v0(), 0.75);
        const double fv = N * vmean;
        const double solid_mass_fraction = model.solid_density() / rho * fv;
        model.SetMoments(solid_mass_fraction, N / model.ScalingFactorNs(), N * smean);

        const auto d = Access::BuildLognormalClosureData(model);

        const double d0 = std::pow(6. / pi * model.v0(), 1. / 3.);
        const double dcSafe = std::max(d.dc_mean, d0);
        const double beta_fm =
            epsCoag * std::sqrt(pi * kB * T / (2. * model.solid_density())) *
            std::sqrt(2. / d.vmean) * std::pow(2. * dcSafe, 2.);

        const double mGas = rho * kB * T / P;
        const double lambdaGas = mu / rho * std::sqrt(pi * mGas / (2. * kB * T));
        const double Cu = 1. + 2.154 * lambdaGas / dcSafe;
        const double beta_cont = 8. * kB * T / (3. * mu) * Cu * dcSafe;
        const double beta_coag = 1.82 * std::max(beta_fm, beta_cont);
        const double lognormal_correction =
            std::exp(2. * (3. / Df - 1.) * std::log(d.Kmean)) *
            Access::LognormalCoagulationIntegralCorrection(d.M, d.sigma);
        const double expected = -0.5 * beta_coag * d.N * d.N *
                                lognormal_correction / model.ScalingFactorNs();

        model.ComputeSources();
        const auto coagulation = model.sources_coagulation();
        const auto total = model.sources();

        source_ok =
            d.valid &&
            expected < 0. &&
            coagulation.size() == 3u &&
            coagulation[0] == 0. &&
            near(coagulation[1], expected) &&
            coagulation[2] == 0. &&
            near(total[1], expected);
    }
    catch (const std::exception&)
    {
        source_ok = false;
    }

    try
    {
        MOM::MetalOxide<MOM::BasicThermoData>::Config base_cfg;
        base_cfg.precursor_species = "TiOH4";
        base_cfg.nucleation_model = "none";
        base_cfg.coagulation_model = 1;
        base_cfg.condensation_model = 0;
        base_cfg.sintering_model = 0;

        auto lognormal_cfg = base_cfg;
        lognormal_cfg.closure_model = "lognormal";

        MOM::MetalOxide<MOM::BasicThermoData> mono(th);
        mono.SetupFromConfig(base_cfg);
        mono.SetViscosity(mu);
        mono.SetState(T, P, Y.data());

        MOM::MetalOxide<MOM::BasicThermoData> logn(th);
        logn.SetupFromConfig(lognormal_cfg);
        logn.SetViscosity(mu);
        logn.SetState(T, P, Y.data());

        const double N = 1.e18;
        const double vmean = 8. * mono.v0();
        const double smean = mono.s0() * std::pow(vmean / mono.v0(), 2. / 3.);
        const double fv = N * vmean;
        const double solid_mass_fraction = mono.solid_density() / rho * fv;

        mono.SetMoments(solid_mass_fraction, N / mono.ScalingFactorNs(), N * smean);
        logn.SetMoments(solid_mass_fraction, N / logn.ScalingFactorNs(), N * smean);
        mono.ComputeSources();
        logn.ComputeSources();

        const auto mono_coag = mono.sources_coagulation();
        const auto logn_coag = logn.sources_coagulation();
        spherical_limit_ok =
            mono_coag.size() == logn_coag.size() &&
            mono_coag[0] == logn_coag[0] &&
            near(mono_coag[1], logn_coag[1]) &&
            mono_coag[2] == logn_coag[2];
    }
    catch (const std::exception&)
    {
        spherical_limit_ok = false;
    }

    const bool ok = correction_ok && source_ok && spherical_limit_ok;

    std::cout << "\n=== MetalOxide lognormal coagulation source ===\n";
    std::cout << (correction_ok
                      ? "  [PASS] Lognormal coagulation integral correction is guarded\n"
                      : "  [FAIL] Lognormal coagulation correction helper is inconsistent\n");
    std::cout << (source_ok
                      ? "  [PASS] Lognormal coagulation matches analytical correction expression\n"
                      : "  [FAIL] Lognormal coagulation source is inconsistent\n");
    std::cout << (spherical_limit_ok
                      ? "  [PASS] Lognormal coagulation recovers monodisperse spherical limit\n"
                      : "  [FAIL] Lognormal coagulation does not recover spherical limit\n");

    return ok;
}

static bool validateMetalOxideLognormalCondensation()
{
    const auto th = buildMetalOxideThermo();
    const auto Y  = X2Y({0.050, 0.950}, th);

    constexpr double T = 1500.;
    constexpr double P = 101325.;
    constexpr double R = 8314.46261815324;
    constexpr double kB = 1.380649e-23;
    constexpr double NavKmol = 6.02214076e26;
    constexpr double pi = 3.141592653589793238462643383279502884;
    constexpr double epsCond = 1.3;
    constexpr double chi = -0.2043;
    constexpr double Df = 1.8;

    const double invMW = std::inner_product(
        Y.begin(),
        Y.end(),
        th.mw.begin(),
        0.,
        std::plus<>(),
        [](double y, double mw) { return y / mw; });
    const double rho = P / (R * T) / invMW;

    auto near = [](double value, double expected, double rtol = 1.e-12)
    {
        const double scale = std::max({1., std::abs(value), std::abs(expected)});
        return std::abs(value - expected) <= rtol * scale;
    };

    auto pow_positive = [](double base, double exponent)
    {
        if (!std::isfinite(base) || !std::isfinite(exponent) || base <= 0.)
            return 0.;
        return std::exp(std::clamp(exponent * std::log(base), -700., 700.));
    };

    bool source_ok = false;
    bool spherical_limit_ok = false;

    try
    {
        MOM::MetalOxide<MOM::BasicThermoData>::Config cfg;
        cfg.precursor_species = "TiOH4";
        cfg.nucleation_model = "none";
        cfg.coagulation_model = 0;
        cfg.condensation_model = 1;
        cfg.sintering_model = 0;
        cfg.closure_model = "lognormal";

        MOM::MetalOxide<MOM::BasicThermoData> model(th);
        model.SetupFromConfig(cfg);
        model.SetState(T, P, Y.data());

        const double N = 5.e17;
        const double vmean = 4096. * model.v0();
        const double smean = model.s0() * std::pow(vmean / model.v0(), 0.75);
        const double fv = N * vmean;
        const double solid_mass_fraction = model.solid_density() / rho * fv;
        model.SetMoments(solid_mass_fraction, N / model.ScalingFactorNs(), N * smean);

        const auto d =
            MOM::detail::MetalOxideTestAccess<MOM::BasicThermoData>::BuildLognormalClosureData(model);

        const double solidDensity = model.solid_density();
        const double vprec = 79.866 / NavKmol / solidDensity;
        const double dprec = std::pow(6. / pi * (th.mw[0] / NavKmol / solidDensity), 1. / 3.);
        const double Nprec = 0.050 * P / (R * T) * NavKmol;

        const double morphology = d.M / 3.;
        const double Kdc = pow_positive(d.Kmean, 3. / Df - 1.);
        const double a_dc = 1. - morphology + (d.M - 2.) / Df;
        const double collision_moment =
            d.dc_mean * d.dc_mean * Kdc * Kdc *
                MOM::detail::MetalOxideTestAccess<MOM::BasicThermoData>::
                    DimensionlessLognormalMoment(2. * a_dc, d.sigma) +
            2. * dprec * d.dc_mean * Kdc *
                MOM::detail::MetalOxideTestAccess<MOM::BasicThermoData>::
                    DimensionlessLognormalMoment(a_dc, d.sigma) +
            dprec * dprec;

        const double delta_s_mean =
            (2. / 3.) * (vprec / d.vmean) * d.smean * pow_positive(std::max(d.npp_mean, 1.), chi);
        const double Kdelta = pow_positive(d.Kmean, 1. + 3. * chi);
        const double a_delta = morphology - 1. + chi * (d.M - 2.);
        const double surface_moment =
            d.dc_mean * d.dc_mean * Kdelta * Kdc * Kdc *
                MOM::detail::MetalOxideTestAccess<MOM::BasicThermoData>::
                    DimensionlessLognormalMoment(a_delta + 2. * a_dc, d.sigma) +
            2. * dprec * d.dc_mean * Kdelta * Kdc *
                MOM::detail::MetalOxideTestAccess<MOM::BasicThermoData>::
                    DimensionlessLognormalMoment(a_delta + a_dc, d.sigma) +
            dprec * dprec * Kdelta *
                MOM::detail::MetalOxideTestAccess<MOM::BasicThermoData>::
                    DimensionlessLognormalMoment(a_delta, d.sigma);

        const double prefactor =
            epsCond * std::sqrt(pi * kB * T / (2. * solidDensity)) *
            std::sqrt(1. / vprec + 1. / d.vmean) * Nprec * d.N;

        const double expected0 = solidDensity / rho * vprec * prefactor * collision_moment;
        const double expected2 = delta_s_mean * prefactor * surface_moment;

        model.ComputeSources();
        const auto condensation = model.sources_condensation();
        const auto total = model.sources();

        source_ok =
            d.valid &&
            expected0 > 0. &&
            expected2 > 0. &&
            condensation.size() == 3u &&
            near(condensation[0], expected0) &&
            condensation[1] == 0. &&
            near(condensation[2], expected2) &&
            near(total[0], expected0) &&
            near(total[2], expected2);
    }
    catch (const std::exception&)
    {
        source_ok = false;
    }

    try
    {
        MOM::MetalOxide<MOM::BasicThermoData>::Config base_cfg;
        base_cfg.precursor_species = "TiOH4";
        base_cfg.nucleation_model = "none";
        base_cfg.coagulation_model = 0;
        base_cfg.condensation_model = 1;
        base_cfg.sintering_model = 0;

        auto lognormal_cfg = base_cfg;
        lognormal_cfg.closure_model = "lognormal";

        MOM::MetalOxide<MOM::BasicThermoData> mono(th);
        mono.SetupFromConfig(base_cfg);
        mono.SetState(T, P, Y.data());

        MOM::MetalOxide<MOM::BasicThermoData> logn(th);
        logn.SetupFromConfig(lognormal_cfg);
        logn.SetState(T, P, Y.data());

        const double N = 1.e18;
        const double vmean = 8. * mono.v0();
        const double smean = mono.s0() * std::pow(vmean / mono.v0(), 2. / 3.);
        const double fv = N * vmean;
        const double solid_mass_fraction = mono.solid_density() / rho * fv;

        mono.SetMoments(solid_mass_fraction, N / mono.ScalingFactorNs(), N * smean);
        logn.SetMoments(solid_mass_fraction, N / logn.ScalingFactorNs(), N * smean);
        mono.ComputeSources();
        logn.ComputeSources();

        const auto mono_cond = mono.sources_condensation();
        const auto logn_cond = logn.sources_condensation();
        spherical_limit_ok =
            mono_cond.size() == logn_cond.size() &&
            near(mono_cond[0], logn_cond[0]) &&
            mono_cond[1] == logn_cond[1] &&
            near(mono_cond[2], logn_cond[2]);
    }
    catch (const std::exception&)
    {
        spherical_limit_ok = false;
    }

    const bool ok = source_ok && spherical_limit_ok;

    std::cout << "\n=== MetalOxide lognormal condensation source ===\n";
    std::cout << (source_ok
                      ? "  [PASS] Lognormal condensation matches analytical closure expression\n"
                      : "  [FAIL] Lognormal condensation source is inconsistent\n");
    std::cout << (spherical_limit_ok
                      ? "  [PASS] Lognormal condensation recovers monodisperse spherical limit\n"
                      : "  [FAIL] Lognormal condensation does not recover spherical limit\n");

    return ok;
}

static bool validateMetalOxideLognormalSintering()
{
    const auto th = buildMetalOxideThermo();
    const auto Y  = X2Y({0.050, 0.950}, th);

    constexpr double T = 1500.;
    constexpr double P = 101325.;
    constexpr double R = 8314.46261815324;

    const double invMW = std::inner_product(
        Y.begin(),
        Y.end(),
        th.mw.begin(),
        0.,
        std::plus<>(),
        [](double y, double mw) { return y / mw; });
    const double rho = P / (R * T) / invMW;

    auto near = [](double value, double expected, double rtol = 1.e-12)
    {
        const double scale = std::max({1., std::abs(value), std::abs(expected)});
        return std::abs(value - expected) <= rtol * scale;
    };

    bool source_ok = false;
    bool deferred_rejected = false;

    try
    {
        MOM::MetalOxide<MOM::BasicThermoData>::Config cfg;
        cfg.precursor_species = "TiOH4";
        cfg.nucleation_model = "none";
        cfg.coagulation_model = 0;
        cfg.condensation_model = 0;
        cfg.sintering_model = 1;
        cfg.closure_model = "lognormal";

        MOM::MetalOxide<MOM::BasicThermoData> model(th);
        model.SetupFromConfig(cfg);
        model.SetState(T, P, Y.data());

        const double N = 5.e17;
        const double vmean = 4096. * model.v0();
        const double smean = model.s0() * std::pow(vmean / model.v0(), 0.75);
        const double fv = N * vmean;
        const double solid_mass_fraction = model.solid_density() / rho * fv;
        model.SetMoments(solid_mass_fraction, N / model.ScalingFactorNs(), N * smean);

        const auto d =
            MOM::detail::MetalOxideTestAccess<MOM::BasicThermoData>::BuildLognormalClosureData(model);

        constexpr double dp_exponent = 4.;
        const double morphology = d.M / 3.;
        const double eta_surface_exponent =
            morphology + dp_exponent * (morphology - 1.);
        const double eta_sphere_exponent =
            2. / 3. + dp_exponent * (morphology - 1.);

        const double expected =
            -(d.Sp * std::pow(d.Kmean, dp_exponent + 1.) *
                  MOM::detail::MetalOxideTestAccess<MOM::BasicThermoData>::
                      DimensionlessLognormalMoment(eta_surface_exponent, d.sigma)
              - d.N * d.ssph_mean * std::pow(d.Kmean, dp_exponent) *
                    MOM::detail::MetalOxideTestAccess<MOM::BasicThermoData>::
                        DimensionlessLognormalMoment(eta_sphere_exponent, d.sigma))
            / std::max(d.tau_s_mean, cfg.sintering_tau_min_s);

        model.ComputeSources();
        const auto sintering = model.sources_sintering();
        const auto total = model.sources();

        source_ok =
            d.valid &&
            expected < 0. &&
            sintering.size() == 3u &&
            sintering[0] == 0. &&
            sintering[1] == 0. &&
            near(sintering[2], expected) &&
            near(total[2], expected);
    }
    catch (const std::exception&)
    {
        source_ok = false;
    }

    try
    {
        MOM::MetalOxide<MOM::BasicThermoData>::Config cfg;
        cfg.closure_model = "lognormal";
        cfg.sintering_deferred = true;

        MOM::MetalOxide<MOM::BasicThermoData> model(th);
        model.SetupFromConfig(cfg);
    }
    catch (const std::invalid_argument&)
    {
        deferred_rejected = true;
    }
    catch (...)
    {
        deferred_rejected = false;
    }

    const bool ok = source_ok && deferred_rejected;

    std::cout << "\n=== MetalOxide lognormal sintering source ===\n";
    std::cout << (source_ok
                      ? "  [PASS] Lognormal sintering matches analytical closure expression\n"
                      : "  [FAIL] Lognormal sintering source is inconsistent\n");
    std::cout << (deferred_rejected
                      ? "  [PASS] Lognormal closure rejects deferred sintering for now\n"
                      : "  [FAIL] Lognormal closure accepted unsupported deferred sintering\n");

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
    all_ok &= validateMetalOxideMonodisperseClosureRegression();
    all_ok &= validateMetalOxideLognormalClosureState();
    all_ok &= validateMetalOxideDimensionlessLognormalMoment();
    all_ok &= validateMetalOxideLognormalLimitingCases();
    all_ok &= validateMetalOxideLognormalPaper0DCases();
    all_ok &= validateMetalOxideLognormalCoagulation();
    all_ok &= validateMetalOxideLognormalCondensation();
    all_ok &= validateMetalOxideLognormalSintering();

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
        model.SetState(T_soot, P_atm, Y_soot.data());
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
        model.SetState(T_soot, P_atm, Y_soot.data());
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
        model.SetState(T_soot, P_atm, Y_soot.data());
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
        model.SetState(T_metaloxide, P_atm, Y_metaloxide.data());
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
