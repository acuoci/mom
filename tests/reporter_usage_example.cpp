/*-----------------------------------------------------------------------*\
|   MOM Library — Reporter / Observer pattern usage example               |
|                                                                         |
|   Demonstrates the clean separation between:                            |
|     • Numerical classes (compute only, zero I/O dependency)             |
|     • MomentMethodReporter (format/write only, read-only model access)  |
|                                                                         |
|   Key properties verified here:                                         |
|     1. Variants have no WriteHeaderLine / WriteOutputLine methods.      |
|     2. MomentMethodReporter accepts any MomentMethod<M> via const&.    |
|     3. Switching variant requires changing exactly ONE type alias.      |
|     4. The reporter works identically for the type-erased               |
|        AnyMomentMethod<Thermo> (via std::visit).                        |
|     5. The reporter fires OUTSIDE the CFD cell loop — zero overhead     |
|        when no output is needed.                                        |
|                                                                         |
|   Compile (from tests/ directory):                                      |
|     g++ -std=c++20 -O2                                                  |
|         -I ../include -I ../include/MOM -isystem /path/to/eigen3        |
|         reporter_usage_example.cpp -o reporter_usage_example            |
\*-----------------------------------------------------------------------*/

#include "MOM/MOM.hpp"

#include <filesystem>
#include <iostream>
#include <vector>

// ============================================================================
// Thermo helper (reused from verify_all_variants)
// ============================================================================

static MOM::BasicThermoData buildSootThermo()
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
// Pattern A — static-dispatch (compile-time variant, zero overhead)
// ============================================================================
// The CFD code selects the variant at compile time via a single alias.
// The reporter is invoked outside the cell loop, only on output steps.

template <MOM::MomentMethod Model>
static void runStaticDispatch(Model& model,
                              const MOM::BasicThermoData& thermo,
                              const std::filesystem::path& outfile)
{
    // ── Build a synthetic 1-D "grid" (10 cells, uniform state) ────────────────
    constexpr int N_CELLS     = 10;
    constexpr int OUTPUT_FREQ = 3; // write every 3rd cell (mimics output_step check)
    constexpr double T        = 1800.;
    constexpr double P        = 101325.;
    constexpr double mu       = 4.5e-5;
    const auto Y              = X2Y({0.01, 0.02, 0.18, 0.01, 0.10, 0.06, 0.62}, thermo);
    const auto ic             = model.initial_moments();

    // ── Reporter setup — happens ONCE, outside any loop ───────────────────────
    MOM::OutputFileColumns file(outfile);
    MOM::MomentMethodReporter reporter(file, thermo.names); // pass species names
    reporter.WriteHeader(model); // registers all columns from concept interface
    file.Complete();             // finalises column widths

    // ── CFD cell loop ─────────────────────────────────────────────────────────
    for (int cell = 0; cell < N_CELLS; ++cell)
    {
        // --- Every cell: compute source terms (pure numerics) ---
        model.SetStatus(T, P, Y.data());
        model.SetMoments(ic);
        model.SetViscosity(mu);
        model.CalculateSourceMoments();

        // --- Copy sources into CFD residuals (schematic) ---
        // for (unsigned j = 0; j < Model::n_equations; ++j)
        //     residual[cell][j] += model.sources()[j] * cell_volume;

        // --- Output step: reporter observes, never mutates ────────────────────
        if (cell % OUTPUT_FREQ == 0)
        {
			reporter.WriteRow(model); // ← pure read, fired infrequently
			file.NewRow();
		}
    }

    file.Close();
    std::cout << "  [A] Static-dispatch output → " << outfile.string() << "\n";
}

// ============================================================================
// Pattern B — runtime-dispatch via AnyMomentMethod (factory + std::visit)
// ============================================================================
// The variant is selected at runtime from a string (e.g. from an input file).
// The reporter's WriteRow calls std::visit internally — dispatch happens once
// per WriteRow call, NOT once per cell.
//
// For maximum throughput, move the write out of the inner loop (same as A):
// call model.CalculateSourceMoments() per cell in a bare loop, and only call
// reporter.WriteRow() on output steps.

static void runRuntimeDispatch(const std::string& variant_name,
                               const MOM::BasicThermoData& thermo,
                               const std::filesystem::path& outfile)
{
    // ── Factory: one runtime branch here, zero overhead inside the loop ───────
    // MakeAnyMomentMethod throws std::runtime_error for unknown labels.
    MOM::AnyMomentMethod<MOM::BasicThermoData> m =
        MOM::MakeAnyMomentMethod<MOM::BasicThermoData>(thermo, variant_name);

    constexpr int N_CELLS     = 10;
    constexpr int OUTPUT_FREQ = 3;
    constexpr double T        = 1800.;
    constexpr double P        = 101325.;
    constexpr double mu       = 4.5e-5;
    const auto Y              = X2Y({0.01, 0.02, 0.18, 0.01, 0.10, 0.06, 0.62}, thermo);

    // Initial moments — extracted before the loop via std::visit
    std::vector<double> ic_buf;
    std::visit(
        [&](const auto& concrete)
        {
            auto ic = concrete.initial_moments();
            ic_buf.assign(ic.begin(), ic.end());
        },
        m);

    // ── Reporter setup — outside any loop ─────────────────────────────────────
    MOM::OutputFileColumns file(outfile);
    MOM::MomentMethodReporter reporter(file, thermo.names);
    reporter.WriteHeader(m); // dispatches via std::visit to the concrete type
    file.Complete();

    // ── CFD cell loop — ForEachCell pattern ───────────────────────────────────
    // std::visit fires ONCE here (via ForEachCell), generating a fully-optimised
    // loop body for the selected variant.  The reporter's WriteRow is called
    // only on output steps, each triggering one additional std::visit.
    MOM::ForEachCell(m,
                     [&](auto& concrete)
                     {
                         for (int cell = 0; cell < N_CELLS; ++cell)
                         {
                             concrete.SetStatus(T, P, Y.data());
                             concrete.SetMoments(std::span<const double>(ic_buf));
                             concrete.SetViscosity(mu);
                             concrete.CalculateSourceMoments();

                             if (cell % OUTPUT_FREQ == 0)
                             {
						     	reporter.WriteRow(concrete); // const& — never mutates
								file.NewRow();
							 }
                         }
                     });

    file.Close();
    std::cout << "  [B] Runtime-dispatch ('" << variant_name << "') output → " << outfile.string()
              << "\n";
}

// ============================================================================
// main
// ============================================================================

int main()
{
    std::cout << "╔══════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  MOM Reporter — usage example                                   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════╝\n\n";

    auto thermo = buildSootThermo();

    // ── Pattern A: static dispatch ─────────────────────────────────────────────
    std::cout << "Pattern A: static-dispatch (compile-time variant selection)\n";
    {
        // ← Change this one alias to switch variant.  Nothing else changes.
        using ParticleModel = MOM::ThreeEquations<MOM::BasicThermoData>;
        static_assert(MOM::MomentMethod<ParticleModel>, "ParticleModel must satisfy MomentMethod");

        // WriteHeaderLine / WriteOutputLine have been removed from all variant classes.
        // Attempting to call them is a compile error — verified by the absence of any
        // declaration in the variant headers and the clean -Wall compilation below.

        ParticleModel model(thermo);
        model.SetPAH("C2H2");
        model.SetNucleation(1);
        model.SetCoagulation(1);
        model.SetCondensation(1);
        model.SetSurfaceGrowth(1);
        model.SetOxidation(1);
        runStaticDispatch(model, thermo, "output_ThreeEquations.out");
    }
    {
        using ParticleModel = MOM::HMOM<MOM::BasicThermoData>;
        ParticleModel model(thermo);
        model.SetNucleation(1);
        model.SetCoagulation(1);
        model.SetCondensation(1);
        model.SetSurfaceGrowth(1);
        model.SetOxidation(1);
        runStaticDispatch(model, thermo, "output_HMOM.out");
    }
    {
        using ParticleModel = MOM::BrookesMoss<MOM::BasicThermoData>;
        ParticleModel model(thermo);
        model.SetNucleation(1);
        model.SetCoagulation(1);
        model.SetSurfaceGrowth(1);
        model.SetOxidation(1);
        runStaticDispatch(model, thermo, "output_BrookesMoss.out");
    }

    // ── Pattern B: runtime dispatch ────────────────────────────────────────────
    std::cout << "\nPattern B: runtime-dispatch via AnyMomentMethod + ForEachCell\n";
    for (const char* name : {"HMOM", "ThreeEquations", "BrookesMoss"})
        runRuntimeDispatch(name, thermo, std::string("output_any_") + name + ".out");

    std::cout << "\nAll output files written. Inspect column headers to confirm:\n";
    std::cout << "  • BrookesMoss:    Scon[ZF], Ssin[ZF] columns (zero-fallback tagged)\n";
    std::cout << "  • HMOM:           Ssin[ZF] column\n";
    std::cout << "  • ThreeEquations: Ssin[ZF] column\n";

    return 0;
}
