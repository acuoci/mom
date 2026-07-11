/*-----------------------------------------------------------------------*\
|    ___                   ____  __  __  ___  _  _______                  |
|   / _ \ _ __   ___ _ __ / ___||  \/  |/ _ \| |/ / ____| _     _         |
|  | | | | '_ \ / _ \ '_ \\___ \| |\/| | | | | ' /|  _| _| |_ _| |_       |
|  | |_| | |_) |  __/ | | |___) | |  | | |_| | . \| |__|_   _|_   _|      |
|   \___/| .__/ \___|_| |_|____/|_|  |_|\___/|_|\_\_____||_|   |_|        |
|        |_|                                                              |
|                                                                         |
|   Author: Alberto Cuoci <alberto.cuoci@polimi.it>                       |
|   CRECK Modeling Lab <https://www.creckmodeling.polimi.it>              |
|   Department of Chemistry, Materials, and Chemical Engineering          |
|   Politecnico di Milano                                                 |
|   P.zza Leonardo da Vinci 32, 20133 Milano                              |
|                                                                         |
|-------------------------------------------------------------------------|
|                                                                         |
|   This file is part of the OpenSMOKEpp library.                         |
|                                                                         |
|   Copyright (C) 2026 Alberto Cuoci.                                     |
|                                                                         |
|   OpenSMOKEpp is free software: you can redistribute it and/or modify   |
|   it under the terms of the GNU General Public License as published by  |
|   the Free Software Foundation, either version 3 of the License, or     |
|   (at your option) any later version.                                   |
|                                                                         |
|   OpenSMOKEpp is distributed in the hope that it will be useful,        |
|   but WITHOUT ANY WARRANTY; without even the implied warranty of        |
|   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         |
|   GNU General Public License for more details.                          |
|                                                                         |
|   You should have received a copy of the GNU General Public License     |
|   along with OpenSMOKEpp. If not, see <https://www.gnu.org/licenses/>.  |
|                                                                         |
\*-----------------------------------------------------------------------*/

#pragma once

// -- Standard library ---------------------------------------------------------
#include <exception>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#if defined(MOM_USE_DICTIONARY)
#include <expected>
#endif

// -- Project headers -----------------------------------------------------------
// MomVariantList.hpp is the single authoritative registry of all variants.
// It transitively provides: all variant headers, MomentMethodConcept.hpp,
// detail::TypeList, AllVariants, and <variant>.
#include "MomVariantList.hpp"

namespace MOM
{

// ============================================================================
// AnyMomentMethod<Thermo> — runtime-selectable moment method
// ============================================================================
//
// A std::variant over all four concrete moment method types. Enables
// runtime selection of the method from an input file or command-line flag
// without duplicating CFD solver code.
//
// Dispatch is via std::visit, which generates a jump-table (one indirect
// branch) — comparable to a vtable call but without aliasing penalties that
// prevent auto-vectorisation of the source computation loop.
//
// USAGE:
//
//   // Construction (method chosen at runtime):
//   MOM::AnyMomentMethod<MyThermo> model =
//       MOM::MakeAnyMomentMethod<MyThermo>(thermo, "HMOM");
//
//   // Preferred per-cell call — one std::visit for the full update:
//   MOM::ComputeCell(model, T, P, Y, mu, cell_moments);
//   auto src = MOM::GetSources(model);       // std::span<const double>
//   auto n   = MOM::GetNEquations(model);    // unsigned
//
//   // Individual setters remain available when only part of the state changes:
//   MOM::SetState(model, T, P, Y);
//   MOM::SetMoments(model, cell_moments);
//   MOM::Compute(model);
//
// ============================================================================

// -- AnyMomentMethod<Thermo> — derived automatically from AllVariants ----------
//
// Expands to std::variant<HMOM<Thermo>, BrookesMoss<Thermo>, ...> with the
// exact set of types registered in MomVariantList.hpp::AllVariants.
// No manual edit required here when adding a new variant.

template <ThermoMap Thermo>
using AnyMomentMethod = typename AllVariants::template AsVariant<Thermo>;

// ============================================================================
// detail::FactoryHelper — compile-time recursive label dispatcher
// ============================================================================
//
// Iterates the registered variants at compile time, checking each type's
// variant_labels member against the runtime label string.  Falls through
// recursively until a match is found or the list is exhausted.
//
// Derived from AllVariants in MakeAnyMomentMethod, so it automatically
// covers every registered variant without manual if-chains.
// ============================================================================

namespace detail
{

/**
 * @brief Base case — empty variant list; no label matched.
 *
 * Throws `std::invalid_argument` with a diagnostic message pointing to
 * `MomVariantList.hpp` for the list of registered variants.
 */
template <template <typename> class... Vs> struct FactoryHelper
{
    template <typename Thermo>
    [[noreturn]] static AnyMomentMethod<Thermo> make(const Thermo&, std::string_view label)
    {
        throw std::invalid_argument(
            "MOM::MakeAnyMomentMethod: unknown method label '" + std::string(label) +
            "'. See MomVariantList.hpp for registered variants and their labels.");
    }
};

/**
 * @brief Recursive case — try `First`'s labels, then recurse into `Rest...`.
 *
 * Iterates `First<Thermo>::variant_labels` at compile time.  On a match,
 * constructs an `AnyMomentMethod<Thermo>` in-place holding `First<Thermo>`.
 * On no match, delegates to `FactoryHelper<Rest...>`.
 */
template <template <typename> class First, template <typename> class... Rest>
struct FactoryHelper<First, Rest...>
{
    template <typename Thermo>
    static AnyMomentMethod<Thermo> make(const Thermo& thermo, std::string_view label)
    {
        for (std::string_view lbl : First<Thermo>::variant_labels)
            if (lbl == label)
                return AnyMomentMethod<Thermo>{std::in_place_type<First<Thermo>>, thermo};
        return FactoryHelper<Rest...>::template make<Thermo>(thermo, label);
    }
};

/**
 * @brief Unpacks `TypeList<Vs...>` → `FactoryHelper<Vs...>::make`.
 *
 * Allows `MakeAnyMomentMethod` to delegate to the correct `FactoryHelper`
 * specialisation without naming the individual variant types explicitly.
 */
template <template <typename> class... Vs, typename Thermo>
inline AnyMomentMethod<Thermo> make_from_type_list(TypeList<Vs...>,
                                                   const Thermo& thermo,
                                                   std::string_view label)
{
    return FactoryHelper<Vs...>::template make<Thermo>(thermo, label);
}

} // namespace detail

/**
 * @brief Construct an `AnyMomentMethod` holding the variant matching @p label.
 *
 * Performs an exact, case-sensitive match against each registered variant's
 * `variant_labels` static member (defined in `MomVariantList.hpp`).
 *
 * @par Example
 * @code
 *   MOM::AnyMomentMethod<MyThermo> model =
 *       MOM::MakeAnyMomentMethod<MyThermo>(thermo, "HMOM");
 * @endcode
 *
 * @tparam Thermo    Must satisfy `ThermoMap`.
 * @param  thermo    Thermodynamics object — must outlive the returned variant.
 * @param  label     Any label registered in a variant's `variant_labels` member.
 * @return `AnyMomentMethod<Thermo>` holding the matched concrete type.
 * @throws std::invalid_argument if @p label is not recognised.
 */
template <ThermoMap Thermo>
[[nodiscard]] AnyMomentMethod<Thermo> MakeAnyMomentMethod(const Thermo& thermo,
                                                          std::string_view label);

/**
 * @brief Deleted overload — prevents a temporary Thermo from being silently bound.
 *
 * `const Thermo&&` is NOT a forwarding reference (it is const-qualified), so it
 * matches rvalues and moved objects but never plain lvalues.  The live lvalue
 * overload above therefore remains reachable for all valid call sites.
 *
 * @note Use this pattern to diagnose mistakes early:
 * @code
 *   // BAD  — thermo lifetime ends before the returned variant is used:
 *   auto m = MOM::MakeAnyMomentMethod(MOM::BasicThermoData{...}, "HMOM"); // deleted ✓
 *
 *   // GOOD — thermo outlives the variant:
 *   MOM::BasicThermoData thermo{...};
 *   auto m = MOM::MakeAnyMomentMethod(thermo, "HMOM");                    // OK ✓
 * @endcode
 */
template <ThermoMap Thermo>
AnyMomentMethod<Thermo> MakeAnyMomentMethod(const Thermo&&, std::string_view) = delete;

/**
 * @name Runtime dispatch helpers for AnyMomentMethod
 *
 * These free functions reproduce the full `MomentMethod` concept interface as
 * `std::visit` wrappers operating on `AnyMomentMethod<Thermo>`.  Each carries
 * O(1) dispatch cost — one indirect branch via the `std::variant` jump table.
 *
 * Use these when the concrete type is not known at compile time (e.g. the method
 * is chosen from an input file).  When the type IS known statically, prefer the
 * direct template interface or `MomentMethod`-constrained overloads — the
 * compiler can then inline and vectorise the full computation.
 * @{
 */

/** @brief Returns the number of transported equations for the active method. */
template <ThermoMap Thermo>
[[nodiscard]] inline unsigned GetNEquations(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) -> unsigned { return mm.n_equations; }, m);
}

/**
 * @brief Injects thermodynamic state into the active method.
 * @param T    Temperature [K].
 * @param P_Pa Pressure [Pa].
 * @param Y    Species mass fractions (pointer, size = n_species).
 */
template <ThermoMap Thermo>
inline void SetState(AnyMomentMethod<Thermo>& m, double T, double P_Pa, const double* Y) noexcept
{
    std::visit([&](auto& mm) { mm.SetStatus(T, P_Pa, Y); }, m);
}

/** @brief Sets the mixture dynamic viscosity [kg/m/s]. */
template <ThermoMap Thermo> inline void SetViscosity(AnyMomentMethod<Thermo>& m, double mu) noexcept
{
    std::visit([mu](auto& mm) { mm.SetViscosity(mu); }, m);
}

/** @brief Sets current moment values from a contiguous span of size `n_equations`. */
template <ThermoMap Thermo>
inline void SetMoments(AnyMomentMethod<Thermo>& m, std::span<const double> moments) noexcept
{
    std::visit([moments](auto& mm) { mm.SetMoments(moments); }, m);
}

/** @brief Computes all source terms and gas-phase consumption. */
template <ThermoMap Thermo> inline void Compute(AnyMomentMethod<Thermo>& m)
{
    std::visit([](auto& mm) { mm.CalculateSourceMoments(); }, m);
}

/**
 * @brief Single-call per-cell entry point for the runtime path.
 *
 * Collapses `SetState` / `SetMoments` / `SetViscosity` / `Compute` into a
 * single `std::visit`, reducing jump-table dispatches from four to one.
 *
 * @note For the compile-time path (type known statically), use the
 *       `MomentMethod`-constrained `ComputeCell` overload in
 *       `MomentMethodConcept.hpp` — it allows full inlining.
 */
template <ThermoMap Thermo>
inline void ComputeCell(AnyMomentMethod<Thermo>& m,
                        double T,
                        double P_Pa,
                        const double* Y,
                        double mu,
                        std::span<const double> moments) noexcept
{
    std::visit(
        [T, P_Pa, Y, mu, moments](auto& mm) noexcept
        {
            mm.SetStatus(T, P_Pa, Y);
            mm.SetMoments(moments);
            mm.SetViscosity(mu);
            mm.CalculateSourceMoments();
        },
        m);
}

/**
 * @brief Hoist variant dispatch outside the cell loop for maximum performance.
 *
 * Calls `std::visit` **once** before the cell loop, then invokes @p callback
 * with the concrete model type as argument.  The callback owns the loop, so all
 * N cells are processed with the same concrete type — zero jump-table overhead
 * inside the hot loop.
 *
 * @code
 *   MOM::ForEachCell(model, [&](auto& m) {
 *       // typeof(m) is, e.g., ThreeEquations<Thermo>&
 *       for (int i = 0; i < n_cells; ++i) {
 *           MOM::ComputeCell(m, T[i], P[i], Y + i*ns, mu[i], M + i*neq);
 *           auto src = m.sources();
 *           std::copy(src.begin(), src.end(), Src + i*neq);
 *       }
 *   });
 * @endcode
 *
 * The compiler generates one fully-optimised loop body per registered variant
 * and selects among them via a single indirect branch, which the branch
 * predictor collapses to zero overhead after the first call.
 *
 * @tparam Thermo       Must satisfy `ThermoMap`.
 * @tparam CellCallback Callable with signature `void(ConcreteModel&)`.
 * @param  m            The runtime-polymorphic model instance.
 * @param  callback     Function receiving the concrete model reference.
 */
template <ThermoMap Thermo, typename CellCallback>
inline void ForEachCell(AnyMomentMethod<Thermo>& m, CellCallback&& callback)
{
    std::visit(std::forward<CellCallback>(callback), m);
}

#if defined(MOM_USE_DICTIONARY)
/**
 * @brief Configure the active variant by parsing the given dictionary.
 *
 * Dispatches to the concrete variant's `ParseConfig(dict)` static member
 * template, then calls `SetupFromConfig()` on success.
 *
 * @tparam Dictionary  OpenSMOKE++ dictionary type (deduced).
 *
 * @throws std::runtime_error if the dictionary grammar check fails (missing
 *         mandatory keyword, unknown keyword, duplicate keyword), if a keyword
 *         value is invalid, or if `SetupFromConfig` raises an exception.
 *         Dictionary errors are always fatal — the solver must not continue
 *         with silently-defaulted parameters.
 */
template <ThermoMap Thermo, typename Dictionary>
inline void SetupFromDictionary(AnyMomentMethod<Thermo>& m, Dictionary& dict)
{
    std::visit(
        [&dict](auto& mm)
        {
            // ParseConfig validates the grammar and reads all keyword values.
            // On failure it returns std::unexpected with a diagnostic string.
            auto cfg = mm.ParseConfig(dict);
            if (!cfg)
                throw std::runtime_error(cfg.error());

            // Apply the validated configuration.
            mm.SetupFromConfig(*cfg);
        },
        m);
}
#endif

/** @brief Returns the thermophoretic model flag (0 = off, 1 = standard). */
template <ThermoMap Thermo>
[[nodiscard]] inline int GetThermophoreticModel(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.thermophoretic_model(); }, m);
}

/** @brief Returns `true` if gas-phase precursor consumption is enabled. */
template <ThermoMap Thermo>
[[nodiscard]] inline bool GetGasConsumption(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.gas_consumption(); }, m);
}

/**
 * @brief Returns a zero-copy span over the total source vector [mol/m³/s].
 * @return Span of size `n_equations`; valid until next `CalculateSourceMoments()`.
 */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double> GetSources(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.sources(); }, m);
}

/**
 * @brief Returns a zero-copy span over the **oxidation-only** source vector.
 *
 * The span points directly into the model's internal `source_oxidation_` storage,
 * so the call is zero-overhead.  Use together with GetSourcesWithoutOxidation()
 * and GetOxidationRateCoefficients() to implement operator splitting.
 *
 * @pre  Compute() must have been called at the current state.
 * @return Span of size `n_equations`; valid until next Compute().
 */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double>
GetOxidationSources(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.sources_oxidation(); }, m);
}

/**
 * @name Per-process activation queries
 *
 * Return the integer model flag for each physical process (0 = off, >0 = active variant).
 * These route through the `model_X()` CRTP dispatchers in `MomentMethodBase`, so the
 * return value is always 0 for models that do not implement that process (e.g.
 * `GetSinteringModel` returns 0 for all soot models; `GetCondensationModel` returns 0
 * for BrookesMoss).
 *
 * Typical CFD usage:
 * @code
 *   if (MOM::GetOxidationModel(mom_) > 0) {
 *       // oxidation is active — apply operator splitting
 *   }
 *   if (MOM::GetSinteringModel(mom_) > 0) {
 *       // sintering sources are available
 *       auto s = MOM::GetSinteringSources(mom_);
 *   }
 * @endcode
 * @{
 */

/** @brief Returns the nucleation model flag (0 = off, >0 = active variant). */
template <ThermoMap Thermo>
[[nodiscard]] inline int GetNucleationModel(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.model_nucleation(); }, m);
}

/** @brief Returns the surface-growth model flag (0 = off, >0 = active variant). */
template <ThermoMap Thermo>
[[nodiscard]] inline int GetGrowthModel(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.model_growth(); }, m);
}

/** @brief Returns the coagulation model flag (0 = off, >0 = active variant). */
template <ThermoMap Thermo>
[[nodiscard]] inline int GetCoagulationModel(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.model_coagulation(); }, m);
}

/** @brief Returns the condensation model flag (0 = off, >0 = active variant). */
template <ThermoMap Thermo>
[[nodiscard]] inline int GetCondensationModel(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.model_condensation(); }, m);
}

/** @brief Returns the oxidation model flag (0 = off, >0 = active variant). */
template <ThermoMap Thermo>
[[nodiscard]] inline int GetOxidationModel(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.model_oxidation(); }, m);
}

/** @brief Returns the sintering model flag (0 = off, >0 = active; MetalOxide only). */
template <ThermoMap Thermo>
[[nodiscard]] inline int GetSinteringModel(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.model_sintering(); }, m);
}

/** @} */

/**
 * @name Per-process source term accessors
 *
 * Zero-copy spans into the model's internal per-process source storage.
 * For models that do not implement a given process, the base-class CRTP fallback
 * returns a span over a static constexpr zero array — no runtime branch, no
 * allocation, same size as `n_equations`.
 *
 * Always check the corresponding `GetXxxModel() > 0` before consuming these
 * values to distinguish "process off" from "process on but rate happens to be zero".
 * @{
 */

/** @brief Zero-copy span over the nucleation source terms [model-specific units]. */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double>
GetNucleationSources(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.sources_nucleation(); }, m);
}

/** @brief Zero-copy span over the surface-growth source terms. */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double>
GetGrowthSources(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.sources_growth(); }, m);
}

/** @brief Zero-copy span over the coagulation source terms. */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double>
GetCoagulationSources(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.sources_coagulation(); }, m);
}

/** @brief Zero-copy span over the condensation source terms. */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double>
GetCondensationSources(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.sources_condensation(); }, m);
}

/** @brief Zero-copy span over the sintering source terms (MetalOxide only). */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double>
GetSinteringSources(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.sources_sintering(); }, m);
}

/** @} */

/**
 * @brief Writes `source_all[i] - source_oxidation[i]` into @p out for each moment.
 *
 * For operator splitting of the stiff oxidation terms: pass these reduced sources
 * to the stiff ODE solver instead of the full GetSources() vector, then apply the
 * oxidation sub-step analytically with GetOxidationRateCoefficients().
 *
 * @par Operator-splitting usage pattern
 * @code
 *   // --- inside Equations() ---
 *   MOM::Compute(mom);
 *   double src_buf[MAX_MOM];
 *   MOM::GetSourcesWithoutOxidation(mom, {src_buf, n_mom});
 *   for (int i = 0; i < n_mom; ++i)
 *       dy(NC + 1 + i) = src_buf[i];   // stiff-safe: no oxidation eigenvalue
 *
 *   // --- after the outer CFD timestep dt has completed ---
 *   double kappa[MAX_MOM], mvals[MAX_MOM];
 *   for (int i = 0; i < n_mom; ++i) mvals[i] = M[i];
 *   MOM::GetOxidationRateCoefficients(mom, {mvals, n_mom}, {kappa, n_mom});
 *   for (int i = 0; i < n_mom; ++i)
 *       M[i] *= std::exp(-kappa[i] * dt);   // exact first-order decay
 * @endcode
 *
 * @pre  Compute() must have been called at the current state.
 * @param[in]  m    Model variant.
 * @param[out] out  Caller-allocated buffer; size must be >= n_equations().
 */
template <ThermoMap Thermo>
inline void GetSourcesWithoutOxidation(const AnyMomentMethod<Thermo>& m,
                                        std::span<double> out) noexcept
{
    std::visit([&out](const auto& mm)
    {
        const auto total = mm.sources();           // source_all_ — zero-copy
        const auto ox    = mm.sources_oxidation(); // source_oxidation_ — zero-copy
        const std::size_t N = std::min(total.size(), out.size());
        for (std::size_t i = 0; i < N; ++i)
            out[i] = total[i] - ox[i];
    }, m);
}

/**
 * @brief Computes the effective first-order oxidation rate coefficient [1/s] per moment.
 *
 * Linearises the (generally nonlinear) oxidation depletion as a first-order decay:
 *
 *   κ_i = max(-source_oxidation_[i], 0) / max(|M_i|, ε)
 *
 * The exact analytical solution for the oxidation sub-step then reads:
 *
 *   M_i(t + Δt) = M_i(t) · exp(−κ_i · Δt)
 *
 * This is unconditionally stable regardless of how large κ_i is — it removes the
 * stiff oxidation eigenvalue from the ODE system entirely.
 *
 * @note  κ_i is evaluated at the **current** stored state (the last Compute() call).
 *        For Lie–Trotter splitting, call this after the ODE step has finished.
 *        For the symmetric Strang splitting, call it at half-time t + Δt/2.
 *
 * @pre   Compute() must have been called at the current state.
 * @param[in]  m                Model variant.
 * @param[in]  current_moments  Transported moment values M_i as they appear in the
 *                              ODE state vector.  Size must be >= n_equations().
 * @param[out] kappa_out        Output buffer for κ_i [1/s].  Size >= n_equations().
 */
template <ThermoMap Thermo>
inline void GetOxidationRateCoefficients(const AnyMomentMethod<Thermo>& m,
                                          std::span<const double>        current_moments,
                                          std::span<double>              kappa_out) noexcept
{
    std::visit([&current_moments, &kappa_out](const auto& mm)
    {
        const auto ox = mm.sources_oxidation();
        const std::size_t N =
            std::min({ox.size(), current_moments.size(), kappa_out.size()});
        constexpr double eps = 1.e-300;
        for (std::size_t i = 0; i < N; ++i)
        {
            // Oxidation is a sink: source_ox[i] ≤ 0.  Rate coeff is non-negative.
            const double neg_src = -ox[i];
            const double M       = std::max(std::abs(current_moments[i]), eps);
            kappa_out[i] = (neg_src > 0.) ? neg_src / M : 0.;
        }
    }, m);
}

/** @brief Returns a zero-copy span over gas-phase consumption terms [kg/m³/s]. */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double> GetOmegaGas(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.omega_gas(); }, m);
}

/**
 * @brief Returns a zero-copy span over the **oxidation-only** gas-phase source vector [kg/m³/s].
 *
 * Points directly into internal storage — zero-overhead.
 * Empty span for models without oxidation gas coupling (TiO2/MetalOxide).
 *
 * @pre Compute() must have been called at the current state.
 */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double>
GetOmegaGasOxidation(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.omega_gas_oxidation(); }, m);
}

/**
 * @brief Writes `omega_gas[k] − omega_gas_oxidation[k]` into @p out for each species.
 *
 * For operator splitting: pass these reduced gas sources inside `Equations()`,
 * then apply the oxidation sub-step analytically after the ODE step completes.
 *
 * @par Post-step gas-phase correction
 * After the outer CFD timestep @p dt has completed and the soot moments have been
 * updated with the exponential decay, apply the integrated oxidation effect on gas
 * species using a correction factor that accounts for the decreasing oxidation rate
 * as soot is consumed:
 *
 * @code
 *   // kappa_eff = effective soot-mass oxidation rate [1/s] from GetOxidationRateCoefficients()
 *   //             (use the mass-fraction moment index: 0 for ThreeEquations, 1 for HMOM)
 *   const double x = kappa_eff * dt;
 *   const double phi = (x > 1e-8) ? (1. - std::exp(-x)) / x : 1.;
 *   auto ox_gas = MOM::GetOmegaGasOxidation(mom_);
 *   for (int k = 0; k < n_species; ++k)
 *       Y[k] += ox_gas[k] / rho * dt * phi;
 * @endcode
 *
 * The factor `phi` ∈ (0, 1] collapses to 1 for small `kappa_eff * dt` (first-order
 * approximation) and falls toward `1/(kappa_eff*dt)` for fast oxidation, correctly
 * bounding the integrated gas consumption to at most the available reactant mass.
 *
 * @pre  Compute() must have been called at the current state.
 * @param[in]  m    Model variant.
 * @param[out] out  Caller-allocated buffer; size must be >= n_species.
 */
template <ThermoMap Thermo>
inline void GetOmegaGasWithoutOxidation(const AnyMomentMethod<Thermo>& m,
                                         std::span<double> out) noexcept
{
    std::visit([&out](const auto& mm)
    {
        const auto total = mm.omega_gas();            // full omega_gas_  — zero-copy
        const auto ox    = mm.omega_gas_oxidation();  // oxidation-only   — zero-copy
        const std::size_t N = std::min(total.size(), out.size());
        for (std::size_t k = 0; k < N; ++k)
            out[k] = total[k] - (k < ox.size() ? ox[k] : 0.);
    }, m);
}

/** @brief Returns the particle volume fraction [-]. */
template <ThermoMap Thermo>
[[nodiscard]] inline double GetVolumeFraction(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.volume_fraction(); }, m);
}

/** @brief Returns the mean primary particle diameter [m]. */
template <ThermoMap Thermo>
[[nodiscard]] inline double GetParticleDiameter(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.particle_diameter(); }, m);
}

/** @brief Returns the total particle number density [#/m³]. */
template <ThermoMap Thermo>
[[nodiscard]] inline double GetParticleNumberDensity(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.particle_number_density(); }, m);
}

/** @brief Returns the particle mass fraction [-]. */
template <ThermoMap Thermo>
[[nodiscard]] inline double GetMassFraction(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.mass_fraction(); }, m);
}

/** @brief Returns the specific surface area [m2/m3]. */
template <ThermoMap Thermo>
[[nodiscard]] inline double GetSpecificSurfaceArea(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.specific_surface(); }, m);
}

/** @brief Returns the collision diameter [m]. */
template <ThermoMap Thermo>
[[nodiscard]] inline double GetCollisionDiameter(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.collision_diameter(); }, m);
}

/** @brief Returns the number of primary particles [-]. */
template <ThermoMap Thermo>
[[nodiscard]] inline double GetNumberPrimaryParticles(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.number_primary_particles(); }, m);
}

/** @brief Returns the particle Schmidt number [-]. */
template <ThermoMap Thermo>
[[nodiscard]] inline double GetSchmidtNumber(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.schmidt_number(); }, m);
}

/** @brief Returns the effective particle diffusion coefficient [kg/m/s]. */
template <ThermoMap Thermo>
[[nodiscard]] inline double GetDiffusionCoefficient(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.diffusion_coefficient(); }, m);
}

/** @brief Returns `true` if the Planck absorption coefficient should be included in radiation. */
template <ThermoMap Thermo>
[[nodiscard]] inline bool GetRadiativeHeatTransfer(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.radiative_heat_transfer(); }, m);
}

/**
 * @brief Returns the Planck mean absorption coefficient of the particle phase [1/m].
 * @param T  Gas temperature [K].
 * @param fv Particle volume fraction [-].
 */
template <ThermoMap Thermo>
[[nodiscard]] inline double GetPlanckCoefficient(const AnyMomentMethod<Thermo>& m,
                                                 double T,
                                                 double fv) noexcept
{
    return std::visit([T, fv](const auto& mm) { return mm.planck_coefficient(T, fv); }, m);
}

/**
 * @brief Returns the initial moment values for solver initialisation.
 * @return Span of size `n_equations` with near-zero seed values.
 */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double> GetInitialMoments(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.initial_moments(); }, m);
}

/** @brief Returns `true` if a gas-closure dummy species has been configured. */
template <ThermoMap Thermo>
[[nodiscard]] inline bool GetClosureDummySpeciesIsActive(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.is_closure_dummy_species(); }, m);
}

/** @brief Returns `true` if the object is active. */
template <ThermoMap Thermo>
[[nodiscard]] inline bool GetIsActive(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.is_active(); }, m);
}

/** @brief Returns the 0-based index of the gas-closure dummy species (−1 if inactive). */
template <ThermoMap Thermo>
[[nodiscard]] inline int GetClosureDummyIndex(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.closure_dummy_index(); }, m);
}

/** @brief Returns the 0-based precursor species index in the thermo map (−1 if unset). */
template <ThermoMap Thermo>
[[nodiscard]] inline int GetPrecursorIndex(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.precursor_index(); }, m);
}

/** @brief Prints the active variant's model configuration to stdout. */
template <ThermoMap Thermo> inline void PrintSummary(const AnyMomentMethod<Thermo>& m)
{
    std::visit([](const auto& mm) { mm.PrintSummary(); }, m);
}

/** @} */

} // namespace MOM

// AnyMomentMethod<Thermo> is a thin std::variant alias — its only template body
// is the MakeAnyMomentMethod factory (AnyMomentMethod.tpp).
// That factory is trivially inlined and does not benefit from pre-compilation,
// so we always include it (no extern template needed here).
#include "AnyMomentMethod.tpp"
