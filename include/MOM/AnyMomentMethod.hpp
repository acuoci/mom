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
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

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
 * @brief Configure the active variant from a key-value dictionary.
 * @return `std::expected<void, std::string>` — error string on failure, never throws.
 */
template <ThermoMap Thermo, typename Dictionary>
[[nodiscard]] inline std::expected<void, std::string> SetupFromDictionary(AnyMomentMethod<Thermo>& m,
                                                                          Dictionary& dict)
{
    return std::visit([&dict](auto& mm) -> std::expected<void, std::string>
                      { return mm.SetupFromDictionary(dict); },
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

/** @brief Returns a zero-copy span over gas-phase consumption terms [kg/m³/s]. */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double> GetOmegaGas(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.omega_gas(); }, m);
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
    return std::visit([](const auto& mm) { return mm.ClosureDummySpeciesIsActive(); }, m);
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
