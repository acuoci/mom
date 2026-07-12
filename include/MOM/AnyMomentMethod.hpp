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

/**
 * @file AnyMomentMethod.hpp
 * @brief Core type alias, factory, and cell-loop helper for AnyMomentMethod.
 *
 * This file defines exactly four things:
 *
 * 1. `AnyMomentMethod<Thermo>` — the runtime-selectable moment method type
 *    (a `std::variant` over all registered concrete variants).
 *
 * 2. `detail::FactoryHelper` — a compile-time recursive label dispatcher used
 *    exclusively by `MakeAnyMomentMethod`.
 *
 * 3. `MakeAnyMomentMethod<Thermo>(thermo, label)` — the factory function that
 *    constructs the right variant from a runtime string label.
 *
 * 4. `ForEachCell(m, callback)` — hoists the variant dispatch outside a cell
 *    loop, so the loop body executes with the concrete type statically known.
 *
 * @par Full interface
 * This file does **not** include the free-function dispatch API.  For the
 * complete public API, include `MOM/MOM.hpp`, which transitively provides:
 *
 * | Header           | Contents                                            |
 * |------------------|-----------------------------------------------------|
 * | `Dispatch.hpp`   | SetState, SetMoments, Compute, ComputeCell, …       |
 * | `Properties.hpp` | GetVolumeFraction, GetParticleDiameter, …           |
 * | `Sources.hpp`    | GetSources, GetNucleationSources, GetOxidationModel, … |
 * | `Splitting.hpp`  | GetSourcesWithoutOxidation, GetOxidationRateCoefficients, … |
 *
 * @par Adding a new variant
 * Edit `MomVariantList.hpp` only — add the new type to `AllVariants`.
 * No changes are required here.
 */

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

// -- Project headers ----------------------------------------------------------
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
// Expands to std::variant<HMOM<Thermo>, BrookesMoss<Thermo>, ...> with the
// exact set of types registered in MomVariantList.hpp::AllVariants.
// Dispatch is via std::visit, which generates a jump table (one indirect
// branch per call) — comparable to a vtable call but without aliasing
// penalties that prevent auto-vectorisation of the source computation loop.
//
// No manual edit required here when adding a new variant.  Edit
// MomVariantList.hpp::AllVariants instead.
// ============================================================================

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
// Used exclusively by MakeAnyMomentMethod below; not part of the public API.
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
 * @note Diagnoses lifetime mistakes at compile time:
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
 * @brief Hoist variant dispatch outside a cell loop for maximum performance.
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
 *           m.SetState(T[i], P[i], Y + i*ns);
 *           m.SetMoments({M + i*neq, neq});
 *           m.SetViscosity(mu[i]);
 *           m.ComputeSources();
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

} // namespace MOM

// AnyMomentMethod<Thermo> is a thin std::variant alias — its only template body
// is the MakeAnyMomentMethod factory (AnyMomentMethod.tpp).
// That factory is trivially inlined and does not benefit from pre-compilation,
// so we always include it (no extern template needed here).
#include "AnyMomentMethod.tpp"
