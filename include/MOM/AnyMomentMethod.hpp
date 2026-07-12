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
 * @brief Runtime-selectable MOM variant type, factory, and dispatch-hoisting helper.
 *
 * `AnyMomentMethod<Thermo>` is a `std::variant` over the concrete variants
 * registered in `MomVariantList.hpp`. Use `MakeAnyMomentMethod()` when the
 * variant name is read from an input file, and use `ForEachCell()` to move the
 * runtime dispatch outside a CFD cell loop.
 *
 * @note Include `MOM/MOM.hpp` for the complete public API.
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
#include "MomVariantList.hpp"

namespace MOM
{

/**
 * @brief Runtime-selectable moment method over all registered concrete variants.
 *
 * @tparam Thermo Thermodynamics backend satisfying `ThermoMap`.
 */
template <ThermoMap Thermo>
using AnyMomentMethod = typename AllVariants::template AsVariant<Thermo>;

namespace detail
{

/**
 * @brief Factory recursion base case used when no registered label matches.
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
 * @brief Factory recursion step over the registered variant list.
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
 * @brief Builds the factory dispatcher from a registered variant type list.
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
 * Moment method objects store a reference to the thermodynamics backend, so the
 * backend must outlive the returned `AnyMomentMethod`.
 *
 * @code
 *   auto m = MOM::MakeAnyMomentMethod(MOM::BasicThermoData{...}, "HMOM"); // error
 *
 *   MOM::BasicThermoData thermo{...};
 *   auto m = MOM::MakeAnyMomentMethod(thermo, "HMOM");                    // OK
 * @endcode
 */
template <ThermoMap Thermo>
AnyMomentMethod<Thermo> MakeAnyMomentMethod(const Thermo&&, std::string_view) = delete;

/**
 * @brief Hoist variant dispatch outside a cell loop for maximum performance.
 *
 * Calls `std::visit` once before the cell loop, then invokes @p callback
 * with the concrete model type as argument.  The callback owns the loop, so all
 * cells are processed with the same statically known type.
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

#include "AnyMomentMethod.tpp"
