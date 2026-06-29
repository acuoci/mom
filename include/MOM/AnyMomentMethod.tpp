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

/**
 * @file AnyMomentMethod.tpp
 * @brief Template implementation of the MakeAnyMomentMethod factory function.
 *
 * This file is `#include`d at the **bottom** of AnyMomentMethod.hpp and must
 * never be compiled as a standalone translation unit.  Separating the body here
 * keeps the factory declaration in the header uncluttered while still allowing
 * the compiler to see the full definition wherever AnyMomentMethod.hpp is
 * included (a requirement for function templates).
 *
 * @par Factory dispatch chain
 *
 * The single line in this file invokes a compile-time recursive template:
 *
 * @code
 *   MakeAnyMomentMethod(thermo, "HMOM")
 *     └─ detail::make_from_type_list(AllVariants{}, thermo, "HMOM")
 *          └─ FactoryHelper<HMOM, BrookesMoss, ThreeEquations, TiO2>
 *                            ::make(thermo, "HMOM")
 *               ├─ HMOM::variant_labels contains "HMOM"? yes → return HMOM<Thermo>{thermo}
 *               └─ (would otherwise recurse into FactoryHelper<BrookesMoss, ...>)
 * @endcode
 *
 * `FactoryHelper` is a variadic template that unpacks `AllVariants` (the
 * canonical type registry in MomVariantList.hpp) via partial specialisation:
 *
 * - **Base case** `FactoryHelper<>` — no types left: throws `std::invalid_argument`.
 * - **Recursive case** `FactoryHelper<Head, Tail...>` — checks whether @p label
 *   appears in `Head::variant_labels` (a `std::array<std::string_view, N>`
 *   member of the concrete variant class).  If yes, constructs `Head<Thermo>`
 *   in place and returns it wrapped in the `std::variant`.  If no, delegates
 *   to `FactoryHelper<Tail...>::make(...)`.
 *
 * The entire iteration is resolved at **compile time** as template
 * instantiation — there is no runtime loop, no virtual dispatch, and no heap
 * allocation beyond the `std::variant` storage itself (which is stack-sized by
 * the compiler to the largest alternative).
 *
 * @par Extensibility
 * Adding a new variant (e.g. `MyModel`) requires:
 * 1. Appending `MyModel` to `AllVariants` in MomVariantList.hpp.
 * 2. Defining `MyModel::variant_labels` with the accepted string keys.
 *
 * No change is needed here or in AnyMomentMethod.hpp.
 *
 * @par Error path
 * If @p label matches no registered variant, `FactoryHelper<>` (the base case)
 * throws `std::invalid_argument` with a descriptive message listing the unknown
 * label.  The exception propagates through `make_from_type_list` and out of
 * `MakeAnyMomentMethod` to the caller; it is the caller's responsibility to
 * catch it (typically at solver initialisation, not in the cell loop).
 */

#pragma once

namespace MOM
{

/**
 * @brief Factory implementation — delegates to the compile-time recursive
 *        FactoryHelper unpacked from AllVariants.
 *
 * See the @ref MakeAnyMomentMethod declaration in AnyMomentMethod.hpp for the
 * full public contract (@tparam, @param, @return, @throws documentation).
 *
 * @par Why this is a separate .tpp file
 * `MakeAnyMomentMethod` is a function template: the compiler must see its
 * body at every call site.  Moving the body here (rather than leaving it
 * inline in the `.hpp`) keeps the header shorter without sacrificing
 * instantiability — both files are included together via the `#include`
 * directive at the bottom of AnyMomentMethod.hpp.
 *
 * @tparam Thermo  Thermodynamics backend satisfying the ThermoMap concept.
 * @param  thermo  Constructed thermo object; stored by reference inside the
 *                 returned variant's active alternative.
 * @param  label   Case-sensitive string key for the desired variant, e.g.
 *                 `"HMOM"`, `"BrookesMoss"`, `"ThreeEquations"`, `"TiO2"`.
 * @return         An `AnyMomentMethod<Thermo>` whose active alternative is the
 *                 variant whose `variant_labels` array contains @p label.
 * @throws std::invalid_argument  If @p label is not found in any registered
 *                                variant's `variant_labels`.
 */
template <ThermoMap Thermo>
AnyMomentMethod<Thermo> MakeAnyMomentMethod(const Thermo& thermo, std::string_view label)
{
    // Unpack AllVariants (TypeList<HMOM, BrookesMoss, ThreeEquations, TiO2>)
    // into FactoryHelper<HMOM, BrookesMoss, ThreeEquations, TiO2>::make().
    // The helper walks the type list recursively at compile time, comparing
    // `label` against each variant's variant_labels at runtime until a match
    // is found or the base-case (empty list) throws.
    return detail::make_from_type_list(AllVariants{}, thermo, label);
}

} // namespace MOM
