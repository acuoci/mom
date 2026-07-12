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
 * @file MomVariantList.hpp
 * @brief Authoritative registry of all MOM particle-method variants.
 *
 * **This is the only file that needs to be edited when adding a new variant.**
 *
 * @par How to register a new variant `FooMOM`
 *
 * 1. Create `include/FooMOM/FooMOM.hpp` — a CRTP subclass of
 *    `MomentMethodBase<FooMOM<Thermo>, NEq>` — and add the required static member:
 *    @code
 *      static constexpr std::array<std::string_view, N> variant_labels
 *          { "FooMOM", "foomom" };
 *    @endcode
 * 2. Add one `#include` line below under "Registered variant headers".
 * 3. Add `FooMOM` to the `AllVariants` type alias at the bottom of this file.
 *
 * That is all. The `AnyMomentMethod` variant type, the `MakeAnyMomentMethod`
 * factory, and all `MomentMethod` concept checks are derived automatically from
 * `AllVariants` — no other file requires modification.
 */

// -- Standard library ---------------------------------------------------------
#include <variant>

// -- Infrastructure ------------------------------------------------------------
#include "MomentMethodConcept.hpp"

// -- Registered variant headers ------------------------------------------------
// Add one #include per new variant here.
#include "HMOM/HMOM.hpp"
#include "BrookesMoss/BrookesMoss.hpp"
#include "ThreeEquations/ThreeEquations.hpp"
#include "MetalOxide/MetalOxide.hpp"

namespace MOM
{

namespace detail
{

/**
 * @brief Compile-time list of uninstantiated variant class templates.
 *
 * Provides two operations consumed by `AnyMomentMethod` and `MakeAnyMomentMethod`:
 *
 * - `TypeList<Vs...>::AsVariant<Thermo>` expands to
 *   `std::variant<Vs<Thermo>...>`, which is the type of `AnyMomentMethod<Thermo>`.
 *
 * - `TypeList<Vs...>::ConceptCheck<TestThermo>` is a struct whose instantiation
 *   `static_assert`s that every `Vs<TestThermo>` satisfies the `MomentMethod`
 *   concept.  Triggered by the explicit instantiation in `MOM.hpp`:
 *   @code
 *     template struct MOM::AllVariants::ConceptCheck<MOM::BasicThermoData>;
 *   @endcode
 *   The fold expression fires individually for each `Vs`, so a compiler error
 *   points at the offending variant by name.
 *
 * @tparam Variants  Uninstantiated template-template parameters — one per variant.
 */
template <template <typename> class... Variants> struct TypeList
{
    /** @brief Number of registered variants (computed at compile time). */
    static constexpr std::size_t size = sizeof...(Variants);

    /**
     * @brief Instantiate as `std::variant<Variants<Thermo>...>`.
     * @tparam Thermo  Thermodynamics backend satisfying `ThermoMap`.
     */
    template <typename Thermo> using AsVariant = std::variant<Variants<Thermo>...>;

    /**
     * @brief Compile-time concept gate — verifies all registered variants.
     *
     * Instantiating `ConceptCheck<T>` triggers a `static_assert` that checks
     * every registered variant against the `MomentMethod<M>` concept.
     * If any variant fails, the error message lists the missing API members.
     *
     * @tparam TestThermo  Any type satisfying `ThermoMap` (typically `BasicThermoData`).
     */
    template <typename TestThermo> struct ConceptCheck
    {
        static_assert((MomentMethod<Variants<TestThermo>> && ...),
                      "[MOM] One or more registered variants in AllVariants do not satisfy "
                      "the MomentMethod concept. Check: SetMoments(span), "
                      "ComputeSources() noexcept, DiffusionCoefficient() const, "
                      "CollisionDiameter(), SpecificSurface(), PrintSummary().");
    };
};

} // namespace detail

/**
 * @brief Canonical registry of all registered MOM variants.
 *
 * Every type listed in `AllVariants`:
 * - Appears as an alternative in `AnyMomentMethod<Thermo>` (the runtime-selectable
 *   `std::variant` type).
 * - Is tried by `MakeAnyMomentMethod` during factory label matching, via each
 *   variant's `static constexpr variant_labels` member.
 * - Is concept-checked by the explicit `ConceptCheck` instantiation in `MOM.hpp`.
 *
 * @note **To add a new variant**: add its class template to this alias and add
 *       its `#include` above.  No other file requires modification.
 */
using AllVariants = detail::TypeList<HMOM, BrookesMoss, ThreeEquations, MetalOxide>;

} // namespace MOM
