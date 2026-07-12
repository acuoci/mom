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
 * @par How to add a new variant `FooMOM` — complete checklist
 *
 * **Step 1 — Implementation (files to create)**
 *
 * Copy `include/NewVariant/` to `include/FooMOM/` and rename every occurrence
 * of `NewVariant` → `FooMOM`.  The skeleton already satisfies the concept
 * interface — fill in the `// TODO:` blocks for your physics.  Key points:
 *
 * - Inherit from `MomentMethodBase<FooMOM<Thermo>, NEq>` with your `NEq`.
 * - Declare `variant_labels` (used by `MakeAnyMomentMethod` for factory dispatch):
 *   @code
 *     static constexpr std::array<std::string_view, N> variant_labels
 *         { "FooMOM", "foomom" };
 *   @endcode
 * - For each physical process you model, uncomment the corresponding
 *   `sources_X_impl()` in the header (see the opt-in table in the skeleton).
 *   The base class's `sources_X()` dispatchers detect `_impl()` at compile time
 *   and return a zero span for processes you do not model — zero overhead.
 * - Implement `SetState()`, `SetMoments()`, `ComputeSources()`,
 *   `diffusion_coefficient()`, all particle-property accessors, and `PrintSummary()`.
 *   See `include/NewVariant/NewVariant.tpp` for documented implementation patterns.
 *
 * **Step 2 — Explicit instantiation (src/ file)**
 *
 * Create `src/FooMOM/FooMOM.cpp`:
 * @code
 *   #include "FooMOM/FooMOM.hpp"
 *   namespace MOM { template class FooMOM<BasicThermoData>; }
 * @endcode
 * This pre-compiles the template for the standard thermo backend and prevents
 * implicit instantiation in user translation units (`extern template` in the header
 * enforces this when `MOM_USE_DICTIONARY` is defined).
 *
 * **Step 3 — Registry (this file — the only required edit)**
 *
 * a. Add `#include "FooMOM/FooMOM.hpp"` below under "Registered variant headers".
 * b. Add `FooMOM` to the `AllVariants` alias at the bottom of this file.
 *
 * That is all for this file.  The `AnyMomentMethod<Thermo>` variant type, the
 * `MakeAnyMomentMethod` factory, and the `ConceptCheck` gate are all derived
 * automatically from `AllVariants` — no other infrastructure file needs editing.
 *
 * **Step 4 — Build system**
 *
 * Add `src/FooMOM/FooMOM.cpp` to `CMakeLists.txt` under the `MOM` target sources.
 * If a separate grammar / dictionary reader is needed, add its `.cpp` there too.
 *
 * **Step 5 — Verification**
 *
 * Uncomment the `static_assert` at the bottom of `include/FooMOM/FooMOM.hpp`:
 * @code
 *   static_assert(MomentMethod<FooMOM<BasicThermoData>>,
 *       "FooMOM does not satisfy MomentMethod — see MomentMethodConcept.hpp");
 * @endcode
 * Then rebuild; any missing members are reported by name.  When the assert passes,
 * `AllVariants::ConceptCheck<BasicThermoData>` (instantiated in `MOM.hpp`) will
 * also verify `FooMOM` alongside the existing variants on every build.
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
 *   checks every `Vs<TestThermo>` against the `MomentMethod` concept, one at a time.
 *   Triggered by the explicit instantiation in `MOM.hpp`:
 *   @code
 *     template struct MOM::AllVariants::ConceptCheck<MOM::BasicThermoData>;
 *   @endcode
 *   A single fold-expression `(check_A && check_B && ...)` would short-circuit and
 *   only report the first failure.  The `CheckOne<V>` inner struct pattern avoids
 *   this: each `CheckOne<V>` carries its own `static_assert`, so the compiler
 *   instantiates ALL of them and emits one error per failing variant — each error
 *   includes the full instantiation chain, which names the offending type.
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
     * @brief Compile-time concept gate — one `static_assert` per registered variant.
     *
     * Instantiating `ConceptCheck<TestThermo>` triggers `CheckOne<V>` for every V
     * in the variant pack.  Each `CheckOne<V>` carries its own `static_assert`, so:
     *   - ALL failing variants are reported in one compilation pass (not just the first).
     *   - The compiler's template-instantiation backtrace names the failing type.
     *   - Adding a new variant to `AllVariants` automatically adds a check here.
     *
     * If a `static_assert` fires, read the instantiation chain above the error
     * to identify the variant, then consult `MomentMethodConcept.hpp` for the
     * full required-expression list.
     *
     * @tparam TestThermo  Any type satisfying `ThermoMap` (typically `BasicThermoData`).
     */
    template <typename TestThermo> struct ConceptCheck
    {
        /**
         * @brief Per-variant concept gate.
         *
         * One specialisation is instantiated per registered variant V.
         * The `static_assert` fires during instantiation, giving the compiler's
         * template backtrace a concrete type to name (V<TestThermo>).
         *
         * @tparam V  An uninstantiated variant class template (e.g. HMOM, BrookesMoss).
         */
        template <template <typename> class V> struct CheckOne
        {
            static_assert(MomentMethod<V<TestThermo>>,
                          "[MOM] A registered variant does not satisfy the MomentMethod "
                          "concept.  See the template instantiation context above this "
                          "message to identify which type failed.  Required members:\n"
                          "  Compile-time: n_equations, variant_labels\n"
                          "  Injectors:    SetState(T,P,Y), SetMoments(span), SetViscosity(mu)\n"
                          "  Computation:  ComputeSources() noexcept\n"
                          "  Sources:      sources(), sources_{nucleation,coagulation,...}(), omega_gas()\n"
                          "  Properties:   volume_fraction(), particle_diameter(), collision_diameter(),\n"
                          "                particle_number_density(), number_primary_particles(),\n"
                          "                mass_fraction(), particle_density(), specific_surface_area()\n"
                          "  Transport:    diffusion_coefficient(), schmidt_number(), thermophoretic_model()\n"
                          "  Status:       is_active(), gas_consumption(), initial_moments()\n"
                          "  Gas coupling: precursor_index(), precursor_concentration(),\n"
                          "                is_closure_dummy_species(), closure_dummy_index()\n"
                          "  Radiation:    radiative_heat_transfer(), planck_coefficient(T, fv)\n"
                          "  Diagnostics:  PrintSummary()\n"
                          "See MomentMethodConcept.hpp for the authoritative requires-expression.");
        };

        // Instantiate CheckOne<V> for every V in the Variants pack.
        // std::tuple forces the compiler to fully instantiate each CheckOne<V>,
        // which triggers its static_assert.  All checks run even when earlier
        // ones fail — no short-circuit — so every non-compliant variant is named.
        // Zero runtime cost: the tuple holds empty structs and is never stored.
        std::tuple<CheckOne<Variants>...> check_all{};
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
