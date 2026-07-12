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
 * `AllVariants` defines the alternatives of `AnyMomentMethod<Thermo>`, the
 * labels searched by `MakeAnyMomentMethod()`, and the compile-time concept gate
 * instantiated by `MOM.hpp`.
 *
 * @note To register a new variant, include its header here and add its class
 *       template to `AllVariants`. The variant must satisfy `MomentMethod`.
 */

// -- Standard library ---------------------------------------------------------
#include <variant>

// -- Infrastructure ------------------------------------------------------------
#include "MomentMethodConcept.hpp"

// -- Registered variant headers ------------------------------------------------
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
     * @brief Compile-time concept gate for registered variants.
     *
     * @tparam TestThermo  Any type satisfying `ThermoMap` (typically `BasicThermoData`).
     */
    template <typename TestThermo> struct ConceptCheck
    {
        /**
         * @brief Per-variant concept check.
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

        // Forces one concept check per registered variant.
        std::tuple<CheckOne<Variants>...> check_all{};
    };
};

} // namespace detail

/**
 * @brief Canonical registry of all registered MOM variants.
 *
 * Every type listed here is available through `AnyMomentMethod<Thermo>` and
 * `MakeAnyMomentMethod()`.
 */
using AllVariants = detail::TypeList<HMOM, BrookesMoss, ThreeEquations, MetalOxide>;

} // namespace MOM
