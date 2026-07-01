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

#include <concepts>
#include <span>

namespace MOM
{

/**
 * @concept MomentMethod
 * @brief Authoritative public contract for all Method of Moments particle models.
 *
 * Any class @p M satisfying `MomentMethod<M>` can be used interchangeably by a
 * CFD solver without modification beyond changing a single type alias:
 *
 * @code
 *   using ParticleModel = MOM::ThreeEquations<MyThermo>;  // ← change only this
 *   static_assert(MOM::MomentMethod<ParticleModel>);
 * @endcode
 *
 * All concrete variants (HMOM, BrookesMoss, ThreeEquations, TiO2) derive from
 * MomentMethodBase<Derived, NEq> and are checked against this concept at compile
 * time.  The concept is intentionally free of implementation details — it specifies
 * only observable behaviour.
 *
 * @par Concept requirements summary
 *
 * **Compile-time:**
 * - `M::n_equations` — unsigned, number of transported moment equations
 *
 * **State injection** (call before CalculateSourceMoments each cell/time-step):
 * - `SetStatus(T, P, Y[])` — thermodynamic state (T [K], P [Pa], Y mass fractions)
 * - `SetMoments(span)` — current moment values
 * - `SetViscosity(mu)` — mixture dynamic viscosity [kg/m/s]
 *
 * **Core computation:**
 * - `CalculateSourceMoments()` — evaluates all source terms (noexcept)
 * - `CalculateOmegaGas()` — evaluates gas-phase consumption terms only (noexcept)
 *
 * **Source output** (zero-copy spans into internal fixed-size storage):
 * - `sources()` — total source vector, size = n_equations
 * - `sources_nucleation()` — nucleation contribution
 * - `sources_coagulation()` — coagulation contribution
 * - `sources_condensation()` — condensation contribution (zero span if not modelled)
 * - `sources_growth()` — surface growth contribution (zero span if not modelled)
 * - `sources_oxidation()` — oxidation contribution (zero span if not modelled)
 * - `sources_sintering()` — sintering contribution (zero span if not modelled)
 * - `omega_gas()` — gas-phase species source terms [kg/m3/s], size = n_species
 *
 * **Particle properties** (derived from current moment values):
 * - `volume_fraction()` — particle volume fraction [-]
 * - `particle_diameter()` — primary particle diameter [m]
 * - `collision_diameter()` — collision (aggregate) diameter [m]
 * - `particle_number_density()` — number density [#/m3]
 * - `mass_fraction()` — particle mass fraction [-]
 * - `ParticleDensity()` — material density [kg/m3]
 * - `specific_surface()` — surface area per unit volume [m2/m3]
 *
 * **Transport:**
 * - `schmidt_number()` — particle Schmidt number [-]
 * - `diffusion_coefficient()` — effective diffusion coefficient [kg/m/s]
 * - `thermophoretic_model()` — thermophoretic model flag (int, 0 = off)
 *
 * **Status / control:**
 * - `is_active()` — true if the method is configured and active
 * - `gas_consumption()` — true if gas-phase consumption is enabled
 * - `initial_moments()` — initialisation values for the moment transport equations
 *
 * **Gas coupling:**
 * - `precursor_index()` — 0-based index of precursor species in the thermo map
 * - `precursor_concentration()` — precursor molar concentration [kmol/m3]
 * - `is_closure_dummy_species_()` — true if a dummy closure species is configured
 * - `closure_dummy_index()` — 0-based index of the dummy closure species
 *
 * **Radiative heat transfer:**
 * - `radiative_heat_transfer()` — true if particles contribute to radiative loss
 * - `planck_coefficient(T, fv)` — Planck mean absorption coefficient [1/m]
 *
 * **Diagnostics:**
 * - `PrintSummary()` — prints model configuration to stdout
 */

template <typename M>
concept MomentMethod =

    // Compile-time constant: number of transported equations
    requires {
        { M::n_equations } -> std::convertible_to<unsigned>;
    }

    &&
    requires(M m, const M cm, double scalar, const double* Y_ptr, std::span<const double> moments_in) {
        // -- State injection ------------------------------------------------
        // Y_ptr is a properly-typed local variable; no null-pointer cast needed.
        // noexcept is required: these setters run every cell iteration.
        { m.SetStatus(scalar, scalar, Y_ptr) } noexcept; // T [K], P [Pa], Y[]
        { m.SetMoments(moments_in) } noexcept;
        { m.SetViscosity(scalar) } noexcept;

        // -- Core computation -----------------------------------------------
        // noexcept is part of the contract: these run in the CFD inner loop
        // and must not carry exception-handling overhead or prevent hoisting.
        { m.CalculateSourceMoments() } noexcept;
        { m.CalculateOmegaGas() } noexcept;

        // -- Source output (zero-copy spans) --------------------------------
        { cm.sources() } -> std::convertible_to<std::span<const double>>;
        { cm.sources_nucleation() } -> std::convertible_to<std::span<const double>>;
        { cm.sources_coagulation() } -> std::convertible_to<std::span<const double>>;
        { cm.sources_condensation() } -> std::convertible_to<std::span<const double>>;
        { cm.sources_growth() } -> std::convertible_to<std::span<const double>>;
        { cm.sources_oxidation() } -> std::convertible_to<std::span<const double>>;
        { cm.sources_sintering() } -> std::convertible_to<std::span<const double>>;
        { cm.omega_gas() } -> std::convertible_to<std::span<const double>>;

        // -- Particle properties --------------------------------------------
        { cm.volume_fraction() } -> std::same_as<double>;
        { cm.particle_diameter() } -> std::same_as<double>;
        { cm.collision_diameter() } -> std::same_as<double>;
        { cm.particle_number_density() } -> std::same_as<double>;
        { cm.mass_fraction() } -> std::same_as<double>;
        { cm.particle_density() } -> std::same_as<double>;
        { cm.specific_surface() } -> std::same_as<double>;

        // -- Transport ------------------------------------------------------
        { cm.schmidt_number() } -> std::same_as<double>;
        { cm.diffusion_coefficient() } -> std::same_as<double>;
        { cm.thermophoretic_model() } -> std::same_as<int>;

        // -- Status / control -----------------------------------------------
        { cm.is_active() } -> std::same_as<bool>;
        { cm.gas_consumption() } -> std::same_as<bool>;
        { cm.initial_moments() } -> std::convertible_to<std::span<const double>>;

        // -- Gas coupling ---------------------------------------------------
        { cm.precursor_index() } -> std::same_as<int>;
        { cm.precursor_concentration() } -> std::same_as<double>;
        { cm.is_closure_dummy_species() } -> std::same_as<bool>;
        { cm.closure_dummy_index() } -> std::same_as<int>;

        // -- Radiative heat transfer ----------------------------------------
        { cm.radiative_heat_transfer() } -> std::same_as<bool>;
        { cm.planck_coefficient(scalar, scalar) } -> std::same_as<double>;

        // -- Diagnostics ----------------------------------------------------
        { cm.PrintSummary() };
    };

/**
 * @concept HasReconstructedNDF
 * @brief Satisfied by models that provide a Pareto + log-normal NDF reconstruction.
 *
 * Currently satisfied by ThreeEquations and TiO2.  Not satisfied by HMOM or
 * BrookesMoss, which do not reconstruct the particle size distribution.
 *
 * Used by MomentMethodReporter::WriteReconstructedNDF to conditionally enable
 * NDF output for capable variants without any modification to the reporter or
 * to non-NDF variants.
 */
template <typename M>
concept HasReconstructedNDF =
    requires(const M& cm) {
        { cm.ReconstructedNDF(0.0, false) }           -> std::same_as<double>;
        { cm.ReconstructedNormalizedNDF(0.0, false) } -> std::same_as<double>;
    };

/**
 * @brief Illustrates the "one line to switch" idiom.
 *
 * The `static_assert` fires at include time with a clear message if the
 * selected type no longer satisfies the concept (e.g. after a refactoring).
 *
 * @code
 *   // In the CFD solver:
 *   #include "MOM/MOM.hpp"
 *   using ParticleModel = MOM::HMOM<MOM::Thermo>;
 *   static_assert(MOM::MomentMethod<ParticleModel>,
 *       "ParticleModel must satisfy MOM::MomentMethod");
 * @endcode
 */

/**
 * @brief Single-call per-cell entry point for moment source computation.
 *
 * Preferred over calling SetStatus / SetMoments / SetViscosity /
 * CalculateSourceMoments individually. Bundling all four into one call lets the
 * compiler keep the object layout in registers across the full per-cell
 * computation without reloading `this` at each function boundary.
 *
 * @note This function is `inline` and `noexcept`. In a release build (-O2/-O3)
 *       the four calls are collapsed to a direct in-place computation with no
 *       function-call overhead.  Use this as the single hot-path entry point in
 *       CFD cell loops.
 *
 * @code
 *   template <MOM::MomentMethod M>
 *   void CellLoop(M& model) {
 *       for (auto& cell : cells) {
 *           MOM::ComputeCell(model, cell.T, cell.P, cell.Y.data(),
 *                            cell.mu, cell.moments);
 *           auto src = model.sources();   // zero-copy span — no allocation
 *       }
 *   }
 * @endcode
 *
 * @tparam M       Any type satisfying MomentMethod<M>.
 * @param  model   The moment method instance to update.
 * @param  T       Gas temperature [K].
 * @param  P_Pa    Gas pressure [Pa].
 * @param  Y       Species mass fractions (pointer, size = thermo.NumberOfSpecies()).
 * @param  mu      Mixture dynamic viscosity [kg/m/s].
 * @param  moments Current moment values (span of size M::n_equations).
 */
template <MomentMethod M>
inline void ComputeCell(
    M& model, double T, double P_Pa, const double* Y, double mu, std::span<const double> moments) noexcept
{
    model.SetStatus(T, P_Pa, Y);
    model.SetMoments(moments);
    model.SetViscosity(mu);
    model.CalculateSourceMoments();
}

} // namespace MOM
