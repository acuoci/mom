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
 * All concrete variants (HMOM, BrookesMoss, ThreeEquations, MetalOxide) derive from
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
        { cm.number_primary_particles() } -> std::same_as<double>;
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
 * @brief Satisfied by models that provide a continuous NDF reconstruction.
 *
 * Currently satisfied by ThreeEquations, MetalOxide, and HMOM (smeared
 * bimodal log-normal).  Not satisfied by BrookesMoss, which carries no NDF
 * reconstruction.
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

// ============================================================================
// Process-capability concepts
// ============================================================================
//
// These concepts formally encode which physical sub-processes a variant
// actively models.  They are used by MomentMethodReporter::WriteHeader to
// apply [ZF] ("zero-fallback") tags to columns whose values are always zero
// for a particular variant — alerting post-processing tools that the column
// is structurally empty, not just transiently zero.
//
// Detection mechanism
// -------------------
// Each variant signals that it models process X by declaring a public
// `sources_X_impl()` method, which MomentMethodBase detects via CRTP to
// route the `sources_X()` accessor.  Variants that do NOT model X omit the
// `_impl()` method; MomentMethodBase then returns a compile-time zero span.
//
// These concepts are the single, named, documented detection point for that
// distinction.  They replace ad-hoc inline requires-expressions scattered
// across the reporter, making the detection logic:
//   1. Centralized — defined here, used everywhere.
//   2. Stable       — if the `_impl()` naming convention ever changes, only
//                     these definitions need updating, not every consumer.
//   3. Readable     — `ModelsNucleation<M>` is self-documenting; an inline
//                     requires-expression is not.
//
// Concept matrix (as of 2026-07)
// ---------------------------------------------------------------
//   Concept              | HMOM | BrookesMoss | ThreeEq | MetalOxide
//   ModelsNucleation     |  Y   |      Y      |    Y    |     Y
//   ModelsCoagulation    |  Y   |      Y      |    Y    |     Y
//   ModelsCondensation   |  Y   |      N      |    Y    |     Y
//   ModelsSurfaceGrowth  |  Y   |      Y      |    Y    |     N
//   ModelsOxidation      |  Y   |      Y      |    Y    |     N
//   ModelsSintering      |  N   |      N      |    N    |     Y
// ---------------------------------------------------------------

/**
 * @concept ModelsNucleation
 * @brief Satisfied by variants that compute and own nucleation source terms.
 *
 * A variant satisfies this concept when it declares `sources_nucleation_impl()`,
 * signalling to MomentMethodBase that `sources_nucleation()` should return the
 * variant's own data rather than a static zero span.
 *
 * Satisfied by: HMOM, BrookesMoss, ThreeEquations, MetalOxide.
 */
template <typename M>
concept ModelsNucleation =
    requires(const M& cm) {
        { cm.sources_nucleation_impl() } -> std::convertible_to<std::span<const double>>;
    };

/**
 * @concept ModelsCoagulation
 * @brief Satisfied by variants that compute and own coagulation source terms.
 *
 * Satisfied by: HMOM, BrookesMoss, ThreeEquations, MetalOxide.
 */
template <typename M>
concept ModelsCoagulation =
    requires(const M& cm) {
        { cm.sources_coagulation_impl() } -> std::convertible_to<std::span<const double>>;
    };

/**
 * @concept ModelsCondensation
 * @brief Satisfied by variants that compute and own condensation source terms.
 *
 * Satisfied by: HMOM, ThreeEquations, MetalOxide.
 * NOT satisfied by: BrookesMoss (condensation not modelled; zero span returned).
 */
template <typename M>
concept ModelsCondensation =
    requires(const M& cm) {
        { cm.sources_condensation_impl() } -> std::convertible_to<std::span<const double>>;
    };

/**
 * @concept ModelsSurfaceGrowth
 * @brief Satisfied by variants that compute and own surface-growth source terms.
 *
 * Satisfied by: HMOM, BrookesMoss, ThreeEquations.
 * NOT satisfied by: MetalOxide (surface growth not modelled; zero span returned).
 */
template <typename M>
concept ModelsSurfaceGrowth =
    requires(const M& cm) {
        { cm.sources_growth_impl() } -> std::convertible_to<std::span<const double>>;
    };

/**
 * @concept ModelsOxidation
 * @brief Satisfied by variants that compute and own oxidation source terms.
 *
 * Satisfied by: HMOM, BrookesMoss, ThreeEquations.
 * NOT satisfied by: MetalOxide (particles are already fully oxidised;
 *                   zero span returned).
 */
template <typename M>
concept ModelsOxidation =
    requires(const M& cm) {
        { cm.sources_oxidation_impl() } -> std::convertible_to<std::span<const double>>;
    };

/**
 * @concept ModelsSintering
 * @brief Satisfied by variants that compute and own sintering source terms.
 *
 * Satisfied by: MetalOxide.
 * NOT satisfied by: HMOM, BrookesMoss, ThreeEquations (zero span returned).
 */
template <typename M>
concept ModelsSintering =
    requires(const M& cm) {
        { cm.sources_sintering_impl() } -> std::convertible_to<std::span<const double>>;
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
