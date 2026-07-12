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
 * @file NewVariant.tpp
 * @brief [SKELETON] Template implementation bodies for NewVariant<Thermo>.
 *
 * Included at the bottom of NewVariant.hpp — never compiled in isolation.
 * Each function is documented with its expected implementation pattern.
 *
 * Naming conventions carried over from existing variants:
 *   - Index  -1   → species absent in the user's mechanism (treat as zero concentration)
 *   - cTot       → total molar concentration [kmol/m3]
 *   - conc_X_    → species molar concentration [kmol/m3]
 *   - source_X_  → per-process Eigen column vector, size NEq
 *   - source_all_→ total (= sum of all active processes), owned by MomentMethodBase
 */

#include <cmath>
#include <iostream>

namespace MOM
{

// =============================================================================
// State injection
// =============================================================================

/**
 * @brief Updates thermodynamic state and extracts species concentrations.
 *
 * Pattern:
 *   1. Call `UpdateMixtureState()` — updates T_, P_Pa_, rho_, MW_ in the base class.
 *   2. Extract precursor and any other species concentrations needed by the physics.
 *   3. Look up species indices once in ApplyConfig (not here); use -1 sentinel checks.
 *
 * @note `noexcept` is part of the MomentMethod contract — this runs every cell.
 *       Do not allocate memory or throw exceptions here.
 */
template <ThermoMap Thermo>
void NewVariant<Thermo>::SetState(double T, double P_Pa, const double* Y) noexcept
{
    // UpdateMixtureState computes T_, P_Pa_, MW_, rho_ from (T, P, Y[]).
    // It returns the total molar concentration cTot [kmol/m3].
    const double cTot = this->UpdateMixtureState(T, P_Pa, Y, thermo_);

    // Extract precursor concentration.
    // SpeciesConcentrationKmolM3 returns 0 when prec_index_ == -1.
    conc_prec_ = this->SpeciesConcentrationKmolM3(prec_index_, Y, cTot, thermo_);

    // TODO: extract other species concentrations your physics require, e.g.:
    // conc_c2h2_ = this->SpeciesConcentrationKmolM3(c2h2_index_, Y, cTot, thermo_);
    // conc_o2_   = this->SpeciesConcentrationKmolM3(o2_index_,   Y, cTot, thermo_);
    // conc_oh_   = this->SpeciesConcentrationKmolM3(oh_index_,   Y, cTot, thermo_);
}

// =============================================================================

/**
 * @brief Unpacks the transported moment span into your physical variables.
 *
 * The contract: `m.size() == n_equations`.  The CFD solver provides whatever
 * is stored in the moment transport fields — it is your responsibility to
 * interpret them correctly and guard against negative values if applicable.
 *
 * @note `noexcept` required — runs every cell.
 */
template <ThermoMap Thermo>
void NewVariant<Thermo>::SetMoments(std::span<const double> m) noexcept
{
    // TODO: unpack m[0], m[1], ... into your physical variables, e.g.:
    // Ys_ = std::max(m[0], 0.);   // soot mass fraction (must be ≥ 0)
    // bs_ = std::max(m[1], 0.);   // normalised nuclei concentration (must be ≥ 0)
    (void)m;   // remove once your variables are filled in
}

// =============================================================================
// Core computation
// =============================================================================

/**
 * @brief Evaluates all moment source terms for the current cell state.
 *
 * Call sequence:
 *   1. Guard: return immediately if !is_active_.
 *   2. Zero accumulators: call ZeroSources() + zero each per-process vector.
 *   3. Invoke per-process helpers (fill source_nucleation_, etc.).
 *   4. Assemble source_all_ = sum of active process vectors.
 *   5. (If gas_consumption_) compute omega_gas_.
 *
 * @note `noexcept` required — runs in the CFD inner loop.  Do not allocate
 *       memory or throw exceptions here.
 */
template <ThermoMap Thermo>
void NewVariant<Thermo>::ComputeSources() noexcept
{
    if (!this->is_active_) return;

    // -- Zero base-class accumulator and per-process vectors ------------------
    this->ZeroSources();
    // TODO: zero your per-process vectors, e.g.:
    // source_nucleation_.setZero();
    // source_coagulation_.setZero();

    // -- Per-process physics ---------------------------------------------------
    // TODO: call your sub-routines in the order that makes physical sense.
    //   SootNucleation();     // fills source_nucleation_
    //   SootCoagulation();    // fills source_coagulation_
    //   SootGrowth();         // fills source_growth_
    //   SootOxidation();      // fills source_oxidation_

    // -- Assemble total source vector -----------------------------------------
    // Add every non-zero per-process vector.
    // source_all_ is a fixed-size Eigen vector of size NEq — no heap allocation.
    // TODO: replace with your actual sub-process vectors:
    // this->source_all_ = source_nucleation_ + source_coagulation_ + ...;

    // -- Gas-phase consumption (optional) --------------------------------------
    // If this->gas_consumption_ is true, populate this->omega_gas_ with
    // per-species mass consumption rates [kg/m3/s].  Keep signs consistent:
    // negative = species consumed by particle growth; positive = released.
    //
    // if (this->gas_consumption_)
    // {
    //     this->omega_gas_.setZero();
    //     // e.g. precursor consumed during nucleation:
    //     if (prec_index_ >= 0)
    //         this->omega_gas_[prec_index_] -= nucleation_rate_ * MW_prec_;
    // }
}

// =============================================================================
// Particle properties
// =============================================================================

/**
 * @brief Particle volume fraction [-].
 *
 * For a soot mass-fraction model: fv = Ys_ * rho_ / rho_particle_.
 * For a number/volume moment model: fv = M1 (third moment / Avogadro).
 */
template <ThermoMap Thermo>
double NewVariant<Thermo>::volume_fraction() const noexcept
{
    // TODO: implement from your transported variables, e.g.:
    // return Ys_ * this->rho_ / this->rho_particle_;
    return 0.;
}

// =============================================================================

/**
 * @brief Mean primary particle diameter [m].
 *
 * For a two-equation mass/number model: dp = SphereDiameter(v_particle)
 * where v_particle = M1 / (rho_particle * N).
 * For BrookesMoss-style scalar models: infer from mass fraction + number density.
 */
template <ThermoMap Thermo>
double NewVariant<Thermo>::particle_diameter() const noexcept
{
    // TODO: compute from your transported state, e.g.:
    // const double N  = particle_number_density();
    // if (N <= 0.) return 0.;
    // const double v  = volume_fraction() / N;    // mean particle volume [m3]
    // return this->SphereDiameter(v);             // (6V/π)^{1/3}
    return 0.;
}

// =============================================================================

/**
 * @brief Collision (aggregate) diameter [m].
 *
 * For non-fractal models equals `particle_diameter()`.
 * For fractal-aggregate models (e.g. HMOM), computed from the collision-diameter
 * model set by SetCollisionDiameterModel().
 */
template <ThermoMap Thermo>
double NewVariant<Thermo>::collision_diameter() const noexcept
{
    // TODO: for simple spherical particles:
    // return particle_diameter();
    return 0.;
}

// =============================================================================

/**
 * @brief Total particle number density [#/m3].
 *
 * Derive from your transported variables.
 */
template <ThermoMap Thermo>
double NewVariant<Thermo>::particle_number_density() const noexcept
{
    // TODO: e.g. for BM-style:
    // return bs_ * this->rho_;   // bs_ = Ns / (rho * Ns_norm) → N = bs_ * rho_
    return 0.;
}

// =============================================================================

/**
 * @brief Mean number of primary spheres per aggregate [-] (≥ 1).
 *
 * Return 1.0 for non-fractal models.  For aggregate models with surface area
 * tracking, use `this->NumberPrimaryParticles(ss, vs)`.
 */
template <ThermoMap Thermo>
double NewVariant<Thermo>::number_primary_particles() const noexcept
{
    return 1.;   // TODO: override for aggregate models
}

// =============================================================================

/**
 * @brief Particle mass fraction [-].
 *
 * For mass-fraction transport: return the transported scalar directly.
 * For number/volume moment models: fv * rho_particle / rho.
 */
template <ThermoMap Thermo>
double NewVariant<Thermo>::mass_fraction() const noexcept
{
    // TODO: e.g. return Ys_;
    return 0.;
}

// =============================================================================

/**
 * @brief Particle specific surface area [m2/m3].
 *
 * Typical approximation: treat all particles as spheres,
 * then SSA = N * π * dp² = 6 * fv / dp.
 * For models that transport surface area directly, return the moment directly.
 */
template <ThermoMap Thermo>
double NewVariant<Thermo>::specific_surface_area() const noexcept
{
    // TODO: e.g.
    // const double dp = particle_diameter();
    // return (dp > 0.) ? 6. * volume_fraction() / dp : 0.;
    return 0.;
}

// =============================================================================
// Transport
// =============================================================================

/**
 * @brief Effective diffusion coefficient [kg/m/s].
 *
 * The Cunningham-corrected Brownian diffusivity `rho * D` is computed by the
 * base-class helper `CunninghamDiffusionCoeff(dc_safe)`.  The Schmidt-number
 * fallback `mu / Sc` is added via std::max, which keeps the coefficient
 * physically reasonable when particle sizes are very small.
 *
 * @note `dc_safe` must be strictly positive — clamp with std::max before calling.
 */
template <ThermoMap Thermo>
double NewVariant<Thermo>::diffusion_coefficient() const noexcept
{
    const double dc = collision_diameter();
    const double dc_safe       = std::max(dc, 1.e-12);
    const double GammaBrownian = this->CunninghamDiffusionCoeff(dc_safe);
    const double GammaSc       = this->mu_ / this->schmidt_number_;
    return std::max(GammaBrownian, GammaSc);
}

// =============================================================================
// Diagnostics
// =============================================================================

/**
 * @brief Prints the model configuration to stdout.
 *
 * Called once after ApplyConfig() to confirm the model has been set up correctly.
 * No format requirements — use whatever is informative for your model.
 */
template <ThermoMap Thermo>
void NewVariant<Thermo>::PrintSummary() const
{
    // TODO: print your model configuration, e.g.:
    std::cout << "\n===== NewVariant Configuration =====\n";
    std::cout << "  Active:           " << (this->is_active_ ? "yes" : "no")  << "\n";
    std::cout << "  NEq:              " << NEq                                 << "\n";
    std::cout << "  Particle density: " << this->rho_particle_ << " kg/m3\n";
    std::cout << "  Precursor index:  " << prec_index_                         << "\n";
    std::cout << "  Schmidt number:   " << this->schmidt_number_               << "\n";
    // TODO: print your variant-specific parameters
    std::cout << "====================================\n\n";
}

// =============================================================================
// Private physics helpers (implement below as you add processes)
// =============================================================================

// template <ThermoMap Thermo>
// void NewVariant<Thermo>::SootNucleation() noexcept
// {
//     // TODO: fill source_nucleation_.
//     //
//     // Nucleation rate [mol/m3/s] — Arrhenius, PAH dimerisation, etc.
//     // Use this->Rgas_mol_ for the gas constant in Arrhenius expressions.
//     // Use conc_prec_ for the precursor concentration [kmol/m3].
// }

// template <ThermoMap Thermo>
// void NewVariant<Thermo>::SootCoagulation() noexcept
// {
//     // TODO: fill source_coagulation_.
// }

// template <ThermoMap Thermo>
// void NewVariant<Thermo>::SootGrowth() noexcept
// {
//     // TODO: fill source_growth_.
// }

// template <ThermoMap Thermo>
// void NewVariant<Thermo>::SootOxidation() noexcept
// {
//     // TODO: fill source_oxidation_.
// }

} // namespace MOM
