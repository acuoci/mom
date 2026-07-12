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
|   Copyright (C) 2026 Alberto Cuoci, Benedetta Franzelli                 |
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

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <span>
#include <sstream>
namespace MOM
{

// ============================================================================
// Constructor
// ============================================================================

template <ThermoMap Thermo> MetalOxide<Thermo>::MetalOxide(const Thermo& thermo) : thermo_(thermo)
{
    // -- Default material / physics constants -----------------------------
    // Config{} keeps TiO2/anatase as the default material. The material can be
    // replaced at setup time for generic metal-oxide use.
    this->rho_particle_ = solid_density_kg_m3_;
    this->planck_model_ = PlanckCoeffModel::None;

    // -- Enhancement factors (not exposed in Config; set before ApplyConfig
    //    so they are ready when Precalculations() runs inside ApplyConfig) --
    epsilon_nuc_  = 2.5;
    epsilon_coag_ = 2.2;
    epsilon_cond_ = 1.3;

    // -- Internal sintering regularisation (not in Config) -----------------
    sintering_activation_np_       = 1.05;
    sintering_activation_width_np_ = 0.05;
    sintering_relative_tolerance_  = 1.e-3;
    sintering_tau_qss_             = 1.e-6;

    // -- Apply all tunable parameter defaults from Config{} ----------------
    // Config{} is the single source of truth for every numerical constant.
    // ApplyConfig calls Precalculations() internally after setting
    // n_formula_units_min_
    // and the other geometry-affecting parameters.
    ApplyConfig(Config{});

    // -- Memory allocation (size depends on thermo_.NumberOfSpecies()) -----
    MemoryAllocation();
}

// ============================================================================
// MemoryAllocation
// ============================================================================

template <ThermoMap Thermo> void MetalOxide<Thermo>::MemoryAllocation()
{
    this->ZeroSources();          // zeros source_all_ (base class)
    source_nucleation_.setZero(); // owned by MetalOxide — zeroed explicitly
    source_coagulation_.setZero();
    source_condensation_.setZero();
    source_sintering_.setZero();

    this->omega_gas_ = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(thermo_.NumberOfSpecies()));

    // Initial moments (IC at numerical floor values)
    const double N0  = std::max(N_min_, 0.0);
    const double fv0 = N0 * v0_;
    const double Y0  = solid_density_kg_m3_ / 1.0 * fv0; // reference gas density 1 kg/m3
    const double S0  = N0 * s0_;

    initial_moments_cache_(0) = Y0;
    initial_moments_cache_(1) = N0 / N0_scaling_;
    initial_moments_cache_(2) = S0;
}

// ============================================================================
// Precalculations  (call whenever n_formula_units_min_ or vprec_/dprec_ change)
// ============================================================================

template <ThermoMap Thermo> void MetalOxide<Thermo>::Precalculations()
{
    v0_ = static_cast<double>(n_formula_units_min_) * solid_formula_unit_volume_m3_;
    d0_ = std::pow(6. * v0_ / this->pi_, 1. / 3.);
    s0_ = this->pi_ * d0_ * d0_;

    v_min_ = v0_;
    S_min_ = N_min_ * s0_;

    // Kernel prefactor for nucleation (temperature-independent part)
    if (vprec_ > 0. && dprec_ > 0.)
        alpha_nuc_ = epsilon_nuc_ * std::sqrt(this->pi_ * this->kB_ / (2. * solid_density_kg_m3_)) *
                     std::sqrt(2. / vprec_) * std::pow(2. * dprec_, 2.);
    else
        alpha_nuc_ = 0.;

    alpha_coag_ = epsilon_coag_ * std::sqrt(this->pi_ * this->kB_ / (2. * solid_density_kg_m3_));

    alpha_cond_ = epsilon_cond_ * std::sqrt(this->pi_ * this->kB_ / (2. * solid_density_kg_m3_));
}

// ============================================================================
// Material configuration
// ============================================================================

template <ThermoMap Thermo>
void MetalOxide<Thermo>::SetSolidMaterial(std::string_view name,
                                    double molecular_weight_kg_kmol,
                                    double density_kg_m3)
{
    if (name.empty())
        throw std::invalid_argument("[MetalOxide] Solid material name cannot be empty.");
    if (!std::isfinite(molecular_weight_kg_kmol) || molecular_weight_kg_kmol <= 0.)
        throw std::invalid_argument("[MetalOxide] Solid molecular weight must be positive.");
    if (!std::isfinite(density_kg_m3) || density_kg_m3 <= 0.)
        throw std::invalid_argument("[MetalOxide] Solid density must be positive.");

    solid_name_                       = std::string{name};
    solid_molecular_weight_kg_kmol_   = molecular_weight_kg_kmol;
    solid_density_kg_m3_              = density_kg_m3;
    solid_formula_unit_mass_kg_       = solid_molecular_weight_kg_kmol_ / this->Nav_kmol_;
    solid_formula_unit_volume_m3_     = solid_formula_unit_mass_kg_ / solid_density_kg_m3_;
    this->rho_particle_               = solid_density_kg_m3_;
    if (precursor_index_ >= 0)
        vprec_ = solid_formula_units_per_precursor_ * solid_formula_unit_volume_m3_;

    Precalculations();
}

template <ThermoMap Thermo>
void MetalOxide<Thermo>::SetSolidFormulaUnitsPerPrecursor(double n)
{
    if (!std::isfinite(n) || n <= 0.)
        throw std::invalid_argument("[MetalOxide] Solid formula units per precursor must be positive.");

    solid_formula_units_per_precursor_ = n;
    vprec_ = (precursor_index_ >= 0) ? solid_formula_units_per_precursor_ * solid_formula_unit_volume_m3_ : 0.;
    Precalculations();
}

template <ThermoMap Thermo>
void MetalOxide<Thermo>::SetNumberOfFormulaUnitsPerNucleatedParticle(unsigned n)
{
    if (n == 0u)
        throw std::invalid_argument(
            "[MetalOxide] Number of formula units per nucleated particle must be positive.");

    n0_ = n;
    Precalculations();
}

template <ThermoMap Thermo>
void MetalOxide<Thermo>::SetMinimumNumberOfFormulaUnits(unsigned n)
{
    if (n == 0u)
        throw std::invalid_argument("[MetalOxide] Minimum number of formula units must be positive.");

    n_formula_units_min_ = n;
    Precalculations();
}

// ============================================================================
// SetStatus  — injects thermodynamic state from the CFD solver
// ============================================================================

template <ThermoMap Thermo>
void MetalOxide<Thermo>::SetStatus(double T, double P_Pa, const double* Y) noexcept
{
    const double cTot = this->template UpdateMixtureState<>(T, P_Pa, Y, thermo_);

    // Precursor mass fraction and concentration
    Y_precursor_ = 0.;
    c_precursor_ = 0.;
    if (precursor_index_ >= 0)
    {
        Y_precursor_ = std::max(Y[precursor_index_], 0.);
        c_precursor_ = this->SpeciesConcentrationKmolM3(precursor_index_, Y, cTot, thermo_);
    }
}

// ============================================================================
// SetMoments
// ============================================================================

template <ThermoMap Thermo> void MetalOxide<Thermo>::SetMoments(std::span<const double> m) noexcept
{
    assert(m.size() == static_cast<std::size_t>(Base::n_equations) &&
           "[MetalOxide::SetMoments] expected exactly 3 moment values.");
    SetMoments(m[0], m[1], m[2]);
}

template <ThermoMap Thermo>
void MetalOxide<Thermo>::SetMoments(double solid_mass_fraction,
                              double scaled_number_density,
                              double surface_area_concentration) noexcept
{
    solid_mass_fraction_ = std::max(solid_mass_fraction, 0.);
    scaled_number_density_ = std::max(scaled_number_density, 0.);
    surface_area_concentration_ = std::max(surface_area_concentration, 0.);
}

// ============================================================================
// SetPrecursor
// ============================================================================

template <ThermoMap Thermo> void MetalOxide<Thermo>::SetPrecursor(std::string_view name)
{
    precursor_species_ = std::string(name);

    if (precursor_species_ == "none")
    {
        precursor_index_ = -1;
        Y_precursor_     = 0.;
        c_precursor_     = 0.;
        W_precursor_     = 0.;
        m_precursor_     = 0.;
        v_precursor_     = 0.;
        d_precursor_     = 0.;
        vprec_           = 0.;
        dprec_           = 0.;
        Precalculations();
        return;
    }

    precursor_index_ = thermo_.IndexOfSpecies(name);
    if (precursor_index_ < 0)
        throw std::runtime_error("[MetalOxide] Precursor species not found in mechanism: " +
                                 precursor_species_);

    const unsigned ui = static_cast<unsigned>(precursor_index_);

    W_precursor_   = thermo_.MolecularWeight(ui);
    m_precursor_   = W_precursor_ / this->Nav_kmol_;

    v_precursor_ = m_precursor_ / solid_density_kg_m3_;
    d_precursor_ = std::pow(6. * v_precursor_ / this->pi_, 1. / 3.);

    // Effective collision geometry used by nucleation/condensation kernels:
    // volume contribution = configured solid formula units per precursor.
    vprec_ = solid_formula_units_per_precursor_ * solid_formula_unit_volume_m3_;
    dprec_ = d_precursor_;

    Precalculations();
}

// ============================================================================
// Gas stoichiometry setup
// ============================================================================

template <ThermoMap Thermo> void MetalOxide<Thermo>::ClearGasStoichiometry() noexcept
{
    gas_stoichiometry_.clear();
}

template <ThermoMap Thermo>
void MetalOxide<Thermo>::AddGasStoichiometryTerm(std::string_view species, double coefficient)
{
    if (!std::isfinite(coefficient))
        throw std::invalid_argument("[MetalOxide] Gas stoichiometric coefficient must be finite.");
    if (std::fabs(coefficient) <= 1.e-14)
        return;

    const int index = thermo_.IndexOfSpecies(species);
    if (index < 0)
        throw std::runtime_error("[MetalOxide] Gas stoichiometry species not found: " +
                                 std::string(species));

    const double mw = thermo_.MolecularWeight(static_cast<unsigned>(index));

    for (auto& term : gas_stoichiometry_)
    {
        if (term.index == index)
        {
            term.coefficient += coefficient;
            return;
        }
    }

    gas_stoichiometry_.push_back(
        RuntimeGasStoichiometryTerm{index, coefficient, mw, std::string(species)});
}

template <ThermoMap Thermo> void MetalOxide<Thermo>::ValidateGasStoichiometryMassBalance() const
{
    if (gas_stoichiometry_.empty() || precursor_index_ < 0)
        return;

    bool has_precursor = false;
    double precursor_coefficient = 0.;
    double gas_mass = 0.;

    for (const auto& term : gas_stoichiometry_)
    {
        gas_mass += term.coefficient * term.molecular_weight_kg_kmol;
        if (term.index == precursor_index_)
        {
            has_precursor = true;
            precursor_coefficient += term.coefficient;
        }
    }

    if (!has_precursor)
        throw std::runtime_error("[MetalOxide] Gas stoichiometry must include precursor species '" +
                                 precursor_species_ + "' with coefficient -1.");

    if (std::fabs(precursor_coefficient + 1.) > 1.e-12)
        throw std::runtime_error("[MetalOxide] Gas stoichiometry is normalized per precursor; species '" +
                                 precursor_species_ + "' must have coefficient -1.");

    const double solid_mass =
        solid_formula_units_per_precursor_ * solid_molecular_weight_kg_kmol_;
    const double residual = gas_mass + solid_mass;
    const double scale = std::max({std::fabs(gas_mass), std::fabs(solid_mass), 1.});

    if (std::fabs(residual) > gas_stoichiometry_mass_tolerance_ * scale)
    {
        std::ostringstream msg;
        msg << "[MetalOxide] Gas stoichiometry is not mass balanced: gas_mass=" << gas_mass
            << " kg/kmol_precursor, solid_mass=" << solid_mass
            << " kg/kmol_precursor, residual=" << residual
            << " kg/kmol_precursor, tolerance=" << gas_stoichiometry_mass_tolerance_;
        throw std::runtime_error(msg.str());
    }
}

template <ThermoMap Thermo>
void MetalOxide<Thermo>::SetGasStoichiometry(std::span<const GasStoichiometryTerm> terms,
                                       double relative_mass_tolerance)
{
    if (!std::isfinite(relative_mass_tolerance) || relative_mass_tolerance < 0.)
        throw std::invalid_argument("[MetalOxide] Gas stoichiometry mass tolerance must be non-negative.");

    gas_stoichiometry_mass_tolerance_ = relative_mass_tolerance;
    ClearGasStoichiometry();

    if (terms.empty())
        return;

    for (const auto& term : terms)
    {
        if (term.species.empty())
            throw std::invalid_argument("[MetalOxide] Gas stoichiometry species name cannot be empty.");
        AddGasStoichiometryTerm(term.species, term.coefficient);
    }

    ValidateGasStoichiometryMassBalance();
}

// ============================================================================
// SetGasClosureDummySpecies
// ============================================================================

template <ThermoMap Thermo> void MetalOxide<Thermo>::SetGasClosureDummySpecies(std::string_view name)
{
    this->closure_dummy_species_ = std::string(name);

    if (this->closure_dummy_species_ == "none")
    {
        this->closure_dummy_index_      = -1;
        this->is_closure_dummy_species_ = false;
        return;
    }

    this->closure_dummy_index_ = thermo_.IndexOfSpecies(this->closure_dummy_species_);
    if (this->closure_dummy_index_ < 0)
        throw std::runtime_error("[MetalOxide] Dummy species not found in mechanism: " +
                                 this->closure_dummy_species_);

    if (this->closure_dummy_index_ == precursor_index_)
        throw std::runtime_error("[MetalOxide] Dummy species cannot be the precursor species.");

    this->is_closure_dummy_species_ = true;
}

// ============================================================================
// Properties  — aggregate particle properties from the three moments
// ============================================================================

template <ThermoMap Thermo>
void MetalOxide<Thermo>::Properties(double& fv,
                              double& dp,
                              double& dc,
                              double& da,
                              double& np,
                              double& ss,
                              double& vs,
                              double& ssph,
                              double& tauS) const noexcept
{
    const double Y = std::max(solid_mass_fraction_, 0.);
    const double N = std::max(scaled_number_density_ * N0_scaling_, 0.);
    const double S = std::max(surface_area_concentration_, 0.);

    // Volume fraction [-]
    fv = this->rho_ / solid_density_kg_m3_ * Y;

    // Number density used for property calculation (regularized)
    const double NStar  = std::max(N, N_min_);
    const double fvStar = std::max(fv, fv_min_);

    // Mean particle volume [m3]
    vs = std::max(fvStar / NStar, v_min_);

    // Spherical surface area for a particle of volume vs [m2]
    ssph = std::pow(36. * this->pi_, 1. / 3.) * std::pow(vs, 2. / 3.);

    // Specific surface per particle [m2]
    const double SStar = std::max(S, S_min_);
    ss                 = std::max(SStar / NStar, ssph);

    // Sphere-equivalent aggregate diameter [m]
    // (diameter of a sphere whose volume equals the aggregate volume)
    da = std::pow(6. * vs / this->pi_, 1. / 3.);
    da = std::max(da, d0_);

    // Primary particle diameter [m]
    // dp ≤ da always (since ss ≥ ssph ensures 6*vs/ss ≤ 6*vs/ssph = da)    
    dp = 6. * vs / ss;
    dp = std::max(dp, d0_);

    // Number of primary particles per aggregate [-]
    np = std::max(std::pow(ss, 3.) / std::pow(vs, 2.) / (36. * this->pi_), 1.);

    // Collision diameter — fractal aggregate (Df = 1.8) [m]
    // dc = dp * np^(1/Df). For np = 1 (sphere) this reduces to dc = dp.
    // Df = 3 (compact sphere) would give a significantly smaller dc for np > 1,
    // corrupting the condensation kernel (dprec + dc)^2 and coagulation (2*dc)^2.
    constexpr double Df = 1.8;
    dc = dp * std::pow(np, 1. / Df);

    // Sintering time scale [s]:  τ_s = As · T^ns · dp^4 · exp(Ts/T)
    tauS = As_ * std::pow(this->T_, ns_) * std::pow(dp, 4.) * std::exp(Ts_ / this->T_);
}

// ============================================================================
// VolumeFraction, MassFraction, ParticleNumberDensity, SpecificSurface
// ============================================================================

template <ThermoMap Thermo> double MetalOxide<Thermo>::volume_fraction() const noexcept
{
    return this->rho_ / solid_density_kg_m3_ * std::max(solid_mass_fraction_, 0.);
}

template <ThermoMap Thermo> double MetalOxide<Thermo>::mass_fraction() const noexcept
{
    return std::max(solid_mass_fraction_, 0.);
}

template <ThermoMap Thermo> double MetalOxide<Thermo>::particle_number_density() const noexcept
{
    return std::max(scaled_number_density_ * N0_scaling_, 0.);
}

template <ThermoMap Thermo> double MetalOxide<Thermo>::specific_surface() const noexcept
{
    return std::max(surface_area_concentration_, 0.);
}

// ============================================================================
// ParticleDiameter, CollisionDiameter, AggregateDiameter,
// NumberOfPrimaryParticles
// ============================================================================

template <ThermoMap Thermo> double MetalOxide<Thermo>::particle_diameter() const noexcept
{
    double fv, dp, dc, da, np, ss, vs, ssph, tauS;
    Properties(fv, dp, dc, da, np, ss, vs, ssph, tauS);
    return dp;
}

template <ThermoMap Thermo> double MetalOxide<Thermo>::collision_diameter() const noexcept
{
    double fv, dp, dc, da, np, ss, vs, ssph, tauS;
    Properties(fv, dp, dc, da, np, ss, vs, ssph, tauS);
    return dc;
}

template <ThermoMap Thermo> double MetalOxide<Thermo>::AggregateDiameter() const noexcept
{
    double fv, dp, dc, da, np, ss, vs, ssph, tauS;
    Properties(fv, dp, dc, da, np, ss, vs, ssph, tauS);
    return da;
}

template <ThermoMap Thermo> double MetalOxide<Thermo>::number_primary_particles() const noexcept
{
    double fv, dp, dc, da, np, ss, vs, ssph, tauS;
    Properties(fv, dp, dc, da, np, ss, vs, ssph, tauS);
    return np;
}

// ============================================================================
// DiffusionCoefficient  — Cunningham-corrected Brownian + Schmidt fallback
// ============================================================================

template <ThermoMap Thermo> double MetalOxide<Thermo>::diffusion_coefficient() const noexcept
{
    double fv, dp, dc, da, np, ss, vs, ssph, tauS;
    Properties(fv, dp, dc, da, np, ss, vs, ssph, tauS);

    const double dcSafe = std::max(dc, d0_);
    const double mGas   = this->rho_ * this->kB_ * this->T_ / this->P_Pa_;
    const double lambdaGas =
        this->mu_ / this->rho_ * std::sqrt(this->pi_ * mGas / (2. * this->kB_ * this->T_));
    const double Cu            = 1. + 2.154 * lambdaGas / dcSafe;
    const double D             = this->kB_ * this->T_ * Cu / (3. * this->pi_ * this->mu_ * dcSafe);
    const double GammaBrownian = this->rho_ * D;
    const double GammaSc       = this->mu_ / this->schmidt_number_;
    return std::max(GammaBrownian, GammaSc);
}

// ============================================================================
// NucleationParticleVolume
// ============================================================================

template <ThermoMap Thermo> double MetalOxide<Thermo>::NucleationParticleVolume() const noexcept
{
    if (nucleation_variant_ == NucleationVariant::Binary)
    {
        if (vprec_ > 0.)
            return 2. * vprec_;
        return 2. * solid_formula_unit_volume_m3_;
    }
    if (nucleation_variant_ == NucleationVariant::FixedCluster)
        return static_cast<double>(n0_) * solid_formula_unit_volume_m3_;

    return v0_;
}

// ============================================================================
// NucleationSourceTerms  — dispatcher
// ============================================================================

template <ThermoMap Thermo> void MetalOxide<Thermo>::NucleationSourceTerms()
{
    if (nucleation_variant_ == NucleationVariant::Binary)
        NucleationSourceTerms_Binary();
    else if (nucleation_variant_ == NucleationVariant::FixedCluster)
        NucleationSourceTerms_FixedCluster();
}

// ============================================================================
// NucleationSourceTerms_Binary
// ============================================================================

template <ThermoMap Thermo> void MetalOxide<Thermo>::NucleationSourceTerms_Binary()
{
    if (c_precursor_ <= 0. || vprec_ <= 0. || dprec_ <= 0.)
        return;

    // Precursor number density [#/m3]
    const double Nprec = c_precursor_ * this->Nav_kmol_;

    // Free-molecular binary collision kernel [m3/s]
    const double beta_nuc = epsilon_nuc_ *
                            std::sqrt(this->pi_ * this->kB_ * this->T_ /
                                      (2. * solid_density_kg_m3_)) *
                            std::sqrt(1. / vprec_ + 1. / vprec_) * std::pow(2. * dprec_, 2.);

    // Rate of nucleation events [#/m3/s]
    const double eventRate = 0.5 * beta_nuc * Nprec * Nprec;

    // Volume source: each event creates one dimer of volume 2*vprec_
    const double source_fv = 2. * vprec_ * eventRate;

    const double source_N = eventRate;

    // Surface source: dimer treated as sphere of volume 2*vprec_
    const double source_S =
        std::pow(18. * this->pi_, 1. / 3.) * std::pow(vprec_, 2. / 3.) * beta_nuc * Nprec * Nprec;

    this->source_nucleation_(0) = solid_density_kg_m3_ / this->rho_ * source_fv;
    this->source_nucleation_(1) = source_N / N0_scaling_;
    this->source_nucleation_(2) = source_S;
}

// ============================================================================
// NucleationSourceTerms_FixedCluster
// ============================================================================

template <ThermoMap Thermo> void MetalOxide<Thermo>::NucleationSourceTerms_FixedCluster()
{
    if (c_precursor_ <= 0.)
        return;

    const double v_cluster = static_cast<double>(n0_) * solid_formula_unit_volume_m3_;
    const double d_cluster    = std::pow(6. * v_cluster / this->pi_, 1. / 3.);

    // Precursor number density [#/m3]
    const double Nprec = c_precursor_ * this->Nav_kmol_;

    // Free-molecular collision kernel for a cluster colliding with itself
    const double beta_nuc = epsilon_nuc_ *
                            std::sqrt(this->pi_ * this->kB_ * this->T_ /
                                      (2. * solid_density_kg_m3_)) *
                            std::sqrt(2. / v_cluster) * std::pow(2. * d_cluster, 2.);

    const double eventRate = 0.5 * beta_nuc * Nprec * Nprec;

    const double source_fv = 2. * v_cluster * eventRate;
    const double source_N  = eventRate;
    const double source_S =
        std::pow(18. * this->pi_, 1. / 3.) * std::pow(v_cluster, 2. / 3.) * beta_nuc * Nprec * Nprec;

    this->source_nucleation_(0) = solid_density_kg_m3_ / this->rho_ * source_fv;
    this->source_nucleation_(1) = source_N / N0_scaling_;
    this->source_nucleation_(2) = source_S;
}

// ============================================================================
// CoagulationSourceTerms
// ============================================================================

template <ThermoMap Thermo> void MetalOxide<Thermo>::CoagulationSourceTerms()
{
    const double N = scaled_number_density_ * N0_scaling_;
    if (N <= N_min_)
        return;

    double fv, dp, dc, da, np, ss, vs, ssph, tauS;
    Properties(fv, dp, dc, da, np, ss, vs, ssph, tauS);

    if (vs <= 0. || dc <= 0.)
        return;

    const double dcSafe = std::max(dc, d0_);

    // Free-molecular kernel [m3/s]
    const double beta_fm = epsilon_coag_ *
                           std::sqrt(this->pi_ * this->kB_ * this->T_ /
                                     (2. * solid_density_kg_m3_)) *
                           std::sqrt(2. / vs) * std::pow(2. * dcSafe, 2.);

    // Continuum kernel with Cunningham slip correction [m3/s]
    const double mGas = this->rho_ * this->kB_ * this->T_ / this->P_Pa_;
    const double lambdaGas =
        this->mu_ / this->rho_ * std::sqrt(this->pi_ * mGas / (2. * this->kB_ * this->T_));
    const double Cu        = 1. + 2.154 * lambdaGas / dcSafe;
    const double beta_cont = 8. * this->kB_ * this->T_ / (3. * this->mu_) * Cu * dcSafe;

    // Transition-regime kernel: harmonic mean, scaled by 1.82
    const double beta_m    = std::max(beta_fm, beta_cont);
    const double beta_coag = 1.82 * beta_m;

    // Number density source [#/m3/s]
    const double OmegaN = -0.5 * beta_coag * N * N;

    this->source_coagulation_(0) = 0.;
    this->source_coagulation_(1) = OmegaN / N0_scaling_;
    this->source_coagulation_(2) = 0.;
}

// ============================================================================
// CondensationSourceTerms
// ============================================================================

template <ThermoMap Thermo> void MetalOxide<Thermo>::CondensationSourceTerms()
{
    if (c_precursor_ <= 0. || vprec_ <= 0. || dprec_ <= 0.)
        return;

    const double N = scaled_number_density_ * N0_scaling_;
    if (N <= N_min_)
        return;

    double fv, dp, dc, da, np, ss, vs, ssph, tauS;
    Properties(fv, dp, dc, da, np, ss, vs, ssph, tauS);

    if (vs <= 0. || dc <= 0.)
        return;

    // Precursor number density [#/m3]
    const double Nprec = c_precursor_ * this->Nav_kmol_;

    // Free-molecular condensation kernel [m3/s]: precursor + particle
    const double beta_cond = epsilon_cond_ *
                             std::sqrt(this->pi_ * this->kB_ * this->T_ /
                                       (2. * solid_density_kg_m3_)) *
                             std::sqrt(1. / vprec_ + 1. / vs) * std::pow(dprec_ + dc, 2.);

    const double BetaNprecN = beta_cond * Nprec * N;

    // Surface area change per condensation event (fractal correction).
    // deltas = (2/3) * (vprec/vs) * ss * np^chi
    // The np^chi factor (chi = -0.2043) accounts for the fractal morphology of aggregates: 
    // a particle with np primaries grows its surface less per unit volume added than a sphere
    // would.  When np = 1 (sphere) the factor is 1. Dropping this term causes deltas to be 
    // overestimated whenever coagulation has produced aggregates (np > 1).
    constexpr double chi = -0.2043;
    const double deltas = (2. / 3.) * (vprec_ / vs) * ss * std::pow(std::max(np, 1.), chi);

    this->source_condensation_(0) = solid_density_kg_m3_ / this->rho_ * vprec_ * BetaNprecN;
    this->source_condensation_(1) = 0.;
    this->source_condensation_(2) = deltas * BetaNprecN;
}

// ============================================================================
// SinteringSourceTerms
// ============================================================================

template <ThermoMap Thermo> void MetalOxide<Thermo>::SinteringSourceTerms()
{
    const double N = scaled_number_density_ * N0_scaling_;
    const double S = surface_area_concentration_;

    if (N <= N_min_ || S <= 0.)
        return;

    double fv, dp, dc, da, np, ss, vs, ssph, tauS;
    Properties(fv, dp, dc, da, np, ss, vs, ssph, tauS);

    // Spherical surface area for the current mean volume
    const double S_sphere = N * ssph;

    // Sintering is only active when surface area exceeds spherical limit
    if (S <= S_sphere)
        return;

    // Primary particle diameter guard
    if (dp < sintering_dp_min_)
        return;

    // Smooth activation based on number of primary particles per aggregate
    double activation = 1.;
    if (sintering_activation_np_ > 1.)
    {
        const double x =
            (np - sintering_activation_np_) / std::max(sintering_activation_width_np_, 1.e-300);
        activation = 0.5 * (1. + std::tanh(x));
    }
    if (activation <= 0.)
        return;

    // Physical sintering time scale [s]  
    const double tauPhysical = std::max(tauS, sintering_tau_min_);

    // Rate of surface area decrease toward spherical limit [m2/m3/s]
    const double OmegaS = -activation * (S - S_sphere) / tauPhysical;

    this->source_sintering_(0) = 0.;
    this->source_sintering_(1) = 0.;
    this->source_sintering_(2) = OmegaS;
}

// ============================================================================
// SinteringDeferredUpdate  — analytical ODE solution for stiff sintering
// ============================================================================

template <ThermoMap Thermo> double MetalOxide<Thermo>::SinteringDeferredUpdate(double dt_ode)
{
    if (sintering_model_ == 0)
        return surface_area_concentration_;

    const double N = scaled_number_density_ * N0_scaling_;
    if (N <= N_min_)
        return surface_area_concentration_;

    double fv, dp, dc, da, np, ss, vs, ssph, tauS;
    Properties(fv, dp, dc, da, np, ss, vs, ssph, tauS);

    const double S_sphere = N * ssph;
    const double S        = surface_area_concentration_;

    if (S <= S_sphere)
        return S;
    if (dp < sintering_dp_min_)
        return S;

    // Smooth activation
    double activation = 1.;
    if (sintering_activation_np_ > 1.)
    {
        const double x =
            (np - sintering_activation_np_) / std::max(sintering_activation_width_np_, 1.e-300);
        activation = 0.5 * (1. + std::tanh(x));
    }
    if (activation <= 0.)
        return S;

    const double tauS_phys = std::max(tauS, sintering_tau_min_);
    const double tauSeff   = tauS_phys / activation;

    // Analytical solution: S(t) = S_sphere + (S0 - S_sphere)*exp(-dt/tau)
    const double S_new = S_sphere + (S - S_sphere) * std::exp(-dt_ode / tauSeff);
    return std::max(S_new, S_sphere);
}

// ============================================================================
// CalculateSourceMoments  — master entry point
// ============================================================================

template <ThermoMap Thermo> void MetalOxide<Thermo>::CalculateSourceMoments() noexcept
{
    this->ZeroSources();          // zeros source_all_ (base class)
    source_nucleation_.setZero(); // owned by MetalOxide — must be zeroed before early return
    source_coagulation_.setZero();
    source_condensation_.setZero();
    source_sintering_.setZero();

    if (!this->is_active_)
    {
        if (this->gas_consumption_)
            this->omega_gas_.setZero();
        return;
    }

    if (nucleation_variant_ != NucleationVariant::Off)
        NucleationSourceTerms();

    if (coagulation_model_ > 0)
        CoagulationSourceTerms();

    if (condensation_model_ > 0)
        CondensationSourceTerms();

    if (sintering_model_ > 0)
        SinteringSourceTerms();

    // Sanitise and accumulate total source
    for (unsigned i = 0; i < 3u; ++i)
    {
        if (!std::isfinite(this->source_nucleation_(i)))
            this->source_nucleation_(i) = 0.;
        if (!std::isfinite(this->source_coagulation_(i)))
            this->source_coagulation_(i) = 0.;
        if (!std::isfinite(this->source_condensation_(i)))
            this->source_condensation_(i) = 0.;
        if (!std::isfinite(this->source_sintering_(i)))
            this->source_sintering_(i) = 0.;

        this->source_all_(i) = this->source_nucleation_(i) + this->source_coagulation_(i) +
                               this->source_condensation_(i);

        if (!is_sintering_deferred_)
            this->source_all_(i) += this->source_sintering_(i);
    }

    if (this->gas_consumption_)
        CalculateOmegaGas_internal();
}

// ============================================================================
// CalculateOmegaGas  — public interface
// ============================================================================

template <ThermoMap Thermo> void MetalOxide<Thermo>::CalculateOmegaGas() noexcept
{
    CalculateOmegaGas_internal();
}

// ============================================================================
// CalculateOmegaGas_internal  — gas-phase consumption [kg/m3/s]
// ============================================================================

template <ThermoMap Thermo> void MetalOxide<Thermo>::CalculateOmegaGas_internal() noexcept
{
    this->omega_gas_.setZero();

    if (!this->gas_consumption_)
        return;
    if (precursor_index_ < 0)
        return;
    if (gas_stoichiometry_.empty())
        return;

    if (solid_formula_units_per_precursor_ <= 0.)
        return;

    // Total solid mass deposition rate from nucleation + condensation [kg/m3/s]
    const double omegaSolid =
        this->rho_ * (this->source_nucleation_(0) + this->source_condensation_(0));

    if (omegaSolid <= 0.)
        return;

    // Molar rate of solid formula-unit deposition [kmol/m3/s]
    const double Rsolid = omegaSolid / solid_molecular_weight_kg_kmol_;

    // Molar rate of precursor consumed [kmol/m3/s]
    // Each precursor molecule yields the configured number of solid formula units.
    const double Rprec = Rsolid / solid_formula_units_per_precursor_;

    for (const auto& term : gas_stoichiometry_)
        this->omega_gas_[term.index] +=
            term.coefficient * Rprec * term.molecular_weight_kg_kmol;

    // Dummy species closure: adjust to enforce sum(omega_gas) = 0
    if (this->is_closure_dummy_species_ && this->closure_dummy_index_ >= 0)
    {
        const int nsp = static_cast<int>(thermo_.NumberOfSpecies());
        double sum    = 0.;
        for (int i = 0; i < nsp; ++i)
            if (i != this->closure_dummy_index_)
                sum += this->omega_gas_[i];
        this->omega_gas_[this->closure_dummy_index_] = -sum;
    }
}

// ============================================================================
// PrintSummary
// ============================================================================

template <ThermoMap Thermo> void MetalOxide<Thermo>::PrintSummary() const
{
    // Helper lambdas — no heap allocation, resolved at compile time
    const auto nuc_str = [](NucleationVariant v) -> const char* {
        switch (v) {
            case NucleationVariant::Off:          return "off";
            case NucleationVariant::Binary:       return "binary (precursor+precursor)";
            case NucleationVariant::FixedCluster: return "fixed cluster";
            default:                               return "unknown";
        }
    };
    const auto thermo_str = [](ThermophoreticModel m) -> const char* {
        switch (m) {
            case ThermophoreticModel::Off:      return "off";
            case ThermophoreticModel::Standard: return "standard";
            default:                             return "unknown";
        }
    };
    const auto planck_str = [](PlanckCoeffModel m) -> const char* {
        switch (m) {
            case PlanckCoeffModel::None:   return "none";
            case PlanckCoeffModel::Smooke: return "Smooke (1989)";
            case PlanckCoeffModel::Kent:   return "Kent & Honnery (1990)";
            case PlanckCoeffModel::Sazhin: return "Sazhin (1994)";
            default:                        return "unknown";
        }
    };

    std::cout
        << "\n"
        << "------------------------------------------------------------------------------------------\n"
        << "                      Solid Oxide Nanoparticle Model Summary\n"
        << "------------------------------------------------------------------------------------------\n"
        << " * Active: " << (this->is_active_ ? "yes" : "no") << "\n"
        << "\n"
        << " [Physical properties]\n"
        << "    + Solid name:                    " << solid_name_ << "\n"
        << "    + Solid density (kg/m3):         " << solid_density_kg_m3_ << "\n"
        << "    + Solid mol. weight (kg/kmol):   " << solid_molecular_weight_kg_kmol_ << "\n"
        << "    + Formula units per precursor:   " << solid_formula_units_per_precursor_ << "\n"
        << "\n"
        << " [Monomer / nucleated particle geometry]\n"
        << "    + formula units/nucleated part.: " << n0_ << "\n"
        << "    + minimum formula units (-):     " << n_formula_units_min_ << "\n"
        << "    + v0 (m3):                       " << v0_ << "\n"
        << "    + d0 (m):                        " << d0_ << "\n"
        << "    + s0 (m2):                       " << s0_ << "\n"
        << "\n"
        << " [Processes]\n"
        << "    + Nucleation:                    " << static_cast<int>(nucleation_variant_) << "  (" << nuc_str(nucleation_variant_) << ")\n"
        << "    + Coagulation:                   " << coagulation_model_ << "\n"
        << "    + Condensation:                  " << condensation_model_ << "\n"
        << "    + Sintering:                     " << sintering_model_ << "\n"
        << "\n"
        << " [Collision enhancement factors]\n"
        << "    + Nucleation (-):                " << epsilon_nuc_ << "\n"
        << "    + Coagulation (-):               " << epsilon_coag_ << "\n"
        << "    + Condensation (-):              " << epsilon_cond_ << "\n"
        << "\n"
        << " [Sintering kinetics]  tau_s = As * T^ns * dp^4 * exp(Ts/T)\n"
        << "    + As (s/K^ns/m4):               " << As_ << "\n"
        << "    + ns (-):                        " << ns_ << "\n"
        << "    + Ts (K):                        " << Ts_ << "\n"
        << "    + dp_min (m):                    " << sintering_dp_min_ << "\n"
        << "\n"
        << " [Species — precursor]\n"
        << "    + Name:                          " << precursor_species_ << "\n";

    if (precursor_index_ >= 0)
    {
        std::cout
            << "    + Index (-):                     " << precursor_index_ << "\n"
            << "    + MW (kg/kmol):                  " << W_precursor_ << "\n"
            << "    + m (kg):                        " << m_precursor_ << "\n"
            << "    + v (m3):                        " << v_precursor_ << "\n"
            << "    + d (m):                         " << d_precursor_ << "\n";
    }

    std::cout
        << "\n"
        << " [Gas stoichiometry]\n"
        << "    + Source:                        explicit\n"
        << "    + Mass tolerance (-):            " << gas_stoichiometry_mass_tolerance_ << "\n";
    if (gas_stoichiometry_.empty())
    {
        std::cout << "    + Terms:                         none\n";
    }
    else
    {
        for (const auto& term : gas_stoichiometry_)
            std::cout << "    + nu(" << term.species << "):"
                      << std::string(term.species.size() < 24 ? 24 - term.species.size() : 1, ' ')
                      << term.coefficient << "\n";
    }

    std::cout
        << "\n"
        << " [Effective precursor]\n"
            << "    + v (m3):                        " << vprec_ << "\n"
            << "    + d (m):                         " << dprec_ << "\n";

    std::cout
        << "\n"
        << " [Transport & radiation]\n"
        << "    + Schmidt number (-):            " << this->schmidt_number_ << "\n"
        << "    + Thermophoretic model:          " << this->thermophoretic_model() << "  (" << thermo_str(this->thermophoretic_model_) << ")\n"
        << "    + Gas consumption:               " << (this->gas_consumption_ ? "yes" : "no") << "\n"
        << "    + Radiative heat transfer:       " << (this->radiative_heat_transfer_ ? "yes" : "no") << "\n"
        << "    + Planck coeff. model:           " << static_cast<int>(this->planck_model_) << "  (" << planck_str(this->planck_model_) << ")\n"
        << "    + Closure dummy species:         " << (this->is_closure_dummy_species_ ? this->closure_dummy_species_ : "none") << "\n"
        << "\n"
        << " [Numerical floors]\n"
        << "    + N_min  (#/m3):                 " << N_min_ << "\n"
        << "    + fv_min (-):                    " << fv_min_ << "\n"
        << "    + v_min  (m3):                   " << v_min_ << "\n"
        << "    + S_min  (m2/m3):                " << S_min_ << "\n"
        << "\n"
        << " [Debug]\n"
        << "    + Debug mode:                    " << (is_debug_mode_ ? "yes" : "no") << "\n"
        << "------------------------------------------------------------------------------------------\n";
}

// ============================================================================
// ApplyConfig  (private — all parameter assignments, no I/O)
// ============================================================================

template <ThermoMap Thermo>
void MetalOxide<Thermo>::ApplyConfig(const Config& cfg)
{
    this->is_active_ = cfg.is_active;

    // -- Solid material ----------------------------------------------------
    this->SetSolidMaterial(cfg.solid_name,
                           cfg.solid_molecular_weight_kg_kmol,
                           cfg.solid_density_kg_m3);
    this->SetSolidFormulaUnitsPerPrecursor(cfg.solid_formula_units_per_precursor);

    // -- Precursor / gas setup ---------------------------------------------
    // Always call SetPrecursor — including when cfg.precursor_species == "none",
    // which resets all precursor-derived geometry (vprec_, dprec_, etc.) to 0.
    // This ensures a subsequent SetupFromConfig("none") can clear a prior precursor.
    this->SetPrecursor(cfg.precursor_species);
    this->SetGasStoichiometry(cfg.gas_stoichiometry,
                              cfg.gas_stoichiometry_mass_tolerance);
    this->SetGasClosureDummySpecies(cfg.gas_closure_dummy_species);
    this->SetGasConsumption(cfg.gas_consumption);

    if (cfg.gas_consumption && precursor_index_ >= 0)
    {
        if (gas_stoichiometry_.empty())
            throw std::runtime_error(
                "[MetalOxide] Gas consumption is enabled but no gas stoichiometry is available. "
                "Provide explicit gas stoichiometry.");
        ValidateGasStoichiometryMassBalance();
    }

    // -- Nucleation model (string → enum → int) ----------------------------
    if (cfg.nucleation_model == "none" || cfg.nucleation_model == "0")
        this->SetNucleation(static_cast<int>(NucleationVariant::Off));
    else if (cfg.nucleation_model == "binary" || cfg.nucleation_model == "1")
        this->SetNucleation(static_cast<int>(NucleationVariant::Binary));
    else if (cfg.nucleation_model == "fixed-cluster" || cfg.nucleation_model == "2")
        this->SetNucleation(static_cast<int>(NucleationVariant::FixedCluster));
    else
        throw std::invalid_argument(
            "[MetalOxide] Invalid nucleation model. Allowed values: none, binary, fixed-cluster.");

    // -- Other process models ----------------------------------------------
    this->SetSintering(cfg.sintering_model);
    this->SetCoagulation(cfg.coagulation_model);
    this->SetCondensation(cfg.condensation_model);
    this->SetThermophoreticModel(cfg.thermophoretic_model);

    // -- Cluster sizes -----------------------------------------------------
    const int minimum_units = cfg.minimum_formula_units;
    const int nucleated_units = cfg.nucleated_particle_formula_units;

    if (minimum_units <= 0)
        throw std::invalid_argument("[MetalOxide] minimum formula units must be positive.");
    if (nucleated_units <= 0)
        throw std::invalid_argument(
            "[MetalOxide] nucleated-particle formula units must be positive.");

    this->SetMinimumNumberOfFormulaUnits(static_cast<unsigned int>(minimum_units));
    this->SetNumberOfFormulaUnitsPerNucleatedParticle(
        static_cast<unsigned int>(nucleated_units));

    // -- Sintering kinetics ------------------------------------------------
    As_ = cfg.sintering_As_s_K_m;
    Ts_ = cfg.sintering_Ts_K;
    ns_ = cfg.sintering_ns;

    // -- Sintering regularisation (Config-exposed subset) ------------------
    // sintering_activation_np_, sintering_activation_width_np_,
    // sintering_relative_tolerance_, sintering_tau_qss_ are internal
    // numerical knobs not exposed in Config — they live in the constructor.
    is_sintering_deferred_ = cfg.sintering_deferred;
    sintering_dp_min_      = cfg.sintering_dp_min_m;
    sintering_tau_min_     = cfg.sintering_tau_min_s;
    sintering_k_max_       = cfg.sintering_k_max_per_s;

    // -- Numerical floors + geometry recompute -----------------------------
    N_min_  = cfg.ns_minimum_per_m3;
    fv_min_ = cfg.fv_minimum;
    Precalculations(); // recompute geometry that depends on N_min_, n_formula_units_min_, epsilon_*

    // -- Transport ---------------------------------------------------------
    this->SetSchmidtNumber(cfg.schmidt_number);

    // -- Debug mode --------------------------------------------------------
    this->is_debug_mode_ = cfg.debug_mode;
}

// ============================================================================
// SetupFromConfig  (public — delegates to ApplyConfig, then prints summary)
// ============================================================================

template <ThermoMap Thermo>
void MetalOxide<Thermo>::SetupFromConfig(const Config& cfg)
{
    ApplyConfig(cfg);
    PrintSummary();
}

// ============================================================================
// NDF reconstruction  (Pareto + log-normal, analogous to ThreeEquations)
// ============================================================================

template <ThermoMap Thermo>
typename MetalOxide<Thermo>::NDFReconstructionData MetalOxide<Thermo>::ReconstructedNDFData(
    bool use_regularized_moments) const
{
    NDFReconstructionData d{}; // all zero, valid = false

    const double Y_raw = std::max(solid_mass_fraction_, 0.);
    const double N_raw = std::max(scaled_number_density_ * N0_scaling_, 0.);

    double Y = Y_raw;
    double N = N_raw;
    if (use_regularized_moments)
    {
        const double fv_raw = this->rho_ / solid_density_kg_m3_ * Y_raw;
        if (fv_raw < fv_min_)
            Y = solid_density_kg_m3_ / this->rho_ * fv_min_;
        N = std::max(N, N_min_);
    }

    if (Y <= 0. || N <= 0.)
        return d;

    const double fv = this->rho_ / solid_density_kg_m3_ * Y;
    if (!std::isfinite(fv) || fv <= 0.)
        return d;

    double nuMean = fv / N;
    if (!std::isfinite(nuMean) || nuMean <= 0.)
        return d;

    const double nuNucl = NucleationParticleVolume();
    if (!std::isfinite(nuNucl) || nuNucl <= 0.)
        return d;
    if (nuMean < nuNucl)
        nuMean = nuNucl;

    // Pareto weight (Eq. 8, Franzelli et al. 2019)
    double alpha = 1.0 - 0.18 * std::pow(nuMean / nuNucl, 0.12);
    alpha        = std::clamp(alpha, 0., 1.);

    const double nbar0 = 8.0 * std::pow(1. - alpha, 2.) * 1.e27; // [1/m3]
    const double sigma = 1.0 + 0.65 * (1. - alpha);              // [-]

    constexpr double eps_min = 1.e-14;
    constexpr double tiny    = 1.e-300;
    constexpr double eps_k   = 1.01;

    double k = 0., nu1mean = 0., nu2mean = 0., mu = 0.;

    if (alpha > eps_min)
    {
        const double k_peak = nbar0 * nuNucl / alpha;
        const double denom  = 1. - alpha * nuNucl / nuMean;
        if (denom <= 0. || !std::isfinite(denom))
            return d;
        k       = std::max({k_peak, eps_k / denom, 1. + 1.e-12});
        nu1mean = k / (k - 1.) * nuNucl;
    }

    if (alpha < (1. - eps_min))
    {
        nu2mean = (nuMean - alpha * nu1mean) / (1. - alpha);
        if (!std::isfinite(nu2mean) || nu2mean <= 0.)
            return d;
        mu = std::log(std::max(nu2mean, tiny)) - 0.5 * sigma * sigma;
    }

    d.valid   = true;
    d.N       = N;
    d.fv      = fv;
    d.nuMean  = nuMean;
    d.nuNucl  = nuNucl;
    d.alpha   = alpha;
    d.nbar0   = nbar0;
    d.sigma   = sigma;
    d.k       = k;
    d.nu1mean = nu1mean;
    d.nu2mean = nu2mean;
    d.mu      = mu;
    return d;
}

template <ThermoMap Thermo>
double MetalOxide<Thermo>::ReconstructedNormalizedNDF(double nu, bool use_regularized_moments) const
{
    if (!std::isfinite(nu) || nu <= 0.)
        return 0.;

    const auto d = ReconstructedNDFData(use_regularized_moments);
    if (!d.valid)
        return 0.;

    const double nuNucl = d.nuNucl;
    double nbar_p = 0., nbar_ln = 0.;

    // Pareto contribution
    if (d.alpha > 0. && d.k > 1. && nu >= nuNucl)
    {
        const double lg   = std::log(d.k) + d.k * std::log(nuNucl) - (d.k + 1.) * std::log(nu);
        const double lmin = std::log(std::numeric_limits<double>::min());
        const double lmax = std::log(std::numeric_limits<double>::max());
        if (lg > lmin && lg < lmax)
            nbar_p = std::exp(lg);
    }

    // Log-normal contribution
    if (d.alpha < 1. && d.sigma > 0. && d.nu2mean > 0.)
    {
        const double z = (std::log(nu) - d.mu) / d.sigma;
        const double lg =
            -std::log(nu) - std::log(d.sigma) - 0.5 * std::log(2. * this->pi_) - 0.5 * z * z;
        const double lmin = std::log(std::numeric_limits<double>::min());
        const double lmax = std::log(std::numeric_limits<double>::max());
        if (lg > lmin && lg < lmax)
            nbar_ln = std::exp(lg);
    }

    const double nbar = d.alpha * nbar_p + (1. - d.alpha) * nbar_ln;
    return (std::isfinite(nbar) && nbar >= 0.) ? nbar : 0.;
}

template <ThermoMap Thermo>
double MetalOxide<Thermo>::ReconstructedNDF(double nu, bool use_regularized_moments) const
{
    const auto d = ReconstructedNDFData(use_regularized_moments);
    if (!d.valid)
        return 0.;
    const double n = d.N * ReconstructedNormalizedNDF(nu, use_regularized_moments);
    return (std::isfinite(n) && n >= 0.) ? n : 0.;
}

template <ThermoMap Thermo>
void MetalOxide<Thermo>::ReconstructedNDF(const Eigen::VectorXd& nu,
                                    Eigen::VectorXd& n,
                                    bool use_regularized_moments) const
{
    n.resize(nu.size());
    for (int i = 0; i < nu.size(); ++i)
        n(i) = ReconstructedNDF(nu(i), use_regularized_moments);
}

#if defined(MOM_USE_DICTIONARY)
// ============================================================================
// ParseConfig — OpenSMOKE++ dictionary → MetalOxide::Config
// ============================================================================

template <ThermoMap Thermo>
template <typename DictType>
std::expected<typename MetalOxide<Thermo>::Config, std::string>
MetalOxide<Thermo>::ParseConfig(DictType& dict)
{
    // SetGrammar validates the dictionary against the grammar rules defined in
    // MetalOxide_Grammar::DefineRules().  On any violation (missing mandatory
    // keyword, unknown keyword, duplicate, conflicting keyword) it calls
    // Dictionary::ErrorMessage() which throws std::runtime_error.
    // That exception is intentionally NOT caught here so it propagates to the
    // caller and stops the simulation.
    MetalOxide_Grammar grammar;
    dict.SetGrammar(grammar);

    Config cfg;

    if (dict.CheckOption("@MetalOxide"))
        dict.ReadBool("@MetalOxide", cfg.is_active);

    if (dict.CheckOption("@Precursor"))
        dict.ReadString("@Precursor", cfg.precursor_species);

    if (dict.CheckOption("@SolidName"))
        dict.ReadString("@SolidName", cfg.solid_name);

    if (dict.CheckOption("@SolidMolecularWeight"))
    {
        double v; std::string u;
        dict.ReadMeasure("@SolidMolecularWeight", v, u);
        if (u == "kg/kmol") cfg.solid_molecular_weight_kg_kmol = v;
        else return std::unexpected(
            std::string{"@SolidMolecularWeight: allowed units: kg/kmol"});
    }

    if (dict.CheckOption("@SolidDensity"))
    {
        double v; std::string u;
        dict.ReadMeasure("@SolidDensity", v, u);
        if (u == "kg/m3")      cfg.solid_density_kg_m3 = v;
        else if (u == "g/cm3") cfg.solid_density_kg_m3 = v * 1000.;
        else return std::unexpected(std::string{"@SolidDensity: allowed units: kg/m3 | g/cm3"});
    }

    if (dict.CheckOption("@SolidFormulaUnitsPerPrecursor"))
        dict.ReadDouble("@SolidFormulaUnitsPerPrecursor",
                        cfg.solid_formula_units_per_precursor);

    if (dict.CheckOption("@GasClosureDummySpecies"))
        dict.ReadString("@GasClosureDummySpecies", cfg.gas_closure_dummy_species);

    if (dict.CheckOption("@GasConsumption"))
        dict.ReadBool("@GasConsumption", cfg.gas_consumption);

    if (dict.CheckOption("@GasStoichiometry"))
    {
        std::string spec;
        dict.ReadString("@GasStoichiometry", spec);
        std::ranges::replace(spec, '"', ' ');
        std::ranges::replace(spec, '\'', ' ');
        std::ranges::replace(spec, ',', ' ');
        std::ranges::replace(spec, ';', ' ');

        std::istringstream iss(spec);
        std::string token;
        while (iss >> token)
        {
            auto pos = token.find(':');
            if (pos == std::string::npos)
                pos = token.find('=');
            if (pos == std::string::npos || pos == 0u || pos + 1u >= token.size())
                return std::unexpected(
                    std::string{"@GasStoichiometry: expected entries like Species:-1"});

            const std::string species = token.substr(0u, pos);
            const std::string coefficient_text = token.substr(pos + 1u);

            try
            {
                std::size_t parsed = 0u;
                const double coefficient = std::stod(coefficient_text, &parsed);
                if (parsed != coefficient_text.size())
                    return std::unexpected(
                        std::string{"@GasStoichiometry: invalid coefficient in entry "} + token);
                cfg.gas_stoichiometry.push_back(GasStoichiometryTerm{species, coefficient});
            }
            catch (const std::exception&)
            {
                return std::unexpected(
                    std::string{"@GasStoichiometry: invalid coefficient in entry "} + token);
            }
        }
    }

    if (dict.CheckOption("@GasStoichiometryMassTolerance"))
        dict.ReadDouble("@GasStoichiometryMassTolerance",
                        cfg.gas_stoichiometry_mass_tolerance);

    if (dict.CheckOption("@NucleationModel"))
        dict.ReadString("@NucleationModel", cfg.nucleation_model);

    if (dict.CheckOption("@SinteringModel"))    dict.ReadInt("@SinteringModel",    cfg.sintering_model);
    if (dict.CheckOption("@CoagulationModel"))  dict.ReadInt("@CoagulationModel",  cfg.coagulation_model);
    if (dict.CheckOption("@CondensationModel")) dict.ReadInt("@CondensationModel", cfg.condensation_model);
    if (dict.CheckOption("@ThermophoreticModel")) dict.ReadInt("@ThermophoreticModel", cfg.thermophoretic_model);

    if (dict.CheckOption("@MinimumFormulaUnits"))
    {
        int n; dict.ReadInt("@MinimumFormulaUnits", n);
        cfg.minimum_formula_units = n;
    }

    if (dict.CheckOption("@NucleatedParticleFormulaUnits"))
    {
        int n; dict.ReadInt("@NucleatedParticleFormulaUnits", n);
        cfg.nucleated_particle_formula_units = n;
    }

    if (dict.CheckOption("@SinteringDeferred"))
        dict.ReadBool("@SinteringDeferred", cfg.sintering_deferred);

    if (dict.CheckOption("@SinteringDpMinimum"))
    {
        double v; std::string u;
        dict.ReadMeasure("@SinteringDpMinimum", v, u);
        if (u == "m")        cfg.sintering_dp_min_m = v;
        else if (u == "mm")  cfg.sintering_dp_min_m = v * 1.e-3;
        else if (u == "nm")  cfg.sintering_dp_min_m = v * 1.e-9;
        else return std::unexpected(std::string{"@SinteringDpMinimum: allowed units: m | mm | nm"});
    }

    if (dict.CheckOption("@SinteringTauMinimum"))
    {
        double v; std::string u;
        dict.ReadMeasure("@SinteringTauMinimum", v, u);
        if (u != "s") return std::unexpected(std::string{"@SinteringTauMinimum: allowed units: s"});
        cfg.sintering_tau_min_s = v;
    }

    if (dict.CheckOption("@SinteringKMaximum"))
    {
        double v; std::string u;
        dict.ReadMeasure("@SinteringKMaximum", v, u);
        if (u != "1/s") return std::unexpected(std::string{"@SinteringKMaximum: allowed units: 1/s"});
        cfg.sintering_k_max_per_s = v;
    }

    if (dict.CheckOption("@As"))
    {
        double v; std::string u;
        dict.ReadMeasure("@As", v, u);
        if (u != "s,K,m") return std::unexpected(std::string{"@As: allowed units: s,K,m"});
        cfg.sintering_As_s_K_m = v;
    }

    if (dict.CheckOption("@Ts"))
    {
        double v; std::string u;
        dict.ReadMeasure("@Ts", v, u);
        if (u != "K") return std::unexpected(std::string{"@Ts: allowed units: K"});
        cfg.sintering_Ts_K = v;
    }

    if (dict.CheckOption("@ns"))
        dict.ReadDouble("@ns", cfg.sintering_ns);

    if (dict.CheckOption("@SchmidtNumber"))
        dict.ReadDouble("@SchmidtNumber", cfg.schmidt_number);

    if (dict.CheckOption("@MinimumNs"))
    {
        double v; std::string u;
        dict.ReadMeasure("@MinimumNs", v, u);
        if (u == "#/m3")       cfg.ns_minimum_per_m3 = v;
        else if (u == "#/cm3") cfg.ns_minimum_per_m3 = v * 1.e6;
        else return std::unexpected(std::string{"@MinimumNs: allowed units: #/m3 | #/cm3"});
    }

    if (dict.CheckOption("@MinimumFv"))
        dict.ReadDouble("@MinimumFv", cfg.fv_minimum);

    if (dict.CheckOption("@DebugMode"))
        dict.ReadBool("@DebugMode", cfg.debug_mode);

    return cfg;
}
#endif // MOM_USE_DICTIONARY

} // namespace MOM
