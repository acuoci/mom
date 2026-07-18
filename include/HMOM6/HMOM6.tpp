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

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <numeric>

/**
 * @file HMOM6.tpp
 * @brief Template implementation for the HMOM6 soot source-term model
 *        (6+1 bivariate MOMIC closure).
 */

namespace MOM
{

// ---------------------------------------------------------------------------
// Local geometry helpers (anonymous namespace — translation-unit private).
// ---------------------------------------------------------------------------
namespace
{

/// K_diam = (6/pi)^(1/3)
inline double K_diam_HMOM6(double pi) noexcept
{
    return std::pow(6. / pi, 1. / 3.);
}

/// K_spher = (36*pi)^(1/3)
inline double K_spher_HMOM6(double pi) noexcept
{
    return std::pow(36. * pi, 1. / 3.);
}

/// Returns fractal geometry coefficients {Av, As, K}.
inline std::array<double, 3> FractalGeometry6(int model, double pi) noexcept
{
    double chi, Av, As, K;
    if (model == 1)
    {
        chi = -0.2043;
        Av  = -2. * chi - 1.;
        As  = 3. * chi;
        K   = (2. / 3.) * std::pow(1. / (36. * pi), chi);
    }
    else
    {
        chi = 0.;
        Av  = -1.;
        As  = 0.;
        K   = 2. / 3.;
    }
    (void)chi;
    return {Av, As, K};
}

/// Returns collision-diameter geometry coefficients {D, Av, As, K}.
inline std::array<double, 4> CollisionGeometry6(int model, double pi) noexcept
{
    double D, Av, As, K;
    if (model == 1)
    {
        D  = 2.;
        Av = 1. / 3.;
        As = 0.;
        K  = std::pow(6. / pi, 1. / 3.); // K_diam
    }
    else
    {
        D  = 1.8;
        Av = 1. - 2. / D;
        As = 3. / D - 1.;
        K  = 6. / std::pow(36. * pi, 1. / D);
    }
    return {D, Av, As, K};
}

} // anonymous namespace

// ===========================================================================
//  Construction & lifecycle
// ===========================================================================

template <ThermoMap Thermo>
HMOM6<Thermo>::HMOM6(const Thermo& thermo) : thermo_(thermo)
{
    // Species indices used by SetState() and HACA kinetics.
    index_H_    = thermo_.IndexOfSpecies("H");
    index_OH_   = thermo_.IndexOfSpecies("OH");
    index_O2_   = thermo_.IndexOfSpecies("O2");
    index_H2_   = thermo_.IndexOfSpecies("H2");
    index_H2O_  = thermo_.IndexOfSpecies("H2O");
    index_CO_   = thermo_.IndexOfSpecies("CO");
    index_C2H2_ = thermo_.IndexOfSpecies("C2H2");

    // Apply default configuration before cell-level calls are possible.
    ApplyConfig(Config{});

    MemoryAllocation();
}

// ===========================================================================
//  Configuration
// ===========================================================================

template <ThermoMap Thermo>
void HMOM6<Thermo>::ApplyConfig(const Config& cfg)
{
    this->is_active_ = cfg.is_active;

    // Particle density and geometry must be set before PAH-derived geometry.
    this->SetParticleDensity(cfg.soot_density_kg_m3);

    this->SetFractalDiameterModel(cfg.fractal_diameter_model);
    this->SetCollisionDiameterModel(cfg.collision_diameter_model);

    this->is_simplified_pah_mass_ = cfg.simplified_pah_mass;
    pah_species_ = std::string(cfg.pah_species);
    Precalculations();

    this->SetGasConsumption(cfg.gas_consumption);
    this->SetGasClosureDummySpecies(cfg.gas_closure_dummy_species);

    this->SetNucleation(static_cast<int>(cfg.nucleation_model));
    this->SetSurfaceGrowth(static_cast<int>(cfg.surface_growth_model));
    this->SetOxidation(static_cast<int>(cfg.oxidation_model));
    this->SetCondensation(static_cast<int>(cfg.condensation_model));
    this->SetCoagulation(static_cast<int>(cfg.coagulation_model));
    this->SetCoagulationContinuous(cfg.continuous_coagulation_model);
    this->SetThermophoreticModel(cfg.thermophoretic_model);

    this->SetRadiativeHeatTransfer(cfg.radiative_heat_transfer);
    this->SetPlanckAbsorptionCoefficient(cfg.planck_coefficient);
    this->SetSchmidtNumber(cfg.schmidt_number);

    this->SetSurfaceDensity(cfg.surface_density_per_m2);
    this->SetSurfaceDensityCorrectionCoefficient(cfg.surface_density_correction);
    this->SetSurfaceDensityCorrectionCoefficientA1(cfg.surf_dens_a1);
    this->SetSurfaceDensityCorrectionCoefficientA2(cfg.surf_dens_a2);
    this->SetSurfaceDensityCorrectionCoefficientB1(cfg.surf_dens_b1);
    this->SetSurfaceDensityCorrectionCoefficientB2(cfg.surf_dens_b2);

    this->SetStickingCoefficientModel(cfg.sticking_model);
    this->SetStickingCoefficientConstant(cfg.sticking_coeff_constant);

    // HACA kinetics: A in cm3/mol/s; activation energies converted to K.
    this->SetA1f(cfg.A1f);  this->Setn1f(cfg.n1f);  this->SetE1f(cfg.E1f);
    this->SetA1b(cfg.A1b);  this->Setn1b(cfg.n1b);  this->SetE1b(cfg.E1b);
    this->SetA2f(cfg.A2f);  this->Setn2f(cfg.n2f);  this->SetE2f(cfg.E2f);
    this->SetA2b(cfg.A2b);  this->Setn2b(cfg.n2b);  this->SetE2b(cfg.E2b);
    this->SetA3f(cfg.A3f);  this->Setn3f(cfg.n3f);  this->SetE3f(cfg.E3f);
    this->SetA3b(cfg.A3b);  this->Setn3b(cfg.n3b);  this->SetE3b(cfg.E3b);
    this->SetA4(cfg.A4);    this->Setn4(cfg.n4);     this->SetE4(cfg.E4);
    this->SetA5(cfg.A5);    this->Setn5(cfg.n5);     this->SetE5(cfg.E5);
    this->eff6_ = cfg.efficiency6;

    this->is_debug_mode_ = cfg.debug_mode;
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SetupFromConfig(const Config& cfg)
{
    ApplyConfig(cfg);
    PrintSummary();
}

// ===========================================================================
//  Memory and geometry setup
// ===========================================================================

template <ThermoMap Thermo>
void HMOM6<Thermo>::MemoryAllocation()
{
    this->ZeroSources();
    source_nucleation_.setZero();
    source_coagulation_.setZero();
    source_condensation_.setZero();
    source_growth_.setZero();
    source_oxidation_.setZero();

    source_coagulation_discrete_   = MomentVector::Zero();
    source_coagulation_ss_         = MomentVector::Zero();
    source_coagulation_sl_         = MomentVector::Zero();
    source_coagulation_ll_         = MomentVector::Zero();
    source_coagulation_continuous_ = MomentVector::Zero();
    source_coagulation_cont_ss_    = MomentVector::Zero();
    source_coagulation_cont_sl_    = MomentVector::Zero();
    source_coagulation_cont_ll_    = MomentVector::Zero();
    source_coagulation_all_        = MomentVector::Zero();

    this->omega_gas_     = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(thermo_.NumberOfSpecies()));
    omega_gas_oxidation_ = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(thermo_.NumberOfSpecies()));

    // Reset large-mode moments and MOMIC state.
    L00_ = L10_ = L01_ = L20_ = L11_ = L02_ = 0.;
    momic_coeffs_.fill(0.);
    momic_valid_ = false;

    // Initial normalised moments for solver start-up.
    //
    // Two-node bimodal initialisation:
    //   small node: N0s particles at (V0, S0)
    //   large node: NLs particles at (gamma*V0, gS*S0),  gS = gamma^{2/3}
    //
    // Normalisation: M_{x,y,norm} = M_{x,y} / (V0^x * S0^y * Nav)
    // so each normalised moment reduces to a pure number-density ratio.
    //
    // Index layout: [0]=M00, [1]=M10, [2]=M01, [3]=M20, [4]=M11, [5]=M02, [6]=N0
    const double Nseed   = 1.e10;
    const double eta     = 0.90;
    const double gamma   = 2.0;
    const double N0s     = eta * Nseed;
    const double NLs     = (1.0 - eta) * Nseed;
    const double gS      = std::pow(gamma, 2. / 3.); // SL = gS * S0

    initial_moments_cache_(0) = Nseed                        / this->Nav_mol_; // M00
    initial_moments_cache_(1) = (N0s + gamma      * NLs)     / this->Nav_mol_; // M10
    initial_moments_cache_(2) = (N0s + gS         * NLs)     / this->Nav_mol_; // M01
    initial_moments_cache_(3) = (N0s + gamma*gamma * NLs)    / this->Nav_mol_; // M20
    initial_moments_cache_(4) = (N0s + gamma*gS   * NLs)     / this->Nav_mol_; // M11
    initial_moments_cache_(5) = (N0s + gS*gS      * NLs)     / this->Nav_mol_; // M02
    initial_moments_cache_(6) = N0s                           / this->Nav_mol_; // N0
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::Precalculations()
{
    Cfm_                = std::sqrt(this->pi_ * this->kB_ / 2.0 / this->rho_particle_);
    const double K_diam = K_diam_HMOM6(this->pi_);
    betaN_TV_           = 2.2 * 4. * this->sqrt2_ * K_diam * K_diam * Cfm_;

    // Update geometry coefficients from the selected sub-models.
    {
        auto [Av, As, K] = FractalGeometry6(static_cast<int>(fractal_diameter_model_), this->pi_);
        Av_fractal_      = Av;
        As_fractal_      = As;
        K_fractal_       = K;
    }
    {
        auto [D, Av, As, K] =
            CollisionGeometry6(static_cast<int>(collision_diameter_model_), this->pi_);
        D_collisional_  = D;
        Av_collisional_ = Av;
        As_collisional_ = As;
        K_collisional_  = K;
    }

    SetPAH(pah_species_);
}

// ===========================================================================
//  Species / PAH setup
// ===========================================================================

template <ThermoMap Thermo>
void HMOM6<Thermo>::SetPAH(std::string_view name)
{
    const auto pah_index = thermo_.IndexOfSpecies(name);
    if (pah_index < 0)
        throw std::runtime_error("[HMOM6] PAH species not found in mechanism: " + std::string(name));

    const auto pah_index_u = static_cast<unsigned>(pah_index);
    const auto ncpah       = static_cast<double>(thermo_.NumberOfCarbonAtoms(pah_index_u));
    const auto nhpah       = static_cast<double>(thermo_.NumberOfHydrogenAtoms(pah_index_u));

    // PAH mass and sphere-equivalent geometry.
    auto mwpah = thermo_.MolecularWeight(pah_index_u); // [kg/kmol]
    if (is_simplified_pah_mass_)
        mwpah = ncpah * this->WC_;

    const auto vpah = mwpah / this->rho_particle_ / this->Nav_kmol_; // [m3]
    const auto dpah = this->SphereDiameter(vpah);                    // [m]
    const auto spah = this->SphereSurface(dpah);                     // [m2]
    const auto mpah = mwpah / this->Nav_kmol_;                       // [kg]

    pah_species_ = std::string(name);
    pah_index_   = pah_index;
    ncpah_       = ncpah;
    nhpah_       = nhpah;
    mwpah_       = mwpah;
    vpah_        = vpah;
    dpah_        = dpah;
    spah_        = spah;
    mpah_        = mpah;

    dimer_volume_  = 2. * vpah_;
    dimer_surface_ = this->SphereSurfaceFromVolume(dimer_volume_);
    V0_            = 2. * dimer_volume_;
    S0_            = this->SphereSurfaceFromVolume(V0_);
    VC2_           = (this->WC_ / this->rho_particle_ / this->Nav_kmol_) * 2.;
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SetGasClosureDummySpecies(std::string_view name)
{
    this->closure_dummy_species_ = std::string(name);

    if (name == "none")
    {
        this->closure_dummy_index_      = -1;
        this->is_closure_dummy_species_ = false;
        return;
    }

    this->closure_dummy_index_ = thermo_.IndexOfSpecies(name);
    if (this->closure_dummy_index_ < 0)
        throw std::runtime_error("[HMOM6] Dummy species not found: " + this->closure_dummy_species_);

    if (this->closure_dummy_index_ == pah_index_)
        throw std::runtime_error("[HMOM6] Dummy species cannot be the same as the PAH precursor.");

    if (this->closure_dummy_index_ == index_H_  || this->closure_dummy_index_ == index_H2_  ||
        this->closure_dummy_index_ == index_O2_ || this->closure_dummy_index_ == index_OH_  ||
        this->closure_dummy_index_ == index_H2O_|| this->closure_dummy_index_ == index_C2H2_)
        throw std::runtime_error("[HMOM6] Dummy species cannot be H, H2, O2, OH, H2O, or C2H2.");

    this->is_closure_dummy_species_ = true;
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SetStickingCoefficientModel(std::string_view label)
{
    if (label == "constant")
        sticking_model_ = StickingModel::Constant;
    else if (label == "pah-dependent")
        sticking_model_ = StickingModel::PAH4;
    else
        throw std::runtime_error("[HMOM6] Unknown sticking coefficient model: " + std::string(label));
}

// ===========================================================================
//  State injection
// ===========================================================================

template <ThermoMap Thermo>
void HMOM6<Thermo>::SetState(double T, double P_Pa, const double* Y) noexcept
{
    const double cTot = this->template UpdateMixtureState<>(T, P_Pa, Y, thermo_);

    // Mass fractions used by the smooth active-site damping factor.
    mass_fraction_H_  = (index_H_  >= 0) ? std::max(Y[index_H_],  0.) : 0.;
    mass_fraction_OH_ = (index_OH_ >= 0) ? std::max(Y[index_OH_], 0.) : 0.;

    conc_H_    = this->SpeciesConcentrationMolCm3(index_H_,    Y, cTot, thermo_);
    conc_OH_   = this->SpeciesConcentrationMolCm3(index_OH_,   Y, cTot, thermo_);
    conc_O2_   = this->SpeciesConcentrationMolCm3(index_O2_,   Y, cTot, thermo_);
    conc_H2_   = this->SpeciesConcentrationMolCm3(index_H2_,   Y, cTot, thermo_);
    conc_H2O_  = this->SpeciesConcentrationMolCm3(index_H2O_,  Y, cTot, thermo_);
    conc_C2H2_ = this->SpeciesConcentrationMolCm3(index_C2H2_, Y, cTot, thermo_);

    // PAH concentration [mol/cm3].
    conc_PAH_  = this->SpeciesConcentrationMolCm3(pah_index_,  Y, cTot, thermo_);
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SetMoments(std::span<const double> m) noexcept
{
    assert(m.size() == static_cast<std::size_t>(Base::n_equations) &&
           "[HMOM6::SetMoments] expected exactly 7 moment values.");
    SetNormalizedMoments(m[0], m[1], m[2], m[3], m[4], m[5], m[6]);
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SetNormalizedMoments(double M00_norm,
                                         double M10_norm,
                                         double M01_norm,
                                         double M20_norm,
                                         double M11_norm,
                                         double M02_norm,
                                         double N0_norm) noexcept
{
    M00_normalized_ = M00_norm;
    M10_normalized_ = M10_norm;
    M01_normalized_ = M01_norm;
    M20_normalized_ = M20_norm;
    M11_normalized_ = M11_norm;
    M02_normalized_ = M02_norm;
    N0_normalized_  = N0_norm;
    GetMoments();
}

// ===========================================================================
//  Moment reconstruction
// ===========================================================================

template <ThermoMap Thermo>
void HMOM6<Thermo>::GetMoments()
{
    // De-normalise: M_{x,y} = M_{x,y,norm} * V0^x * S0^y * Nav
    double M00_raw = M00_normalized_ * this->Nav_mol_;
    double M10_raw = M10_normalized_ * V0_         * this->Nav_mol_;
    double M01_raw = M01_normalized_ * S0_         * this->Nav_mol_;
    double M20_raw = M20_normalized_ * V0_ * V0_   * this->Nav_mol_;
    double M11_raw = M11_normalized_ * V0_ * S0_   * this->Nav_mol_;
    double M02_raw = M02_normalized_ * S0_ * S0_   * this->Nav_mol_;
    double N0_raw  = N0_normalized_  * this->Nav_mol_;

    // NaN guards — replace non-finite values with zero.
    if (!std::isfinite(M00_raw)) M00_raw = 0.;
    if (!std::isfinite(M10_raw)) M10_raw = 0.;
    if (!std::isfinite(M01_raw)) M01_raw = 0.;
    if (!std::isfinite(M20_raw)) M20_raw = 0.;
    if (!std::isfinite(M11_raw)) M11_raw = 0.;
    if (!std::isfinite(M02_raw)) M02_raw = 0.;
    if (!std::isfinite(N0_raw))  N0_raw  = 0.;

    // All moments are non-negative by physics; clamp numerical noise.
    M00_raw = std::max(M00_raw, 0.);
    M10_raw = std::max(M10_raw, 0.);
    M01_raw = std::max(M01_raw, 0.);
    M20_raw = std::max(M20_raw, 0.);
    M11_raw = std::max(M11_raw, 0.);
    M02_raw = std::max(M02_raw, 0.);
    N0_raw  = std::max(N0_raw,  0.);

    // If the primary moments are below numerical floors treat as particle-free.
    if (M00_raw < kSootNumberFloor || M10_raw < kSootVolumeFloor || M01_raw < kSootSurfaceFloor)
    {
        M00_ = M10_ = M01_ = M20_ = M11_ = M02_ = N0_ = 0.;
        L00_ = L10_ = L01_ = L20_ = L11_ = L02_ = 0.;
        momic_valid_ = false;
        return;
    }

    // Clamp N0 so that every large-mode moment L_{x,y} = M_{x,y} - N0*V0^x*S0^y >= 0.
    // This guarantees that the MOMIC log system is well-posed.
    N0_raw = std::min(N0_raw, M00_raw);
    if (V0_ > 0. && S0_ > 0.)
    {
        N0_raw = std::min(N0_raw, M10_raw / V0_);
        N0_raw = std::min(N0_raw, M01_raw / S0_);
        N0_raw = std::min(N0_raw, M20_raw / (V0_ * V0_));
        N0_raw = std::min(N0_raw, M11_raw / (V0_ * S0_));
        N0_raw = std::min(N0_raw, M02_raw / (S0_ * S0_));
    }
    N0_raw = std::max(N0_raw, 0.);

    M00_ = M00_raw;
    M10_ = M10_raw;
    M01_ = M01_raw;
    M20_ = M20_raw;
    M11_ = M11_raw;
    M02_ = M02_raw;
    N0_  = N0_raw;

    // Large-mode moments: L_{x,y} = M_{x,y} - N0 * V0^x * S0^y
    L00_ = M00_ - N0_;
    L10_ = M10_ - V0_       * N0_;
    L01_ = M01_ - S0_       * N0_;
    L20_ = M20_ - V0_*V0_   * N0_;
    L11_ = M11_ - V0_*S0_   * N0_;
    L02_ = M02_ - S0_*S0_   * N0_;

    // Tolerance correction: numerical discretisation may produce small negative
    // residuals. Flush to zero if within 1e-12 relative to the parent moment.
    auto toleranceFloor = [](double& L, double ref)
    {
        if (L < 0. && std::fabs(L) < 1.e-12 * std::max(ref, 1.))
            L = 0.;
        L = std::max(L, 0.);
    };
    toleranceFloor(L00_, M00_);
    toleranceFloor(L10_, M10_);
    toleranceFloor(L01_, M01_);
    toleranceFloor(L20_, M20_);
    toleranceFloor(L11_, M11_);
    toleranceFloor(L02_, M02_);

    // With the large-mode moments available, solve the MOMIC polynomial.
    ComputeMOMICCoefficients();
}

// ===========================================================================
//  MOMIC closure
// ===========================================================================

template <ThermoMap Thermo>
void HMOM6<Thermo>::ComputeMOMICCoefficients()
{
    // TODO: implement
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::MOMICPolynomial(double x, double y) const noexcept
{
    // TODO: implement
    return 0.;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::GetMoment(double x, double y) const noexcept
{
    // TODO: implement
    return 0.;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::GetMissingMoment(double x, double y) const noexcept
{
    // TODO: implement
    return 0.;
}

// ===========================================================================
//  Helper predicates and utilities
// ===========================================================================

template <ThermoMap Thermo>
bool HMOM6<Thermo>::HasSoot() const noexcept
{
    // TODO: implement
    return false;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::SafePowPositive(double x, double a) const noexcept
{
    // TODO: implement
    return 0.;
}

// ===========================================================================
//  Kinetics helpers
// ===========================================================================

template <ThermoMap Thermo>
double HMOM6<Thermo>::GetBetaC() const noexcept
{
    // TODO: implement
    return 0.;
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::DimerConcentration()
{
    // TODO: implement
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::PAHDimerizationRate() const noexcept
{
    // TODO: implement
    return 0.;
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::CalculateAlphaCoefficient()
{
    // TODO: implement
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootKineticConstants()
{
    // TODO: implement
}

// ===========================================================================
//  Source terms — individual processes
// ===========================================================================

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootNucleationM7()
{
    // TODO: implement
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootSurfaceGrowthM7()
{
    // TODO: implement
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootOxidationM7()
{
    // TODO: implement
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootCondensationM7()
{
    // TODO: implement
}

// ===========================================================================
//  Coagulation — discrete (free-molecular) regime
// ===========================================================================

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootCoagulationM7()
{
    // TODO: implement
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootCoagulationSmallSmallM7()
{
    // TODO: implement
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootCoagulationSmallLargeM7()
{
    // TODO: implement
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootCoagulationLargeLargeM7()
{
    // TODO: implement
}

// ===========================================================================
//  Coagulation — continuum correction
// ===========================================================================

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootCoagulationContinuousM7()
{
    // TODO: implement
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootCoagulationContinuousSmallSmallM7(double lambda, double betai0)
{
    // TODO: implement
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootCoagulationContinuousSmallLargeM7(double lambda, double betai0)
{
    // TODO: implement
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootCoagulationContinuousLargeLargeM7(double lambda, double betai0)
{
    // TODO: implement
}

// ===========================================================================
//  Main source-term orchestrator
// ===========================================================================

template <ThermoMap Thermo>
void HMOM6<Thermo>::ComputeSources() noexcept
{
    // TODO: implement
}

// ===========================================================================
//  Gas-phase consumption
// ===========================================================================

template <ThermoMap Thermo>
void HMOM6<Thermo>::CalculateOmegaGas() noexcept
{
    // TODO: implement
}

// ===========================================================================
//  Particle properties
// ===========================================================================

template <ThermoMap Thermo>
double HMOM6<Thermo>::volume_fraction() const noexcept
{
    // TODO: implement
    return 0.;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::particle_diameter() const noexcept
{
    // TODO: implement
    return 0.;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::collision_diameter() const noexcept
{
    // TODO: implement
    return 0.;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::particle_number_density() const noexcept
{
    // TODO: implement
    return 0.;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::mass_fraction() const noexcept
{
    // TODO: implement
    return 0.;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::specific_surface_area() const noexcept
{
    // TODO: implement
    return 0.;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::number_primary_particles() const noexcept
{
    // TODO: implement
    return 0.;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::diffusion_coefficient() const noexcept
{
    // TODO: implement
    return 0.;
}

// ===========================================================================
//  Diagnostics
// ===========================================================================

template <ThermoMap Thermo>
void HMOM6<Thermo>::PrintSummary() const
{
    // TODO: implement
}

// ===========================================================================
//  HACA kinetics setters
// ===========================================================================

template <ThermoMap Thermo> void HMOM6<Thermo>::SetA1f(double v) noexcept { A1f_ = v; }
template <ThermoMap Thermo> void HMOM6<Thermo>::SetA1b(double v) noexcept { A1b_ = v; }
template <ThermoMap Thermo> void HMOM6<Thermo>::SetA2f(double v) noexcept { A2f_ = v; }
template <ThermoMap Thermo> void HMOM6<Thermo>::SetA2b(double v) noexcept { A2b_ = v; }
template <ThermoMap Thermo> void HMOM6<Thermo>::SetA3f(double v) noexcept { A3f_ = v; }
template <ThermoMap Thermo> void HMOM6<Thermo>::SetA3b(double v) noexcept { A3b_ = v; }
template <ThermoMap Thermo> void HMOM6<Thermo>::SetA4(double v) noexcept  { A4_  = v; }
template <ThermoMap Thermo> void HMOM6<Thermo>::SetA5(double v) noexcept  { A5_  = v; }

template <ThermoMap Thermo> void HMOM6<Thermo>::SetE1f(double kJ) noexcept { E1f_ = kJ * 1e3 / this->Rgas_mol_; }
template <ThermoMap Thermo> void HMOM6<Thermo>::SetE1b(double kJ) noexcept { E1b_ = kJ * 1e3 / this->Rgas_mol_; }
template <ThermoMap Thermo> void HMOM6<Thermo>::SetE2f(double kJ) noexcept { E2f_ = kJ * 1e3 / this->Rgas_mol_; }
template <ThermoMap Thermo> void HMOM6<Thermo>::SetE2b(double kJ) noexcept { E2b_ = kJ * 1e3 / this->Rgas_mol_; }
template <ThermoMap Thermo> void HMOM6<Thermo>::SetE3f(double kJ) noexcept { E3f_ = kJ * 1e3 / this->Rgas_mol_; }
template <ThermoMap Thermo> void HMOM6<Thermo>::SetE3b(double kJ) noexcept { E3b_ = kJ * 1e3 / this->Rgas_mol_; }
template <ThermoMap Thermo> void HMOM6<Thermo>::SetE4(double kJ) noexcept  { E4_  = kJ * 1e3 / this->Rgas_mol_; }
template <ThermoMap Thermo> void HMOM6<Thermo>::SetE5(double kJ) noexcept  { E5_  = kJ * 1e3 / this->Rgas_mol_; }

template <ThermoMap Thermo> void HMOM6<Thermo>::Setn1f(double v) noexcept { n1f_ = v; }
template <ThermoMap Thermo> void HMOM6<Thermo>::Setn1b(double v) noexcept { n1b_ = v; }
template <ThermoMap Thermo> void HMOM6<Thermo>::Setn2f(double v) noexcept { n2f_ = v; }
template <ThermoMap Thermo> void HMOM6<Thermo>::Setn2b(double v) noexcept { n2b_ = v; }
template <ThermoMap Thermo> void HMOM6<Thermo>::Setn3f(double v) noexcept { n3f_ = v; }
template <ThermoMap Thermo> void HMOM6<Thermo>::Setn3b(double v) noexcept { n3b_ = v; }
template <ThermoMap Thermo> void HMOM6<Thermo>::Setn4(double v) noexcept  { n4_  = v; }
template <ThermoMap Thermo> void HMOM6<Thermo>::Setn5(double v) noexcept  { n5_  = v; }

// ===========================================================================
//  Dictionary-based configuration (optional)
// ===========================================================================

#if defined(MOM_USE_DICTIONARY)

template <ThermoMap Thermo>
template <typename DictType>
std::expected<typename HMOM6<Thermo>::Config, std::string>
HMOM6<Thermo>::ParseConfig(DictType& dict)
{
    // TODO: implement (mirrors HMOM::ParseConfig with HMOM6_Grammar)
    return Config{};
}

#endif // MOM_USE_DICTIONARY

} // namespace MOM
