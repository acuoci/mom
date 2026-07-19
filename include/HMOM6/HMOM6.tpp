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
    momic_valid_ = false;
    momic_coeffs_.fill(0.);

    // All six large-mode moments must be strictly positive so that their
    // logarithms are finite.  kMOMICEps (= 1e-300) guards against log(0).
    if (L00_ < kMOMICEps || L10_ < kMOMICEps || L01_ < kMOMICEps ||
        L20_ < kMOMICEps || L11_ < kMOMICEps || L02_ < kMOMICEps)
        return;

    const double lnL00 = std::log(L00_);
    const double lnL10 = std::log(L10_);
    const double lnL01 = std::log(L01_);
    const double lnL20 = std::log(L20_);
    const double lnL11 = std::log(L11_);
    const double lnL02 = std::log(L02_);

    // Analytical closed-form solution of the 6x6 linear system
    //   ln(L_{x,y}) = p(x,y)   at  (x,y) ∈ {(0,0),(1,0),(0,1),(2,0),(1,1),(0,2)}
    //
    // Polynomial basis (Mueller 2009, R=2, Eq. 6):
    //   p(x,y) = a00 + a10·y + a11·x + a20·y² + a21·x·y + a22·x²
    //
    // Evaluation at the six integer moment orders gives the system:
    //   p(0,0) = a00                                    = lnL00  ...(1)
    //   p(1,0) = a00 +      a11 +           a22         = lnL10  ...(2)
    //   p(0,1) = a00 + a10 +      a20                  = lnL01  ...(3)
    //   p(2,0) = a00 +    2·a11 +         4·a22         = lnL20  ...(4)
    //   p(1,1) = a00 + a10 + a11 + a20 + a21 + a22     = lnL11  ...(5)
    //   p(0,2) = a00 + 2·a10 +    4·a20                = lnL02  ...(6)
    //
    // Solution:
    //   From (1):           a00 = lnL00
    //   From (4)-2·(2):     a22 = (lnL00 - 2·lnL10 + lnL20) / 2
    //   From (2):           a11 = lnL10 - lnL00 - a22
    //   From (6)-2·(3):     a20 = (lnL00 - 2·lnL01 + lnL02) / 2
    //   From (3):           a10 = lnL01 - lnL00 - a20
    //   From (5):           a21 = lnL11 - a00 - a10 - a11 - a20 - a22

    const double a00 = lnL00;
    const double a22 = 0.5 * (lnL00 - 2. * lnL10 + lnL20);
    const double a11 = lnL10 - lnL00 - a22;
    const double a20 = 0.5 * (lnL00 - 2. * lnL01 + lnL02);
    const double a10 = lnL01 - lnL00 - a20;
    const double a21 = lnL11 - a00 - a10 - a11 - a20 - a22;

    // All coefficients must be finite (guards against degenerate L distributions).
    if (!std::isfinite(a00) || !std::isfinite(a10) || !std::isfinite(a11) ||
        !std::isfinite(a20) || !std::isfinite(a21) || !std::isfinite(a22))
        return;

    momic_coeffs_[0] = a00;  // a00
    momic_coeffs_[1] = a10;  // a10  (coefficient of y)
    momic_coeffs_[2] = a11;  // a11  (coefficient of x)
    momic_coeffs_[3] = a20;  // a20  (coefficient of y²)
    momic_coeffs_[4] = a21;  // a21  (coefficient of x·y)
    momic_coeffs_[5] = a22;  // a22  (coefficient of x²)
    momic_valid_ = true;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::MOMICPolynomial(double x, double y) const noexcept
{
    // p(x,y) = a00 + a10·y + a11·x + a20·y² + a21·x·y + a22·x²
    // momic_coeffs_: [0]=a00, [1]=a10, [2]=a11, [3]=a20, [4]=a21, [5]=a22
    return momic_coeffs_[0]
         + momic_coeffs_[1] * y
         + momic_coeffs_[2] * x
         + momic_coeffs_[3] * y * y
         + momic_coeffs_[4] * x * y
         + momic_coeffs_[5] * x * x;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::GetMoment(double x, double y) const noexcept
{
    if (!HasSoot())
        return 0.;

    double moment = 0.;

    // Small-mode (delta function at V0, S0) contribution: N0 · V0^x · S0^y
    if (N0_ > 0.)
    {
        const double small = N0_ * SafePowPositive(V0_, x) * SafePowPositive(S0_, y);
        if (std::isfinite(small))
            moment += small;
    }

    // Large-mode (MOMIC polynomial) contribution: exp(p(x, y))
    if (momic_valid_)
    {
        const double lp = MOMICPolynomial(x, y);
        // Guard against overflow (exp > ~DBL_MAX) and underflow (negligible).
        double large = 0.;
        if      (lp >  700.) large = std::exp(700.);
        else if (lp > -700.) large = std::exp(lp);
        // lp <= -700 → large ≈ 0, already initialised to 0.
        if (std::isfinite(large))
            moment += large;
    }

    if (!std::isfinite(moment) || moment < 0.)
        return 0.;
    return moment;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::GetMissingMoment(double x, double y) const noexcept
{
    // Large-mode only: exp(p(x, y)).
    // Returns 0 when soot is absent or the MOMIC system could not be solved.
    if (!HasSoot() || !momic_valid_)
        return 0.;

    const double lp = MOMICPolynomial(x, y);
    double moment = 0.;
    if      (lp >  700.) moment = std::exp(700.);
    else if (lp > -700.) moment = std::exp(lp);
    // lp <= -700 → negligible large-mode contribution → moment stays 0.

    if (!std::isfinite(moment) || moment < 0.)
        return 0.;
    return moment;
}

// ===========================================================================
//  Helper predicates and utilities
// ===========================================================================

template <ThermoMap Thermo>
bool HMOM6<Thermo>::HasSoot() const noexcept
{
    return (M00_ > kSootNumberFloor && M10_ > kSootVolumeFloor && M01_ > kSootSurfaceFloor &&
            std::isfinite(M00_) && std::isfinite(M10_) && std::isfinite(M01_));
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::SafePowPositive(double x, double a) const noexcept
{
    if (!std::isfinite(x) || x <= 0.)
        return 0.;
    const double y = a * std::log(x);
    if (y >  700.) return std::exp(700.);
    if (y < -700.) return 0.;
    return std::exp(y);
}

// ===========================================================================
//  Kinetics helpers
// ===========================================================================

template <ThermoMap Thermo>
double HMOM6<Thermo>::GetBetaC() const noexcept
{
    // Free-molecular collision rate between the PAH dimer and the soot population.
    // This is the integral β_C = Cfm·√T · Σ_k (geometric cross-section terms),
    // identical in structure to HMOM — the only difference is that GetMoment()
    // now uses the MOMIC polynomial for the large mode instead of two-delta.
    //
    // Geometric mean collision kernel (free-molecular, Eq. 28 of Mueller 2009):
    //   β(d_dim, d_p) ∝ (d_dim^{-1/2} + d_p^{-1/2}) · (d_dim + d_p)^2
    // Expressed via collision-diameter model:  d_c = K_coll · V^{Av} · S^{As}
    // and dimer diameter:                      d_dim = K_diam · V_dim^{1/3}

    const double K_diam = K_diam_HMOM6(this->pi_);

    const double betaC =
        K_collisional_ * K_collisional_ * std::pow(dimer_volume_, -3. / 6.) *
            GetMoment(2. * Av_collisional_, 2. * As_collisional_) +
        2. * K_diam * K_collisional_ * std::pow(dimer_volume_, -1. / 6.) *
            GetMoment(Av_collisional_, As_collisional_) +
        K_diam * K_diam * std::pow(dimer_volume_, 1. / 6.) *
            GetMoment(0., 0.) +
        0.5 * K_collisional_ * K_collisional_ * std::pow(dimer_volume_, 3. / 6.) *
            GetMoment(2. * Av_collisional_ - 1., 2. * As_collisional_) +
        2. * K_diam * K_collisional_ * std::pow(dimer_volume_, 5. / 6.) *
            GetMoment(Av_collisional_ - 1., As_collisional_) +
        0.5 * K_diam * K_diam * std::pow(dimer_volume_, 7. / 6.) *
            GetMoment(-1., 0.);

    return Cfm_ * std::sqrt(this->T_) * betaC;
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::DimerConcentration()
{
    conc_DIMER_        = 0.;
    dimerization_rate_ = 0.;

    betaN_ = betaN_TV_ * std::sqrt(this->T_) * std::pow(dimer_volume_, 1. / 6.);

    double betaC = 0.;
    if (condensation_model_ > 0)
        betaC = GetBetaC();

    if (betaN_ <= 0. || !std::isfinite(betaN_))
        return;

    double stickCoeff = sticking_coeff_constant_;
    if (sticking_model_ == StickingModel::PAH4)
        stickCoeff = sticking_coeff_constant_ * std::pow(mwpah_, 4.);

    const double KfmPAH = betaN_TV_ * std::sqrt(this->T_) * std::pow(vpah_, 1. / 6.);
    if (KfmPAH <= 0. || !std::isfinite(KfmPAH))
        return;
    if (betaC < 0. || !std::isfinite(betaC))
        return;

    // PAH dimerization: 2 PAH → dimer (collision-limited).
    const double beta_pah_pah = 0.5 * stickCoeff * KfmPAH;
    const double C_PAH        = std::max(conc_PAH_, 0.) * 1.e6; // mol/cm3 → mol/m3
    const double reduced_rate  = beta_pah_pah * C_PAH * C_PAH;

    dimerization_rate_ = reduced_rate;
    if (dimerization_rate_ <= 0. || !std::isfinite(dimerization_rate_))
    {
        dimerization_rate_ = 0.;
        return;
    }

    // Steady-state dimer number density from the quadratic balance:
    //   Jpah = betaN * Ndim^2 + betaC * Ndim
    // where Jpah = dimerization_rate * Nav^2.
    const double Jpah = dimerization_rate_ * this->Nav_mol_ * this->Nav_mol_;
    if (Jpah <= 0. || !std::isfinite(Jpah))
        return;

    const double discriminant = betaC * betaC + 4. * betaN_ * Jpah;
    if (discriminant <= 0. || !std::isfinite(discriminant))
        return;

    double Ndim = 2. * Jpah / (betaC + std::sqrt(discriminant));
    if (Ndim < 0. || !std::isfinite(Ndim))
        Ndim = 0.;

    conc_DIMER_ = Ndim / this->Nav_mol_;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::PAHDimerizationRate() const noexcept
{
    // Convert from [mol/m3/s] to [kmol/m3/s] for gas-phase coupling convention.
    return 2. * dimerization_rate_ * this->Nav_mol_ / 1000.;
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::CalculateAlphaCoefficient()
{
    alpha_ = 1.;
    if (!std::isfinite(this->T_) || this->T_ <= 0.)
        return;

    const double a = surf_dens_a1_ + surf_dens_a2_ * this->T_;
    const double b = surf_dens_b1_ + surf_dens_b2_ * this->T_;

    // Large-mode mean volume: VL = L10 / L00 (MOMIC equivalent of NLVL/NL).
    double VL = V0_;
    if (L00_ > kSootNumberFloor && L10_ > kSootVolumeFloor)
        VL = L10_ / L00_;
    if (!std::isfinite(VL) || VL <= 0.)
        VL = V0_;

    const double mC  = this->WC_ / this->Nav_kmol_;
    const double mu1 = VL * this->rho_particle_ / mC;
    if (!std::isfinite(mu1) || mu1 <= 1.)
        return;

    const double logMu1 = std::log(mu1);
    if (!std::isfinite(logMu1) || std::fabs(logMu1) < 1.e-12)
        return;

    // Appel et al. surface-density correction for mature soot particles.
    alpha_ = std::tanh(a / logMu1 + b);
    if (!std::isfinite(alpha_))
        alpha_ = 1.;
    alpha_ = std::max(0., std::min(alpha_, 1.));
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootKineticConstants()
{
    const double T   = this->T_;
    const double k1f = A1f_ * std::pow(T, n1f_) * std::exp(-E1f_ / T);
    const double k1b = A1b_ * std::pow(T, n1b_) * std::exp(-E1b_ / T);
    const double k2f = A2f_ * std::pow(T, n2f_) * std::exp(-E2f_ / T);
    const double k2b = A2b_ * std::pow(T, n2b_) * std::exp(-E2b_ / T);
    const double k3f = A3f_ * std::pow(T, n3f_) * std::exp(-E3f_ / T);
    const double k3b = A3b_ * std::pow(T, n3b_) * std::exp(-E3b_ / T);
    const double k4  = A4_  * std::pow(T, n4_)  * std::exp(-E4_  / T);

    // Active-site fraction (HACA steady-state):
    //   χ* / (1 + χ*),  χ* = (k1f[OH] + k2f[H] + k3f) / (k1b[H2O] + k2b[H2] + k3b[H] + k4[C2H2])
    const double ratio = k1b * conc_H2O_ + k2b * conc_H2_ + k3b * conc_H_ + k4 * conc_C2H2_;
    double conc_sootStar = 0.;
    if (ratio > 0.)
        conc_sootStar = (k1f * conc_OH_ + k2f * conc_H_ + k3f) / ratio;

    // Smooth low-radical damping: suppress HACA when both H and OH pools are depleted.
    {
        constexpr double YH_threshold  = 2.e-9;
        constexpr double YOH_threshold = 2.e-8;
        const double rampH  = 0.5 * (1. + std::tanh((mass_fraction_H_  - YH_threshold)
                                                     / (0.5 * YH_threshold)));
        const double rampOH = 0.5 * (1. + std::tanh((mass_fraction_OH_ - YOH_threshold)
                                                     / (0.5 * YOH_threshold)));
        conc_sootStar *= rampH * rampOH;
    }

    conc_sootStar = conc_sootStar / (1. + conc_sootStar);
    conc_sootStar = std::max(conc_sootStar, 0.);

    ksg_    = k4 * conc_C2H2_ * conc_sootStar;
    const double k6 = 8.94 * eff6_ * std::sqrt(T) * this->Nav_mol_;
    kox_O2_ = A5_ * std::pow(T, n5_) * std::exp(-E5_ / T) * conc_O2_ * conc_sootStar;
    kox_OH_ = (0.5 / (alpha_ * surface_density_)) * k6 * conc_OH_;
    kox_    = kox_O2_ + kox_OH_;
}

// ===========================================================================
//  Source terms — individual processes
// ===========================================================================

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootNucleationM7()
{
    this->source_nucleation_.setZero();

    // Nucleation rate: J = 0.5 * β_N * [DIMER]²  [particles/m³/s]
    // Normalised source for M_{x,y,norm} = M_{x,y} / (V0^x · S0^y · Nav):
    //   d(M_{x,y,norm})/dt|_nuc = J · (V0^x · S0^y) / (V0^x · S0^y · Nav) = J / Nav
    //
    // The factor of V0^x · S0^y cancels for every (x,y), so all seven normalised
    // moment sources from nucleation are identical — including N0 (nucleated
    // particles always enter the small mode at (V0, S0)).
    double nuc_N0 = 0.;
    if (conc_DIMER_ > 0. && betaN_ > 0. && std::isfinite(conc_DIMER_) && std::isfinite(betaN_))
        nuc_N0 = 0.5 * betaN_ * conc_DIMER_ * conc_DIMER_ * this->Nav_mol_;
    if (!std::isfinite(nuc_N0) || nuc_N0 < 0.)
        nuc_N0 = 0.;

    for (unsigned i = 0; i < 7u; ++i)
        this->source_nucleation_(i) = nuc_N0;
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootSurfaceGrowthM7()
{
    // Surface growth adds volume δV = VC2_ per C2H2 reaction at an active surface site.
    // General source for the normalised moment M_{x,y,norm} (Mueller 2009, Eq. 37):
    //
    //   dṀ_{x,y}/dt|_sg = ksg · χ · δV · [ x · M_{x-1, y+1}
    //                                       + y · K_frac · M_{x+Av, y+As+1} ]
    //
    //   source_growth_(i) = dṀ_{x,y}/dt|_sg / (V0^x · S0^y · Nav)
    //
    // where χ = alpha_ * surface_density_ (active-site surface density, corrected by
    // the Appel/Frenklach alpha factor), K_frac = K_fractal_, Av = Av_fractal_,
    // As = As_fractal_.
    //
    // Index mapping: 0→M00, 1→M10, 2→M01, 3→M20, 4→M11, 5→M02, 6→N0.
    //
    // M00 (x=0, y=0): both prefactors x and y vanish → source = 0.
    // N0 (small-mode number): treated identically to HMOM4 — small-mode particles at
    //   the fixed point (V0, S0) are swept out at rate ksg·χ·S0·N0.

    this->source_growth_.setZero();
    if (!HasSoot())
        return;

    const double chi  = alpha_ * surface_density_;
    const double coeff = ksg_ * chi * VC2_;          // [ksg · χ · δV]
    const double Nav  = this->Nav_mol_;

    // --- M00 (x=0, y=0): zero (particle number conserved by surface growth)
    this->source_growth_(0) = 0.;

    // --- M10 (x=1, y=0): 1·M_{0,1} / (V0·Nav)
    this->source_growth_(1) =
        coeff * GetMoment(0., 1.) / Nav / V0_;

    // --- M01 (x=0, y=1): 1·K_frac·M_{Av_f, As_f+2} / (S0·Nav)
    this->source_growth_(2) =
        coeff * K_fractal_ * GetMoment(Av_fractal_, As_fractal_ + 2.) / Nav / S0_;

    // --- M20 (x=2, y=0): 2·M_{1,1} / (V0²·Nav)
    this->source_growth_(3) =
        coeff * 2. * GetMoment(1., 1.) / Nav / (V0_ * V0_);

    // --- M11 (x=1, y=1): [M_{0,2} + K_frac·M_{Av_f+1, As_f+2}] / (V0·S0·Nav)
    this->source_growth_(4) =
        coeff * (GetMoment(0., 2.) + K_fractal_ * GetMoment(Av_fractal_ + 1., As_fractal_ + 2.))
        / Nav / (V0_ * S0_);

    // --- M02 (x=0, y=2): 2·K_frac·M_{Av_f, As_f+3} / (S0²·Nav)
    this->source_growth_(5) =
        coeff * 2. * K_fractal_ * GetMoment(Av_fractal_, As_fractal_ + 3.) / Nav / (S0_ * S0_);

    // --- N0 (small-mode, index 6): fixed-point sweep-out rate — same as HMOM4 index 3.
    //   Small particles at (V0, S0) grow away from the fixed node at rate ksg·χ·S0 per particle.
    //   Note: no VC2_ factor here; this is the phenomenological sweep-out rate from the
    //   HMOM delta-function approximation (Mueller 2009, Eq. 31).
    this->source_growth_(6) = -ksg_ * chi * S0_ * N0_ / Nav;
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootOxidationM7()
{
    // Oxidation removes volume from each particle at a rate proportional to its
    // surface area (HACA-type model):
    //
    //   dV/dt|ox  = −coeff · S
    //   dS/dt|ox  = (∂S/∂V) · (dV/dt) = Kf · V^{Av_f} · S^{As_f+1} · (−coeff · S)
    //             = −coeff · Kf · V^{Av_f} · S^{As_f+2}
    //
    // where  coeff = kox · χ · VC2,  χ = α · surface_density.
    //
    // Applying the bivariate transport theorem:
    //
    //   d(M_{x,y})/dt|ox = −coeff · [ x · M(x−1, y+1)
    //                                + y · Kf · M(x+Av_f, y+As_f+1) ]
    //
    // (derived by differentiating V^x · S^y along the particle trajectory and
    //  integrating over the NDF — exactly the same algebra as SootSurfaceGrowthM7
    //  with opposite sign and coeff_ox in place of coeff_sg.)
    //
    // The normalised source is divided by V0^x · S0^y · Nav.
    //
    // Special cases:
    //   M00 (x=0, y=0): the transport term vanishes (no particle creation/destruction
    //     from volume/surface evolution alone).  Particle loss occurs via the boundary
    //     flux at V=0: small-mode particles oxidise completely at rate
    //       dn0/dt|ox = −N0 · coeff · S0 / V0
    //     Normalised: source_(0) = −coeff · N0 · S0 / (V0 · Nav).
    //   N0 (index 6): identical to M00 — both track particle count loss from the
    //     small mode.
    //   M10 (x=1): GetMoment(0,1) includes the small-mode term N0·S0, which
    //     correctly accounts for the volume removed by disappearing small-mode
    //     particles (V0 per particle × rate N0·coeff·S0/V0).
    //
    // Index mapping: 0→M00, 1→M10, 2→M01, 3→M20, 4→M11, 5→M02, 6→N0.

    this->source_oxidation_.setZero();
    if (!HasSoot())
        return;

    const double chi   = alpha_ * surface_density_;
    const double coeff = kox_ * chi * VC2_;
    const double Nav   = this->Nav_mol_;
    const double Av_f  = Av_fractal_;
    const double As_f  = As_fractal_;
    const double Kf    = K_fractal_;

    // --- M00 (x=0, y=0): small-mode particle loss (boundary flux at V→0) ---
    this->source_oxidation_(0) = -coeff * N0_ * S0_ / V0_ / Nav;

    // --- M10 (x=1, y=0): 1·M(0,1) / (V0·Nav) ---
    this->source_oxidation_(1) = -coeff * GetMoment(0., 1.) / (V0_ * Nav);

    // --- M01 (x=0, y=1): 1·Kf·M(Av_f, As_f+2) / (S0·Nav) ---
    //   [second term: y+As_f+1 = 1+As_f+1 = As_f+2, x+Av_f = Av_f]
    this->source_oxidation_(2) = -coeff * Kf * GetMoment(Av_f, As_f + 2.) / (S0_ * Nav);

    // --- M20 (x=2, y=0): 2·M(1,1) / (V0²·Nav) ---
    this->source_oxidation_(3) =
        -2. * coeff * GetMoment(1., 1.) / (V0_ * V0_ * Nav);

    // --- M11 (x=1, y=1): [M(0,2) + Kf·M(Av_f+1, As_f+2)] / (V0·S0·Nav) ---
    //   volume term:  1·M(0,  2)
    //   surface term: 1·Kf·M(1+Av_f, 1+As_f+1) = Kf·M(Av_f+1, As_f+2)
    this->source_oxidation_(4) =
        -coeff * (GetMoment(0., 2.) + Kf * GetMoment(Av_f + 1., As_f + 2.))
        / (V0_ * S0_ * Nav);

    // --- M02 (x=0, y=2): 2·Kf·M(Av_f, As_f+3) / (S0²·Nav) ---
    //   [second term: y+As_f+1 = 2+As_f+1 = As_f+3, x+Av_f = Av_f]
    this->source_oxidation_(5) =
        -2. * coeff * Kf * GetMoment(Av_f, As_f + 3.) / (S0_ * S0_ * Nav);

    // --- N0 (index 6): same as M00 — small-mode particle loss ---
    this->source_oxidation_(6) = this->source_oxidation_(0);
}

template <ThermoMap Thermo>
std::span<const double>
HMOM6<Thermo>::kappa_oxidation(std::span<const double> current_moments) const noexcept
{
    // Step 1: populate kappa_oxidation_ using the standard first-order formula
    //   κ_i = max(-source_oxidation[i], 0) / max(|M_i|, ε)
    // The base-class method calls derived().sources_oxidation() (= source_oxidation_)
    // and writes the result into the protected kappa_oxidation_ cache.
    (void)Base::kappa_oxidation(current_moments);

    // Step 2: apply Cauchy-Schwarz realizability bounds to the second-order kappas.
    //
    // Why this is needed
    // ------------------
    // The M02 source requires GetMoment(Av_f, As_f+3).  For the spherical model
    // (Av_f=-1, As_f=0) this evaluates the MOMIC quadratic polynomial at y=3,
    // outside its calibrated domain y ∈ {0,1,2}.  The quadratic term a20·9 vs a20·4
    // at the calibration boundary y=2 can over-extrapolate GetMoment(-1,3) by 10^5–10^7×
    // for broad soot distributions (a20 = σ_S²/2 >> 0).  The resulting κ₅ is so large
    // that the splitting step M02 *= exp(-κ₅·dt) drives M02 to zero in one step,
    // making L02 < 0 and invalidating the MOMIC closure.
    //
    // Physical justification for the bounds
    // --------------------------------------
    // For any moment set arising from a positive NDF, the Cauchy-Schwarz inequality
    // requires  M_{2i,0}·M_{0,0} ≥ M_{i,0}²  etc.  After exponential-decay splitting
    // (M_j_new = M_j · exp(-κ_j·dt)), the inequality is preserved iff:
    //   κ_3 + κ_0 ≤ 2·κ_1   →  κ_3 ≤ max(0, 2·κ_1 − κ_0)
    //   κ_5 + κ_0 ≤ 2·κ_2   →  κ_5 ≤ max(0, 2·κ_2 − κ_0)
    //   2·κ_4     ≤ κ_3+κ_5  →  κ_4 ≤ max(0, (κ_3+κ_5)/2)
    //
    // These bounds do NOT change the source terms; they only guard the splitting path.

    auto& k = this->kappa_oxidation_;  // Eigen vector of size 7

    const double k0 = k(0);  // M00 kappa
    const double k1 = k(1);  // M10 kappa
    const double k2 = k(2);  // M01 kappa

    // -- M20: kappa_3 ≤ 2·κ₁ − κ₀  (M20·M00 ≥ M10²)
    const double k3_max = std::max(0., 2. * k1 - k0);
    k(3) = std::min(k(3), k3_max);

    // -- M02: kappa_5 ≤ 2·κ₂ − κ₀  (M02·M00 ≥ M01²)  ← primary fix
    const double k5_max = std::max(0., 2. * k2 - k0);
    k(5) = std::min(k(5), k5_max);

    // -- M11: kappa_4 ≤ (κ₃+κ₅)/2  (M11² ≤ M20·M02; uses updated k(3), k(5))
    const double k4_max = std::max(0., 0.5 * (k(3) + k(5)));
    k(4) = std::min(k(4), k4_max);

    const std::size_t N = std::min(
        static_cast<std::size_t>(k.size()), current_moments.size());
    return {k.data(), N};
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootCondensationM7()
{
    // PAH dimer condensation onto existing soot particles (free-molecular regime).
    //
    // Free-molecular collision kernel (beta approximated to first order in Vd/V):
    //   beta(Vd, V) = Cfm*sqrt(T)*Vd^{-1/2} * (1 + 0.5*Vd/V) * (D_DIM + D_c(V,S))^2
    // where D_DIM = K_diam*Vd^{1/3} and D_c = K_collisional*V^{Av_c}*S^{As_c}.
    //
    // The change in a bivariate moment M_{x,y} per dimer-particle collision is
    // linearised in Vd (consistent with the HMOM4 treatment):
    //   Delta(V^x * S^y) ≈ Vd * [ x * V^{x-1} * S^y
    //                             + y * K_frac * V^{x+Av_f} * S^{y+As_f} ]
    // where the second term comes from dS/dV = K_frac * V^{Av_f} * S^{As_f+1}.
    //
    // This yields the normalised condensation source:
    //   source[M_{x,y,norm}] = Cfm*sqrtVd*cd*sqrtT
    //                        * { x * CI(x-1, y)  +  y * Kf * CI(x+Av_f, y+As_f) }
    //                        / (V0^x * S0^y)
    //
    // where the collision integral CI(a, b) is defined as:
    //   CI(a,b) = D_DIM^2  * [M(a,   b)        + 0.5*Vd * M(a-1,       b)]
    //           + K_c^2    * [M(a+2Av_c, b+2As_c) + 0.5*Vd * M(a-1+2Av_c, b+2As_c)]
    //           + 2*D*K_c  * [M(a+Av_c,  b+As_c)  + 0.5*Vd * M(a-1+Av_c,  b+As_c)]
    //
    // Index mapping: 0→M00, 1→M10, 2→M01, 3→M20, 4→M11, 5→M02, 6→N0.
    //
    // The N0 source (index 6) captures the depletion of small-mode particles by
    // dimer collision (identical in structure to HMOM4 source_condensation_(3)).

    this->source_condensation_.setZero();
    if (!HasSoot())
        return;

    const double K_diam = K_diam_HMOM6(this->pi_);
    const double D_DIM  = K_diam * std::pow(dimer_volume_, 1. / 3.);
    const double D_NUCL = K_diam * std::pow(V0_, 1. / 3.);
    const double Vd     = dimer_volume_;
    const double sqrtDv = std::sqrt(Vd);
    const double sqrtT  = std::sqrt(this->T_);
    const double cd     = conc_DIMER_;

    // Collision integral CI(a, b) — encapsulates the three-term kernel expansion.
    // Expanding (D_DIM + K_c*V^{Av_c}*S^{As_c})^2 and the (1+0.5*Vd/V) factor.
    const double Av_c = Av_collisional_;
    const double As_c = As_collisional_;
    const double K_c  = K_collisional_;

    auto CI = [&](double a, double b) noexcept -> double {
        return
            D_DIM * D_DIM *
                (GetMoment(a,              b) + 0.5 * Vd * GetMoment(a - 1.,          b)) +
            K_c * K_c *
                (GetMoment(a + 2.*Av_c, b + 2.*As_c) +
                 0.5 * Vd * GetMoment(a - 1. + 2.*Av_c, b + 2.*As_c)) +
            2. * D_DIM * K_c *
                (GetMoment(a + Av_c,    b + As_c) +
                 0.5 * Vd * GetMoment(a - 1. + Av_c,    b + As_c));
    };

    // Pre-factor shared by all large-mode collision terms.
    const double pre = Cfm_ * sqrtDv * cd * sqrtT;

    const double Av_f = Av_fractal_;
    const double As_f = As_fractal_;
    const double Kf   = K_fractal_;

    // --- M00 (x=0, y=0): no change (particle count conserved by condensation)
    this->source_condensation_(0) = 0.;

    // --- M10 (x=1, y=0): x-term 1*CI(0,0) / V0
    this->source_condensation_(1) = pre * CI(0., 0.) / V0_;

    // --- M01 (x=0, y=1): y-term 1*Kf*CI(Av_f, As_f+1) / S0
    this->source_condensation_(2) = pre * Kf * CI(Av_f, As_f + 1.) / S0_;

    // --- M20 (x=2, y=0): x-term 2*CI(1,0) / V0^2
    this->source_condensation_(3) = pre * 2. * CI(1., 0.) / (V0_ * V0_);

    // --- M11 (x=1, y=1): x-term CI(0,1) + y-term Kf*CI(Av_f+1, As_f+1), / (V0*S0)
    this->source_condensation_(4) =
        pre * (CI(0., 1.) + Kf * CI(Av_f + 1., As_f + 1.)) / (V0_ * S0_);

    // --- M02 (x=0, y=2): y-term 2*Kf*CI(Av_f, As_f+2) / S0^2
    this->source_condensation_(5) = pre * 2. * Kf * CI(Av_f, As_f + 2.) / (S0_ * S0_);

    // --- N0 (small-mode, index 6): depletion of small-mode particles by dimer collisions.
    //   d(N0)/dt|_cond = -beta(Vd, V0) * cd * N0
    //   where beta(Vd,V0) = Cfm/sqrtVd * sqrtT * (1 + 0.5*Vd/V0) * (D_DIM+D_NUCL)^2
    //   Identical to HMOM4 source_condensation_(3) — same physics, just re-indexed.
    this->source_condensation_(6) =
        -Cfm_ / sqrtDv * cd * sqrtT *
        (1. + 0.5 * Vd / V0_) * (D_DIM + D_NUCL) * (D_DIM + D_NUCL) * N0_;
}

// ===========================================================================
//  Coagulation — discrete (free-molecular) regime
// ===========================================================================

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootCoagulationM7()
{
    // Each sub-function guards itself with HasSoot() and zeroes its output.
    SootCoagulationSmallSmallM7();
    SootCoagulationSmallLargeM7();
    SootCoagulationLargeLargeM7();
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootCoagulationSmallSmallM7()
{
    source_coagulation_ss_.setZero();
    if (!HasSoot()) return;

    // Collision diameter of a nucleation-size particle.
    const double K_diam  = K_diam_HMOM6(this->pi_);
    const double DcNUCL  = K_diam * std::pow(V0_, 1. / 3.);

    // Free-molecular beta for two (V0, S0) particles (factor 2.20 = 2 * 1.10).
    // Same expression as HMOM4::SootCoagulationSmallSmallM4.
    const double beta00 = 2.20 * Cfm_ * std::sqrt(2. / V0_)
                          * std::pow(2. * DcNUCL, 2.)
                          * std::sqrt(this->T_);

    // Surface area of the coalesced (2·V0, S00) particle.
    const double S00 = this->SphereSurfaceFromVolume(2. * V0_);

    // Base rate term (negative: net loss of one particle per collision).
    // ss0 = d(M00_norm)/dt from small-small = -0.5 * beta00 * N0^2 / Nav
    const double ss0  = -0.5 * beta00 * N0_ * N0_ / this->Nav_mol_;
    const double R    = S00 / S0_;   // surface ratio of coalesced to primary

    // Unified formula:  source[M_{x,y}] = -ss0 * (2^x * R^y  - 2)
    //   M00 (x=0,y=0): -ss0*(1-2)   =  ss0          < 0  (net -1 particle)
    //   M10 (x=1,y=0): -ss0*(2-2)   =  0            (volume conserved)
    //   M01 (x=0,y=1): -ss0*(R-2)              < 0  (surface lost on merger)
    //   M20 (x=2,y=0): -ss0*(4-2)   = -2*ss0   > 0  (second volume moment grows)
    //   M11 (x=1,y=1): -ss0*(2R-2)             > 0
    //   M02 (x=0,y=2): -ss0*(R²-2)             > 0
    //   N0  (special):  2*ss0                   < 0  (both particles leave small mode)

    source_coagulation_ss_(0) =  ss0;                       // M00
    source_coagulation_ss_(1) =  0.;                        // M10
    source_coagulation_ss_(2) = -ss0 * (R - 2.);            // M01
    source_coagulation_ss_(3) = -2. * ss0;                  // M20
    source_coagulation_ss_(4) = -ss0 * (2. * R - 2.);       // M11
    source_coagulation_ss_(5) = -ss0 * (R * R - 2.);        // M02
    source_coagulation_ss_(6) =  2. * ss0;                  // N0
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootCoagulationSmallLargeM7()
{
    source_coagulation_sl_.setZero();
    if (!HasSoot()) return;

    const double K_diam       = K_diam_HMOM6(this->pi_);
    const double DcNUCL       = K_diam * std::pow(V0_, 1. / 3.);
    const double sqrtT        = std::sqrt(this->T_);
    const double Av_c         = Av_collisional_;
    const double As_c         = As_collisional_;
    const double Kc           = K_collisional_;
    const double Av_f         = Av_fractal_;
    const double As_f         = As_fractal_;
    const double Kf           = K_fractal_;
    const double V0inv_sqrt   = std::pow(V0_, -0.5);

    // Geometric-mean approximation for the weighted collision integral
    //   Ψ(a,b) = ∫ V_L^a · S_L^b · β(V0, V_L) · n_L dV dS
    //          ≈ 2.2 · Cfm · √T · √( p0(a,b) · p1(a,b) )
    //
    // with (Mueller 2009):
    //   p0(a,b) = V0^{-½} · [ Kc²·M^L(2Av_c+a-½, 2As_c+b)
    //                        + Dc²·M^L(a-½,         b)
    //                        + 2·Dc·Kc·M^L(Av_c+a-½, As_c+b) ]
    //   p1(a,b) = same with (a-½) → (a+½)  + V0·p0(a,b)
    //
    // Guard: returns 0 if p0·p1 < 0 (numerical noise prevention).
    auto sqrtPsi = [&](double a, double b) noexcept -> double {
        const double p0 = V0inv_sqrt * (
            Kc * Kc     * GetMissingMoment(2.*Av_c + a - 0.5, 2.*As_c + b) +
            DcNUCL * DcNUCL * GetMissingMoment(a - 0.5,              b) +
            2. * DcNUCL * Kc * GetMissingMoment(Av_c + a - 0.5, As_c + b));
        const double p1 = V0inv_sqrt * (
            Kc * Kc     * GetMissingMoment(2.*Av_c + a + 0.5, 2.*As_c + b) +
            DcNUCL * DcNUCL * GetMissingMoment(a + 0.5,              b) +
            2. * DcNUCL * Kc * GetMissingMoment(Av_c + a + 0.5, As_c + b))
            + V0_ * p0;
        return (p0 * p1 >= 0.) ? std::sqrt(p0 * p1) : 0.;
    };

    // Common prefactor: 2.2·Cfm·√T·N0/Nav  [mol·m^{-3}·s^{-1}·(m³/s)^{-1}]
    const double coeff = 2.2 * Cfm_ * sqrtT * N0_ / this->Nav_mol_;

    // M00: Δ = −1 per collision (two particles → one)
    source_coagulation_sl_(0) = -coeff * sqrtPsi(0., 0.);

    // M10: total volume conserved — small-mode loss = large-mode gain → 0
    source_coagulation_sl_(1) = 0.;

    // M01: Δ(S)/S0 = Kf·V0·V_L^{Av_f}·S_L^{As_f+1}/S0 − 1
    //   source = sl_(0)  +  coeff·Kf·V0/S0·Ψ(Av_f, As_f+1)
    source_coagulation_sl_(2) = source_coagulation_sl_(0)
        + coeff * Kf * V0_ / S0_ * sqrtPsi(Av_f, As_f + 1.);

    // M20: Δ(V²)/V0² = 2·V_L/V0  (exact: (V_L+V0)²−V_L²−V0² = 2V_L·V0)
    //   source = +2·coeff/V0·Ψ(1, 0)   [always positive]
    source_coagulation_sl_(3) = 2. * coeff / V0_ * sqrtPsi(1., 0.);

    // M11: Δ(V·S)/(V0·S0) = (S_L/S0−1) + Kf·V_L^{Av_f+1}·S_L^{As_f+1}/S0
    //   source = sl_(0)  +  coeff/S0·[Ψ(0,1) + Kf·Ψ(Av_f+1, As_f+1)]
    source_coagulation_sl_(4) = source_coagulation_sl_(0)
        + coeff / S0_ * (sqrtPsi(0., 1.) + Kf * sqrtPsi(Av_f + 1., As_f + 1.));

    // M02: Δ(S²)/S0² = 2·Kf·V0·V_L^{Av_f}·S_L^{As_f+2}/S0² − 1
    //   source = sl_(0)  +  2·coeff·Kf·V0/S0²·Ψ(Av_f, As_f+2)
    source_coagulation_sl_(5) = source_coagulation_sl_(0)
        + 2. * coeff * Kf * V0_ / (S0_ * S0_) * sqrtPsi(Av_f, As_f + 2.);

    // N0: one small-mode particle consumed per collision (same rate as M00)
    source_coagulation_sl_(6) = source_coagulation_sl_(0);
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootCoagulationLargeLargeM7()
{
    source_coagulation_ll_.setZero();
    if (!HasSoot()) return;

    // For large-large coagulation both collision diameters are fractal
    // (Dc = Kc·V^{Av_c}·S^{As_c} >> DcNUCL), so no DcNUCL terms appear.
    //
    // Free-molecular beta for two large particles (Mueller 2009, §3.2):
    //   β(V1,V2) ≈ Cfm·√T · Kc²·(V1^{Av_c}·S1^{As_c} + V2^{Av_c}·S2^{As_c})²
    //              · √(1/V1 + 1/V2)
    //
    // Geometric-mean approximation for the symmetric double integral
    //   ∫∫ β(V1,V2)·n_L(V1)·n_L(V2) dV1 dV2 ≈ 2.2·Cfm·√T·√(psi0·psi1)
    //
    // psi0 and psi1 bracket the integral using products of large-mode moments:
    const double Av_c = Av_collisional_;
    const double As_c = As_collisional_;
    const double Kc   = K_collisional_;

    const double psi0 = 2. * Kc * Kc * (
        GetMissingMoment(2.*Av_c - 0.5, 2.*As_c) * GetMissingMoment(-0.5, 0.) +
        GetMissingMoment(Av_c   - 0.5, As_c)    * GetMissingMoment(Av_c - 0.5, As_c));

    const double psi1 = 2. * Kc * Kc * (
        GetMissingMoment(2.*Av_c + 0.5, 2.*As_c) * GetMissingMoment(-0.5, 0.) +
        GetMissingMoment(Av_c   + 0.5, As_c)    * GetMissingMoment(Av_c - 0.5, As_c) +
        GetMissingMoment(2.*Av_c - 0.5, 2.*As_c) * GetMissingMoment( 0.5, 0.));

    // M00: −½ per collision pair (0.5 from symmetric double-counting).
    // All other sources (M10, M01, M20, M11, M02, N0) remain zero:
    //   M10 is exactly conserved; M01/M20/M11/M02 changes are approximated as
    //   negligible (same convention as HMOM4); N0 is unaffected by L-L.
    if (psi0 * psi1 >= 0.)
        source_coagulation_ll_(0) =
            -0.5 * 2.2 * Cfm_ * std::sqrt(this->T_) / this->Nav_mol_
            * std::sqrt(psi0 * psi1);
}

// ===========================================================================
//  Coagulation — continuum correction
// ===========================================================================

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootCoagulationContinuousM7()
{
    // Mean free path λ for air at standard conditions, temperature- and
    // pressure-corrected (identical physical constants as HMOM4).
    // λ₀ = kB·T₀ / (√2·π·d_air²·P₀)  with d_air = 200 pm, P₀ = 101325 Pa
    double lambda = 8.2057e-5 / this->sqrt2_ / std::pow(200.e-12, 2.) / this->Nav_mol_;
    lambda        = 1.257 * lambda * this->T_ / (this->P_Pa_ / 101325.);

    // Stokes–Einstein prefactor: βᵢ₀ = 2·kB·T / (3·μ)
    const double betai0 = 2. * this->kB_ * this->T_ / 3. / this->mu_;

    SootCoagulationContinuousSmallSmallM7(lambda, betai0);
    SootCoagulationContinuousSmallLargeM7(lambda, betai0);
    SootCoagulationContinuousLargeLargeM7(lambda, betai0);
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootCoagulationContinuousSmallSmallM7(double lambda, double betai0)
{
    source_coagulation_cont_ss_.setZero();
    if (!HasSoot()) return;

    // Continuum (Smoluchowski) kernel with Cunningham slip correction.
    // For two identical (V0, S0) particles:
    //   Dc   = K_diam · V0^{1/3}           (collision = spherical nucleation diameter)
    //   CC0  = 1 + lambda/Dc               (Cunningham correction, Kn = lambda/Dc)
    //   beta00 = betai0 · (CC1/Dc1 + CC2/Dc2) · (Dc1 + Dc2)
    //          = betai0 · (2·CC0/Dc) · (2·Dc)   [both particles identical]
    //          = 4 · betai0 · CC0
    const double K_diam = K_diam_HMOM6(this->pi_);
    const double DcNUCL = K_diam * std::pow(V0_, 1. / 3.);
    const double CC0    = 1. + lambda / DcNUCL;
    const double beta00 = betai0 * (2. * CC0 / DcNUCL) * (2. * DcNUCL);

    // Coalesced particle surface area (spherical).
    const double S00 = this->SphereSurfaceFromVolume(2. * V0_);

    // Base rate (= ss0 in the free-molecular case, same sign convention).
    const double ss0 = -0.5 * beta00 * N0_ * N0_ / this->Nav_mol_;
    const double R   = S00 / S0_;   // surface ratio of coalesced to primary

    // Unified formula: source[M_{x,y}] = -ss0 · (2^x · R^y − 2)
    // Identical structure to SootCoagulationSmallSmallM7; only beta00 differs.
    source_coagulation_cont_ss_(0) =  ss0;                       // M00
    source_coagulation_cont_ss_(1) =  0.;                        // M10 (volume conserved)
    source_coagulation_cont_ss_(2) = -ss0 * (R - 2.);            // M01
    source_coagulation_cont_ss_(3) = -2. * ss0;                  // M20
    source_coagulation_cont_ss_(4) = -ss0 * (2. * R - 2.);       // M11
    source_coagulation_cont_ss_(5) = -ss0 * (R * R - 2.);        // M02
    source_coagulation_cont_ss_(6) =  2. * ss0;                  // N0
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootCoagulationContinuousSmallLargeM7(double lambda, double betai0)
{
    source_coagulation_cont_sl_.setZero();
    if (!HasSoot()) return;

    const double K_diam = K_diam_HMOM6(this->pi_);
    const double DcNUCL = K_diam * std::pow(V0_, 1. / 3.);
    const double Av_c   = Av_collisional_;
    const double As_c   = As_collisional_;
    const double Kc     = K_collisional_;
    const double Av_f   = Av_fractal_;
    const double As_f   = As_fractal_;
    const double Kf     = K_fractal_;
    const double Nav    = this->Nav_mol_;

    // Continuum (Smoluchowski + Cunningham) collision integral for the small
    // particle (V0, DcNUCL) against the large-mode distribution n_L.
    //
    // β(V0,Vₗ)/βᵢ₀ = (CC₀/Dc₀ + CC_L/Dc_L) · (Dc₀ + Dc_L)
    //
    // where CC = 1 + λ/Dc and Dc_L = Kc · V^{Av_c} · S^{As_c}.
    //
    // Expanding and integrating against n_L with moment shift (da, db):
    //   CI(da,db) = (2 + λ/Dc₀)         · M^L(da,         db)
    //             + (Dc₀ + λ)/Kc         · M^L(-Av_c+da, -As_c+db)
    //             + (1 + λ/Dc₀)·Kc/Dc₀  · M^L(+Av_c+da, +As_c+db)
    //             + λ·Dc₀·Kc²            · M^L(-2Av_c+da, -2As_c+db)
    //
    // Recovered exactly from HMOM4 at (da,db)=(0,0) for M00 and (Av_f,As_f+1) for M01.
    auto CI = [&](double da, double db) noexcept -> double {
        return
            (2. + lambda / DcNUCL)       * GetMissingMoment(da,              db) +
            (DcNUCL + lambda) / Kc       * GetMissingMoment(-Av_c + da, -As_c + db) +
            (1. + lambda / DcNUCL) * Kc / DcNUCL
                                         * GetMissingMoment( Av_c + da,  As_c + db) +
            lambda * DcNUCL * Kc * Kc   * GetMissingMoment(-2.*Av_c + da, -2.*As_c + db);
    };

    // M00: Δ = −1 per collision
    source_coagulation_cont_sl_(0) = -N0_ / Nav * betai0 * CI(0., 0.);

    // M10: volume conserved
    source_coagulation_cont_sl_(1) = 0.;

    // M01: Δ(S)/S0 = Kf·V0·V_L^{Av_f}·S_L^{As_f+1}/S0 − 1
    //   source = cont_sl_(0)  +  N0·V0·Kf/S0/Nav · βᵢ₀ · CI(Av_f, As_f+1)
    source_coagulation_cont_sl_(2) = source_coagulation_cont_sl_(0)
        + N0_ * V0_ * Kf / S0_ / Nav * betai0 * CI(Av_f, As_f + 1.);

    // M20: Δ(V²)/V0² = 2·V_L/V0 (no constant term — V0² cancels)
    //   source = +2·N0·βᵢ₀/(V0·Nav) · CI(1, 0)   [always positive]
    source_coagulation_cont_sl_(3) = 2. * N0_ * betai0 / (V0_ * Nav) * CI(1., 0.);

    // M11: Δ(V·S)/(V0·S0) = (S_L/S0−1) + Kf·V_L^{Av_f+1}·S_L^{As_f+1}/S0
    //   source = cont_sl_(0)  +  N0·βᵢ₀/(S0·Nav) · [CI(0,1) + Kf·CI(Av_f+1, As_f+1)]
    source_coagulation_cont_sl_(4) = source_coagulation_cont_sl_(0)
        + N0_ * betai0 / (S0_ * Nav) * (CI(0., 1.) + Kf * CI(Av_f + 1., As_f + 1.));

    // M02: Δ(S²)/S0² = 2·Kf·V0·V_L^{Av_f}·S_L^{As_f+2}/S0² − 1
    //   source = cont_sl_(0)  +  2·N0·βᵢ₀·Kf·V0/(S0²·Nav) · CI(Av_f, As_f+2)
    source_coagulation_cont_sl_(5) = source_coagulation_cont_sl_(0)
        + 2. * N0_ * betai0 * Kf * V0_ / (S0_ * S0_ * Nav) * CI(Av_f, As_f + 2.);

    // N0: one small-mode particle consumed per collision (same rate as M00)
    source_coagulation_cont_sl_(6) = source_coagulation_cont_sl_(0);
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::SootCoagulationContinuousLargeLargeM7(double lambda, double betai0)
{
    source_coagulation_cont_ll_.setZero();
    if (!HasSoot()) return;

    // Continuum L-L kernel uses only the fractal collision diameter
    // Dc = Kc·V^{Av_c}·S^{As_c} (DcNUCL is irrelevant for large-large).
    //
    // Only M00 is computed (same HMOM convention as the free-molecular L-L):
    //   M10  = 0  (volume exactly conserved)
    //   M01/M20/M11/M02 = 0  (surface and higher-order L-L changes neglected)
    //   N0   = 0  (small mode unaffected by L-L)
    //
    // The double integral  ½ ∫∫ β(V1,V2)·n_L·n_L dV1 dV2  for M00 evaluates
    // to the same sum-of-moment-product expression as HMOM4; GetMissingMoment
    // now uses the 6-coefficient bivariate MOMIC polynomial transparently.
    const double Av_c = Av_collisional_;
    const double As_c = As_collisional_;
    const double Kc   = K_collisional_;

    const double val =
        GetMissingMoment(0., 0.) * GetMissingMoment(0., 0.) +
        lambda / Kc *
            (GetMissingMoment(0.,   0.)  * GetMissingMoment(-Av_c,    -As_c) +
             GetMissingMoment(Av_c, As_c) * GetMissingMoment(-2.*Av_c, -2.*As_c) +
             GetMissingMoment(Av_c, As_c) * GetMissingMoment(-Av_c,    -As_c));

    // Guard: val should be non-negative; skip if numerical noise makes it negative.
    if (val > 0.)
        source_coagulation_cont_ll_(0) = -0.5 * betai0 / this->Nav_mol_ * val;
}

// ===========================================================================
//  Main source-term orchestrator
// ===========================================================================

template <ThermoMap Thermo>
void HMOM6<Thermo>::ComputeSources() noexcept
{
    // Zero all per-process source vectors and the base-class totals.
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

    // When the variant is disabled, leave all sources at zero (gas consumption
    // is already zeroed by ZeroSources above).
    if (!this->is_active_)
    {
        if (this->gas_consumption_)
            this->omega_gas_.setZero();
        return;
    }

    // --- Temperature-dependent alpha correction (if enabled) ---
    if (surface_density_correction_)
        CalculateAlphaCoefficient();

    // --- PAH dimer concentration (required by nucleation and condensation) ---
    DimerConcentration();

    // --- Per-process sources ---
    if (nucleation_model_ > 0)
        SootNucleationM7();

    if (surface_growth_model_ > 0 || oxidation_model_ > 0)
    {
        SootKineticConstants();
        if (surface_growth_model_ > 0)
            SootSurfaceGrowthM7();
        if (oxidation_model_ > 0)
            SootOxidationM7();
    }

    if (condensation_model_ > 0)
        SootCondensationM7();

    if (coagulation_model_ > 0)
        SootCoagulationM7();

    if (coagulation_continuous_model_ > 0)
        SootCoagulationContinuousM7();

    // --- Sanitize and combine ---
    // NaN-guard each per-process entry; then combine discrete and continuous
    // coagulation via the harmonic-mean interpolation used in HMOM4.
    for (unsigned i = 0; i < 7u; ++i)
    {
        auto nanZ = [](double& v) noexcept
        {
            if (std::isnan(v))
                v = 0.;
        };

        nanZ(source_nucleation_(i));
        nanZ(source_growth_(i));
        nanZ(source_oxidation_(i));
        nanZ(source_condensation_(i));
        nanZ(source_coagulation_ss_(i));
        nanZ(source_coagulation_sl_(i));
        nanZ(source_coagulation_ll_(i));
        nanZ(source_coagulation_cont_ss_(i));
        nanZ(source_coagulation_cont_sl_(i));
        nanZ(source_coagulation_cont_ll_(i));

        const double disc =
            source_coagulation_ss_(i) +
            source_coagulation_sl_(i) +
            source_coagulation_ll_(i);

        const double cont =
            source_coagulation_cont_ss_(i) +
            source_coagulation_cont_sl_(i) +
            source_coagulation_cont_ll_(i);

        source_coagulation_discrete_(i)   = disc;
        source_coagulation_continuous_(i) = cont;

        // Harmonic-mean interpolation between free-molecular (disc) and
        // continuum (cont) coagulation — identical to the HMOM4 formulation.
        if (disc == 0.)
            source_coagulation_all_(i) = cont;
        else if (cont == 0.)
            source_coagulation_all_(i) = disc;
        else
            source_coagulation_all_(i) = cont * disc / (cont + disc);

        source_coagulation_(i) = source_coagulation_all_(i);

        this->source_all_(i) =
            source_nucleation_(i)      +
            source_growth_(i)          +
            source_oxidation_(i)       +
            source_condensation_(i)    +
            source_coagulation_all_(i);
    }

    if (this->gas_consumption_)
        CalculateOmegaGas();
}

// ===========================================================================
//  Gas-phase consumption
// ===========================================================================

template <ThermoMap Thermo>
void HMOM6<Thermo>::CalculateOmegaGas() noexcept
{
    // Gas-phase consumption rates [kg/m3/s] from soot processes.
    // Logic and channel decomposition are identical to HMOM4::CalculateOmegaGas;
    // the only structural difference is the moment index convention (7 equations
    // instead of 4), but the M10 source is still at index 1 in both variants.
    //
    // Sign convention: negative = consumed, positive = produced.

    this->omega_gas_.setZero();
    omega_gas_oxidation_.setZero();
    if (!this->gas_consumption_)
        return;

    // --- Nucleation: PAH precursor consumption ---
    if (nucleation_model_ > 0 && pah_index_ >= 0)
    {
        const double R_PAH = PAHDimerizationRate(); // [kmol/m3/s] → converted below
        if (R_PAH > 0. && std::isfinite(R_PAH))
        {
            const double MW_PAH = thermo_.MolecularWeight(static_cast<unsigned>(pah_index_));
            this->omega_gas_[static_cast<unsigned>(pah_index_)] -= R_PAH * MW_PAH;
        }
    }

    // --- Surface growth: C2H2 consumed, H2 produced ---
    // R_C2H2 [mol/m3/s]: source_growth_(1) is the M10_norm source;
    // multiplying by V0/VC2 recovers the molar C2H2 consumption rate.
    if (surface_growth_model_ > 0 && VC2_ > 0. && index_C2H2_ >= 0)
    {
        const double R_C2H2 = source_growth_(1) * V0_ / VC2_;
        if (R_C2H2 > 0. && std::isfinite(R_C2H2))
        {
            const double MW_C2H2 = thermo_.MolecularWeight(static_cast<unsigned>(index_C2H2_));
            this->omega_gas_[static_cast<unsigned>(index_C2H2_)] -= R_C2H2 * MW_C2H2 / 1000.;

            if (index_H2_ >= 0)
            {
                const double MW_H2 = thermo_.MolecularWeight(static_cast<unsigned>(index_H2_));
                this->omega_gas_[static_cast<unsigned>(index_H2_)] += R_C2H2 * MW_H2 / 1000.;
            }
        }
    }

    // --- Oxidation: O2 and OH channels ---
    // R_oxid_C2 [mol C2-pairs/m3/s]: rate of soot C-pairs removed by oxidation,
    // derived from the M10_norm oxidation source (index 1) in the same way as growth.
    if (oxidation_model_ > 0 && VC2_ > 0.)
    {
        // Snapshot before oxidation contributions are added, for the
        // oxidation-only extraction used by operator splitting.
        const Eigen::VectorXd omega_snap_before_ox = this->omega_gas_;

        const double R_oxid_C2 = -source_oxidation_(1) * V0_ / VC2_;
        if (R_oxid_C2 > 0. && std::isfinite(R_oxid_C2))
        {
            const double kox_sum = kox_O2_ + kox_OH_;
            if (kox_sum > 0. && std::isfinite(kox_sum))
            {
                const double fO2 = kox_O2_ / kox_sum;
                const double fOH = kox_OH_ / kox_sum;

                // O2 channel: C(soot) + O2 → 2 CO
                if (index_O2_ >= 0 && fO2 > 0.)
                {
                    const double R_O2  = fO2 * R_oxid_C2;
                    const double MW_O2 = thermo_.MolecularWeight(static_cast<unsigned>(index_O2_));
                    this->omega_gas_[static_cast<unsigned>(index_O2_)] -= R_O2 * MW_O2 / 1000.;
                    if (index_CO_ >= 0)
                    {
                        const double MW_CO = thermo_.MolecularWeight(static_cast<unsigned>(index_CO_));
                        this->omega_gas_[static_cast<unsigned>(index_CO_)] += R_O2 * 2. * MW_CO / 1000.;
                    }
                }

                // OH channel: C(soot) + OH → CO + 0.5 H2
                if (index_OH_ >= 0 && fOH > 0.)
                {
                    const double R_OH  = fOH * R_oxid_C2;
                    const double MW_OH = thermo_.MolecularWeight(static_cast<unsigned>(index_OH_));
                    this->omega_gas_[static_cast<unsigned>(index_OH_)] -= R_OH * MW_OH / 1000.;
                    if (index_CO_ >= 0)
                    {
                        const double MW_CO = thermo_.MolecularWeight(static_cast<unsigned>(index_CO_));
                        this->omega_gas_[static_cast<unsigned>(index_CO_)] += R_OH * MW_CO / 1000.;
                    }
                    if (index_H2_ >= 0)
                    {
                        const double MW_H2 = thermo_.MolecularWeight(static_cast<unsigned>(index_H2_));
                        this->omega_gas_[static_cast<unsigned>(index_H2_)] += R_OH * 0.5 * MW_H2 / 1000.;
                    }
                }
            }
        }

        // Capture the oxidation-only contribution for operator splitting.
        omega_gas_oxidation_ = this->omega_gas_ - omega_snap_before_ox;
    }

    // --- Closure dummy species (mass conservation) ---
    // If a closure dummy species is in use, its omega_gas entry is set to
    // minus the sum of all other entries so that total mass is conserved.
    if (this->is_closure_dummy_species_ && this->closure_dummy_index_ >= 0)
    {
        double sum = 0.;
        for (Eigen::Index i = 0; i < this->omega_gas_.size(); ++i)
            if (i != static_cast<Eigen::Index>(this->closure_dummy_index_))
                sum += this->omega_gas_[i];
        this->omega_gas_[static_cast<Eigen::Index>(this->closure_dummy_index_)] = -sum;

        double sum_ox = 0.;
        for (Eigen::Index i = 0; i < omega_gas_oxidation_.size(); ++i)
            if (i != static_cast<Eigen::Index>(this->closure_dummy_index_))
                sum_ox += omega_gas_oxidation_[i];
        omega_gas_oxidation_[static_cast<Eigen::Index>(this->closure_dummy_index_)] = -sum_ox;
    }
}

// ===========================================================================
//  Particle properties
// ===========================================================================

template <ThermoMap Thermo>
double HMOM6<Thermo>::volume_fraction() const noexcept
{
    // Total soot volume fraction fv = M_{1,0} = ∫ V · n dV dS  [m³/m³].
    const double fv = GetMoment(1., 0.);
    return (!std::isfinite(fv) || fv < kSootVolumeFloor) ? 0. : fv;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::particle_diameter() const noexcept
{
    // Sauter mean diameter: dp = 6 V / S (holds for volume-equivalent sphere).
    const double V = GetMoment(1., 0.);
    const double S = GetMoment(0., 1.);
    if (!std::isfinite(V) || !std::isfinite(S) || V <= kSootVolumeFloor || S <= kSootSurfaceFloor)
        return 0.;
    const double dp = 6. * V / S;
    return (std::isfinite(dp) && dp > 0.) ? dp : 0.;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::collision_diameter() const noexcept
{
    // Mean collision diameter: dc = K_c · <V^{Av_c} · S^{As_c}> / N
    //   where <·> is the number-weighted mean over the full NDF.
    const double N = GetMoment(0., 0.);
    if (!std::isfinite(N) || N <= kSootNumberFloor)
        return 0.;
    const double dc = K_collisional_ * GetMoment(Av_collisional_, As_collisional_) / N;
    return (std::isfinite(dc) && dc > 0.) ? dc : 0.;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::particle_number_density() const noexcept
{
    // Total particle number density N = M_{0,0} = ∫ n dV dS  [#/m³].
    const double n = GetMoment(0., 0.);
    return (!std::isfinite(n) || n < kSootNumberFloor) ? 0. : n;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::mass_fraction() const noexcept
{
    // Soot mass fraction: Ys = rho_soot / rho_mix · fv.
    return this->rho_particle_ / this->rho_ * volume_fraction();
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::specific_surface_area() const noexcept
{
    // Total soot surface area density [m²/m³]: M_{0,1} = ∫ S · n dV dS.
    const double Ss = GetMoment(0., 1.);
    return (!std::isfinite(Ss) || Ss <= 0.) ? 0. : Ss;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::number_primary_particles() const noexcept
{
    // Mean number of primary particles per aggregate:
    //   np = (K_spher)^{-3} · M_{-2,3} / N
    // where K_spher = (36·π)^{1/3} and M_{-2,3} = ∫ V^{-2}·S³·n dV dS.
    // This formula is exact for spherical primaries arranged in mass-fractal
    // aggregates and is the MOMIC extension of the HMOM4 expression.
    const double N = GetMoment(0., 0.);
    if (!std::isfinite(N) || N <= kSootNumberFloor)
        return 0.;
    const double M = GetMoment(-2., 3.);
    if (!std::isfinite(M) || M <= 0.)
        return 0.;
    const double K_spher = K_spher_HMOM6(this->pi_);
    const double np      = std::pow(K_spher, -3.) * M / N;
    return (std::isfinite(np) && np >= 1.) ? np : 1.;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::diffusion_coefficient() const noexcept
{
    // Brownian diffusion coefficient via Cunningham-corrected Stokes–Einstein.
    // Falls back to kinetic-theory value mu/Sc when dc is sub-nanometre.
    const double dc_safe       = std::max(collision_diameter(), 1.e-12);
    const double GammaBrownian = this->CunninghamDiffusionCoeff(dc_safe);
    return std::max(GammaBrownian, this->mu_ / this->schmidt_number_);
}

// ===========================================================================
//  Morphological statistics and property convenience accessors
// ===========================================================================

template <ThermoMap Thermo>
double HMOM6<Thermo>::soot_large_mean_volume() const noexcept
{
    return (L00_ > kSootNumberFloor && L10_ > kSootVolumeFloor) ? L10_ / L00_ : V0_;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::soot_large_mean_surface() const noexcept
{
    return (L00_ > kSootNumberFloor && L01_ > kSootSurfaceFloor) ? L01_ / L00_ : S0_;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::soot_large_primary_particle_diameter() const noexcept
{
    const double VL = soot_large_mean_volume();
    const double SL = soot_large_mean_surface();
    return (SL > 0.) ? 6. * VL / SL : 0.;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::soot_large_primary_particle_number() const noexcept
{
    const double VL = soot_large_mean_volume();
    const double SL = soot_large_mean_surface();
    if (SL <= 0. || VL <= 0.)
        return 1.;
    return std::max(1., std::pow(SL, 3.) / (36. * this->pi_ * VL * VL));
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::LogGeomStdDevFromMoments(double M0, double M1, double M2) const noexcept
{
    if (M0 <= 0. || M1 <= 0. || M2 <= 0.)
        return 0.;
    const double ratio = M0 * M2 / (M1 * M1);
    return (ratio > 1.) ? std::sqrt(std::log(ratio)) : 0.;
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::soot_log_geom_std_dev_primary_particle_diameter() const noexcept
{
    // dp ~ 6V/S; moments of the dp distribution via bivariate MOMIC.
    return LogGeomStdDevFromMoments(GetMoment(0., 0.), GetMoment(-1., 1.5), GetMoment(-2., 3.));
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::soot_log_geom_std_dev_primary_particle_number() const noexcept
{
    // np ~ (S/S_primary)^3/36π/V^2; moments reduce to M(1,0), M(2,0).
    return LogGeomStdDevFromMoments(GetMoment(0., 0.), GetMoment(1., 0.), GetMoment(2., 0.));
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::soot_large_log_geom_std_dev_primary_particle_diameter() const noexcept
{
    if (!HasSoot() || L00_ <= kSootNumberFloor)
        return 0.;
    const double M0L = GetMissingMoment(0.,  0.);
    const double M1L = GetMissingMoment(1., -1.);
    const double M2L = GetMissingMoment(2., -2.);
    return LogGeomStdDevFromMoments(M0L, M1L, M2L);
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::soot_large_log_geom_std_dev_primary_particle_number() const noexcept
{
    if (!HasSoot() || L00_ <= kSootNumberFloor)
        return 0.;
    const double M0L = GetMissingMoment( 0., 0.);
    const double M1L = GetMissingMoment(-2., 3.);
    const double M2L = GetMissingMoment(-4., 6.);
    return LogGeomStdDevFromMoments(M0L, M1L, M2L);
}

template <ThermoMap Thermo>
double HMOM6<Thermo>::soot_d63() const noexcept
{
    const double M10 = GetMoment(1., 0.);
    const double M43 = GetMoment(4. / 3., 0.);
    if (M10 <= 0. || !std::isfinite(M10))
        return 0.;
    return 6. * std::pow(M43 / M10, 1. / 3.);
}

template <ThermoMap Thermo>
void HMOM6<Thermo>::Properties(
    double& fv, double& dp, double& dc, double& np, double& ss, double& vs) const noexcept
{
    const double N = GetMoment(0., 0.);
    const double V = GetMoment(1., 0.);
    const double S = GetMoment(0., 1.);

    if (!std::isfinite(N) || !std::isfinite(V) || !std::isfinite(S) || N <= 0. || V <= 0. || S <= 0.)
    {
        fv = 0.;
        vs = V0_;
        ss = S0_;
        dp = 6. * V0_ / S0_;
        np = 1.;
        dc = K_collisional_ * std::pow(V0_, Av_collisional_) * std::pow(S0_, As_collisional_);
        return;
    }
    fv = V;
    vs = V / N;
    ss = S / N;
    dp = 6. * vs / ss;
    np = std::pow(ss, 3.) / (36. * this->pi_ * vs * vs);
    if (!std::isfinite(np) || np < 1.)
        np = 1.;
    dc = K_collisional_ * std::pow(vs, Av_collisional_) * std::pow(ss, As_collisional_);
    if (!std::isfinite(dc) || dc <= 0.)
        dc = dp;
}

// ===========================================================================
//  Diagnostics
// ===========================================================================

template <ThermoMap Thermo>
void HMOM6<Thermo>::PrintSummary() const
{
    const auto sticking_str = [](StickingModel m) -> const char* {
        switch (m) {
            case StickingModel::Constant: return "constant";
            case StickingModel::PAH4:     return "PAH 4-ring";
            default:                       return "unknown";
        }
    };
    const auto frac_str = [](FractalDiameterModel m) -> const char* {
        switch (m) {
            case FractalDiameterModel::Model0: return "Mueller et al. (2009)";
            case FractalDiameterModel::Model1: return "Attili et al. (2014)";
            default:                            return "unknown";
        }
    };
    const auto coll_str = [](CollisionDiameterModel m) -> const char* {
        switch (m) {
            case CollisionDiameterModel::Model1: return "N^{1/3} dp (Mueller 2009)";
            case CollisionDiameterModel::Model2: return "fractal geometry (Attili 2014)";
            default:                              return "unknown";
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
        << "             HMOM6 — Hybrid Method of Moments (6+1 bivariate MOMIC) Summary\n"
        << "------------------------------------------------------------------------------------------\n"
        << " * Active: " << (this->is_active_ ? "yes" : "no") << "\n"
        << "\n"
        << " [Physical properties]\n"
        << "    + Soot density (kg/m3):          " << this->rho_particle_ << "\n"
        << "\n"
        << " [Species — PAH precursor]\n"
        << "    + PAH species:                   " << pah_species_ << "\n"
        << "\n"
        << " [Nucleated particle geometry]\n"
        << "    + V0 (m3):                       " << V0_ << "\n"
        << "    + S0 (m2):                       " << S0_ << "\n"
        << "\n"
        << " [MOMIC geometry — fractal (∂S/∂V = Kf·V^Av_f·S^(As_f+1))]\n"
        << "    + Fractal diameter model:        " << static_cast<int>(fractal_diameter_model_)
                                                   << "  (" << frac_str(fractal_diameter_model_) << ")\n"
        << "    + Av_fractal:                    " << Av_fractal_ << "\n"
        << "    + As_fractal:                    " << As_fractal_ << "\n"
        << "    + K_fractal:                     " << K_fractal_ << "\n"
        << "\n"
        << " [MOMIC geometry — collisional (dc = Kc·V^Av_c·S^As_c)]\n"
        << "    + Collision diameter model:      " << static_cast<int>(collision_diameter_model_)
                                                   << "  (" << coll_str(collision_diameter_model_) << ")\n"
        << "    + Av_collisional:                " << Av_collisional_ << "\n"
        << "    + As_collisional:                " << As_collisional_ << "\n"
        << "    + K_collisional:                 " << K_collisional_ << "\n"
        << "    + D_collisional:                 " << D_collisional_ << "\n"
        << "\n"
        << " [Processes]\n"
        << "    + Nucleation:                    " << nucleation_model_ << "\n"
        << "    + Condensation:                  " << condensation_model_ << "\n"
        << "    + Surface growth:                " << surface_growth_model_ << "\n"
        << "    + Oxidation:                     " << oxidation_model_ << "\n"
        << "    + Coagulation:                   " << coagulation_model_ << "\n"
        << "    + Coagulation (continuous):      " << coagulation_continuous_model_ << "\n"
        << "    + Sticking model:                " << static_cast<int>(sticking_model_)
                                                   << "  (" << sticking_str(sticking_model_) << ")\n"
        << "    + Sticking coeff. (-):           " << sticking_coeff_constant_ << "\n"
        << "\n"
        << " [HACA surface kinetics]  A [cm3/mol/s or 1/s],  n [-],  E [kJ/mol]\n"
        << "    + R1f (H-abs. fwd):  A=" << A1f_ << "  n=" << n1f_ << "  E=" << E1f_ * this->Rgas_mol_ / 1000. << "\n"
        << "    + R1b (H-abs. rev):  A=" << A1b_ << "  n=" << n1b_ << "  E=" << E1b_ * this->Rgas_mol_ / 1000. << "\n"
        << "    + R2f (OH-ox. fwd):  A=" << A2f_ << "  n=" << n2f_ << "  E=" << E2f_ * this->Rgas_mol_ / 1000. << "\n"
        << "    + R2b (OH-ox. rev):  A=" << A2b_ << "  n=" << n2b_ << "  E=" << E2b_ * this->Rgas_mol_ / 1000. << "\n"
        << "    + R3f (O2-ox. fwd):  A=" << A3f_ << "  n=" << n3f_ << "  E=" << E3f_ * this->Rgas_mol_ / 1000. << "\n"
        << "    + R3b (O2-ox. rev):  A=" << A3b_ << "  n=" << n3b_ << "  E=" << E3b_ * this->Rgas_mol_ / 1000. << "\n"
        << "    + R4  (C2H2 add.):   A=" << A4_  << "  n=" << n4_  << "  E=" << E4_  * this->Rgas_mol_ / 1000. << "\n"
        << "    + R5  (O-ox.):       A=" << A5_  << "  n=" << n5_  << "  E=" << E5_  * this->Rgas_mol_ / 1000. << "\n"
        << "    + eff6 (O-rad.):     " << eff6_ << "\n"
        << "\n"
        << " [Surface density]\n"
        << "    + Chi (#/m2):                    " << surface_density_ << "\n"
        << "    + T-dependent correction:        " << (surface_density_correction_ ? "yes" : "no") << "\n"
        << "\n"
        << " [Transport & radiation]\n"
        << "    + Schmidt number (-):            " << this->schmidt_number_ << "\n"
        << "    + Thermophoretic model:          " << static_cast<int>(this->thermophoretic_model())
                                                   << "  (" << thermo_str(this->thermophoretic_model_) << ")\n"
        << "    + Gas consumption:               " << (this->gas_consumption_ ? "yes" : "no") << "\n"
        << "    + Radiative heat transfer:       " << (this->radiative_heat_transfer_ ? "yes" : "no") << "\n"
        << "    + Planck coeff. model:           " << static_cast<int>(this->planck_model_)
                                                   << "  (" << planck_str(this->planck_model_) << ")\n"
        << "    + Closure dummy species:         "
                                                   << (this->is_closure_dummy_species_ ? this->closure_dummy_species_ : "none") << "\n"
        << "\n"
        << " [Numerical floors]\n"
        << "    + N floor (#/m3):                " << kSootNumberFloor << "\n"
        << "    + fv floor (-):                  " << kSootVolumeFloor << "\n"
        << "    + S floor (m2/m3):               " << kSootSurfaceFloor << "\n"
        << "\n"
        << " [Debug]\n"
        << "    + Debug mode:                    " << (is_debug_mode_ ? "yes" : "no") << "\n"
        << "------------------------------------------------------------------------------------------\n";
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
    HMOM6_Grammar grammar;
    dict.SetGrammar(grammar);

    Config cfg; // start from library defaults

    if (dict.CheckOption("@HMOM6"))
        dict.ReadBool("@HMOM6", cfg.is_active);

    if (dict.CheckOption("@FractalDiameterModel"))
        dict.ReadInt("@FractalDiameterModel", cfg.fractal_diameter_model);

    if (dict.CheckOption("@CollisionDiameterModel"))
        dict.ReadInt("@CollisionDiameterModel", cfg.collision_diameter_model);

    if (dict.CheckOption("@GasClosureDummySpecies"))
        dict.ReadString("@GasClosureDummySpecies", cfg.gas_closure_dummy_species);

    if (dict.CheckOption("@GasConsumption"))
        dict.ReadBool("@GasConsumption", cfg.gas_consumption);

    if (dict.CheckOption("@SimplifiedPAHMass"))
        dict.ReadBool("@SimplifiedPAHMass", cfg.simplified_pah_mass);

    if (dict.CheckOption("@PAH"))
        dict.ReadString("@PAH", cfg.pah_species);

    if (dict.CheckOption("@NucleationModel"))
        { int _tmp; dict.ReadInt("@NucleationModel", _tmp); cfg.nucleation_model = static_cast<NucleationModel>(_tmp); }

    if (dict.CheckOption("@SurfaceGrowthModel"))
        { int _tmp; dict.ReadInt("@SurfaceGrowthModel", _tmp); cfg.surface_growth_model = static_cast<SurfaceGrowthModel>(_tmp); }

    if (dict.CheckOption("@OxidationModel"))
        { int _tmp; dict.ReadInt("@OxidationModel", _tmp); cfg.oxidation_model = static_cast<OxidationModel>(_tmp); }

    if (dict.CheckOption("@CondensationModel"))
        { int _tmp; dict.ReadInt("@CondensationModel", _tmp); cfg.condensation_model = static_cast<CondensationModel>(_tmp); }

    if (dict.CheckOption("@CoagulationModel"))
        { int _tmp; dict.ReadInt("@CoagulationModel", _tmp); cfg.coagulation_model = static_cast<CoagulationModel>(_tmp); }

    if (dict.CheckOption("@ContinuousCoagulationModel"))
        dict.ReadInt("@ContinuousCoagulationModel", cfg.continuous_coagulation_model);

    if (dict.CheckOption("@ThermophoreticModel"))
        { int _tmp; dict.ReadInt("@ThermophoreticModel", _tmp); cfg.thermophoretic_model = static_cast<ThermophoreticModel>(_tmp); }

    if (dict.CheckOption("@RadiativeHeatTransfer"))
        dict.ReadBool("@RadiativeHeatTransfer", cfg.radiative_heat_transfer);

    if (dict.CheckOption("@PlanckCoefficient"))
        dict.ReadString("@PlanckCoefficient", cfg.planck_coefficient);

    if (dict.CheckOption("@SchmidtNumber"))
        dict.ReadDouble("@SchmidtNumber", cfg.schmidt_number);

    if (dict.CheckOption("@SootDensity"))
    {
        double v; std::string u;
        dict.ReadMeasure("@SootDensity", v, u);
        if (u == "kg/m3")      cfg.soot_density_kg_m3 = v;
        else if (u == "g/cm3") cfg.soot_density_kg_m3 = v * 1000.;
        else return std::unexpected(std::string{"@SootDensity: allowed units: kg/m3 | g/cm3"});
    }

    if (dict.CheckOption("@SurfaceDensity"))
    {
        double v; std::string u;
        dict.ReadMeasure("@SurfaceDensity", v, u);
        if (u == "#/m2")       cfg.surface_density_per_m2 = v;
        else if (u == "#/cm2") cfg.surface_density_per_m2 = v * 1.e4;
        else if (u == "#/mm2") cfg.surface_density_per_m2 = v * 1.e6;
        else return std::unexpected(std::string{"@SurfaceDensity: allowed units: #/m2 | #/cm2 | #/mm2"});
    }

    if (dict.CheckOption("@SurfaceDensityCorrectionCoefficient"))
        dict.ReadBool("@SurfaceDensityCorrectionCoefficient", cfg.surface_density_correction);
    if (dict.CheckOption("@SurfaceDensityCorrectionCoefficientA1"))
        dict.ReadDouble("@SurfaceDensityCorrectionCoefficientA1", cfg.surf_dens_a1);
    if (dict.CheckOption("@SurfaceDensityCorrectionCoefficientA2"))
        dict.ReadDouble("@SurfaceDensityCorrectionCoefficientA2", cfg.surf_dens_a2);
    if (dict.CheckOption("@SurfaceDensityCorrectionCoefficientB1"))
        dict.ReadDouble("@SurfaceDensityCorrectionCoefficientB1", cfg.surf_dens_b1);
    if (dict.CheckOption("@SurfaceDensityCorrectionCoefficientB2"))
        dict.ReadDouble("@SurfaceDensityCorrectionCoefficientB2", cfg.surf_dens_b2);

    // HACA frequency factors [cm3/mol/s].
    {
        double v; std::string u;
        auto readA = [&](const char* key, double& field) -> std::expected<void, std::string>
        {
            if (dict.CheckOption(key)) {
                dict.ReadMeasure(key, v, u);
                if (u != "cm3,mol,s")
                    return std::unexpected(std::string{key} + ": allowed units: cm3,mol,s");
                field = v;
            }
            return {};
        };
        if (auto r = readA("@A1f", cfg.A1f); !r) return std::unexpected(r.error());
        if (auto r = readA("@A1b", cfg.A1b); !r) return std::unexpected(r.error());
        if (auto r = readA("@A2f", cfg.A2f); !r) return std::unexpected(r.error());
        if (auto r = readA("@A2b", cfg.A2b); !r) return std::unexpected(r.error());
        if (auto r = readA("@A3f", cfg.A3f); !r) return std::unexpected(r.error());
        if (auto r = readA("@A3b", cfg.A3b); !r) return std::unexpected(r.error());
        if (auto r = readA("@A4",  cfg.A4);  !r) return std::unexpected(r.error());
        if (auto r = readA("@A5",  cfg.A5);  !r) return std::unexpected(r.error());
    }

    // HACA activation energies [kJ/mol].
    {
        double v; std::string u;
        auto readE = [&](const char* key, double& field) -> std::expected<void, std::string>
        {
            if (dict.CheckOption(key)) {
                dict.ReadMeasure(key, v, u);
                if (u != "kJ/mol")
                    return std::unexpected(std::string{key} + ": allowed units: kJ/mol");
                field = v;
            }
            return {};
        };
        if (auto r = readE("@E1f", cfg.E1f); !r) return std::unexpected(r.error());
        if (auto r = readE("@E1b", cfg.E1b); !r) return std::unexpected(r.error());
        if (auto r = readE("@E2f", cfg.E2f); !r) return std::unexpected(r.error());
        if (auto r = readE("@E2b", cfg.E2b); !r) return std::unexpected(r.error());
        if (auto r = readE("@E3f", cfg.E3f); !r) return std::unexpected(r.error());
        if (auto r = readE("@E3b", cfg.E3b); !r) return std::unexpected(r.error());
        if (auto r = readE("@E4",  cfg.E4);  !r) return std::unexpected(r.error());
        if (auto r = readE("@E5",  cfg.E5);  !r) return std::unexpected(r.error());
    }

    // Temperature exponents.
    if (dict.CheckOption("@n1f")) dict.ReadDouble("@n1f", cfg.n1f);
    if (dict.CheckOption("@n1b")) dict.ReadDouble("@n1b", cfg.n1b);
    if (dict.CheckOption("@n2f")) dict.ReadDouble("@n2f", cfg.n2f);
    if (dict.CheckOption("@n2b")) dict.ReadDouble("@n2b", cfg.n2b);
    if (dict.CheckOption("@n3f")) dict.ReadDouble("@n3f", cfg.n3f);
    if (dict.CheckOption("@n3b")) dict.ReadDouble("@n3b", cfg.n3b);
    if (dict.CheckOption("@n4"))  dict.ReadDouble("@n4",  cfg.n4);
    if (dict.CheckOption("@n5"))  dict.ReadDouble("@n5",  cfg.n5);

    if (dict.CheckOption("@Efficiency6"))
        dict.ReadDouble("@Efficiency6", cfg.efficiency6);

    if (dict.CheckOption("@StickingCoefficientModel"))
        dict.ReadString("@StickingCoefficientModel", cfg.sticking_model);

    if (dict.CheckOption("@StickingCoefficientConstant"))
        dict.ReadDouble("@StickingCoefficientConstant", cfg.sticking_coeff_constant);

    if (dict.CheckOption("@DebugMode"))
        dict.ReadBool("@DebugMode", cfg.debug_mode);

    return cfg;
}

#endif // MOM_USE_DICTIONARY

} // namespace MOM
