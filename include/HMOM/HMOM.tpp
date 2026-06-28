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

#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <numeric>
#if defined(MOM_USE_DICTIONARY)
#include "HMOM_Grammar.h"
#endif

namespace MOM {

// ============================================================================
// Local constants
// ============================================================================
namespace {
    // HMOM uses R in [J/mol/K] for HACA kinetics (stored E are in [K] = E_kJ/mol * 1e3 / R_J_mol)
    constexpr double R_J_mol_HMOM = 8.31446261815324;

    // K_diam = (6/pi)^(1/3), K_spher = (36*pi)^(1/3)
    inline double K_diam_HMOM(double pi) noexcept  { return std::pow(6./pi, 1./3.); }
    inline double K_spher_HMOM(double pi) noexcept { return std::pow(36.*pi, 1./3.); }

    // Compute fractal geometry members from model flag (must match old SetFractalDiameterModel)
    // Returns {Av, As, K}
    inline std::array<double,3> FractalGeometry(int model, double pi) noexcept
    {
        double chi, Av, As, K;
        if (model == 1) {
            chi = -0.2043;
            Av  = -2.*chi - 1.;
            As  =  3.*chi;
            K   =  (2./3.) * std::pow(1./(36.*pi), chi);
        } else {
            chi = 0.;
            Av  = -1.;
            As  =  0.;
            K   =  2./3.;
        }
        (void)chi;
        return {Av, As, K};
    }

    // Compute collision-diameter geometry from model flag (must match old SetCollisionDiameterModel)
    // Returns {D, Av, As, K}
    inline std::array<double,4> CollisionGeometry(int model, double pi) noexcept
    {
        double D, Av, As, K;
        if (model == 1) {
            D  = 2.;
            Av = 1./3.;
            As = 0.;
            K  = std::pow(6./pi, 1./3.);   // K_diam
        } else {
            D  = 1.8;
            Av = 1. - 2./D;
            As = 3./D - 1.;
            K  = 6. / std::pow(36.*pi, 1./D);
        }
        return {D, Av, As, K};
    }
} // anonymous namespace

// ============================================================================
// Constructor
// ============================================================================

template <ThermoMap Thermo>
HMOM<Thermo>::HMOM(const Thermo& thermo)
    : thermo_(thermo)
{
    const double K_diam  = K_diam_HMOM(this->pi_);

    // ── CRTP base state ───────────────────────────────────────────────────
    this->is_active_               = true;
    this->gas_consumption_         = true;
    this->radiative_heat_transfer_ = true;
    this->planck_model_            = PlanckCoeffModel::Smooke;
    this->SetThermophoreticModel(1);
    this->schmidt_number_          = 50.;
    this->rho_particle_            = 1800.;
    this->dummy_species_           = "none";
    this->dummy_index_             = -1;
    this->dummy_species_closure_   = false;

    // ── Free-molecular pre-factors ─────────────────────────────────────────
    Cfm_      = std::sqrt(this->pi_ * this->kB_ / 2.0 / this->rho_particle_);
    betaN_TV_ = 2.2 * 4. * this->sqrt2_ * K_diam * K_diam * Cfm_;

    // ── Fractal / collision geometry (defaults: Model1 / Model2) ──────────
    // The HMOM.hpp inline setters only store the enum — they don't compute the
    // geometry members.  We call them for the enum, then populate the geometry.
    SetFractalDiameterModel(1);
    {
        auto [Av, As, K] = FractalGeometry(1, this->pi_);
        Av_fractal_ = Av; As_fractal_ = As; K_fractal_ = K;
    }
    SetCollisionDiameterModel(2);
    {
        auto [D, Av, As, K] = CollisionGeometry(2, this->pi_);
        D_collisional_ = D; Av_collisional_ = Av; As_collisional_ = As; K_collisional_ = K;
    }

    // ── HACA kinetics (default values; E stored as E_kJ/mol * 1e3 / R) ────
    A1f_ = 6.72e1;   n1f_ = 3.33;   E1f_ = 6.09   * 1e3 / R_J_mol_HMOM;
    A1b_ = 6.44e-1;  n1b_ = 3.79;   E1b_ = 27.96  * 1e3 / R_J_mol_HMOM;
    A2f_ = 1.00e8;   n2f_ = 1.80;   E2f_ = 68.42  * 1e3 / R_J_mol_HMOM;
    A2b_ = 8.68e4;   n2b_ = 2.36;   E2b_ = 25.46  * 1e3 / R_J_mol_HMOM;
    A3f_ = 1.13e16;  n3f_ = -0.06;  E3f_ = 476.05 * 1e3 / R_J_mol_HMOM;
    A3b_ = 4.17e13;  n3b_ = 0.15;   E3b_ = 0.00   * 1e3 / R_J_mol_HMOM;
    A4_  = 2.52e9;   n4_  = 1.10;   E4_  = 17.13  * 1e3 / R_J_mol_HMOM;
    A5_  = 2.20e12;  n5_  = 0.00;   E5_  = 31.38  * 1e3 / R_J_mol_HMOM;
    eff6_= 0.13;

    // ── Surface density correction (off by default) ────────────────────────
    surface_density_correction_ = false;
    surface_density_           = 1.7e19;   // [#/m2]
    surf_dens_a1_ = 12.65;
    surf_dens_a2_ = 0.00563;
    surf_dens_b1_ = 1.38;
    surf_dens_b2_ = 0.00069;
    alpha_        = 1.;

    // ── Model switches (all on by default) ────────────────────────────────
    nucleation_model_             = 1;
    condensation_model_           = 1;
    surface_growth_model_         = 1;
    oxidation_model_              = 1;
    coagulation_model_            = 1;
    coagulation_continuous_model_ = 1;

    // ── Sticking coefficient ───────────────────────────────────────────────
    sticking_model_           = StickingModel::Constant;
    sticking_coeff_constant_  = 2.e-3;

	is_simplified_pah_mass_ = false;
    is_debug_mode_          = false;


    // ── Species indices (0-based) ─────────────────────────────────────────
    index_H_    = thermo_.IndexOfSpecies("H");
    index_OH_   = thermo_.IndexOfSpecies("OH");
    index_O2_   = thermo_.IndexOfSpecies("O2");
    index_H2_   = thermo_.IndexOfSpecies("H2");
    index_H2O_  = thermo_.IndexOfSpecies("H2O");
    index_C2H2_ = thermo_.IndexOfSpecies("C2H2");

    // ── PAH (default: C2H2) ───────────────────────────────────────────────
    pah_species_ = "C2H2";
    SetPAH("C2H2");

    // ── Memory + initial moments ───────────────────────────────────────────
    MemoryAllocation();
}

// ============================================================================
// MemoryAllocation
// ============================================================================

template <ThermoMap Thermo>
void HMOM<Thermo>::MemoryAllocation()
{
    this->ZeroSources();

    source_coagulation_discrete_   = MomentVector::Zero();
    source_coagulation_ss_         = MomentVector::Zero();
    source_coagulation_sl_         = MomentVector::Zero();
    source_coagulation_ll_         = MomentVector::Zero();
    source_coagulation_continuous_ = MomentVector::Zero();
    source_coagulation_cont_ss_    = MomentVector::Zero();
    source_coagulation_cont_sl_    = MomentVector::Zero();
    source_coagulation_cont_ll_    = MomentVector::Zero();
    source_coagulation_all_        = MomentVector::Zero();

    this->omega_gas_ = Eigen::VectorXd::Zero(
        static_cast<Eigen::Index>(thermo_.NumberOfSpecies()));

    NL_   = 0.; NLVL_ = 0.; NLSL_ = 0.;

    // Initial normalised moments (seed: N=1e10, eta=0.9, gamma=2)
    const double Nseed = 1.e10;
    const double eta   = 0.90;
    const double gamma = 2.0;
    const double N0s   = eta * Nseed;
    const double NLs   = (1.0 - eta) * Nseed;
    const double gS    = std::pow(gamma, 2./3.);

    initial_moments_cache_(0) = Nseed / this->Nav_mol_;
    initial_moments_cache_(1) = (N0s + gamma * NLs) / this->Nav_mol_;
    initial_moments_cache_(2) = (N0s + gS * NLs)   / this->Nav_mol_;
    initial_moments_cache_(3) = N0s / this->Nav_mol_;
}

// ============================================================================
// Precalculations (called when PAH or density changes)
// ============================================================================

template <ThermoMap Thermo>
void HMOM<Thermo>::Precalculations()
{
    Cfm_      = std::sqrt(this->pi_ * this->kB_ / 2.0 / this->rho_particle_);
    const double K_diam = K_diam_HMOM(this->pi_);
    betaN_TV_ = 2.2 * 4. * this->sqrt2_ * K_diam * K_diam * Cfm_;

    // Re-apply geometry from stored enums
    {
        auto [Av, As, K] = FractalGeometry(static_cast<int>(fractal_diameter_model_), this->pi_);
        Av_fractal_ = Av; As_fractal_ = As; K_fractal_ = K;
    }
    {
        auto [D, Av, As, K] = CollisionGeometry(static_cast<int>(collision_diameter_model_), this->pi_);
        D_collisional_ = D; Av_collisional_ = Av; As_collisional_ = As; K_collisional_ = K;
    }

    SetPAH(pah_species_);
}

// ============================================================================
// SetPAH
// ============================================================================

template <ThermoMap Thermo>
void HMOM<Thermo>::SetPAH(std::string_view name)
{
    pah_species_ = std::string(name);
    pah_index_   = thermo_.IndexOfSpecies(name);
    ncpah_     = static_cast<double>(thermo_.NumberOfCarbonAtoms(pah_index_));
    nhpah_     = static_cast<double>(thermo_.NumberOfHydrogenAtoms(pah_index_));

    // ── PAH mass and geometry ─────────────────────────────────────────────
    mwpah_ = thermo_.MolecularWeight(pah_index_);             // [kg/kmol]
    if (is_simplified_pah_mass_)
        mwpah_ = ncpah_ * this->WC_;

    vpah_  = mwpah_ / this->rho_particle_ / this->Nav_kmol_;    // [m3]
    dpah_  = std::pow(6./this->pi_ * vpah_, 1./3.);             // [m]
    spah_  = this->pi_ * dpah_ * dpah_;                         // [m2]
    mpah_  = mwpah_ / this->Nav_kmol_;                          // [kg]

    const double K_spher = K_spher_HMOM(this->pi_);

    dimer_volume_  = 2. * vpah_;
    dimer_surface_ = K_spher * std::pow(dimer_volume_, 2./3.);
    V0_  = 2. * dimer_volume_;
    S0_  = K_spher * std::pow(V0_, 2./3.);
    VC2_ = (this->WC_ / this->rho_particle_ / this->Nav_kmol_) * 2.;
}

// ============================================================================
// SetGasClosureDummySpecies
// ============================================================================

template <ThermoMap Thermo>
void HMOM<Thermo>::SetGasClosureDummySpecies(std::string_view name)
{
    this->dummy_species_ = std::string(name);

    if (name == "none") {
        this->dummy_index_ = -1;
        this->dummy_species_closure_ = false;
        return;
    }

    this->dummy_index_ = thermo_.IndexOfSpecies(name);
    if (this->dummy_index_ < 0)
        throw std::runtime_error("[HMOM] Dummy species not found: " + this->dummy_species_);

    if (this->dummy_index_ == pah_index_)
        throw std::runtime_error("[HMOM] Dummy species cannot be the same as the PAH precursor.");

    if (this->dummy_index_ == index_H_   || this->dummy_index_ == index_H2_  ||
        this->dummy_index_ == index_O2_  || this->dummy_index_ == index_OH_  ||
        this->dummy_index_ == index_H2O_ || this->dummy_index_ == index_C2H2_)
        throw std::runtime_error("[HMOM] Dummy species cannot be H, H2, O2, OH, H2O, or C2H2.");

    this->dummy_species_closure_ = true;
}

// ============================================================================
// SetStickingCoefficientModel
// ============================================================================

template <ThermoMap Thermo>
void HMOM<Thermo>::SetStickingCoefficientModel(std::string_view label)
{
    if (label == "constant")
        sticking_model_ = StickingModel::Constant;
    else if (label == "pah-dependent")
        sticking_model_ = StickingModel::PAH4;
    else
        throw std::runtime_error("[HMOM] Unknown sticking coefficient model: " + std::string(label));
}

// ============================================================================
// SetStatus
// ============================================================================

template <ThermoMap Thermo>
void HMOM<Thermo>::SetStatus(double T, double P_Pa, const double* Y) noexcept
{
    this->T_    = T;
    this->P_Pa_ = P_Pa;

    // MW from mass fractions
    double invMW = 0.;
    for (unsigned i = 0; i < thermo_.NumberOfSpecies(); ++i)
        invMW += Y[i] / thermo_.MolecularWeight(i);
    this->MW_ = 1./invMW;

    const double cTot = P_Pa / (this->Rgas_ * T);   // [kmol/m3]
    this->rho_ = cTot * this->MW_;

    // Save mass fractions needed for HACA threshold
    mass_fraction_H_  = (index_H_  >= 0) ? std::max(Y[index_H_],  0.) : 0.;
    mass_fraction_OH_ = (index_OH_ >= 0) ? std::max(Y[index_OH_], 0.) : 0.;

    // Helper: concentration in [mol/cm3] = [kmol/m3] / 1e3
    auto concMolCm3 = [&](int idx) -> double {
        if (idx < 0) return 0.;
        return cTot * std::max(Y[idx], 0.) * this->MW_ / thermo_.MolecularWeight(static_cast<unsigned>(idx)) / 1.e3;
    };

    conc_H_    = concMolCm3(index_H_);
    conc_OH_   = concMolCm3(index_OH_);
    conc_O2_   = concMolCm3(index_O2_);
    conc_H2_   = concMolCm3(index_H2_);
    conc_H2O_  = concMolCm3(index_H2O_);
    conc_C2H2_ = concMolCm3(index_C2H2_);

    // PAH [mol/cm3]
    if (pah_index_ >= 0) {
        conc_PAH_ = cTot * std::max(Y[pah_index_], 0.) * this->MW_
                    / thermo_.MolecularWeight(static_cast<unsigned>(pah_index_)) / 1.e3;
    } else {
        conc_PAH_ = 0.;
    }
}

// ============================================================================
// SetMoments / SetNormalizedMoments
// ============================================================================

template <ThermoMap Thermo>
void HMOM<Thermo>::SetMoments(std::span<const double> m) noexcept
{
    SetNormalizedMoments(m[0], m[1], m[2], m[3]);
}

template <ThermoMap Thermo>
void HMOM<Thermo>::SetNormalizedMoments(double M00_norm, double M10_norm,
                                         double M01_norm, double N0_norm) noexcept
{
    M00_normalized_ = M00_norm;
    M10_normalized_ = M10_norm;
    M01_normalized_ = M01_norm;
    N0_normalized_  = N0_norm;
    GetMoments();
}

// ============================================================================
// GetMoments — denormalize + sanitize
// ============================================================================

template <ThermoMap Thermo>
void HMOM<Thermo>::GetMoments()
{
    double M00_raw = M00_normalized_ * this->Nav_mol_;
    double M10_raw = M10_normalized_ * V0_ * this->Nav_mol_;
    double M01_raw = M01_normalized_ * S0_ * this->Nav_mol_;
    double N0_raw  = N0_normalized_  * this->Nav_mol_;

    if (!std::isfinite(M00_raw)) M00_raw = 0.;
    if (!std::isfinite(M10_raw)) M10_raw = 0.;
    if (!std::isfinite(M01_raw)) M01_raw = 0.;
    if (!std::isfinite(N0_raw))  N0_raw  = 0.;

    M00_raw = std::max(M00_raw, 0.);
    M10_raw = std::max(M10_raw, 0.);
    M01_raw = std::max(M01_raw, 0.);
    N0_raw  = std::max(N0_raw,  0.);

    if (M00_raw < kSootNumberFloor || M10_raw < kSootVolumeFloor || M01_raw < kSootSurfaceFloor) {
        M00_ = M10_ = M01_ = N0_ = NL_ = NLVL_ = NLSL_ = 0.;
        return;
    }

    N0_raw = std::min(N0_raw, M00_raw);
    if (V0_ > 0.) N0_raw = std::min(N0_raw, M10_raw / V0_);
    if (S0_ > 0.) N0_raw = std::min(N0_raw, M01_raw / S0_);
    N0_raw = std::max(N0_raw, 0.);

    M00_ = M00_raw; M10_ = M10_raw; M01_ = M01_raw; N0_ = N0_raw;

    NL_   = M00_ - N0_;
    NLVL_ = M10_ - V0_ * N0_;
    NLSL_ = M01_ - S0_ * N0_;

    if (NL_   < 0. && std::fabs(NL_)   < 1.e-12 * std::max(M00_, 1.)) NL_   = 0.;
    if (NLVL_ < 0. && std::fabs(NLVL_) < 1.e-12 * std::max(M10_, 1.)) NLVL_ = 0.;
    if (NLSL_ < 0. && std::fabs(NLSL_) < 1.e-12 * std::max(M01_, 1.)) NLSL_ = 0.;

    NL_   = std::max(NL_,   0.);
    NLVL_ = std::max(NLVL_, 0.);
    NLSL_ = std::max(NLSL_, 0.);
}

// ============================================================================
// HasSoot
// ============================================================================

template <ThermoMap Thermo>
bool HMOM<Thermo>::HasSoot() const noexcept
{
    return (M00_ > kSootNumberFloor && M10_ > kSootVolumeFloor && M01_ > kSootSurfaceFloor &&
            std::isfinite(M00_) && std::isfinite(M10_) && std::isfinite(M01_));
}

// ============================================================================
// SafePowPositive
// ============================================================================

template <ThermoMap Thermo>
double HMOM<Thermo>::SafePowPositive(double x, double a) const noexcept
{
    if (!std::isfinite(x) || x <= 0.) return 0.;
    const double y = a * std::log(x);
    if (y >  700.) return std::exp(700.);
    if (y < -700.) return 0.;
    return std::exp(y);
}

// ============================================================================
// GetMoment / GetMissingMoment
// ============================================================================

template <ThermoMap Thermo>
double HMOM<Thermo>::GetMoment(double i, double j) const noexcept
{
    if (!HasSoot()) return 0.;

    double moment = 0.;
    if (N0_ > 0.) {
        const double small = N0_ * SafePowPositive(V0_, i) * SafePowPositive(S0_, j);
        if (std::isfinite(small)) moment += small;
    }
    if (NL_ > kSootNumberFloor && NLVL_ > kSootVolumeFloor && NLSL_ > kSootSurfaceFloor) {
        const double VL = NLVL_ / NL_;
        const double SL = NLSL_ / NL_;
        const double large = NL_ * SafePowPositive(VL, i) * SafePowPositive(SL, j);
        if (std::isfinite(large)) moment += large;
    }
    if (!std::isfinite(moment) || moment < 0.) return 0.;
    return moment;
}

template <ThermoMap Thermo>
double HMOM<Thermo>::GetMissingMoment(double i, double j) const noexcept
{
    if (!HasSoot()) return 0.;
    if (NL_ <= kSootNumberFloor || NLVL_ <= kSootVolumeFloor || NLSL_ <= kSootSurfaceFloor)
        return 0.;
    const double VL = NLVL_ / NL_;
    const double SL = NLSL_ / NL_;
    const double moment = NL_ * SafePowPositive(VL, i) * SafePowPositive(SL, j);
    if (!std::isfinite(moment) || moment < 0.) return 0.;
    return moment;
}

// ============================================================================
// GetBetaC
// ============================================================================

template <ThermoMap Thermo>
double HMOM<Thermo>::GetBetaC() const noexcept
{
    const double K_diam = K_diam_HMOM(this->pi_);

    const double betaC =
        std::pow(K_collisional_, 2.) * std::pow(dimer_volume_, -3./6.) * GetMoment(2.*Av_collisional_, 2.*As_collisional_) +
        2.*K_diam*K_collisional_     * std::pow(dimer_volume_, -1./6.) * GetMoment(Av_collisional_, As_collisional_) +
        std::pow(K_diam, 2.)         * std::pow(dimer_volume_,  1./6.) * GetMoment(0., 0.) +
        0.5*std::pow(K_collisional_,2.) * std::pow(dimer_volume_,  3./6.) * GetMoment(2.*Av_collisional_-1., 2.*As_collisional_) +
        2.*K_diam*K_collisional_        * std::pow(dimer_volume_,  5./6.) * GetMoment(Av_collisional_-1., As_collisional_) +
        0.5*std::pow(K_diam, 2.)        * std::pow(dimer_volume_,  7./6.) * GetMoment(-1., 0.);

    return Cfm_ * std::sqrt(this->T_) * betaC;
}

// ============================================================================
// DimerConcentration
// ============================================================================

template <ThermoMap Thermo>
void HMOM<Thermo>::DimerConcentration()
{
    conc_DIMER_        = 0.;
    dimerization_rate_ = 0.;

    betaN_ = betaN_TV_ * std::sqrt(this->T_) * std::pow(dimer_volume_, 1./6.);

    double betaC = 0.;
    if (condensation_model_ > 0) betaC = GetBetaC();

    if (betaN_ <= 0. || !std::isfinite(betaN_)) return;

    double stickCoeff = sticking_coeff_constant_;
    if (sticking_model_ == StickingModel::PAH4)
        stickCoeff = sticking_coeff_constant_ * std::pow(mwpah_, 4.);

    const double KfmPAH = betaN_TV_ * std::sqrt(this->T_) * std::pow(vpah_, 1./6.);
    if (KfmPAH <= 0. || !std::isfinite(KfmPAH)) return;
    if (betaC  <  0. || !std::isfinite(betaC))  return;

    const double beta_pah_pah = 0.5 * stickCoeff * KfmPAH;
    const double C_PAH = std::max(conc_PAH_, 0.) * 1.e6;    // mol/cm3 -> mol/m3
    const double reduced_rate = beta_pah_pah * C_PAH * C_PAH;

    dimerization_rate_ = reduced_rate;
    if (dimerization_rate_ <= 0. || !std::isfinite(dimerization_rate_)) {
        dimerization_rate_ = 0.; return;
    }

    const double Jpah = dimerization_rate_ * this->Nav_mol_ * this->Nav_mol_;
    if (Jpah <= 0. || !std::isfinite(Jpah)) return;

    const double discriminant = betaC*betaC + 4.*betaN_*Jpah;
    if (discriminant <= 0. || !std::isfinite(discriminant)) return;

    double Ndim = 2.*Jpah / (betaC + std::sqrt(discriminant));
    if (Ndim < 0. || !std::isfinite(Ndim)) Ndim = 0.;

    conc_DIMER_ = Ndim / this->Nav_mol_;
}

// ============================================================================
// PAHDimerizationRate [kmol/m3/s]
// ============================================================================

template <ThermoMap Thermo>
double HMOM<Thermo>::PAHDimerizationRate() const noexcept
{
    return 2. * dimerization_rate_ * this->Nav_mol_ / 1000.;
}

// ============================================================================
// CalculateAlphaCoefficient
// ============================================================================

template <ThermoMap Thermo>
void HMOM<Thermo>::CalculateAlphaCoefficient()
{
    alpha_ = 1.;
    if (!std::isfinite(this->T_) || this->T_ <= 0.) return;

    const double a = surf_dens_a1_ + surf_dens_a2_ * this->T_;
    const double b = surf_dens_b1_ + surf_dens_b2_ * this->T_;

    double VL = V0_;
    if (NL_ > kSootNumberFloor && NLVL_ > kSootVolumeFloor) VL = NLVL_ / NL_;
    if (!std::isfinite(VL) || VL <= 0.) VL = V0_;

    const double mC   = this->WC_ / this->Nav_kmol_;
    const double mu1  = VL * this->rho_particle_ / mC;
    if (!std::isfinite(mu1) || mu1 <= 1.) return;

    const double logMu1 = std::log(mu1);
    if (!std::isfinite(logMu1) || std::fabs(logMu1) < 1.e-12) return;

    alpha_ = std::tanh(a / logMu1 + b);
    if (!std::isfinite(alpha_)) alpha_ = 1.;
    alpha_ = std::max(0., std::min(alpha_, 1.));
}

// ============================================================================
// SootKineticConstants
// ============================================================================

template <ThermoMap Thermo>
void HMOM<Thermo>::SootKineticConstants()
{
    const double T = this->T_;
    const double k1f = A1f_ * std::pow(T, n1f_) * std::exp(-E1f_ / T);
    const double k1b = A1b_ * std::pow(T, n1b_) * std::exp(-E1b_ / T);
    const double k2f = A2f_ * std::pow(T, n2f_) * std::exp(-E2f_ / T);
    const double k2b = A2b_ * std::pow(T, n2b_) * std::exp(-E2b_ / T);
    const double k3f = A3f_ * std::pow(T, n3f_) * std::exp(-E3f_ / T);
    const double k3b = A3b_ * std::pow(T, n3b_) * std::exp(-E3b_ / T);
    const double k4  = A4_  * std::pow(T, n4_)  * std::exp(-E4_  / T);

    const double ratio = k1b*conc_H2O_ + k2b*conc_H2_ + k3b*conc_H_ + k4*conc_C2H2_;
    double conc_sootStar = 0.;
    if (ratio > 0.)
        conc_sootStar = (k1f*conc_OH_ + k2f*conc_H_ + k3f) / ratio;

    if (mass_fraction_H_ < 2.e-9 && mass_fraction_OH_ < 2.e-8)
        conc_sootStar = 0.;

    conc_sootStar = conc_sootStar / (1. + conc_sootStar);
    conc_sootStar = std::max(conc_sootStar, 0.);

    ksg_    = k4 * conc_C2H2_ * conc_sootStar;
    const double k6 = 8.94 * eff6_ * std::sqrt(T) * this->Nav_mol_;
    kox_O2_ = A5_ * std::pow(T, n5_) * std::exp(-E5_ / T) * conc_O2_ * conc_sootStar;
    kox_OH_ = (0.5 / (alpha_ * surface_density_)) * k6 * conc_OH_;
    kox_    = kox_O2_ + kox_OH_;
}

// ============================================================================
// SootNucleationM4
// ============================================================================

template <ThermoMap Thermo>
void HMOM<Thermo>::SootNucleationM4()
{
    this->source_nucleation_.setZero();
    double nuc_N0 = 0.;
    if (conc_DIMER_ > 0. && betaN_ > 0. &&
        std::isfinite(conc_DIMER_) && std::isfinite(betaN_))
        nuc_N0 = 0.5 * betaN_ * conc_DIMER_ * conc_DIMER_ * this->Nav_mol_;
    if (!std::isfinite(nuc_N0) || nuc_N0 < 0.) nuc_N0 = 0.;
    for (unsigned i = 0; i < 4u; ++i)
        this->source_nucleation_(i) = nuc_N0;
}

// ============================================================================
// SootSurfaceGrowthM4
// ============================================================================

template <ThermoMap Thermo>
void HMOM<Thermo>::SootSurfaceGrowthM4()
{
    if (!HasSoot()) return;

    this->source_growth_(0) = 0.;
    this->source_growth_(1) = ksg_ * (alpha_*surface_density_) * VC2_
                              * GetMoment(0., 1.) / this->Nav_mol_ / V0_;
    this->source_growth_(2) = ksg_ * (alpha_*surface_density_) * VC2_
                              * K_fractal_ * GetMoment(Av_fractal_, As_fractal_+2.) / this->Nav_mol_ / S0_;
    this->source_growth_(3) = -ksg_ * (alpha_*surface_density_) * S0_ * N0_ / this->Nav_mol_;
}

// ============================================================================
// SootOxidationM4
// ============================================================================

template <ThermoMap Thermo>
void HMOM<Thermo>::SootOxidationM4()
{
    if (!HasSoot()) return;

    const double coeff = kox_ * (alpha_*surface_density_) * VC2_;
    const double M01   = GetMissingMoment(0., 1.);
    const double M_12  = GetMissingMoment(-1., 2.);

    this->source_oxidation_(0) = -coeff * N0_ * S0_ / V0_ / this->Nav_mol_;
    this->source_oxidation_(1) = -coeff / V0_ / this->Nav_mol_ * M01;
    this->source_oxidation_(2) = -coeff / S0_ / this->Nav_mol_
                                  * (S0_*S0_*N0_/V0_ + 2./3.*(M_12 - S0_*N0_));
    this->source_oxidation_(3) = this->source_oxidation_(0);
}

// ============================================================================
// SootCondensationM4
// ============================================================================

template <ThermoMap Thermo>
void HMOM<Thermo>::SootCondensationM4()
{
    if (!HasSoot()) return;

    const double K_diam  = K_diam_HMOM(this->pi_);
    const double D_DIM   = K_diam * std::pow(dimer_volume_, 1./3.);
    const double D_NUCL  = K_diam * std::pow(V0_,           1./3.);
    const double sqrtDv  = std::sqrt(dimer_volume_);
    const double sqrtT   = std::sqrt(this->T_);
    const double cd      = conc_DIMER_;

    this->source_condensation_(0) = 0.;

    this->source_condensation_(1) =
        D_DIM*D_DIM * (GetMoment(0.,0.) + 0.5*dimer_volume_*GetMoment(-1.,0.)) +
        K_collisional_*K_collisional_ * (GetMoment(2.*Av_collisional_,2.*As_collisional_)
                                       + 0.5*dimer_volume_*GetMoment(2.*Av_collisional_-1.,2.*As_collisional_)) +
        2.*D_DIM*K_collisional_ * (GetMoment(Av_collisional_,As_collisional_)
                                 + 0.5*dimer_volume_*GetMoment(Av_collisional_-1.,As_collisional_));
    this->source_condensation_(1) = Cfm_ * sqrtDv * cd * sqrtT * this->source_condensation_(1) / V0_;

    this->source_condensation_(2) =
        D_DIM*D_DIM * (GetMoment(Av_fractal_,As_fractal_+1.)
                     + 0.5*dimer_volume_*GetMoment(Av_fractal_-1.,As_fractal_+1.)) +
        K_collisional_*K_collisional_ * (GetMoment(Av_fractal_+2.*Av_collisional_,As_fractal_+1.+2.*As_collisional_)
                                       + 0.5*dimer_volume_*GetMoment(Av_fractal_+2.*Av_collisional_-1.,As_fractal_+1.+2.*As_collisional_)) +
        2.*D_DIM*K_collisional_ * (GetMoment(Av_fractal_+Av_collisional_,As_fractal_+1.+As_collisional_)
                                 + 0.5*dimer_volume_*GetMoment(Av_fractal_+Av_collisional_-1.,As_fractal_+1.+As_collisional_));
    this->source_condensation_(2) = Cfm_ * sqrtDv * cd * sqrtT * this->source_condensation_(2) / S0_ * K_fractal_;

    this->source_condensation_(3) = -Cfm_ / sqrtDv * cd * sqrtT
                                    * (1. + 0.5*dimer_volume_/V0_)
                                    * (D_DIM + D_NUCL)*(D_DIM + D_NUCL) * N0_;
}

// ============================================================================
// SootCoagulationM4 and variants
// ============================================================================

template <ThermoMap Thermo>
void HMOM<Thermo>::SootCoagulationM4()
{
    if (!HasSoot()) return;
    SootCoagulationSmallSmallM4();
    SootCoagulationSmallLargeM4();
    SootCoagulationLargeLargeM4();
}

template <ThermoMap Thermo>
void HMOM<Thermo>::SootCoagulationSmallSmallM4()
{
    const double K_diam  = K_diam_HMOM(this->pi_);
    const double K_spher = K_spher_HMOM(this->pi_);
    const double DcNUCL  = K_diam * std::pow(V0_, 1./3.);
    const double S00     = K_spher * std::pow(2.*V0_, 2./3.);
    const double beta00  = 2.20*Cfm_*std::sqrt(2./V0_)*std::pow(2.*DcNUCL,2.)*std::sqrt(this->T_);

    source_coagulation_ss_(0) = -0.5*beta00*N0_*N0_/this->Nav_mol_;
    source_coagulation_ss_(1) = 0.;
    source_coagulation_ss_(2) = 0.5*beta00*S00/S0_*N0_*N0_/this->Nav_mol_ + 2.*source_coagulation_ss_(0);
    source_coagulation_ss_(3) = 2.*source_coagulation_ss_(0);
}

template <ThermoMap Thermo>
void HMOM<Thermo>::SootCoagulationSmallLargeM4()
{
    const double K_diam = K_diam_HMOM(this->pi_);
    const double DcNUCL = K_diam * std::pow(V0_, 1./3.);
    const double sqrtT  = std::sqrt(this->T_);

    {
        const double psi0 = std::pow(V0_,-1./2.)*(
            K_collisional_*K_collisional_*GetMissingMoment(2.*Av_collisional_-0.5,2.*As_collisional_) +
            DcNUCL*DcNUCL               *GetMissingMoment(-0.5,0.) +
            2.*DcNUCL*K_collisional_    *GetMissingMoment(Av_collisional_-0.5,As_collisional_));

        double psi1 = std::pow(V0_,-1./2.)*(
            K_collisional_*K_collisional_*GetMissingMoment(2.*Av_collisional_+0.5,2.*As_collisional_) +
            DcNUCL*DcNUCL               *GetMissingMoment(0.5,0.) +
            2.*DcNUCL*K_collisional_    *GetMissingMoment(Av_collisional_+0.5,As_collisional_));
        psi1 += V0_*psi0;

        source_coagulation_sl_(0) = 0.;
        if (psi0*psi1 >= 0.)
            source_coagulation_sl_(0) = -2.2*Cfm_*sqrtT*N0_/this->Nav_mol_*std::sqrt(psi0*psi1);
    }

    source_coagulation_sl_(1) = 0.;

    {
        const double psi0 = std::pow(V0_,-1./2.)*(
            K_collisional_*K_collisional_*GetMissingMoment(2.*Av_collisional_-0.5+Av_fractal_,2.*As_collisional_+As_fractal_+1.) +
            DcNUCL*DcNUCL               *GetMissingMoment(-0.5+Av_fractal_,As_fractal_+1.) +
            2.*DcNUCL*K_collisional_    *GetMissingMoment(Av_collisional_-0.5+Av_fractal_,As_collisional_+As_fractal_+1.));

        double psi1 = std::pow(V0_,-1./2.)*(
            K_collisional_*K_collisional_*GetMissingMoment(2.*Av_collisional_+0.5+Av_fractal_,2.*As_collisional_+As_fractal_+1.) +
            DcNUCL*DcNUCL               *GetMissingMoment(0.5+Av_fractal_,As_fractal_+1.) +
            2.*DcNUCL*K_collisional_    *GetMissingMoment(Av_collisional_+0.5+Av_fractal_,As_collisional_+As_fractal_+1.));
        psi1 += V0_*psi0;

        source_coagulation_sl_(2) = source_coagulation_sl_(0);
        if (psi0*psi1 >= 0.)
            source_coagulation_sl_(2) += 2.2*Cfm_*sqrtT*N0_*V0_*K_fractal_/S0_/this->Nav_mol_*std::sqrt(psi0*psi1);
    }

    source_coagulation_sl_(3) = source_coagulation_sl_(0);
}

template <ThermoMap Thermo>
void HMOM<Thermo>::SootCoagulationLargeLargeM4()
{
    source_coagulation_ll_.setZero();

    const double psi0 = 2.*K_collisional_*K_collisional_*(
        GetMissingMoment(2.*Av_collisional_-0.5,2.*As_collisional_)*GetMissingMoment(-0.5,0.) +
        GetMissingMoment(Av_collisional_-0.5,As_collisional_)*GetMissingMoment(Av_collisional_-0.5,As_collisional_));

    const double psi1 = 2.*K_collisional_*K_collisional_*(
        GetMissingMoment(2.*Av_collisional_+0.5,2.*As_collisional_)*GetMissingMoment(-0.5,0.) +
        GetMissingMoment(Av_collisional_+0.5,As_collisional_)*GetMissingMoment(Av_collisional_-0.5,As_collisional_) +
        GetMissingMoment(2.*Av_collisional_-0.5,2.*As_collisional_)*GetMissingMoment(0.5,0.));

    if (psi0*psi1 >= 0.)
        source_coagulation_ll_(0) = -0.5*2.2*Cfm_*std::sqrt(this->T_)/this->Nav_mol_*std::sqrt(psi0*psi1);
}

// ============================================================================
// SootCoagulationContinuousM4 and variants
// ============================================================================

template <ThermoMap Thermo>
void HMOM<Thermo>::SootCoagulationContinuousM4()
{
    double lambda = 8.2057e-5 / this->sqrt2_ / std::pow(200.e-12, 2.) / this->Nav_mol_;
    lambda = 1.257 * lambda * this->T_ / (this->P_Pa_ / 101325.);

    SootCoagulationContinuousSmallSmallM4(lambda);
    SootCoagulationContinuousSmallLargeM4(lambda);
    SootCoagulationContinuousLargeLargeM4(lambda);
}

template <ThermoMap Thermo>
void HMOM<Thermo>::SootCoagulationContinuousSmallSmallM4(double lambda)
{
    const double K_diam  = K_diam_HMOM(this->pi_);
    const double K_spher = K_spher_HMOM(this->pi_);
    const double DcNUCL  = K_diam * std::pow(V0_, 1./3.);
    const double S00     = K_spher * std::pow(2.*V0_, 2./3.);
    const double CC0     = 1. + lambda / DcNUCL;
    const double beta00  = 2.*this->kB_*this->T_/3./this->mu_ * (2.*CC0/DcNUCL) * (2.*DcNUCL);

    source_coagulation_cont_ss_(0) = -0.5*beta00*N0_*N0_/this->Nav_mol_;
    source_coagulation_cont_ss_(1) = 0.;
    source_coagulation_cont_ss_(2) = 0.5*beta00*S00/S0_*N0_*N0_/this->Nav_mol_ + 2.*source_coagulation_cont_ss_(0);
    source_coagulation_cont_ss_(3) = 2.*source_coagulation_cont_ss_(0);
}

template <ThermoMap Thermo>
void HMOM<Thermo>::SootCoagulationContinuousSmallLargeM4(double lambda)
{
    const double K_diam = K_diam_HMOM(this->pi_);
    const double DcNUCL = K_diam * std::pow(V0_, 1./3.);
    const double betai0 = 2.*this->kB_*this->T_/3./this->mu_;

    source_coagulation_cont_sl_(0) =
        (2. + lambda/DcNUCL) *GetMissingMoment(0.,0.) +
        (DcNUCL+lambda)/K_collisional_ *GetMissingMoment(-Av_collisional_,-As_collisional_) +
        (1.+lambda/DcNUCL)*K_collisional_/DcNUCL *GetMissingMoment(Av_collisional_,As_collisional_) +
        lambda*DcNUCL*K_collisional_*K_collisional_ *GetMissingMoment(-2.*Av_collisional_,-2.*As_collisional_);
    source_coagulation_cont_sl_(0) *= -N0_/this->Nav_mol_*betai0;

    source_coagulation_cont_sl_(1) = 0.;

    source_coagulation_cont_sl_(2) =
        (2.+lambda/DcNUCL) *GetMissingMoment(Av_fractal_,As_fractal_+1.) +
        (DcNUCL+lambda)/K_collisional_ *GetMissingMoment(-Av_collisional_+Av_fractal_,-As_collisional_+As_fractal_+1.) +
        (1.+lambda/DcNUCL)*K_collisional_/DcNUCL *GetMissingMoment(Av_collisional_+Av_fractal_,As_collisional_+As_fractal_+1.) +
        lambda*DcNUCL*K_collisional_*K_collisional_ *GetMissingMoment(-2.*Av_collisional_+Av_fractal_,-2.*As_collisional_+As_fractal_+1.);
    source_coagulation_cont_sl_(2) =
        N0_*V0_*K_fractal_/S0_/this->Nav_mol_*betai0*source_coagulation_cont_sl_(2)
        + source_coagulation_cont_sl_(0);

    source_coagulation_cont_sl_(3) = source_coagulation_cont_sl_(0);
}

template <ThermoMap Thermo>
void HMOM<Thermo>::SootCoagulationContinuousLargeLargeM4(double lambda)
{
    const double betai0 = 2.*this->kB_*this->T_/3./this->mu_;

    const double val =
        GetMissingMoment(0.,0.)*GetMissingMoment(0.,0.) +
        lambda/K_collisional_*(
            GetMissingMoment(0.,0.)*GetMissingMoment(-Av_collisional_,-As_collisional_) +
            GetMissingMoment(Av_collisional_,As_collisional_)*GetMissingMoment(-2.*Av_collisional_,-2.*As_collisional_) +
            GetMissingMoment(Av_collisional_,As_collisional_)*GetMissingMoment(-Av_collisional_,-As_collisional_));

    source_coagulation_cont_ll_(0) = -0.5*betai0/this->Nav_mol_*val;
    source_coagulation_cont_ll_(1) = 0.;
    source_coagulation_cont_ll_(2) = 0.;
    source_coagulation_cont_ll_(3) = 0.;
}

// ============================================================================
// CalculateSourceMoments
// ============================================================================

template <ThermoMap Thermo>
void HMOM<Thermo>::CalculateSourceMoments() noexcept
{
    this->ZeroSources();
    source_coagulation_discrete_   = MomentVector::Zero();
    source_coagulation_ss_         = MomentVector::Zero();
    source_coagulation_sl_         = MomentVector::Zero();
    source_coagulation_ll_         = MomentVector::Zero();
    source_coagulation_continuous_ = MomentVector::Zero();
    source_coagulation_cont_ss_    = MomentVector::Zero();
    source_coagulation_cont_sl_    = MomentVector::Zero();
    source_coagulation_cont_ll_    = MomentVector::Zero();
    source_coagulation_all_        = MomentVector::Zero();

    if (!this->is_active_) return;

    if (surface_density_correction_) CalculateAlphaCoefficient();

    DimerConcentration();

    if (nucleation_model_ > 0) SootNucleationM4();

    if (surface_growth_model_ > 0 || oxidation_model_ > 0) {
        SootKineticConstants();
        if (surface_growth_model_ > 0) SootSurfaceGrowthM4();
        if (oxidation_model_       > 0) SootOxidationM4();
    }

    if (condensation_model_           > 0) SootCondensationM4();
    if (coagulation_model_            > 0) SootCoagulationM4();
    if (coagulation_continuous_model_ > 0) SootCoagulationContinuousM4();

    // Sanitize and combine
    for (unsigned i = 0; i < 4u; ++i) {
        auto nanZ = [](double& v){ if (std::isnan(v)) v = 0.; };
        nanZ(this->source_nucleation_(i));
        nanZ(this->source_growth_(i));
        nanZ(this->source_oxidation_(i));
        nanZ(this->source_condensation_(i));
        nanZ(source_coagulation_ss_(i));
        nanZ(source_coagulation_sl_(i));
        nanZ(source_coagulation_ll_(i));
        nanZ(source_coagulation_cont_ss_(i));
        nanZ(source_coagulation_cont_sl_(i));
        nanZ(source_coagulation_cont_ll_(i));

        const double disc = source_coagulation_ss_(i) + source_coagulation_sl_(i) + source_coagulation_ll_(i);
        const double cont = source_coagulation_cont_ss_(i) + source_coagulation_cont_sl_(i) + source_coagulation_cont_ll_(i);

        source_coagulation_discrete_(i)   = disc;
        source_coagulation_continuous_(i) = cont;

        if (disc == 0.)
            source_coagulation_all_(i) = cont;
        else if (cont == 0.)
            source_coagulation_all_(i) = disc;
        else
            source_coagulation_all_(i) = cont * disc / (cont + disc);

        this->source_coagulation_(i) = source_coagulation_all_(i);

        this->source_all_(i) = this->source_nucleation_(i)
                              + this->source_growth_(i)
                              + this->source_oxidation_(i)
                              + this->source_condensation_(i)
                              + source_coagulation_all_(i);
    }

    if (this->gas_consumption_) CalculateOmegaGas();
}

// ============================================================================
// CalculateOmegaGas
// ============================================================================

template <ThermoMap Thermo>
void HMOM<Thermo>::CalculateOmegaGas() noexcept
{
    this->omega_gas_.setZero();
    if (!this->gas_consumption_) return;

    if (nucleation_model_ > 0 && pah_index_ >= 0) {
        const double R_PAH = PAHDimerizationRate();   // [kmol/m3/s]
        if (R_PAH > 0. && std::isfinite(R_PAH)) {
            const double mw_pah = thermo_.MolecularWeight(static_cast<unsigned>(pah_index_));
            this->omega_gas_[static_cast<unsigned>(pah_index_)] -= R_PAH * mw_pah;
        }
    }

    if (surface_growth_model_ > 0 && VC2_ > 0. && index_C2H2_ >= 0) {
        const double R_C2H2 = this->source_growth_(1) * V0_ / VC2_;   // [mol/m3/s]
        if (R_C2H2 > 0. && std::isfinite(R_C2H2)) {
            const double kg_per_mol = thermo_.MolecularWeight(static_cast<unsigned>(index_C2H2_)) / 1000.;
            this->omega_gas_[static_cast<unsigned>(index_C2H2_)] -= R_C2H2 * kg_per_mol;
        }
    }

    if (oxidation_model_ > 0 && VC2_ > 0.) {
        const double R_oxid_C2 = -this->source_oxidation_(1) * V0_ / VC2_;   // [mol/m3/s]
        if (R_oxid_C2 > 0. && std::isfinite(R_oxid_C2)) {
            const double kox_sum = kox_O2_ + kox_OH_;
            if (kox_sum > 0. && std::isfinite(kox_sum)) {
                const double fO2 = kox_O2_ / kox_sum;
                const double fOH = kox_OH_ / kox_sum;
                if (index_O2_ >= 0 && fO2 > 0.) {
                    const double kg_per_mol = thermo_.MolecularWeight(static_cast<unsigned>(index_O2_)) / 1000.;
                    this->omega_gas_[static_cast<unsigned>(index_O2_)] -= fO2 * R_oxid_C2 * kg_per_mol;
                }
                if (index_OH_ >= 0 && fOH > 0.) {
                    const double kg_per_mol = thermo_.MolecularWeight(static_cast<unsigned>(index_OH_)) / 1000.;
                    this->omega_gas_[static_cast<unsigned>(index_OH_)] -= fOH * R_oxid_C2 * kg_per_mol;
                }
            }
        }
    }

    if (this->dummy_species_closure_ && this->dummy_index_ >= 0) {
        double sum = 0.;
        for (Eigen::Index i = 0; i < this->omega_gas_.size(); ++i)
            if (i != static_cast<Eigen::Index>(this->dummy_index_))
                sum += this->omega_gas_[i];
        this->omega_gas_[this->dummy_index_] = -sum;
    }
}

// ============================================================================
// Particle property accessors
// ============================================================================

template <ThermoMap Thermo>
double HMOM<Thermo>::VolumeFraction() const noexcept
{
    const double fv = GetMoment(1.,0.);
    return (!std::isfinite(fv) || fv < kSootVolumeFloor) ? 0. : fv;
}

template <ThermoMap Thermo>
double HMOM<Thermo>::ParticleDiameter() const noexcept
{
    const double V = GetMoment(1.,0.);
    const double S = GetMoment(0.,1.);
    if (!std::isfinite(V)||!std::isfinite(S)||V<=kSootVolumeFloor||S<=kSootSurfaceFloor) return 0.;
    const double dp = 6.*V/S;
    return (std::isfinite(dp)&&dp>0.) ? dp : 0.;
}

template <ThermoMap Thermo>
double HMOM<Thermo>::CollisionDiameter() const noexcept
{
    const double N = GetMoment(0.,0.);
    if (!std::isfinite(N)||N<=kSootNumberFloor) return 0.;
    const double dc = K_collisional_ * GetMoment(Av_collisional_,As_collisional_) / N;
    return (std::isfinite(dc)&&dc>0.) ? dc : 0.;
}

template <ThermoMap Thermo>
double HMOM<Thermo>::ParticleNumberDensity() const noexcept
{
    const double n = GetMoment(0.,0.);
    return (!std::isfinite(n)||n<kSootNumberFloor) ? 0. : n;
}

template <ThermoMap Thermo>
double HMOM<Thermo>::MassFraction() const noexcept
{
    return this->rho_particle_ / this->rho_ * VolumeFraction();
}

template <ThermoMap Thermo>
double HMOM<Thermo>::SpecificSurface() const noexcept
{
    const double Ss = GetMoment(0.,1.);
    return (!std::isfinite(Ss)||Ss<=0.) ? 0. : Ss;
}

template <ThermoMap Thermo>
double HMOM<Thermo>::NumberOfPrimaryParticles() const noexcept
{
    const double N = GetMoment(0.,0.);
    if (!std::isfinite(N)||N<=kSootNumberFloor) return 0.;
    const double M = GetMoment(-2.,3.);
    if (!std::isfinite(M)||M<=0.) return 0.;
    const double K_spher = K_spher_HMOM(this->pi_);
    const double np = std::pow(K_spher,-3.)*M/N;
    return (std::isfinite(np)&&np>=1.) ? np : 1.;
}

template <ThermoMap Thermo>
double HMOM<Thermo>::DiffusionCoefficient() const noexcept
{
    const double m      = this->rho_*this->kB_*this->T_/this->P_Pa_;
    const double lambda = this->mu_/this->rho_*std::sqrt(this->pi_*m/(2.*this->kB_*this->T_));
    const double dc     = std::max(CollisionDiameter(), 1.e-12);
    const double Cu     = 1. + 2.154*lambda/dc;
    const double D      = this->kB_*this->T_*Cu/(3.*this->pi_*this->mu_*dc);
    return std::max(this->rho_*D, this->mu_/this->schmidt_number_);
}

// ============================================================================
// Properties
// ============================================================================

template <ThermoMap Thermo>
void HMOM<Thermo>::Properties(double& fv, double& dp, double& dc,
                               double& np, double& ss, double& vs) const noexcept
{
    const double N = GetMoment(0.,0.);
    const double V = GetMoment(1.,0.);
    const double S = GetMoment(0.,1.);

    if (!std::isfinite(N)||!std::isfinite(V)||!std::isfinite(S)||N<=0.||V<=0.||S<=0.) {
        fv = 0.; vs = V0_; ss = S0_;
        dp = 6.*V0_/S0_; np = 1.;
        dc = K_collisional_*std::pow(V0_,Av_collisional_)*std::pow(S0_,As_collisional_);
        return;
    }
    fv = V; vs = V/N; ss = S/N; dp = 6.*vs/ss;
    np = std::pow(ss,3.)/(36.*this->pi_*vs*vs);
    if (!std::isfinite(np)||np<1.) np = 1.;
    dc = K_collisional_*std::pow(vs,Av_collisional_)*std::pow(ss,As_collisional_);
    if (!std::isfinite(dc)||dc<=0.) dc = dp;
}

// ============================================================================
// Write on files
// ============================================================================

template <ThermoMap Thermo>
void HMOM<Thermo>::WriteHeaderLine(MOM::OutputFileColumns& fOutput, const unsigned int precision)
{
	// Main variables (common to every method)
	fOutput.AddColumn("Ys[-]", precision);
	fOutput.AddColumn("Ns[#/m3]", precision);
	fOutput.AddColumn("Ss[m2/m3]", precision);
	fOutput.AddColumn("fv[-]", precision);
	fOutput.AddColumn("dp[nm]", precision);
	fOutput.AddColumn("dc[nm]", precision);
	fOutput.AddColumn("np[nm]", precision);

	// Gas consumption (formation rates)
	fOutput.AddColumn("Gas[kg/m3/s]", precision);
	fOutput.AddColumn("PAH[kg/m3/s]", precision);
	fOutput.AddColumn("C2H2[kg/m3/s]", precision);
	fOutput.AddColumn("H2[kg/m3/s]", precision);
	fOutput.AddColumn("H[kg/m3/s]", precision);
	fOutput.AddColumn("OH[kg/m3/s]", precision);
	fOutput.AddColumn("O2[kg/m3/s]", precision);

	// Specific quantities
	fOutput.AddColumn("ss[m2/#]", precision);
	fOutput.AddColumn("vs[m3/#]", precision);

	// Diffusion coefficient
	fOutput.AddColumn("Gamma[m2/s]", precision);

	// Additional statistical quantities
	fOutput.AddColumn("N0[#/m3]", precision);
	fOutput.AddColumn("NL[#/m3]", precision);
	fOutput.AddColumn("alphaL[-]", precision);
	fOutput.AddColumn("dpL[nm]", precision);
	fOutput.AddColumn("npL[-]", precision);	
	fOutput.AddColumn("d63[nm]", precision);
	fOutput.AddColumn("sigma_dp[-]", precision);
	fOutput.AddColumn("sigma_np[-]", precision);
	fOutput.AddColumn("sigma_dp_L[-]", precision);
	fOutput.AddColumn("sigma_np_L[-]", precision);
	fOutput.AddColumn("gsd_dp[-]", precision);
	fOutput.AddColumn("gsd_np[-]", precision);
	fOutput.AddColumn("gsd_dp_L[-]", precision);
	fOutput.AddColumn("gsd_np_L[-]", precision);	

	// Specific quantities
	for (unsigned int j = 0; j < this->n_equations; j++)
	{
		std::stringstream label;	label << j;
		std::string title = "M(" + label.str() + ")[mol/m3]";
		fOutput.AddColumn(title, precision);
	}

	for (unsigned int j = 0; j < this->n_equations; j++)
	{
		std::stringstream label;	label << j;
		std::string title = "Sall(" + label.str() + ")[mol/m3/s]";
		fOutput.AddColumn(title, precision);
	}

	for (unsigned int j = 0; j < this->n_equations; j++)
	{
		std::stringstream label;	label << j;
		std::string title = "Snuc(" + label.str() + ")[mol/m3/s]";
		fOutput.AddColumn(title, precision);
	}

	for (unsigned int j = 0; j < this->n_equations; j++)
	{
		std::stringstream label;	label << j;
		std::string title = "Sgro(" + label.str() + ")[mol/m3/s]";
		fOutput.AddColumn(title, precision);
	}

	for (unsigned int j = 0; j < this->n_equations; j++)
	{
		std::stringstream label;	label << j;
		std::string title = "Soxi(" + label.str() + ")[mol/m3/s]";
		fOutput.AddColumn(title, precision);
	}

	for (unsigned int j = 0; j < this->n_equations; j++)
	{
		std::stringstream label;	label << j;
		std::string title = "Scon(" + label.str() + ")[mol/m3/s]";
		fOutput.AddColumn(title, precision);
	}

	for (unsigned int j = 0; j < this->n_equations; j++)
	{
		std::stringstream label;	label << j;
		std::string title = "ScoaTot(" + label.str() + ")[mol/m3/s]";
		fOutput.AddColumn(title, precision);
	}

	for (unsigned int j = 0; j < this->n_equations; j++)
	{
		std::stringstream label;	label << j;
		std::string title = "ScoaDis(" + label.str() + ")[mol/m3/s]";
		fOutput.AddColumn(title, precision);
	}

	for (unsigned int j = 0; j < this->n_equations; j++)
	{
		std::stringstream label;	label << j;
		std::string title = "ScoaDisSS(" + label.str() + ")[mol/m3/s]";
		fOutput.AddColumn(title, precision);
	}

	for (unsigned int j = 0; j < this->n_equations; j++)
	{
		std::stringstream label;	label << j;
		std::string title = "ScoaDisSL(" + label.str() + ")[mol/m3/s]";
		fOutput.AddColumn(title, precision);
	}

	for (unsigned int j = 0; j < this->n_equations; j++)
	{
		std::stringstream label;	label << j;
		std::string title = "ScoaDisLL(" + label.str() + ")[mol/m3/s]";
		fOutput.AddColumn(title, precision);
	}

	for (unsigned int j = 0; j < this->n_equations; j++)
	{
		std::stringstream label;	label << j;
		std::string title = "ScoaCon(" + label.str() + ")[mol/m3/s]";
		fOutput.AddColumn(title, precision);
	}

	for (unsigned int j = 0; j < this->n_equations; j++)
	{
		std::stringstream label;	label << j;
		std::string title = "ScoaConSS(" + label.str() + ")[mol/m3/s]";
		fOutput.AddColumn(title, precision);
	}

	for (unsigned int j = 0; j < this->n_equations; j++)
	{
		std::stringstream label;	label << j;
		std::string title = "ScoaConSL(" + label.str() + ")[mol/m3/s]";
		fOutput.AddColumn(title, precision);
	}

	for (unsigned int j = 0; j < this->n_equations; j++)
	{
		std::stringstream label;	label << j;
		std::string title = "ScoaConLL(" + label.str() + ")[mol/m3/s]";
		fOutput.AddColumn(title, precision);
	}
}

template <ThermoMap Thermo>
void HMOM<Thermo>::WriteOutputLine( MOM::OutputFileColumns& fOutput,
							const double T, const double P_Pa, const double* Y, const double mu,
							const double* M)
{
	SetNormalizedMoments(M[0], M[1], M[2], M[3]);
	SetStatus(T, P_Pa, Y);
	this->SetViscosity(mu);
	CalculateSourceMoments();
	
	// Soot properties
	fOutput << MassFraction();
	fOutput << ParticleNumberDensity();
	fOutput << SpecificSurface();
	fOutput << VolumeFraction();
	fOutput << ParticleDiameter()*1e9;
	fOutput << CollisionDiameter()*1e9;
	fOutput << NumberOfPrimaryParticles();

	// Gas consumption (formation rates)
	fOutput << this->omega_gas_.sum();
	fOutput << this->omega_gas_[pah_index_];
	fOutput << this->omega_gas_[index_C2H2_];
	fOutput << this->omega_gas_[index_H2_];
	fOutput << this->omega_gas_[index_H_];
	fOutput << this->omega_gas_[index_OH_];
	fOutput << this->omega_gas_[index_O2_];

	// Specific quantities
	fOutput << this->SootMeanParticleSurface();
	fOutput << this->SootMeanParticleVolume();

	// Diffusion coefficient (m2/s)
	fOutput << DiffusionCoefficient()/this->rho_;

	// Additional statistical quantities
	fOutput << SootSmallParticleNumberDensity();
	fOutput << SootLargeParticleNumberDensity();
	fOutput << SootLargeParticleFraction();
	fOutput << SootLargePrimaryParticleDiameter() * 1.e9;
	fOutput << SootLargeNumberOfPrimaryParticles();
	fOutput << SootD63() * 1.e9;
	fOutput << SootLogGeometricStdDevPrimaryParticleDiameter();
	fOutput << SootLogGeometricStdDevPrimaryParticleNumber();
	fOutput << SootLargeLogGeometricStdDevPrimaryParticleDiameter();
	fOutput << SootLargeLogGeometricStdDevPrimaryParticleNumber();
	fOutput << std::exp(SootLogGeometricStdDevPrimaryParticleDiameter());
	fOutput << std::exp(SootLogGeometricStdDevPrimaryParticleNumber());
	fOutput << std::exp(SootLargeLogGeometricStdDevPrimaryParticleDiameter());
	fOutput << std::exp(SootLargeLogGeometricStdDevPrimaryParticleNumber());

                        
	// Soot moments
	for (unsigned int j = 0; j < this->n_equations; j++)
		fOutput << M[j];

	// Source terms (overall)
	for (unsigned int j = 0; j < this->n_equations; j++)
		fOutput << this->source_all_[j];

	// Source terms (nucleation)
	for (unsigned int j = 0; j < this->n_equations; j++)
		fOutput << this->source_nucleation_[j];

	// Source terms (surface growth)
	for (unsigned int j = 0; j < this->n_equations; j++)
		fOutput << this->source_growth_[j];

	// Source terms (oxidation)
	for (unsigned int j = 0; j < this->n_equations; j++)
		fOutput << this->source_oxidation_[j];

	// Source terms (condensation)
	for (unsigned int j = 0; j < this->n_equations; j++)
		fOutput << this->source_condensation_[j];

	// Source terms (coagulation)
	for (unsigned int j = 0; j < this->n_equations; j++)
		fOutput << this->source_coagulation_all_[j];

	// Source terms (coagulation discrete)
	for (unsigned int j = 0; j < this->n_equations; j++)
		fOutput << this->source_coagulation_discrete_[j];

	// Source terms (coagulation discrete)
	for (unsigned int j = 0; j < this->n_equations; j++)
		fOutput << this->source_coagulation_ss_[j];

	// Source terms (coagulation discrete)
	for (unsigned int j = 0; j < this->n_equations; j++)
		fOutput << this->source_coagulation_sl_[j];

	// Source terms (coagulation discrete)
	for (unsigned int j = 0; j < this->n_equations; j++)
		fOutput << this->source_coagulation_ll_[j];

	// Source terms (coagulation discrete)
	for (unsigned int j = 0; j < this->n_equations; j++)
		fOutput << this->source_coagulation_continuous_[j];

	// Source terms (coagulation discrete)
	for (unsigned int j = 0; j < this->n_equations; j++)
		fOutput << this->source_coagulation_cont_ss_[j];

	// Source terms (coagulation discrete)
	for (unsigned int j = 0; j < this->n_equations; j++)
		fOutput << this->source_coagulation_cont_sl_[j];

	// Source terms (coagulation discrete)
	for (unsigned int j = 0; j < this->n_equations; j++)
		fOutput << this->source_coagulation_cont_ll_[j];
					
}

// ============================================================================
// SetupFromDictionary 
// ============================================================================

#if defined(MOM_USE_DICTIONARY)

template <ThermoMap Thermo>
template <typename Dictionary>
std::expected<void, std::string>
HMOM<Thermo>::SetupFromDictionary(Dictionary& dict)
{
	HMOM_Grammar grammar;
	dict.SetGrammar(grammar);

	if (dict.CheckOption("@HMOM") == true)
		dict.ReadBool("@HMOM", this->is_active_);

	if (dict.CheckOption("@FractalDiameterModel") == true)
	{
		int flag;
		dict.ReadInt("@FractalDiameterModel", flag);
		this->SetFractalDiameterModel(flag);
	}

	if (dict.CheckOption("@CollisionDiameterModel") == true)
	{
		int flag;
		dict.ReadInt("@CollisionDiameterModel", flag);
		this->SetCollisionDiameterModel(flag);
	}

	if (dict.CheckOption("@GasClosureDummySpecies") == true)
	{
		std::string species;
		dict.ReadString("@GasClosureDummySpecies", species);
		this->SetGasClosureDummySpecies(species);
	}		
	
	if (dict.CheckOption("@GasConsumption") == true)
	{
		bool flag;
		dict.ReadBool("@GasConsumption", flag);
		this->SetGasConsumption(flag);
	}

	if (dict.CheckOption("@SimplifiedPAHMass") == true)
		dict.ReadBool("@SimplifiedPAHMass", this->is_simplified_pah_mass_);

	if (dict.CheckOption("@PAH") == true)
	{
		std::string name;
		dict.ReadString("@PAH", name);
		this->SetPAH(name);
	}

	if (dict.CheckOption("@NucleationModel") == true)
	{
		int flag;
		dict.ReadInt("@NucleationModel", flag);
		this->SetNucleation(flag);
	}

	if (dict.CheckOption("@SurfaceGrowthModel") == true)
	{
		int flag;
		dict.ReadInt("@SurfaceGrowthModel", flag);
		this->SetSurfaceGrowth(flag);
	}

	if (dict.CheckOption("@OxidationModel") == true)
	{
		int flag;
		dict.ReadInt("@OxidationModel", flag);
		this->SetOxidation(flag);
	}

	if (dict.CheckOption("@CondensationModel") == true)
	{
		int flag;
		dict.ReadInt("@CondensationModel", flag);
		this->SetCondensation(flag);
	}

	if (dict.CheckOption("@CoagulationModel") == true)
	{
		int flag;
		dict.ReadInt("@CoagulationModel", flag);
		this->SetCoagulation(flag);
	}

	if (dict.CheckOption("@ContinuousCoagulationModel") == true)
	{
		int flag;
		dict.ReadInt("@ContinuousCoagulationModel", flag);
		this->SetCoagulationContinuous(flag);
	}

	if (dict.CheckOption("@ThermophoreticModel") == true)
	{
		int flag;
		dict.ReadInt("@ThermophoreticModel", flag);
		this->SetThermophoreticModel(flag);
	}

	if (dict.CheckOption("@RadiativeHeatTransfer") == true)
		dict.ReadBool("@RadiativeHeatTransfer", this->radiative_heat_transfer_);

	if (dict.CheckOption("@PlanckCoefficient") == true)
	{
		std::string flag;
		dict.ReadString("@PlanckCoefficient", flag);
		this->SetPlanckAbsorptionCoefficient(flag);
	}

	if (dict.CheckOption("@SchmidtNumber") == true)
	{
		double value;
		dict.ReadDouble("@SchmidtNumber", value);
		this->SetSchmidtNumber(value);
	}

	if (dict.CheckOption("@SootDensity") == true)
	{
		double value;
		std::string units;
		dict.ReadMeasure("@SootDensity", value, units);

		if (units == "kg/m3")		value = value;
		else if (units == "g/cm3")	value *= 1000.;
		else return std::unexpected("@SootDensity: allowed units kg/m3 | g/cm3");

		this->SetSootDensity(value);
	}


	if (dict.CheckOption("@SurfaceDensity") == true)
	{
		double value;
		std::string units;
		dict.ReadMeasure("@SurfaceDensity", value, units);

		if (units == "#/m2")		value = value;
		else if (units == "#/cm2")	value *= 1.e4;
		else if (units == "#/mm2")	value *= 1.e6;
		else return std::unexpected("@SurfaceDensity: allowed units #/m2 | #/cm2 | #/mm2");

		this->SetSurfaceDensity(value);
	}

	if (dict.CheckOption("@SurfaceDensityCorrectionCoefficient") == true)
	{
		bool value;
		dict.ReadBool("@SurfaceDensityCorrectionCoefficient", value);
		this->SetSurfaceDensityCorrectionCoefficient(value);
	}

	if (dict.CheckOption("@SurfaceDensityCorrectionCoefficientA1") == true)
	{
		double value;
		dict.ReadDouble("@SurfaceDensityCorrectionCoefficientA1", value);
		this->SetSurfaceDensityCorrectionCoefficientA1(value);
	}

	if (dict.CheckOption("@SurfaceDensityCorrectionCoefficientA2") == true)
	{
		double value;
		dict.ReadDouble("@SurfaceDensityCorrectionCoefficientA2", value);
		this->SetSurfaceDensityCorrectionCoefficientA2(value);
	}

	if (dict.CheckOption("@SurfaceDensityCorrectionCoefficientB1") == true)
	{
		double value;
		dict.ReadDouble("@SurfaceDensityCorrectionCoefficientB1", value);
		this->SetSurfaceDensityCorrectionCoefficientB1(value);
	}

	if (dict.CheckOption("@SurfaceDensityCorrectionCoefficientB2") == true)
	{
		double value;
		dict.ReadDouble("@SurfaceDensityCorrectionCoefficientB2", value);
		this->SetSurfaceDensityCorrectionCoefficientB2(value);
	}


	// Frequency factors
	{ 
		double value;
		std::string units;

		if ( dict.CheckOption("@A1f") == true ) 
		{
			dict.ReadMeasure("@A1f", value, units);
			if (units != "cm3,mol,s")	return std::unexpected("Allowed frequency factor units: cm3,mol,s");
			this->A1f_ = value;
		}
		if ( dict.CheckOption("@A1b") == true ) 
		{
			dict.ReadMeasure("@A1b", value, units);
			if (units != "cm3,mol,s")	return std::unexpected("Allowed frequency factor units: cm3,mol,s");
			this->A1b_ = value;
		}
		if ( dict.CheckOption("@A2f") == true ) 
		{
			dict.ReadMeasure("@A2f", value, units);
			if (units != "cm3,mol,s")	return std::unexpected("Allowed frequency factor units: cm3,mol,s");
			this->A2f_ = value;
		}
		if ( dict.CheckOption("@A2b") == true ) 
		{
			dict.ReadMeasure("@A2b", value, units);
			if (units != "cm3,mol,s")	return std::unexpected("Allowed frequency factor units: cm3,mol,s");
			this->A2b_ = value;
		}
		if ( dict.CheckOption("@A3f") == true ) 
		{
			dict.ReadMeasure("@A3f", value, units);
			if (units != "cm3,mol,s")	return std::unexpected("Allowed frequency factor units: cm3,mol,s");
			this->A3f_ = value;
		}
		if ( dict.CheckOption("@A3b") == true ) 
		{
			dict.ReadMeasure("@A3b", value, units);
			if (units != "cm3,mol,s")	return std::unexpected("Allowed frequency factor units: cm3,mol,s");				
			this->A3b_ = value;
		}
		if ( dict.CheckOption("@A4") == true ) 
		{
			dict.ReadMeasure("@A4", value, units);
			if (units != "cm3,mol,s")	return std::unexpected("Allowed frequency factor units: cm3,mol,s");
			this->A4_ = value;
		}
		if ( dict.CheckOption("@A5") == true ) 
		{
			dict.ReadMeasure("@A5", value, units);
			if (units != "cm3,mol,s")	return std::unexpected("Allowed frequency factor units: cm3,mol,s");
			this->A5_ = value;
		}
	}

	// Activation energies
	{
		double value;
		std::string units;

		if ( dict.CheckOption("@E1f") == true ) 
		{
			dict.ReadMeasure("@E1f", value, units);
			if (units != "kJ/mol")	return std::unexpected("Allowed activation energy units: kJ/mol");
			this->E1f_ = value* 1e3 / this->Rgas_;
		}
		if ( dict.CheckOption("@E1b") == true ) 
		{
			dict.ReadMeasure("@E1b", value, units);
			if (units != "kJ/mol")	return std::unexpected("Allowed activation energy units: kJ/mol");
			this->E1b_ = value* 1e3 / this->Rgas_;
		}
		if ( dict.CheckOption("@E2f") == true ) 
		{
			dict.ReadMeasure("@E2f", value, units);
			if (units != "kJ/mol")	return std::unexpected("Allowed activation energy units: kJ/mol");
			this->E2f_ = value* 1e3 / this->Rgas_;
		}
		if ( dict.CheckOption("@E2b") == true ) 
		{
			dict.ReadMeasure("@E2b", value, units);
			if (units != "kJ/mol")	return std::unexpected("Allowed activation energy units: kJ/mol");
			this->E2b_ = value* 1e3 / this->Rgas_;
		}
		if ( dict.CheckOption("@E3f") == true ) 
		{
			dict.ReadMeasure("@E3f", value, units);
			if (units != "kJ/mol")	return std::unexpected("Allowed activation energy units: kJ/mol");
			this->E3f_ = value* 1e3 / this->Rgas_;
		}
		if ( dict.CheckOption("@E3b") == true ) 
		{
			dict.ReadMeasure("@E3b", value, units);
			if (units != "kJ/mol")	return std::unexpected("Allowed activation energy units: kJ/mol");
			this->E3b_ = value* 1e3 / this->Rgas_;
		}
		if ( dict.CheckOption("@E4") == true ) 
		{
			dict.ReadMeasure("@E4", value, units);
			if (units != "kJ/mol")	return std::unexpected("Allowed activation energy units: kJ/mol");
			this->E4_ = value* 1e3 / this->Rgas_;
		}
		if ( dict.CheckOption("@E5") == true ) 
		{
			dict.ReadMeasure("@E5", value, units);
			if (units != "kJ/mol")	return std::unexpected("Allowed activation energy units: kJ/mol");
			this->E5_ = value* 1e3 / this->Rgas_;
		}
	}

	// Temperature exponents
	{
		double value;

		if ( dict.CheckOption("@n1f") == true ) 
		{
			dict.ReadDouble("@n1f", value);
			this->n1f_ = value;
		}
		if ( dict.CheckOption("@n1b") == true )
		{
			dict.ReadDouble("@n1b", value);
			this->n1b_ = value;
		}
		if ( dict.CheckOption("@n2f") == true ) 
		{
			dict.ReadDouble("@n2f", value);
			this->n2f_ = value;
		}
		if ( dict.CheckOption("@n2b") == true ) 
		{
			dict.ReadDouble("@n2b", value);
			this->n2b_ = value;
		}
		if ( dict.CheckOption("@n3f") == true ) 
		{
			dict.ReadDouble("@n3f", value);
			this->n3f_ = value;
		}
		if ( dict.CheckOption("@n3b") == true ) 
		{
			dict.ReadDouble("@n3b", value);
			this->n3b_ = value;
		}
		if ( dict.CheckOption("@n4") == true )  
		{
			dict.ReadDouble("@n4",  value);
			this->n4_  = value;
		}
		if ( dict.CheckOption("@n5") == true ) 
		{
			dict.ReadDouble("@n5",  value);
			this->n5_  = value;
		}
	}

	if (dict.CheckOption("@Efficiency6") == true)
	{
		double value;
		dict.ReadDouble("@Efficiency6", value);
		this->eff6_ = value;
	}

	if (dict.CheckOption("@StickingCoefficientModel") == true)
	{
		std::string model;
		dict.ReadString("@StickingCoefficientModel", model);
		this->SetStickingCoefficientModel(model);
	}

	if (dict.CheckOption("@StickingCoefficientConstant") == true)
	{
		double value;
		dict.ReadDouble("@StickingCoefficientConstant", value);
		this->SetStickingCoefficientConstant(value);
	}		

	if (dict.CheckOption("@DebugMode") == true)
		dict.ReadBool("@DebugMode", this->is_debug_mode_);	

    PrintSummary();
    return {};
}
#endif  // MOM_USE_DICTIONARY expected

// ============================================================================
// NDF two-node reconstruction
// ============================================================================

template <ThermoMap Thermo>
std::array<typename HMOM<Thermo>::NDFNode, 2>
HMOM<Thermo>::NumberDensityFunctionNodes() const
{
    std::array<NDFNode, 2> nodes{};

    nodes[0].number_density    = N0_;
    nodes[0].volume            = V0_;
    nodes[0].surface           = S0_;
    nodes[0].primary_diameter  = 6.*V0_/S0_;
    nodes[0].primary_particles = 1.;
    nodes[0].collision_diameter= K_collisional_*std::pow(V0_,Av_collisional_)*std::pow(S0_,As_collisional_);

    nodes[1].number_density = NL_;
    if (NL_ > kSootNumberFloor && NLVL_ > kSootVolumeFloor && NLSL_ > kSootSurfaceFloor) {
        const double VL = NLVL_/NL_;
        const double SL = NLSL_/NL_;
        nodes[1].volume            = VL;
        nodes[1].surface           = SL;
        nodes[1].primary_diameter  = 6.*VL/SL;
        nodes[1].primary_particles = std::max(1., std::pow(SL,3.)/(36.*this->pi_*VL*VL));
        nodes[1].collision_diameter= K_collisional_*std::pow(VL,Av_collisional_)*std::pow(SL,As_collisional_);
    } else {
        nodes[1] = nodes[0];
        nodes[1].number_density = NL_;
    }
    return nodes;
}

// ============================================================================
// NDF accessors
// ============================================================================

template <ThermoMap Thermo>
double HMOM<Thermo>::SootSmallParticleNumberDensity() const noexcept { return N0_; }

template <ThermoMap Thermo>
double HMOM<Thermo>::SootLargeParticleNumberDensity() const noexcept { return NL_; }

template <ThermoMap Thermo>
double HMOM<Thermo>::SootLargeParticleFraction() const noexcept
{
    const double t = N0_ + NL_;
    return (t > 0.) ? NL_/t : 0.;
}

template <ThermoMap Thermo>
double HMOM<Thermo>::SootSmallParticleFraction() const noexcept
{
    const double t = N0_ + NL_;
    return (t > 0.) ? N0_/t : 1.;
}

template <ThermoMap Thermo>
double HMOM<Thermo>::SootMeanParticleVolume() const noexcept
{
	if (!std::isfinite(M00_) || !std::isfinite(M10_) || M00_ <= 0.0 || M10_ <= 0.0)
		return V0_;

	return M10_ / M00_;   // [m3/#]
}


template <ThermoMap Thermo>
double HMOM<Thermo>::SootMeanParticleSurface() const noexcept
{
	if (!std::isfinite(M00_) || !std::isfinite(M01_) || M00_ <= 0.0 || M01_ <= 0.0)
		return S0_;

	return M01_ / M00_;   // [m2/#]
}

template <ThermoMap Thermo>
double HMOM<Thermo>::SootLargeMeanParticleVolume() const noexcept
{
    return (NL_ > kSootNumberFloor && NLVL_ > kSootVolumeFloor) ? NLVL_/NL_ : V0_;
}

template <ThermoMap Thermo>
double HMOM<Thermo>::SootLargeMeanParticleSurface() const noexcept
{
    return (NL_ > kSootNumberFloor && NLSL_ > kSootSurfaceFloor) ? NLSL_/NL_ : S0_;
}

template <ThermoMap Thermo>
double HMOM<Thermo>::SootLargePrimaryParticleDiameter() const noexcept
{
    const double VL = SootLargeMeanParticleVolume();
    const double SL = SootLargeMeanParticleSurface();
    return (SL > 0.) ? 6.*VL/SL : 0.;
}

template <ThermoMap Thermo>
double HMOM<Thermo>::SootLargeNumberOfPrimaryParticles() const noexcept
{
    const double VL = SootLargeMeanParticleVolume();
    const double SL = SootLargeMeanParticleSurface();
    if (SL <= 0. || VL <= 0.) return 1.;
    return std::max(1., std::pow(SL,3.)/(36.*this->pi_*VL*VL));
}

template <ThermoMap Thermo>
double HMOM<Thermo>::LogGeomStdDevFromMoments(double M0, double M1, double M2) const noexcept
{
    if (M0 <= 0. || M1 <= 0. || M2 <= 0.) return 0.;
    const double ratio = M0*M2/(M1*M1);
    return (ratio > 1.) ? std::sqrt(std::log(ratio)) : 0.;
}

template <ThermoMap Thermo>
double HMOM<Thermo>::SootLogGeometricStdDevPrimaryParticleDiameter() const noexcept
{
    return LogGeomStdDevFromMoments(GetMoment(0.,0.), GetMoment(-1.,1.5), GetMoment(-2.,3.));
}

template <ThermoMap Thermo>
double HMOM<Thermo>::SootLogGeometricStdDevPrimaryParticleNumber() const noexcept
{
    return LogGeomStdDevFromMoments(GetMoment(0.,0.), GetMoment(1.,0.), GetMoment(2.,0.));
}

template <ThermoMap Thermo>
double HMOM<Thermo>::SootLargeLogGeometricStdDevPrimaryParticleDiameter() const noexcept
{
	if (!HasSoot() || NL_ <= kSootNumberFloor)
		return 0.0;

	const double M0L = GetMissingMoment(0.0, 0.0);
	const double M1L = GetMissingMoment(1.0, -1.0);
	const double M2L = GetMissingMoment(2.0, -2.0);

	return LogGeomStdDevFromMoments(M0L, M1L, M2L);
}

template <ThermoMap Thermo>
double HMOM<Thermo>::SootLargeLogGeometricStdDevPrimaryParticleNumber() const noexcept
{
	if (!HasSoot() || NL_ <= kSootNumberFloor)
		return 0.0;

	const double M0L = GetMissingMoment(0.0, 0.0);
	const double M1L = GetMissingMoment(-2.0, 3.0);
	const double M2L = GetMissingMoment(-4.0, 6.0);

	return LogGeomStdDevFromMoments(M0L, M1L, M2L);
}

template <ThermoMap Thermo>
double HMOM<Thermo>::SootD63() const noexcept
{
    const double M10 = GetMoment(1.,0.);
    const double M43 = GetMoment(4./3., 0.);
    if (M10 <= 0. || !std::isfinite(M10)) return 0.;
    return 6. * std::pow(M43/M10, 1./3.);
}

// ============================================================================
// Coagulation breakdown span accessors
// ============================================================================

template <ThermoMap Thermo>
std::span<const double> HMOM<Thermo>::sources_coagulation_discrete() const noexcept
{ return { source_coagulation_discrete_.data(), 4u }; }

template <ThermoMap Thermo>
std::span<const double> HMOM<Thermo>::sources_coagulation_discrete_ss() const noexcept
{ return { source_coagulation_ss_.data(), 4u }; }

template <ThermoMap Thermo>
std::span<const double> HMOM<Thermo>::sources_coagulation_discrete_sl() const noexcept
{ return { source_coagulation_sl_.data(), 4u }; }

template <ThermoMap Thermo>
std::span<const double> HMOM<Thermo>::sources_coagulation_discrete_ll() const noexcept
{ return { source_coagulation_ll_.data(), 4u }; }

template <ThermoMap Thermo>
std::span<const double> HMOM<Thermo>::sources_coagulation_continuous() const noexcept
{ return { source_coagulation_continuous_.data(), 4u }; }

template <ThermoMap Thermo>
std::span<const double> HMOM<Thermo>::sources_coagulation_continuous_ss() const noexcept
{ return { source_coagulation_cont_ss_.data(), 4u }; }

template <ThermoMap Thermo>
std::span<const double> HMOM<Thermo>::sources_coagulation_continuous_sl() const noexcept
{ return { source_coagulation_cont_sl_.data(), 4u }; }

template <ThermoMap Thermo>
std::span<const double> HMOM<Thermo>::sources_coagulation_continuous_ll() const noexcept
{ return { source_coagulation_cont_ll_.data(), 4u }; }

// ============================================================================
// HACA parameter setters
// ============================================================================

template <ThermoMap Thermo> void HMOM<Thermo>::SetA1f(double v) noexcept { A1f_ = v; }
template <ThermoMap Thermo> void HMOM<Thermo>::SetA1b(double v) noexcept { A1b_ = v; }
template <ThermoMap Thermo> void HMOM<Thermo>::SetA2f(double v) noexcept { A2f_ = v; }
template <ThermoMap Thermo> void HMOM<Thermo>::SetA2b(double v) noexcept { A2b_ = v; }
template <ThermoMap Thermo> void HMOM<Thermo>::SetA3f(double v) noexcept { A3f_ = v; }
template <ThermoMap Thermo> void HMOM<Thermo>::SetA3b(double v) noexcept { A3b_ = v; }
template <ThermoMap Thermo> void HMOM<Thermo>::SetA4 (double v) noexcept { A4_  = v; }
template <ThermoMap Thermo> void HMOM<Thermo>::SetA5 (double v) noexcept { A5_  = v; }

template <ThermoMap Thermo> void HMOM<Thermo>::SetE1f(double kJ) noexcept { E1f_ = kJ*1e3/R_J_mol_HMOM; }
template <ThermoMap Thermo> void HMOM<Thermo>::SetE1b(double kJ) noexcept { E1b_ = kJ*1e3/R_J_mol_HMOM; }
template <ThermoMap Thermo> void HMOM<Thermo>::SetE2f(double kJ) noexcept { E2f_ = kJ*1e3/R_J_mol_HMOM; }
template <ThermoMap Thermo> void HMOM<Thermo>::SetE2b(double kJ) noexcept { E2b_ = kJ*1e3/R_J_mol_HMOM; }
template <ThermoMap Thermo> void HMOM<Thermo>::SetE3f(double kJ) noexcept { E3f_ = kJ*1e3/R_J_mol_HMOM; }
template <ThermoMap Thermo> void HMOM<Thermo>::SetE3b(double kJ) noexcept { E3b_ = kJ*1e3/R_J_mol_HMOM; }
template <ThermoMap Thermo> void HMOM<Thermo>::SetE4 (double kJ) noexcept { E4_  = kJ*1e3/R_J_mol_HMOM; }
template <ThermoMap Thermo> void HMOM<Thermo>::SetE5 (double kJ) noexcept { E5_  = kJ*1e3/R_J_mol_HMOM; }

template <ThermoMap Thermo> void HMOM<Thermo>::Setn1f(double v) noexcept { n1f_ = v; }
template <ThermoMap Thermo> void HMOM<Thermo>::Setn1b(double v) noexcept { n1b_ = v; }
template <ThermoMap Thermo> void HMOM<Thermo>::Setn2f(double v) noexcept { n2f_ = v; }
template <ThermoMap Thermo> void HMOM<Thermo>::Setn2b(double v) noexcept { n2b_ = v; }
template <ThermoMap Thermo> void HMOM<Thermo>::Setn3f(double v) noexcept { n3f_ = v; }
template <ThermoMap Thermo> void HMOM<Thermo>::Setn3b(double v) noexcept { n3b_ = v; }
template <ThermoMap Thermo> void HMOM<Thermo>::Setn4 (double v) noexcept { n4_  = v; }
template <ThermoMap Thermo> void HMOM<Thermo>::Setn5 (double v) noexcept { n5_  = v; }

// ============================================================================
// PrintSummary
// ============================================================================

template <ThermoMap Thermo>
void HMOM<Thermo>::PrintSummary() const
{
    std::cout << "\n-------------------------------------------------------------------\n"
              << " HMOM — Hybrid Method of Moments\n"
              << "-------------------------------------------------------------------\n"
              << " PAH species   : " << pah_species_ << "\n"
              << " V0 [m3]       : " << V0_ << "\n"
              << " S0 [m2]       : " << S0_ << "\n"
              << " HACA kinetics (A [cm3/mol/s], E [kJ/mol]):\n"
              << "   R1f: " << A1f_ << " " << n1f_ << " " << E1f_*R_J_mol_HMOM/1000. << "\n"
              << "   R1b: " << A1b_ << " " << n1b_ << " " << E1b_*R_J_mol_HMOM/1000. << "\n"
              << "   R2f: " << A2f_ << " " << n2f_ << " " << E2f_*R_J_mol_HMOM/1000. << "\n"
              << "   R2b: " << A2b_ << " " << n2b_ << " " << E2b_*R_J_mol_HMOM/1000. << "\n"
              << "   R3f: " << A3f_ << " " << n3f_ << " " << E3f_*R_J_mol_HMOM/1000. << "\n"
              << "   R3b: " << A3b_ << " " << n3b_ << " " << E3b_*R_J_mol_HMOM/1000. << "\n"
              << "   R4 : " << A4_  << " " << n4_  << " " << E4_ *R_J_mol_HMOM/1000. << "\n"
              << "   R5 : " << A5_  << " " << n5_  << " " << E5_ *R_J_mol_HMOM/1000. << "\n"
              << "   eff6: " << eff6_ << "\n"
              << " Surface density [#/m2]: " << surface_density_ << "\n"
              << " Models: nuc=" << nucleation_model_
              << " cond=" << condensation_model_
              << " sg=" << surface_growth_model_
              << " ox=" << oxidation_model_
              << " coag=" << coagulation_model_
              << " coag_cont=" << coagulation_continuous_model_ << "\n"
              << " Sc=" << this->schmidt_number_
              << "  sticking=" << sticking_coeff_constant_ << "\n"
              << "-------------------------------------------------------------------\n\n";
}

} // namespace MOM
