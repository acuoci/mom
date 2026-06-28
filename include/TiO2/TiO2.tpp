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
#include <span>
#include <numeric>
#if defined(MOM_USE_DICTIONARY)
#include "TiO2_Grammar.h"
#endif

namespace MOM {

// ============================================================================
// Constructor
// ============================================================================

template <ThermoMap Thermo>
TiO2<Thermo>::TiO2(const Thermo& thermo)
    : thermo_(thermo)
{
    // ── CRTP base state ───────────────────────────────────────────────────
    this->is_active_            = true;
    this->gas_consumption_      = false;
    this->rho_particle_         = rhoTiO2_;
    this->planck_model_         = PlanckCoeffModel::None;   // TiO2 is dielectric
    this->SetThermophoreticModel(1);
    this->schmidt_number_       = 50.;
    this->dummy_species_        = "none";
    this->dummy_index_          = -1;
    this->dummy_species_closure_= false;

    // ── Nucleation parameters ─────────────────────────────────────────────
    nTiO2_min_   = 2;
    n0_          = 5;
    epsilon_nuc_ = 2.5;

    // ── Coagulation / condensation ────────────────────────────────────────
    epsilon_coag_ = 2.2;
    epsilon_cond_ = 1.3;

    // ── Model flags ───────────────────────────────────────────────────────
    nucleation_variant_ = NucleationVariant::Binary;
    coagulation_model_  = 1;
    condensation_model_ = 1;
    sintering_model_    = 1;

    // ── Sintering kinetics ────────────────────────────────────────────────
    As_ = 7.44e16;
    ns_ = 1.0;
    // Ts_ is already initialised to -31000. by the class default; do NOT override.

    // ── Numerical floors ──────────────────────────────────────────────────
    N_min_  = 1.e3;
    fv_min_ = 1.e-16;

    // ── Precursor (none initially) ────────────────────────────────────────
    precursor_species_ = "none";
    precursor_index_   = -1;
    nti_precursor_ = nh_precursor_ = no_precursor_ = nc_precursor_ = 0.;
    Y_precursor_   = 0.;
    c_precursor_   = 0.;
    W_precursor_   = 0.;
    m_precursor_   = 0.;
    v_precursor_   = 0.;
    d_precursor_   = 0.;
    vprec_         = 0.;
    dprec_         = 0.;

    // ── Gas consumption stoichiometry indices ─────────────────────────────
    H2O_index_ = CO2_index_ = O2_index_ = -1;
    W_H2O_ = W_CO2_ = W_O2_ = 0.;
    nu_H2O_from_prec_ = nu_CO2_from_prec_ = nu_O2_from_prec_ = 0.;

    // ── Sintering regularisation ──────────────────────────────────────────
    sintering_dp_min_              = 2.e-9;
    sintering_activation_np_       = 1.05;
    sintering_activation_width_np_ = 0.05;
    sintering_tau_min_             = 1.e-10;
    sintering_k_max_               = 1.e6;
    sintering_relative_tolerance_  = 1.e-3;
    sintering_tau_qss_             = 1.e-6;
    is_sintering_deferred_         = false;

	//
	is_debug_mode_ = false;

    // ── Geometry, floors, and initial-moments cache ───────────────────────
    Precalculations();
    MemoryAllocation();
}

// ============================================================================
// MemoryAllocation
// ============================================================================

template <ThermoMap Thermo>
void TiO2<Thermo>::MemoryAllocation()
{
    this->ZeroSources();               // zeros source_all_, omega_gas_ (base class)
    source_nucleation_.setZero();     // owned by TiO2 — zeroed explicitly
    source_coagulation_.setZero();
    source_condensation_.setZero();
    source_sintering_.setZero();

    this->omega_gas_ = Eigen::VectorXd::Zero(
        static_cast<Eigen::Index>(thermo_.NumberOfSpecies()));

    // Initial moments (IC at numerical floor values)
    const double N0  = std::max(N_min_, 0.0);
    const double fv0 = N0 * v0_;
    const double Y0  = rhoTiO2_ / 1.0 * fv0;   // reference gas density 1 kg/m3
    const double S0  = N0 * s0_;

    initial_moments_cache_(0) = Y0;
    initial_moments_cache_(1) = N0 / N0_scaling_;
    initial_moments_cache_(2) = S0;
}

// ============================================================================
// Precalculations  (call whenever nTiO2_min_ or vprec_/dprec_ change)
// ============================================================================

template <ThermoMap Thermo>
void TiO2<Thermo>::Precalculations()
{
    const double v_TiO2 = m_TiO2_ / rhoTiO2_;

    v0_ = static_cast<double>(nTiO2_min_) * v_TiO2;
    d0_ = std::pow(6. * v0_ / this->pi_, 1. / 3.);
    s0_ = this->pi_ * d0_ * d0_;

    v_min_ = v0_;
    S_min_ = N_min_ * s0_;

    // Kernel prefactor for nucleation (temperature-independent part)
    if (vprec_ > 0. && dprec_ > 0.)
        alpha_nuc_ = epsilon_nuc_
            * std::sqrt(this->pi_ * this->kB_ / (2. * rhoTiO2_))
            * std::sqrt(2. / vprec_)
            * std::pow(2. * dprec_, 2.);
    else
        alpha_nuc_ = 0.;

    alpha_coag_ = epsilon_coag_
        * std::sqrt(this->pi_ * this->kB_ / (2. * rhoTiO2_));

    alpha_cond_ = epsilon_cond_
        * std::sqrt(this->pi_ * this->kB_ / (2. * rhoTiO2_));
}

// ============================================================================
// SetStatus  — injects thermodynamic state from the CFD solver
// ============================================================================

template <ThermoMap Thermo>
void TiO2<Thermo>::SetStatus(double T, double P_Pa, const double* Y) noexcept
{
    this->T_    = T;
    this->P_Pa_ = P_Pa;

    // Mixture molecular weight  (1/MW = sum Yi/MWi)
    {
        double inv = 0.;
        for (unsigned i = 0; i < thermo_.NumberOfSpecies(); ++i)
            inv += Y[i] / thermo_.MolecularWeight(i);
        this->MW_ = (inv > 1.e-300) ? 1. / inv : 1.;
    }

    const double cTot = this->P_Pa_ / (this->Rgas_ * this->T_);   // [kmol/m3]
    this->rho_ = cTot * this->MW_;                                  // [kg/m3]

    // Precursor mass fraction and concentration
    Y_precursor_ = 0.;
    c_precursor_ = 0.;
    if (precursor_index_ >= 0) {
        Y_precursor_ = std::max(Y[precursor_index_], 0.);
        c_precursor_ = cTot * Y_precursor_ * this->MW_
                       / thermo_.MolecularWeight(static_cast<unsigned>(precursor_index_));
    }
}

// ============================================================================
// SetMoments
// ============================================================================

template <ThermoMap Thermo>
void TiO2<Thermo>::SetMoments(std::span<const double> m) noexcept
{
    SetMoments(m[0], m[1], m[2]);
}

template <ThermoMap Thermo>
void TiO2<Thermo>::SetMoments(double YTiO2, double NTiO2N, double STiO2) noexcept
{
    YTiO2_  = std::max(YTiO2,  0.);
    NTiO2N_ = std::max(NTiO2N, 0.);
    STiO2_  = std::max(STiO2,  0.);
}

// ============================================================================
// SetPrecursor
// ============================================================================

template <ThermoMap Thermo>
void TiO2<Thermo>::SetPrecursor(std::string_view name)
{
    precursor_species_ = std::string(name);

    if (precursor_species_ == "none") {
        precursor_index_  = -1;
        nti_precursor_    = 0.;
        nh_precursor_     = 0.;
        no_precursor_     = 0.;
        nc_precursor_     = 0.;
        Y_precursor_      = 0.;
        c_precursor_      = 0.;
        W_precursor_      = 0.;
        m_precursor_      = 0.;
        v_precursor_      = 0.;
        d_precursor_      = 0.;
        vprec_            = 0.;
        dprec_            = 0.;
        Precalculations();
        return;
    }

    precursor_index_ = thermo_.IndexOfSpecies(name);
    if (precursor_index_ < 0)
        throw std::runtime_error("[TiO2] Precursor species not found in mechanism: "
                                 + precursor_species_);

    const unsigned ui = static_cast<unsigned>(precursor_index_);

    W_precursor_  = thermo_.MolecularWeight(ui);
    m_precursor_  = W_precursor_ / this->Nav_kmol_;
    nti_precursor_= thermo_.NumberOfTitaniumAtoms(ui);
    nh_precursor_ = thermo_.NumberOfHydrogenAtoms(ui);
    no_precursor_ = thermo_.NumberOfOxygenAtoms(ui);
    nc_precursor_ = thermo_.NumberOfCarbonAtoms(ui);

    if (nti_precursor_ <= 0.)
        throw std::runtime_error("[TiO2] Precursor species has no Ti atoms: "
                                 + precursor_species_);

    v_precursor_ = m_precursor_ / rhoTiO2_;
    d_precursor_ = std::pow(6. * v_precursor_ / this->pi_, 1. / 3.);

    // Effective collision geometry used by nucleation/condensation kernels:
    // volume contribution = one TiO2 unit per Ti atom in the precursor molecule
    const double v_TiO2_local = m_TiO2_ / rhoTiO2_;
    vprec_ = nti_precursor_ * v_TiO2_local;
    dprec_ = d_precursor_;

    SetupGasConsumptionStoichiometry();
    Precalculations();
}

// ============================================================================
// SetupGasConsumptionStoichiometry
// ============================================================================
//
// Atom balance for precursor Ti_a C_b H_c O_d decomposing to TiO2(s):
//
//   Ti_a C_b H_c O_d + x O2  →  a TiO2(s) + b CO2 + (c/2) H2O
//
//   O balance: d + 2x = 2a + 2b + c/2
//              x = (2a + 2b + c/2 - d) / 2
//
// nu_O2_from_prec_ is positive when O2 is produced, negative when consumed.
// (x > 0 means O2 is consumed, so we store -x)

template <ThermoMap Thermo>
void TiO2<Thermo>::SetupGasConsumptionStoichiometry()
{
    nu_H2O_from_prec_ = 0.;
    nu_CO2_from_prec_ = 0.;
    nu_O2_from_prec_  = 0.;
    H2O_index_ = CO2_index_ = O2_index_ = -1;
    W_H2O_ = W_CO2_ = W_O2_ = 0.;

    if (precursor_index_ < 0 || nti_precursor_ <= 0.) return;

    const double a = nti_precursor_;
    const double b = nc_precursor_;
    const double c = nh_precursor_;
    const double d = no_precursor_;

    // Stoichiometric coefficients per precursor molecule
    nu_H2O_from_prec_ = 0.5 * c;
    nu_CO2_from_prec_ = b;

    // x = moles of O2 consumed per precursor molecule
    // Positive x = O2 consumed; store as negative convention (consumed < 0)
    const double x_O2 = (2. * a + 2. * b + 0.5 * c - d) / 2.;
    nu_O2_from_prec_  = -x_O2;   // negative = consumed, positive = produced

    // Look up gas species indices and molecular weights
    H2O_index_ = thermo_.IndexOfSpecies("H2O");
    if (H2O_index_ >= 0)
        W_H2O_ = thermo_.MolecularWeight(static_cast<unsigned>(H2O_index_));

    CO2_index_ = thermo_.IndexOfSpecies("CO2");
    if (CO2_index_ >= 0)
        W_CO2_ = thermo_.MolecularWeight(static_cast<unsigned>(CO2_index_));

    O2_index_ = thermo_.IndexOfSpecies("O2");
    if (O2_index_ >= 0)
        W_O2_ = thermo_.MolecularWeight(static_cast<unsigned>(O2_index_));
}

// ============================================================================
// SetGasClosureDummySpecies
// ============================================================================

template <ThermoMap Thermo>
void TiO2<Thermo>::SetGasClosureDummySpecies(std::string_view name)
{
    this->dummy_species_ = std::string(name);

    if (this->dummy_species_ == "none") {
        this->dummy_index_          = -1;
        this->dummy_species_closure_= false;
        return;
    }

    this->dummy_index_ = thermo_.IndexOfSpecies(this->dummy_species_);
    if (this->dummy_index_ < 0)
        throw std::runtime_error("[TiO2] Dummy species not found in mechanism: "
                                 + this->dummy_species_);

    if (this->dummy_index_ == precursor_index_)
        throw std::runtime_error("[TiO2] Dummy species cannot be the precursor species.");

    this->dummy_species_closure_ = true;
}

// ============================================================================
// Properties  — aggregate particle properties from the three moments
// ============================================================================

template <ThermoMap Thermo>
void TiO2<Thermo>::Properties(double& fv,   double& dp, double& dc,
                                double& da,   double& np, double& ss,
                                double& vs,   double& ssph,
                                double& tauS) const noexcept
{
    const double Y = std::max(YTiO2_,              0.);
    const double N = std::max(NTiO2N_ * N0_scaling_, 0.);
    const double S = std::max(STiO2_,              0.);

    // Volume fraction [-]
    fv = this->rho_ / rhoTiO2_ * Y;

    // Number density used for property calculation (regularized)
    const double NStar = std::max(N, N_min_);
    const double fvStar= std::max(fv, fv_min_);

    // Mean particle volume [m3]
    vs = std::max(fvStar / NStar, v_min_);

    // Spherical surface area for a particle of volume vs [m2]
    ssph = std::pow(36. * this->pi_, 1. / 3.) * std::pow(vs, 2. / 3.);

    // Specific surface per particle [m2]
    const double SStar = std::max(S, S_min_);
    ss = std::max(SStar / NStar, ssph);

    // Primary particle diameter [m]
    dp = 6. * vs / ss;
    dp = std::max(dp, d0_);

    // Number of primary particles per aggregate [-]
    np = std::max(std::pow(ss, 3.) / std::pow(vs, 2.) / (36. * this->pi_), 1.);

    // Collision (aggregate) diameter — same as primary for TiO2 (Df not used here)
    dc = dp * std::pow(np, 1. / 3.);   // Df = 3 (compact aggregates)

    // Mobility/aggregate diameter [m]
    da = dc;

    // Sintering time scale [s] — uses new sign convention: exp(-Ts_/T) = exp(31000/T)
    tauS = As_ * std::pow(this->T_, ns_) * std::pow(dp, 4.)
           * std::exp(-Ts_ / this->T_);
}

// ============================================================================
// VolumeFraction, MassFraction, ParticleNumberDensity, SpecificSurface
// ============================================================================

template <ThermoMap Thermo>
double TiO2<Thermo>::VolumeFraction() const noexcept
{
    return this->rho_ / rhoTiO2_ * std::max(YTiO2_, 0.);
}

template <ThermoMap Thermo>
double TiO2<Thermo>::MassFraction() const noexcept
{
    return std::max(YTiO2_, 0.);
}

template <ThermoMap Thermo>
double TiO2<Thermo>::ParticleNumberDensity() const noexcept
{
    return std::max(NTiO2N_ * N0_scaling_, 0.);
}

template <ThermoMap Thermo>
double TiO2<Thermo>::SpecificSurface() const noexcept
{
    return std::max(STiO2_, 0.);
}

// ============================================================================
// ParticleDiameter, CollisionDiameter, AggregateDiameter,
// NumberOfPrimaryParticles
// ============================================================================

template <ThermoMap Thermo>
double TiO2<Thermo>::ParticleDiameter() const noexcept
{
    double fv, dp, dc, da, np, ss, vs, ssph, tauS;
    Properties(fv, dp, dc, da, np, ss, vs, ssph, tauS);
    return dp;
}

template <ThermoMap Thermo>
double TiO2<Thermo>::CollisionDiameter() const noexcept
{
    double fv, dp, dc, da, np, ss, vs, ssph, tauS;
    Properties(fv, dp, dc, da, np, ss, vs, ssph, tauS);
    return dc;
}

template <ThermoMap Thermo>
double TiO2<Thermo>::AggregateDiameter() const noexcept
{
    double fv, dp, dc, da, np, ss, vs, ssph, tauS;
    Properties(fv, dp, dc, da, np, ss, vs, ssph, tauS);
    return da;
}

template <ThermoMap Thermo>
double TiO2<Thermo>::NumberOfPrimaryParticles() const noexcept
{
    double fv, dp, dc, da, np, ss, vs, ssph, tauS;
    Properties(fv, dp, dc, da, np, ss, vs, ssph, tauS);
    return np;
}

// ============================================================================
// DiffusionCoefficient  — Cunningham-corrected Brownian + Schmidt fallback
// ============================================================================

template <ThermoMap Thermo>
double TiO2<Thermo>::DiffusionCoefficient() const noexcept
{
    double fv, dp, dc, da, np, ss, vs, ssph, tauS;
    Properties(fv, dp, dc, da, np, ss, vs, ssph, tauS);

    const double dcSafe   = std::max(dc, d0_);
    const double mGas     = this->rho_ * this->kB_ * this->T_ / this->P_Pa_;
    const double lambdaGas= this->mu_ / this->rho_
                            * std::sqrt(this->pi_ * mGas / (2. * this->kB_ * this->T_));
    const double Cu       = 1. + 2.154 * lambdaGas / dcSafe;
    const double D        = this->kB_ * this->T_ * Cu
                            / (3. * this->pi_ * this->mu_ * dcSafe);
    const double GammaBrownian = this->rho_ * D;
    const double GammaSc       = this->mu_ / this->schmidt_number_;
    return std::max(GammaBrownian, GammaSc);
}

// ============================================================================
// NucleationParticleVolume
// ============================================================================

template <ThermoMap Thermo>
double TiO2<Thermo>::NucleationParticleVolume() const noexcept
{
    const double v_TiO2_local = m_TiO2_ / rhoTiO2_;

    if (nucleation_variant_ == NucleationVariant::Binary) {
        if (vprec_ > 0.) return 2. * vprec_;
        return 2. * v_TiO2_local;
    }
    if (nucleation_variant_ == NucleationVariant::FixedCluster)
        return static_cast<double>(n0_) * v_TiO2_local;

    return v0_;
}

// ============================================================================
// NucleationSourceTerms  — dispatcher
// ============================================================================

template <ThermoMap Thermo>
void TiO2<Thermo>::NucleationSourceTerms()
{
    if (nucleation_variant_ == NucleationVariant::Binary)
        NucleationSourceTerms_Binary();
    else if (nucleation_variant_ == NucleationVariant::FixedCluster)
        NucleationSourceTerms_FixedCluster();
}

// ============================================================================
// NucleationSourceTerms_Binary
// ============================================================================

template <ThermoMap Thermo>
void TiO2<Thermo>::NucleationSourceTerms_Binary()
{
    if (c_precursor_ <= 0. || vprec_ <= 0. || dprec_ <= 0.) return;

    // Precursor number density [#/m3]
    const double Nprec = c_precursor_ * this->Nav_kmol_;

    // Free-molecular binary collision kernel [m3/s]
    const double beta_nuc = epsilon_nuc_
        * std::sqrt(this->pi_ * this->kB_ * this->T_ / (2. * rhoTiO2_))
        * std::sqrt(1. / vprec_ + 1. / vprec_)
        * std::pow(2. * dprec_, 2.);

    // Rate of nucleation events [#/m3/s]
    const double eventRate = 0.5 * beta_nuc * Nprec * Nprec;

    // Volume source: each event creates one dimer of volume 2*vprec_
    const double source_fv = 2. * vprec_ * eventRate;

    const double source_N  = eventRate;

    // Surface source: dimer treated as sphere of volume 2*vprec_
    const double source_S  = std::pow(18. * this->pi_, 1. / 3.)
                             * std::pow(vprec_, 2. / 3.)
                             * beta_nuc * Nprec * Nprec;

    this->source_nucleation_(0) = rhoTiO2_ / this->rho_ * source_fv;
    this->source_nucleation_(1) = source_N / N0_scaling_;
    this->source_nucleation_(2) = source_S;
}

// ============================================================================
// NucleationSourceTerms_FixedCluster
// ============================================================================

template <ThermoMap Thermo>
void TiO2<Thermo>::NucleationSourceTerms_FixedCluster()
{
    if (c_precursor_ <= 0.) return;

    const double v_TiO2_local = m_TiO2_ / rhoTiO2_;
    const double v_cluster    = static_cast<double>(n0_) * v_TiO2_local;
    const double d_cluster    = std::pow(6. * v_cluster / this->pi_, 1. / 3.);

    // Precursor number density [#/m3]
    const double Nprec = c_precursor_ * this->Nav_kmol_;

    // Free-molecular collision kernel for a cluster colliding with itself
    const double beta_nuc = epsilon_nuc_
        * std::sqrt(this->pi_ * this->kB_ * this->T_ / (2. * rhoTiO2_))
        * std::sqrt(2. / v_cluster)
        * std::pow(2. * d_cluster, 2.);

    const double eventRate = 0.5 * beta_nuc * Nprec * Nprec;

    const double source_fv = 2. * v_cluster * eventRate;
    const double source_N  = eventRate;
    const double source_S  = std::pow(18. * this->pi_, 1. / 3.)
                             * std::pow(v_cluster, 2. / 3.)
                             * beta_nuc * Nprec * Nprec;

    this->source_nucleation_(0) = rhoTiO2_ / this->rho_ * source_fv;
    this->source_nucleation_(1) = source_N / N0_scaling_;
    this->source_nucleation_(2) = source_S;
}

// ============================================================================
// CoagulationSourceTerms
// ============================================================================

template <ThermoMap Thermo>
void TiO2<Thermo>::CoagulationSourceTerms()
{
    const double N = NTiO2N_ * N0_scaling_;
    if (N <= N_min_) return;

    double fv, dp, dc, da, np, ss, vs, ssph, tauS;
    Properties(fv, dp, dc, da, np, ss, vs, ssph, tauS);

    if (vs <= 0. || dc <= 0.) return;

    const double dcSafe = std::max(dc, d0_);

    // Free-molecular kernel [m3/s]
    const double beta_fm = epsilon_coag_
        * std::sqrt(this->pi_ * this->kB_ * this->T_ / (2. * rhoTiO2_))
        * std::sqrt(2. / vs)
        * std::pow(2. * dcSafe, 2.);

    // Continuum kernel with Cunningham slip correction [m3/s]
    const double mGas     = this->rho_ * this->kB_ * this->T_ / this->P_Pa_;
    const double lambdaGas= this->mu_ / this->rho_
                            * std::sqrt(this->pi_ * mGas / (2. * this->kB_ * this->T_));
    const double Cu       = 1. + 2.154 * lambdaGas / dcSafe;
    const double beta_cont= 8. * this->kB_ * this->T_ / (3. * this->mu_) * Cu * dcSafe;

    // Transition-regime kernel: harmonic mean, scaled by 1.82
    const double beta_m   = std::max(beta_fm, beta_cont);
    const double beta_coag= 1.82 * beta_m;

    // Number density source [#/m3/s]
    const double OmegaN   = -0.5 * beta_coag * N * N;

    this->source_coagulation_(0) = 0.;
    this->source_coagulation_(1) = OmegaN / N0_scaling_;
    this->source_coagulation_(2) = 0.;
}

// ============================================================================
// CondensationSourceTerms
// ============================================================================

template <ThermoMap Thermo>
void TiO2<Thermo>::CondensationSourceTerms()
{
    if (c_precursor_ <= 0. || vprec_ <= 0. || dprec_ <= 0.) return;

    const double N = NTiO2N_ * N0_scaling_;
    if (N <= N_min_) return;

    double fv, dp, dc, da, np, ss, vs, ssph, tauS;
    Properties(fv, dp, dc, da, np, ss, vs, ssph, tauS);

    if (vs <= 0. || dc <= 0.) return;

    // Precursor number density [#/m3]
    const double Nprec = c_precursor_ * this->Nav_kmol_;

    // Free-molecular condensation kernel [m3/s]: precursor + particle
    const double beta_cond = epsilon_cond_
        * std::sqrt(this->pi_ * this->kB_ * this->T_ / (2. * rhoTiO2_))
        * std::sqrt(1. / vprec_ + 1. / vs)
        * std::pow(dprec_ + dc, 2.);

    const double BetaNprecN = beta_cond * Nprec * N;

    // Surface area change per condensation event
    // (precursor adds vprec_ volume; surface treated as spherical increment)
    const double deltas = (2. / 3.) * (vprec_ / vs) * ss;

    this->source_condensation_(0) = rhoTiO2_ / this->rho_ * vprec_ * BetaNprecN;
    this->source_condensation_(1) = 0.;
    this->source_condensation_(2) = deltas * BetaNprecN;
}

// ============================================================================
// SinteringSourceTerms
// ============================================================================

template <ThermoMap Thermo>
void TiO2<Thermo>::SinteringSourceTerms()
{
    const double N = NTiO2N_ * N0_scaling_;
    const double S = STiO2_;

    if (N <= N_min_ || S <= 0.) return;

    double fv, dp, dc, da, np, ss, vs, ssph, tauS;
    Properties(fv, dp, dc, da, np, ss, vs, ssph, tauS);

    // Spherical surface area for the current mean volume
    const double S_sphere = N * ssph;

    // Sintering is only active when surface area exceeds spherical limit
    if (S <= S_sphere) return;

    // Primary particle diameter guard
    if (dp < sintering_dp_min_) return;

    // Smooth activation based on number of primary particles per aggregate
    double activation = 1.;
    if (sintering_activation_np_ > 1.) {
        const double x = (np - sintering_activation_np_)
                         / std::max(sintering_activation_width_np_, 1.e-300);
        activation = 0.5 * (1. + std::tanh(x));
    }
    if (activation <= 0.) return;

    // Physical sintering time scale [s]  (exp(-Ts_/T) = exp(31000/T))
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

template <ThermoMap Thermo>
double TiO2<Thermo>::SinteringDeferredUpdate(double dt_ode)
{
    if (sintering_model_ == 0) return STiO2_;

    const double N = NTiO2N_ * N0_scaling_;
    if (N <= N_min_) return STiO2_;

    double fv, dp, dc, da, np, ss, vs, ssph, tauS;
    Properties(fv, dp, dc, da, np, ss, vs, ssph, tauS);

    const double S_sphere = N * ssph;
    const double S        = STiO2_;

    if (S <= S_sphere) return S;
    if (dp < sintering_dp_min_) return S;

    // Smooth activation
    double activation = 1.;
    if (sintering_activation_np_ > 1.) {
        const double x = (np - sintering_activation_np_)
                         / std::max(sintering_activation_width_np_, 1.e-300);
        activation = 0.5 * (1. + std::tanh(x));
    }
    if (activation <= 0.) return S;

    // exp(-Ts_/T) = exp(31000/T) because Ts_ = -31000
    const double tauS_phys = std::max(tauS, sintering_tau_min_);
    const double tauSeff   = tauS_phys / activation;

    // Analytical solution: S(t) = S_sphere + (S0 - S_sphere)*exp(-dt/tau)
    const double S_new = S_sphere + (S - S_sphere) * std::exp(-dt_ode / tauSeff);
    return std::max(S_new, S_sphere);
}

// ============================================================================
// CalculateSourceMoments  — master entry point
// ============================================================================

template <ThermoMap Thermo>
void TiO2<Thermo>::CalculateSourceMoments() noexcept
{
    this->ZeroSources();               // zeros source_all_, omega_gas_ (base class)
    source_nucleation_.setZero();     // owned by TiO2 — must be zeroed before early return
    source_coagulation_.setZero();
    source_condensation_.setZero();
    source_sintering_.setZero();

    if (!this->is_active_) return;

    if (nucleation_variant_ != NucleationVariant::Off)
        NucleationSourceTerms();

    if (coagulation_model_ > 0)
        CoagulationSourceTerms();

    if (condensation_model_ > 0)
        CondensationSourceTerms();

    if (sintering_model_ > 0)
        SinteringSourceTerms();

    // Sanitise and accumulate total source
    for (unsigned i = 0; i < 3u; ++i) {
        if (!std::isfinite(this->source_nucleation_(i)))
            this->source_nucleation_(i) = 0.;
        if (!std::isfinite(this->source_coagulation_(i)))
            this->source_coagulation_(i) = 0.;
        if (!std::isfinite(this->source_condensation_(i)))
            this->source_condensation_(i) = 0.;
        if (!std::isfinite(this->source_sintering_(i)))
            this->source_sintering_(i) = 0.;

        this->source_all_(i) = this->source_nucleation_(i)
                              + this->source_coagulation_(i)
                              + this->source_condensation_(i);

        if (!is_sintering_deferred_)
            this->source_all_(i) += this->source_sintering_(i);
    }

    if (this->gas_consumption_)
        CalculateOmegaGas_internal();
}

// ============================================================================
// CalculateOmegaGas  — public interface
// ============================================================================

template <ThermoMap Thermo>
void TiO2<Thermo>::CalculateOmegaGas() noexcept
{
    CalculateOmegaGas_internal();
}

// ============================================================================
// CalculateOmegaGas_internal  — gas-phase consumption [kg/m3/s]
// ============================================================================

template <ThermoMap Thermo>
void TiO2<Thermo>::CalculateOmegaGas_internal() noexcept
{
    this->omega_gas_.setZero();

    if (!this->gas_consumption_) return;
    if (precursor_index_ < 0)   return;

    // nti_precursor_ must be positive — validated during SetPrecursor() setup.
    assert(nti_precursor_ > 0. &&
           "[TiO2] CalculateOmegaGas_internal: precursor has zero Ti atoms.");

    // Total TiO2 mass deposition rate from nucleation + condensation [kg/m3/s]
    const double omegaTiO2 = this->rho_
        * (this->source_nucleation_(0) + this->source_condensation_(0));

    if (omegaTiO2 <= 0.) return;

    // Molar rate of TiO2 deposition [kmol/m3/s]
    const double RTiO2 = omegaTiO2 / W_TiO2_;

    // Molar rate of precursor consumed [kmol/m3/s]
    // Each precursor molecule yields nti_precursor_ TiO2 formula units
    const double Rprec = RTiO2 / nti_precursor_;

    // Precursor consumption
    this->omega_gas_[precursor_index_] -= Rprec * W_precursor_;

    // H2O production
    if (nu_H2O_from_prec_ > 0. && H2O_index_ >= 0)
        this->omega_gas_[H2O_index_] += nu_H2O_from_prec_ * Rprec * W_H2O_;

    // CO2 production
    if (nu_CO2_from_prec_ > 0. && CO2_index_ >= 0)
        this->omega_gas_[CO2_index_] += nu_CO2_from_prec_ * Rprec * W_CO2_;

    // O2 production/consumption (nu_O2_from_prec_ negative = consumed)
    if (std::fabs(nu_O2_from_prec_) > 1.e-14 && O2_index_ >= 0)
        this->omega_gas_[O2_index_] += nu_O2_from_prec_ * Rprec * W_O2_;

    // Dummy species closure: adjust to enforce sum(omega_gas) = 0
    if (this->dummy_species_closure_ && this->dummy_index_ >= 0) {
        const int nsp = static_cast<int>(thermo_.NumberOfSpecies());
        double sum = 0.;
        for (int i = 0; i < nsp; ++i)
            if (i != this->dummy_index_)
                sum += this->omega_gas_[i];
        this->omega_gas_[this->dummy_index_] = -sum;
    }
}

// ============================================================================
// PrintSummary
// ============================================================================

template <ThermoMap Thermo>
void TiO2<Thermo>::PrintSummary() const
{
    std::cout
        << "\n"
        << "------------------------------------------------------------------------------------------\n"
        << "                         TiO2 Nanoparticle Model Summary\n"
        << "------------------------------------------------------------------------------------------\n"
        << " * TiO2 density (kg/m3):         " << rhoTiO2_          << "\n"
        << " * TiO2 mol. weight (kg/kmol):   " << W_TiO2_           << "\n"
        << "\n"
        << " * Monomer geometry\n"
        << "    + nTiO2_min (-):    " << nTiO2_min_ << "\n"
        << "    + v0 (m3):          " << v0_         << "\n"
        << "    + d0 (m):           " << d0_         << "\n"
        << "    + s0 (m2):          " << s0_         << "\n"
        << "\n"
        << " * Processes\n"
        << "    + Nucleation:   " << static_cast<int>(nucleation_variant_) << "\n"
        << "    + Coagulation:  " << coagulation_model_  << "\n"
        << "    + Condensation: " << condensation_model_ << "\n"
        << "    + Sintering:    " << sintering_model_    << "\n"
        << "\n"
        << " * Collision enhancement factors\n"
        << "    + Nucleation:   " << epsilon_nuc_  << "\n"
        << "    + Coagulation:  " << epsilon_coag_ << "\n"
        << "    + Condensation: " << epsilon_cond_ << "\n"
        << "\n"
        << " * Sintering kinetics (tau_s = As * T^ns * dp^4 * exp(-Ts/T))\n"
        << "    + As (s/K/m4): " << As_ << "\n"
        << "    + ns (-):       " << ns_ << "\n"
        << "    + Ts (K):       " << Ts_ << "\n"
        << "\n"
        << " * Precursor: " << precursor_species_ << "\n";

    if (precursor_index_ >= 0) {
        std::cout
            << "    + index:  " << precursor_index_  << "\n"
            << "    + MW:     " << W_precursor_       << " kg/kmol\n"
            << "    + nTi:    " << nti_precursor_     << "\n"
            << "    + nH:     " << nh_precursor_      << "\n"
            << "    + nO:     " << no_precursor_      << "\n"
            << "    + nC:     " << nc_precursor_      << "\n"
            << "    + nu_H2O: " << nu_H2O_from_prec_  << "\n"
            << "    + nu_CO2: " << nu_CO2_from_prec_  << "\n"
            << "    + nu_O2:  " << nu_O2_from_prec_   << "\n";
    }

    std::cout
        << "\n"
        << " * Numerical floors\n"
        << "    + N_min  (#/m3):  " << N_min_  << "\n"
        << "    + fv_min (-):     " << fv_min_ << "\n"
        << "    + v_min  (m3):    " << v_min_  << "\n"
        << "    + S_min  (m2/m3): " << S_min_  << "\n"
        << "------------------------------------------------------------------------------------------\n";
}



// ============================================================================
// Write on files
// ============================================================================

template <ThermoMap Thermo>
void TiO2<Thermo>::WriteHeaderLine(MOM::OutputFileColumns& fOutput, const unsigned int precision)
{
	// Main variables (common to every method)
	fOutput.AddColumn("Ys[-]", precision);
	fOutput.AddColumn("Ns[#/m3]", precision);
	fOutput.AddColumn("Ss[m2/m3]", precision);

	fOutput.AddColumn("fv[-]", precision);
	fOutput.AddColumn("dp[nm]", precision);
	fOutput.AddColumn("dc[nm]", precision);
	fOutput.AddColumn("da[nm]", precision);
	fOutput.AddColumn("np[-]", precision);

	// Gas consumption (formation rates)
	fOutput.AddColumn("Gas[kg/m3/s]", precision);
	fOutput.AddColumn("Prec[kg/m3/s]", precision);

	// Specific quantities
	fOutput.AddColumn("ss[m2/#]", precision);
	fOutput.AddColumn("vs[m3/#]", precision);
	fOutput.AddColumn("taus[s]", precision);

	// Diffusion coefficient
	fOutput.AddColumn("Gamma[m2/s]", precision);

	// Number density function reconstruction parameters
	{
		fOutput.AddColumn("alpha[-]", precision);
		fOutput.AddColumn("nbar0[1/m3]", precision);
		fOutput.AddColumn("sigma[-]", precision);
		fOutput.AddColumn("kPareto[-]", precision);
		fOutput.AddColumn("nu1mean[m3/#]", precision);
		fOutput.AddColumn("nu2mean[m3/#]", precision);
		fOutput.AddColumn("mu[log(m3)]", precision);
	}

	// Overall source terms
	fOutput.AddColumn("Sall_Y[1/s]", precision);
	fOutput.AddColumn("Sall_Nnorm[1/s]", precision);
	fOutput.AddColumn("Sall_S[1/m/s]", precision);

	fOutput.AddColumn("Snuc_Y[1/s]", precision);
	fOutput.AddColumn("Snuc_Nnorm[1/s]", precision);
	fOutput.AddColumn("Snuc_S[1/m/s]", precision);

	fOutput.AddColumn("Ssin_Y[1/s]", precision);
	fOutput.AddColumn("Ssin_Nnorm[1/s]", precision);
	fOutput.AddColumn("Ssin_S[1/m/s]", precision);

	fOutput.AddColumn("Scon_Y[1/s]", precision);
	fOutput.AddColumn("Scon_Nnorm[1/s]", precision);
	fOutput.AddColumn("Scon_S[1/m/s]", precision);

	fOutput.AddColumn("Scoa_Y[1/s]", precision);
	fOutput.AddColumn("Scoa_Nnorm[1/s]", precision);
	fOutput.AddColumn("Scoa_S[1/m/s]", precision);
}

template <ThermoMap Thermo>
void TiO2<Thermo>::WriteOutputLine( 	MOM::OutputFileColumns& fOutput,
										const double T, const double P_Pa, const double* Y, const double mu,
										const double* M)
{
	SetMoments(M[0], M[1], M[2]);
	SetStatus(T, P_Pa, Y);
	this->SetViscosity(mu);
	CalculateSourceMoments();

	double fv, dp, dc, da, np, ss, vs, ssph, tauS;
	Properties(fv, dp, dc, da, np, ss, vs, ssph, tauS);

	fOutput << MassFraction();
	fOutput << ParticleNumberDensity();
	fOutput << SpecificSurface();

	fOutput << fv;
	fOutput << dp*1.e9;
	fOutput << dc*1.e9;
	fOutput << da*1.e9;
	fOutput << np;

	// Gas consumption (formation rates)
	fOutput << this->omega_gas_.sum();
	if (precursor_index_ >= 0)	fOutput << this->omega_gas_[precursor_index_];
	else						fOutput << 0.0;
	
	// Specific quantities
	fOutput << ss;
	fOutput << vs;
	fOutput << tauS;

	// Diffusion coefficient (m2/s)
	fOutput << DiffusionCoefficient()/this->rho_;

	// Reconstructed NDF
	{
		NDFReconstructionData ndf = ReconstructedNDFData();

		fOutput << ndf.alpha;
		fOutput << ndf.nbar0;
		fOutput << ndf.sigma;
		fOutput << ndf.k;
		fOutput << ndf.nu1mean;
		fOutput << ndf.nu2mean;
		fOutput << ndf.mu;
	}

	// Source terms (overall)
	for (unsigned int j=0; j<this->n_equations; ++j)
		fOutput << this->source_all_[j];

	// Source terms (nucleation)
	for (unsigned int j=0; j<this->n_equations; ++j)
		fOutput << this->source_nucleation_[j];

	// Source terms (sintering)
	for (unsigned int j=0; j<this->n_equations; ++j)
		fOutput << this->source_sintering_[j];			

	// Source terms (condensation)
	for (unsigned int j=0; j<this->n_equations; ++j)
		fOutput << this->source_condensation_[j];

	// Source terms (coagulation)
	for (unsigned int j=0; j<this->n_equations; ++j)
		fOutput << this->source_coagulation_[j];							
}

// ============================================================================
// SetupFromDictionary 
// ============================================================================

#if defined(MOM_USE_DICTIONARY)

template <ThermoMap Thermo>
template <typename Dictionary>
std::expected<void, std::string>
TiO2<Thermo>::SetupFromDictionary(Dictionary& dict)
{
	TiO2_Grammar grammar;
	dict.SetGrammar(grammar);

	if (dict.CheckOption("@TiO2") == true)
		dict.ReadBool("@TiO2", this->is_active_);

	if (dict.CheckOption("@Precursor") == true)
	{
		dict.ReadString("@Precursor", this->precursor_species_);
		this->SetPrecursor(this->precursor_species_);
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


	if (dict.CheckOption("@NucleationModel") == true)
	{
		std::string value;
		dict.ReadString("@NucleationModel", value);

		if (value == "none" || value == "0")
			nucleation_variant_ = NucleationVariant::Off;
		else if (value == "binary" || value == "1")
    		nucleation_variant_ = NucleationVariant::Binary;
		else if (value == "fixed-cluster" || value == "2")
    		nucleation_variant_ = NucleationVariant::FixedCluster;
		else
			return std::unexpected("Unknown @NucleationModel: " + value);

		this->SetNucleation(static_cast<int>(nucleation_variant_));
	}

	if (dict.CheckOption("@SinteringModel") == true)
	{
		int flag;
		dict.ReadInt("@SinteringModel", flag);
		this->SetSintering(flag);
	}

	if (dict.CheckOption("@CoagulationModel") == true)
	{
		int flag;
		dict.ReadInt("@CoagulationModel", flag);
		this->SetCoagulation(flag);
	}

	if (dict.CheckOption("@CondensationModel") == true)
	{
		int flag;
		dict.ReadInt("@CondensationModel", flag);
		this->SetCondensation(flag);
	}		

	if (dict.CheckOption("@ThermophoreticModel") == true)
	{
		int flag;
		dict.ReadInt("@ThermophoreticModel", flag);
		this->SetThermophoreticModel(flag);
	}		

	if (dict.CheckOption("@MinimumTiO2Units") == true)
	{
		int n;
		dict.ReadInt("@MinimumTiO2Units", n);
		this->SetMinimumNumberOfTiO2Units(static_cast<unsigned int>(n));
	}

	if (dict.CheckOption("@NucleatedParticleTiO2Units") == true)
	{
		int n;
		dict.ReadInt("@NucleatedParticleTiO2Units", n);
		this->SetNumberOfTiO2UnitsPerNucleatedParticle(static_cast<unsigned int>(n));
	}

	if (dict.CheckOption("@SinteringDeferred") == true)
		dict.ReadBool("@SinteringDeferred", this->is_sintering_deferred_);	

	if (dict.CheckOption("@SinteringDpMinimum") == true)
	{
		double value;
		std::string units;
		dict.ReadMeasure("@SinteringDpMinimum", value, units);
		if (units == "m")		this->sintering_dp_min_ = value;
		else if (units == "mm")	this->sintering_dp_min_ = value*1.e-3;
		else return std::unexpected("Allowed units for @SinteringDpMinimum: m | mm");
	}

	if (dict.CheckOption("@SinteringTauMinimum") == true)
	{
		double value;
		std::string units;
		dict.ReadMeasure("@SinteringTauMinimum", value, units);
		if (units == "s")		this->sintering_tau_min_ = value;
		else return std::unexpected("Allowed units for @SinteringTauMinimum: s");
	}	
	
	if (dict.CheckOption("@SinteringKMaximum") == true)
	{
		double value;
		std::string units;
		dict.ReadMeasure("@SinteringKMaximum", value, units);
		if (units == "1/s")		this->sintering_k_max_ = value;
		else return std::unexpected("Allowed units for @SinteringKMaximum: 1/s");
	}			

	// Frequency factors
	{ 
		double value;
		std::string units;

		if ( dict.CheckOption("@As") == true ) 
		{
			dict.ReadMeasure("@As", value, units);
			if (units != "s,K,m")	return std::unexpected("Allowed frequency factor units: s,K");
			this->As_ = value;
		}
	}

	// Activation temperatures
	{
		double value;
		std::string units;

		if ( dict.CheckOption("@Ts") == true ) 
		{
			dict.ReadMeasure("@Ts", value, units);
			if (units != "K")	return std::unexpected("Allowed activation temperature units: K");
			this->Ts_ = value;
		}
	}

	// Temperature exponents
	{
		double value;

		if ( dict.CheckOption("@ns") == true ) 
		{
			dict.ReadDouble("@ns", value);
			this->ns_ = value;
		}
	}

	if (dict.CheckOption("@SchmidtNumber") == true)
	{
		double value;
		dict.ReadDouble("@SchmidtNumber", value);
		this->SetSchmidtNumber(value);
	}

	// Minimum values for properties calculation 
	if ( dict.CheckOption("@MinimumNs") == true )
	{
		double value;
		std::string units;

		dict.ReadMeasure("@MinimumNs", value, units);
		if (units == "#/m3")		this->N_min_ = value;
		else if (units == "#/cm3")	this->N_min_ *= 1.e6;
		else return std::unexpected("Allowed units for @MinimumNs: #/m3 | #/cm3");

		Precalculations();
	}	
	
	// Minimum values for properties calculation 
	if ( dict.CheckOption("@MinimumFv") == true )
	{
		double value;
		dict.ReadDouble("@MinimumFv", value);
		this->fv_min_ = value;
	}			

	if (dict.CheckOption("@DebugMode") == true)
		dict.ReadBool("@DebugMode", this->is_debug_mode_);

    PrintSummary();
    return {};
}
#endif  // MOM_USE_DICTIONARY expected


// ============================================================================
// NDF reconstruction  (Pareto + log-normal, analogous to ThreeEquations)
// ============================================================================

template <ThermoMap Thermo>
typename TiO2<Thermo>::NDFReconstructionData
TiO2<Thermo>::ReconstructedNDFData(bool use_regularized_moments) const
{
    NDFReconstructionData d{};   // all zero, valid = false

    const double Y_raw = std::max(YTiO2_,               0.);
    const double N_raw = std::max(NTiO2N_ * N0_scaling_, 0.);

    double Y = Y_raw;
    double N = N_raw;
    if (use_regularized_moments) {
        const double fv_raw = this->rho_ / rhoTiO2_ * Y_raw;
        if (fv_raw < fv_min_) Y = rhoTiO2_ / this->rho_ * fv_min_;
        N = std::max(N, N_min_);
    }

    if (Y <= 0. || N <= 0.) return d;

    const double fv = this->rho_ / rhoTiO2_ * Y;
    if (!std::isfinite(fv) || fv <= 0.) return d;

    double nuMean = fv / N;
    if (!std::isfinite(nuMean) || nuMean <= 0.) return d;

    const double nuNucl = NucleationParticleVolume();
    if (!std::isfinite(nuNucl) || nuNucl <= 0.) return d;
    if (nuMean < nuNucl) nuMean = nuNucl;

    // Pareto weight (Eq. 8, Franzelli et al. 2019)
    double alpha = 1.0 - 0.18 * std::pow(nuMean / nuNucl, 0.12);
    alpha = std::clamp(alpha, 0., 1.);

    const double nbar0 = 8.0 * std::pow(1. - alpha, 2.) * 1.e27;   // [1/m3]
    const double sigma = 1.0 + 0.65 * (1. - alpha);                 // [-]

    constexpr double eps_min = 1.e-14;
    constexpr double tiny    = 1.e-300;
    constexpr double eps_k   = 1.01;

    double k = 0., nu1mean = 0., nu2mean = 0., mu = 0.;

    if (alpha > eps_min) {
        const double k_peak  = nbar0 * nuNucl / alpha;
        const double denom   = 1. - alpha * nuNucl / nuMean;
        if (denom <= 0. || !std::isfinite(denom)) return d;
        k       = std::max({ k_peak, eps_k / denom, 1. + 1.e-12 });
        nu1mean = k / (k - 1.) * nuNucl;
    }

    if (alpha < (1. - eps_min)) {
        nu2mean = (nuMean - alpha * nu1mean) / (1. - alpha);
        if (!std::isfinite(nu2mean) || nu2mean <= 0.) return d;
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
double TiO2<Thermo>::ReconstructedNormalizedNDF(
    double nu, bool use_regularized_moments) const
{
    if (!std::isfinite(nu) || nu <= 0.) return 0.;

    const auto d = ReconstructedNDFData(use_regularized_moments);
    if (!d.valid) return 0.;

    const double nuNucl = d.nuNucl;
    double nbar_p = 0., nbar_ln = 0.;

    // Pareto contribution
    if (d.alpha > 0. && d.k > 1. && nu >= nuNucl) {
        const double lg = std::log(d.k)
                        + d.k * std::log(nuNucl)
                        - (d.k + 1.) * std::log(nu);
        const double lmin = std::log(std::numeric_limits<double>::min());
        const double lmax = std::log(std::numeric_limits<double>::max());
        if (lg > lmin && lg < lmax) nbar_p = std::exp(lg);
    }

    // Log-normal contribution
    if (d.alpha < 1. && d.sigma > 0. && d.nu2mean > 0.) {
        const double z  = (std::log(nu) - d.mu) / d.sigma;
        const double lg = -std::log(nu)
                        - std::log(d.sigma)
                        - 0.5 * std::log(2. * this->pi_)
                        - 0.5 * z * z;
        const double lmin = std::log(std::numeric_limits<double>::min());
        const double lmax = std::log(std::numeric_limits<double>::max());
        if (lg > lmin && lg < lmax) nbar_ln = std::exp(lg);
    }

    const double nbar = d.alpha * nbar_p + (1. - d.alpha) * nbar_ln;
    return (std::isfinite(nbar) && nbar >= 0.) ? nbar : 0.;
}

template <ThermoMap Thermo>
double TiO2<Thermo>::ReconstructedNDF(
    double nu, bool use_regularized_moments) const
{
    const auto d = ReconstructedNDFData(use_regularized_moments);
    if (!d.valid) return 0.;
    const double n = d.N * ReconstructedNormalizedNDF(nu, use_regularized_moments);
    return (std::isfinite(n) && n >= 0.) ? n : 0.;
}

template <ThermoMap Thermo>
void TiO2<Thermo>::ReconstructedNDF(
    const Eigen::VectorXd& nu, Eigen::VectorXd& n,
    bool use_regularized_moments) const
{
    n.resize(nu.size());
    for (int i = 0; i < nu.size(); ++i)
        n(i) = ReconstructedNDF(nu(i), use_regularized_moments);
}

} // namespace MOM
