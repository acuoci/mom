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
|   Copyright (C) 2026 Alberto Cuoci                                      |
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
#if defined(MOM_USE_DICTIONARY)
#include "BrookesMoss_Grammar.h"
#endif

namespace MOM {

// ============================================================================
// Constructor
// ============================================================================

template <ThermoMap Thermo>
BrookesMoss<Thermo>::BrookesMoss(const Thermo& thermo)
    : thermo_(thermo)
{
    // ── CRTP base state ───────────────────────────────────────────────────
    this->is_active_               = true;
    this->gas_consumption_         = false;
    this->radiative_heat_transfer_ = true;
    this->schmidt_number_          = 50.;
    this->rho_particle_            = 1800.;
    this->planck_model_            = PlanckCoeffModel::Smooke;
    this->SetThermophoreticModel(1);
    this->dummy_species_           = "none";
    this->dummy_index_             = -1;
    this->dummy_species_closure_   = false;

    // ── Particle properties ───────────────────────────────────────────────
    dp_      = 1.e-9;
    mwp_     = 144.;
    Ns_norm_ = 1.e15;
    Ys_min_  = 1.e-30;
    bs_min_  = 1.e-30;

    // ── Model constants ───────────────────────────────────────────────────
    Calpha_  = 54.;        Talpha_ = 21000.;
    Cbeta_   = 1.0;
    Cgamma_  = 11700.;     Tgamma_ = 12100.;
    Comega_  = 105.8125;   etaColl_= 0.04;
    Coxid_   = 0.015;
    exp_l_   = 1.;  exp_m_ = 1.;  exp_n_ = 1.;

    // ── BM-Hall extended constants ─────────────────────────────────────────
    Calpha1_MBH_ = 127. * std::pow(10., 8.88);
    Calpha2_MBH_ = 178. * std::pow(10., 9.50);
    Talpha1_MBH_ = 4378.;
    Talpha2_MBH_ = 6390.;
    Comega2_MBH_ = 8903.51;
    Tomega2_MBH_ = 19778.;

    // ── Model flags ───────────────────────────────────────────────────────
    nucleation_variant_   = NucleationVariant::Off;
    oxidation_variant_    = OxidationVariant::Off;
    surface_growth_model_ = 0;
    coagulation_model_    = 0;

    // ── Gas consumption intermediates ─────────────────────────────────────
    dMdt_nucleation_       = 0.;
    dMdt_nucleation_BMH_1_ = 0.;
    dMdt_nucleation_BMH_2_ = 0.;
    dMdt_surface_growth_   = 0.;
    dMdt_oxidation_        = 0.;

    // ── Default precursor: C2H2 ───────────────────────────────────────────
    prec_species_ = "C2H2";
    prec_index_   = thermo_.IndexOfSpecies("C2H2");
    if (prec_index_ >= 0) {
        prec_nc_ = static_cast<double>(
            thermo_.NumberOfCarbonAtoms(static_cast<unsigned>(prec_index_)));
        prec_nh_ = static_cast<double>(
            thermo_.NumberOfHydrogenAtoms(static_cast<unsigned>(prec_index_)));
    } else {
        prec_nc_ = 2.;
        prec_nh_ = 2.;
    }
    conc_prec_ = 0.;

    // ── Default surface growth species: C2H2 ──────────────────────────────
    sg_species_ = "C2H2";
    sg_index_   = prec_index_;
    sg_nc_      = prec_nc_;
    sg_nh_      = prec_nh_;
    conc_sg_    = 0.;

    // ── Key species indices (0-based, -1 if absent) ───────────────────────
    index_H_    = thermo_.IndexOfSpecies("H");
    index_OH_   = thermo_.IndexOfSpecies("OH");
    index_O2_   = thermo_.IndexOfSpecies("O2");
    index_H2_   = thermo_.IndexOfSpecies("H2");
    index_H2O_  = thermo_.IndexOfSpecies("H2O");
    index_C2H2_ = thermo_.IndexOfSpecies("C2H2");
    index_C6H5_ = thermo_.IndexOfSpecies("A1-");
    index_C6H6_ = thermo_.IndexOfSpecies("A1");

    // ── Key species concentrations ────────────────────────────────────────
    conc_H_ = conc_OH_ = conc_O2_ = conc_H2_ = conc_H2O_ = conc_C2H2_ = 0.;
    Y_C2H2_ = Y_C6H5_ = Y_C6H6_ = Y_H2_ = Y_OH_ = Y_O2_ = 0.;

    MemoryAllocation();
}

// ============================================================================
// MemoryAllocation
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::MemoryAllocation()
{
    this->ZeroSources();
    this->omega_gas_ = Eigen::VectorXd::Zero(
        static_cast<Eigen::Index>(thermo_.NumberOfSpecies()));

    // Default initial moment values
    initial_moments_cache_(0) = 1.e-21;   // Ys_init [-]
    initial_moments_cache_(1) = 1.e-12;   // bs_init [m3/kg]
}

// ============================================================================
// SetStatus
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::SetStatus(double T, double P_Pa, const double* Y) noexcept
{
    this->T_    = T;
    this->P_Pa_ = P_Pa;

    // Mixture molecular weight [kg/kmol]  (1/MW = sum Yi/MWi)
    {
        const unsigned nsp = thermo_.NumberOfSpecies();
        double inv_MW = 0.;
        for (unsigned k = 0; k < nsp; ++k)
            inv_MW += Y[k] / thermo_.MolecularWeight(k);
        this->MW_ = (inv_MW > 1.e-300) ? 1. / inv_MW : 1.;
    }

    // Total molar concentration [kmol/m3] and mixture density [kg/m3]
    const double cTot = this->P_Pa_ / (this->Rgas_ * this->T_);
    this->rho_        = cTot * this->MW_;

    // Helper lambda: molar concentration of species idx [kmol/m3]
    auto conc = [&](int idx) -> double {
        if (idx < 0) return 0.;
        return cTot * std::max(Y[idx], 0.) * this->MW_
               / thermo_.MolecularWeight(static_cast<unsigned>(idx));
    };

    // Precursor concentration [kmol/m3]
    conc_prec_ = (prec_index_ >= 0)
        ? cTot * std::max(Y[prec_index_], 0.) * this->MW_
          / thermo_.MolecularWeight(static_cast<unsigned>(prec_index_))
        : 0.;

    // Surface growth species concentration [kmol/m3]
    conc_sg_ = (sg_index_ >= 0)
        ? cTot * std::max(Y[sg_index_], 0.) * this->MW_
          / thermo_.MolecularWeight(static_cast<unsigned>(sg_index_))
        : 0.;

    // Key species concentrations [kmol/m3]
    conc_H_    = conc(index_H_);
    conc_OH_   = conc(index_OH_);
    conc_O2_   = conc(index_O2_);
    conc_H2_   = conc(index_H2_);
    conc_H2O_  = conc(index_H2O_);
    conc_C2H2_ = conc(index_C2H2_);

    // Mass fractions needed for BM-Hall expressions
    Y_C2H2_ = (index_C2H2_ >= 0) ? std::max(Y[index_C2H2_], 0.) : 0.;
    Y_C6H5_ = (index_C6H5_ >= 0) ? std::max(Y[index_C6H5_], 0.) : 0.;
    Y_C6H6_ = (index_C6H6_ >= 0) ? std::max(Y[index_C6H6_], 0.) : 0.;
    Y_H2_   = (index_H2_   >= 0) ? std::max(Y[index_H2_],   0.) : 0.;
    Y_OH_   = (index_OH_   >= 0) ? std::max(Y[index_OH_],   0.) : 0.;
    Y_O2_   = (index_O2_   >= 0) ? std::max(Y[index_O2_],   0.) : 0.;
}

// ============================================================================
// SetMoments
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::SetMoments(std::span<const double> m) noexcept
{
    SetMoments(m[0], m[1]);
}

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::SetMoments(double Ys, double bs) noexcept
{
    Ys_ = std::max(Ys, 0.);
    bs_ = std::max(bs, 0.);
}

// ============================================================================
// SetPrecursors
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::SetPrecursors(std::string_view name)
{
    prec_species_ = std::string(name);
    prec_index_   = thermo_.IndexOfSpecies(prec_species_);

    if (prec_index_ < 0)
        throw std::runtime_error("[BrookesMoss] Precursor species not found in mechanism: "
                                 + prec_species_);

    prec_nc_ = static_cast<double>(
        thermo_.NumberOfCarbonAtoms(static_cast<unsigned>(prec_index_)));
    prec_nh_ = static_cast<double>(
        thermo_.NumberOfHydrogenAtoms(static_cast<unsigned>(prec_index_)));

    if (prec_nc_ <= 0.)
        throw std::runtime_error("[BrookesMoss] Precursor species has no carbon atoms: "
                                 + prec_species_);
}

// ============================================================================
// SetSurfaceGrowthSpecies
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::SetSurfaceGrowthSpecies(std::string_view name)
{
    sg_species_ = std::string(name);
    sg_index_   = thermo_.IndexOfSpecies(sg_species_);

    if (sg_index_ < 0)
        throw std::runtime_error("[BrookesMoss] Surface growth species not found in mechanism: "
                                 + sg_species_);

    sg_nc_ = static_cast<double>(
        thermo_.NumberOfCarbonAtoms(static_cast<unsigned>(sg_index_)));
    sg_nh_ = static_cast<double>(
        thermo_.NumberOfHydrogenAtoms(static_cast<unsigned>(sg_index_)));

    if (sg_nc_ <= 0.)
        throw std::runtime_error("[BrookesMoss] Surface growth species has no carbon atoms: "
                                 + sg_species_);
}

// ============================================================================
// SetGasClosureDummySpecies
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::SetGasClosureDummySpecies(std::string_view name)
{
    this->dummy_species_ = std::string(name);

    if (this->dummy_species_ == "none") {
        this->dummy_index_           = -1;
        this->dummy_species_closure_ = false;
        return;
    }

    this->dummy_index_ = thermo_.IndexOfSpecies(this->dummy_species_);

    if (this->dummy_index_ < 0)
        throw std::runtime_error("[BrookesMoss] Dummy species not found in mechanism: "
                                 + this->dummy_species_);

    this->dummy_species_closure_ = true;
}

// ============================================================================
// SetNucleation (string variant)
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::SetNucleation(std::string_view label)
{
    if (label == "none") {
        nucleation_variant_ = NucleationVariant::Off;
    }
    else if (label == "BrookesMoss") {
        nucleation_variant_ = NucleationVariant::BrookesMoss;
        mwp_                = 144.;
    }
    else if (label == "BrookesMossHall") {
        nucleation_variant_ = NucleationVariant::BrookesMossHall;
        mwp_                = 1200.;
        CheckBrookesMossHallSpecies();
    }
    else {
        throw std::runtime_error("[BrookesMoss] Unknown nucleation model: "
                                 + std::string(label)
                                 + ". Allowed: none | BrookesMoss | BrookesMossHall");
    }
}

// ============================================================================
// SetOxidation (string variant)
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::SetOxidation(std::string_view label)
{
    if (label == "none") {
        oxidation_variant_ = OxidationVariant::Off;
    }
    else if (label == "BrookesMoss") {
        oxidation_variant_ = OxidationVariant::BrookesMoss;
        etaColl_           = 0.04;
        Coxid_             = 0.015;
    }
    else if (label == "BrookesMossHall") {
        oxidation_variant_ = OxidationVariant::BrookesMossHall;
        etaColl_           = 0.13;
        Coxid_             = 0.015;
        CheckBrookesMossHallSpecies();
    }
    else {
        throw std::runtime_error("[BrookesMoss] Unknown oxidation model: "
                                 + std::string(label)
                                 + ". Allowed: none | BrookesMoss | BrookesMossHall");
    }
}

// ============================================================================
// CheckBrookesMossHallSpecies
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::CheckBrookesMossHallSpecies()
{
    // Update all BMH-relevant indices (0-based, no -1 subtraction)
    index_C6H5_ = thermo_.IndexOfSpecies("A1-");
    index_C6H6_ = thermo_.IndexOfSpecies("A1");

    if (index_C6H5_ < 0)
        throw std::runtime_error(
            "[BrookesMoss] BrookesMossHall requires species A1- "
            "(phenyl radical) in the mechanism.");
    if (index_C6H6_ < 0)
        throw std::runtime_error(
            "[BrookesMoss] BrookesMossHall requires species A1 "
            "(benzene) in the mechanism.");

    // Adjust surface growth activation temperature for BMH
    Cgamma_ = 9000.6;
}

// ============================================================================
// CalculateSourceMoments
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::CalculateSourceMoments() noexcept
{
    // Reset all source vectors and gas sources
    this->ZeroSources();

    // Reset intermediate quantities
    dMdt_nucleation_       = 0.;
    dMdt_nucleation_BMH_1_ = 0.;
    dMdt_nucleation_BMH_2_ = 0.;
    dMdt_surface_growth_   = 0.;
    dMdt_oxidation_        = 0.;

    // Sub-process source terms
    if (nucleation_variant_   != NucleationVariant::Off) NucleationSourceTerms();
    if (surface_growth_model_ >  0)                      SurfaceGrowthSourceTerms();
    if (oxidation_variant_    != OxidationVariant::Off)  OxidationSourceTerms();
    if (coagulation_model_    >  0)                      CoagulationSourceTerms();

    // Sanitise and accumulate total source
    for (unsigned i = 0; i < 2u; ++i) {
        if (!std::isfinite(this->source_nucleation_(i)))  this->source_nucleation_(i)  = 0.;
        if (!std::isfinite(this->source_growth_(i)))      this->source_growth_(i)      = 0.;
        if (!std::isfinite(this->source_oxidation_(i)))   this->source_oxidation_(i)   = 0.;
        if (!std::isfinite(this->source_coagulation_(i))) this->source_coagulation_(i) = 0.;

        this->source_all_(i) = this->source_nucleation_(i)
                             + this->source_growth_(i)
                             + this->source_oxidation_(i)
                             + this->source_coagulation_(i);
    }

    CalculateOmegaGas();
}

// ============================================================================
// NucleationSourceTerms  (dispatcher)
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::NucleationSourceTerms()
{
    if (nucleation_variant_ == NucleationVariant::BrookesMoss)
        NucleationSourceTerms_BM();
    else if (nucleation_variant_ == NucleationVariant::BrookesMossHall)
        NucleationSourceTerms_BMH();
}

// ============================================================================
// NucleationSourceTerms_BM  — Brookes & Moss (1999)
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::NucleationSourceTerms_BM()
{
    if (conc_prec_ <= 0.) return;

    // Particle number formation rate [#/m3/s]
    const double dNdt = Calpha_ * this->Nav_kmol_
                      * std::pow(conc_prec_, exp_l_)
                      * std::exp(-Talpha_ / this->T_);

    // Soot mass formation rate [kg/m3/s]:  dMdt = mwp * dNdt / Nav
    const double dMdt = mwp_ * dNdt / this->Nav_kmol_;
    dMdt_nucleation_  = dMdt;

    // Source for Ys [1/s]  =  dMdt / rho
    this->source_nucleation_(0) = dMdt / this->rho_;

    // Source for bs [m3/kg/s]  =  dNdt / (rho * Ns_norm)
    this->source_nucleation_(1) = dNdt / (this->rho_ * Ns_norm_);
}

// ============================================================================
// NucleationSourceTerms_BMH  — Hall et al. (2016)
// Two-channel inception: C2H2 (channel 1) + A1/C6H6 (channel 2)
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::NucleationSourceTerms_BMH()
{
    dMdt_nucleation_BMH_1_ = 0.;
    dMdt_nucleation_BMH_2_ = 0.;

    // Channel 1: C2H2-based inception
    if (conc_C2H2_ > 0.) {
        const double dNdt1 = Calpha1_MBH_
                           * std::pow(conc_C2H2_, exp_l_)
                           * std::exp(-Talpha1_MBH_ / this->T_);
        dMdt_nucleation_BMH_1_ = mwp_ * dNdt1 / this->Nav_kmol_;
    }

    // Channel 2: A1 (benzene / C6H6) based inception
    if (index_C6H6_ >= 0) {
        const double cTot   = this->P_Pa_ / (this->Rgas_ * this->T_);
        const double conc_A1 = cTot * Y_C6H6_ * this->MW_
                              / thermo_.MolecularWeight(static_cast<unsigned>(index_C6H6_));
        if (conc_A1 > 0.) {
            const double dNdt2 = Calpha2_MBH_
                               * std::pow(conc_A1, exp_l_)
                               * std::exp(-Talpha2_MBH_ / this->T_);
            dMdt_nucleation_BMH_2_ = mwp_ * dNdt2 / this->Nav_kmol_;
        }
    }

    // Total nucleation mass rate [kg/m3/s]
    const double dMdt   = dMdt_nucleation_BMH_1_ + dMdt_nucleation_BMH_2_;
    dMdt_nucleation_    = dMdt;

    // Total number rate [#/m3/s]
    const double dNdt_total = (mwp_ > 0.) ? dMdt * this->Nav_kmol_ / mwp_ : 0.;

    // Source for Ys [1/s]
    this->source_nucleation_(0) = dMdt / this->rho_;

    // Source for bs [m3/kg/s]
    this->source_nucleation_(1) = dNdt_total / (this->rho_ * Ns_norm_);
}

// ============================================================================
// SurfaceGrowthSourceTerms
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::SurfaceGrowthSourceTerms()
{
    // Soot mass and number concentrations
    const double M = std::max(Ys_, 0.) * this->rho_;            // [kg/m3]
    const double N = std::max(bs_, 0.) * this->rho_ * Ns_norm_; // [#/m3]

    if (M <= 0. || N <= 0. || conc_sg_ <= 0.) {
        dMdt_surface_growth_ = 0.;
        return;
    }

    // Particle surface area per unit volume [m2/m3]:
    //   A = (pi * N)^(1/3) * (6 * M / rho_p)^(2/3)
    const double A = std::pow(this->pi_ * N, 1. / 3.)
                   * std::pow(6. * M / this->rho_particle_, 2. / 3.);

    // Surface growth mass rate [kg/m3/s]
    const double dMdt = Cgamma_
                      * std::pow(conc_sg_, exp_m_)
                      * std::exp(-Tgamma_ / this->T_)
                      * std::pow(A, exp_n_);

    dMdt_surface_growth_ = dMdt;

    // Source for Ys [1/s]
    this->source_growth_(0) = dMdt / this->rho_;

    // No change in number density from surface growth
    this->source_growth_(1) = 0.;
}

// ============================================================================
// CoagulationSourceTerms  — free-molecular regime
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::CoagulationSourceTerms()
{
    const double N  = std::max(bs_, 0.) * this->rho_ * Ns_norm_; // [#/m3]
    const double dp = ParticleDiameter();                          // [m]

    if (N <= 0. || dp <= 0.) return;

    // Free-molecular coagulation kernel:
    //   dNdt = Cbeta * sqrt(24 * R * T / rho_p / Nav) * sqrt(dp) * N^2
    const double dNdt = Cbeta_
                      * std::sqrt(24. * this->Rgas_ * this->T_
                                  / this->rho_particle_ / this->Nav_kmol_)
                      * std::sqrt(dp)
                      * N * N;

    // No change in soot mass from coagulation
    this->source_coagulation_(0) = 0.;

    // Source for bs [m3/kg/s]:  -dNdt / (rho * Ns_norm)
    this->source_coagulation_(1) = -1. / this->rho_ * (dNdt / Ns_norm_);
}

// ============================================================================
// OxidationSourceTerms  (dispatcher)
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::OxidationSourceTerms()
{
    if (oxidation_variant_ == OxidationVariant::BrookesMoss)
        OxidationSourceTerms_BM();
    else if (oxidation_variant_ == OxidationVariant::BrookesMossHall)
        OxidationSourceTerms_BMH();
}

// ============================================================================
// OxidationSourceTerms_BM  — Brookes & Moss (1999), OH attack
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::OxidationSourceTerms_BM()
{
    const double M = std::max(Ys_, 0.) * this->rho_;            // [kg/m3]
    const double N = std::max(bs_, 0.) * this->rho_ * Ns_norm_; // [#/m3]

    if (M <= 0. || N <= 0. || conc_OH_ <= 0.) return;

    // Particle surface area per unit volume [m2/m3]
    const double A = std::pow(this->pi_ * N, 1. / 3.)
                   * std::pow(6. * M / this->rho_particle_, 2. / 3.);

    // Oxidation mass rate [kg/m3/s]
    //   dMdt = Coxid * Comega * etaColl * [OH] * sqrt(T) * A
    const double dMdt_oxid = Coxid_ * Comega_ * etaColl_
                           * conc_OH_
                           * std::sqrt(this->T_)
                           * A;

    dMdt_oxidation_ = dMdt_oxid;

    // Source for Ys [1/s]  (negative: consumption)
    this->source_oxidation_(0) = -dMdt_oxid / this->rho_;

    // No change in number density from oxidation (mass loss only)
    this->source_oxidation_(1) = 0.;
}

// ============================================================================
// OxidationSourceTerms_BMH  — Hall et al. (2016), OH + O2 channels
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::OxidationSourceTerms_BMH()
{
    const double M = std::max(Ys_, 0.) * this->rho_;            // [kg/m3]
    const double N = std::max(bs_, 0.) * this->rho_ * Ns_norm_; // [#/m3]

    if (M <= 0. || N <= 0.) return;

    // Particle surface area per unit volume [m2/m3]
    const double A = std::pow(this->pi_ * N, 1. / 3.)
                   * std::pow(6. * M / this->rho_particle_, 2. / 3.);

    double dMdt_oxid = 0.;

    // Channel 1: OH oxidation (BM form with BMH collision efficiency etaColl_=0.13)
    if (conc_OH_ > 0.) {
        dMdt_oxid += Coxid_ * Comega_ * etaColl_
                   * conc_OH_
                   * std::sqrt(this->T_)
                   * A;
    }

    // Channel 2: O2 oxidation (Arrhenius-type with BMH constants)
    if (conc_O2_ > 0.) {
        dMdt_oxid += Comega2_MBH_
                   * conc_O2_
                   * std::exp(-Tomega2_MBH_ / this->T_)
                   * A;
    }

    dMdt_oxidation_ = dMdt_oxid;

    // Source for Ys [1/s]  (negative: consumption)
    this->source_oxidation_(0) = -dMdt_oxid / this->rho_;

    // No change in number density from oxidation
    this->source_oxidation_(1) = 0.;
}

// ============================================================================
// CalculateOmegaGas
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::CalculateOmegaGas() noexcept
{
    this->omega_gas_.setZero();
    if (!this->gas_consumption_) return;

    const int nsp = static_cast<int>(thermo_.NumberOfSpecies());

    // Helper: add mass source [kg/m3/s] directly for species idx
    auto AddMass = [&](int idx, double omega_kg_m3_s) {
        if (idx < 0 || idx >= nsp || !std::isfinite(omega_kg_m3_s)) return;
        this->omega_gas_[idx] += omega_kg_m3_s;
    };

    // ── Nucleation coupling ───────────────────────────────────────────────
    //
    // For every kg of soot formed:
    //   precursor consumed:  omega_prec = -dMdt * MW_prec / (nc_prec * WC_)
    //   H2 produced:         omega_H2   = +dMdt * (nh_prec/2) * MW_H2 / (nc_prec * WC_)
    //
    // [kg/m3/s] = [kg/m3/s] * [kmol_prec/kg_soot] * [kg/kmol_prec]

    if (nucleation_variant_ == NucleationVariant::BrookesMoss) {
        if (dMdt_nucleation_ > 0. && prec_index_ >= 0 && prec_nc_ > 0.) {
            const double MW_prec          = thermo_.MolecularWeight(
                                                static_cast<unsigned>(prec_index_));
            const double prec_per_kg_soot = 1. / (prec_nc_ * this->WC_);

            AddMass(prec_index_,
                    -dMdt_nucleation_ * prec_per_kg_soot * MW_prec);

            if (index_H2_ >= 0 && prec_nh_ > 0.) {
                const double MW_H2 = thermo_.MolecularWeight(
                                         static_cast<unsigned>(index_H2_));
                AddMass(index_H2_,
                        dMdt_nucleation_ * prec_per_kg_soot * (prec_nh_ / 2.) * MW_H2);
            }
        }
    }
    else if (nucleation_variant_ == NucleationVariant::BrookesMossHall) {
        // Channel 1: C2H2-based
        if (dMdt_nucleation_BMH_1_ > 0. && prec_index_ >= 0 && prec_nc_ > 0.) {
            const double MW_prec          = thermo_.MolecularWeight(
                                                static_cast<unsigned>(prec_index_));
            const double prec_per_kg_soot = 1. / (prec_nc_ * this->WC_);

            AddMass(prec_index_,
                    -dMdt_nucleation_BMH_1_ * prec_per_kg_soot * MW_prec);

            if (index_H2_ >= 0 && prec_nh_ > 0.) {
                const double MW_H2 = thermo_.MolecularWeight(
                                         static_cast<unsigned>(index_H2_));
                AddMass(index_H2_,
                        dMdt_nucleation_BMH_1_ * prec_per_kg_soot * (prec_nh_ / 2.) * MW_H2);
            }
        }

        // Channel 2: A1 (benzene)-based
        if (dMdt_nucleation_BMH_2_ > 0. && index_C6H6_ >= 0) {
            const double nc_A1 = static_cast<double>(
                thermo_.NumberOfCarbonAtoms(static_cast<unsigned>(index_C6H6_)));
            const double nh_A1 = static_cast<double>(
                thermo_.NumberOfHydrogenAtoms(static_cast<unsigned>(index_C6H6_)));

            if (nc_A1 > 0.) {
                const double MW_A1          = thermo_.MolecularWeight(
                                                  static_cast<unsigned>(index_C6H6_));
                const double A1_per_kg_soot = 1. / (nc_A1 * this->WC_);

                AddMass(index_C6H6_,
                        -dMdt_nucleation_BMH_2_ * A1_per_kg_soot * MW_A1);

                if (index_H2_ >= 0 && nh_A1 > 0.) {
                    const double MW_H2 = thermo_.MolecularWeight(
                                             static_cast<unsigned>(index_H2_));
                    AddMass(index_H2_,
                            dMdt_nucleation_BMH_2_ * A1_per_kg_soot * (nh_A1 / 2.) * MW_H2);
                }
            }
        }
    }

    // ── Surface growth coupling ───────────────────────────────────────────
    if (dMdt_surface_growth_ > 0. && sg_index_ >= 0 && sg_nc_ > 0.) {
        const double MW_sg          = thermo_.MolecularWeight(
                                          static_cast<unsigned>(sg_index_));
        const double sg_per_kg_soot = 1. / (sg_nc_ * this->WC_);

        AddMass(sg_index_,
                -dMdt_surface_growth_ * sg_per_kg_soot * MW_sg);

        // H2 production from H atoms in surface growth species
        if (index_H2_ >= 0 && sg_nh_ > 0.) {
            const double MW_H2 = thermo_.MolecularWeight(
                                     static_cast<unsigned>(index_H2_));
            AddMass(index_H2_,
                    dMdt_surface_growth_ * sg_per_kg_soot * (sg_nh_ / 2.) * MW_H2);
        }
    }

    // ── Dummy species closure (enforce mass conservation) ─────────────────
    if (this->dummy_species_closure_) {
        // dummy_index_ must be set during setup — a negative value is a programming error.
        assert(this->dummy_index_ >= 0 &&
               "[BrookesMoss::CalculateOmegaGas] dummy_index_ not set.");
        double sum = 0.;
        for (int i = 0; i < nsp; ++i)
            if (i != this->dummy_index_) sum += this->omega_gas_[i];
        this->omega_gas_[this->dummy_index_] = -sum;
    }
}

// ============================================================================
// Property methods
// ============================================================================

template <ThermoMap Thermo>
double BrookesMoss<Thermo>::VolumeFraction() const noexcept
{
    return this->rho_ / this->rho_particle_ * std::max(Ys_, 0.);
}

template <ThermoMap Thermo>
double BrookesMoss<Thermo>::MassFraction() const noexcept
{
    return std::max(Ys_, 0.);
}

template <ThermoMap Thermo>
double BrookesMoss<Thermo>::ParticleNumberDensity() const noexcept
{
    return std::max(bs_, 0.) * this->rho_ * Ns_norm_;
}

template <ThermoMap Thermo>
double BrookesMoss<Thermo>::ParticleDiameter() const noexcept
{
    const double M = std::max(Ys_, 0.) * this->rho_;            // [kg/m3]
    const double N = std::max(bs_, 0.) * this->rho_ * Ns_norm_; // [#/m3]

    if (N <= 0. || M <= 0.) return 0.;

    return std::pow(6. * M / (this->pi_ * this->rho_particle_ * N), 1. / 3.);
}

template <ThermoMap Thermo>
double BrookesMoss<Thermo>::CollisionDiameter() const noexcept
{
    // For BrookesMoss the collision diameter equals the mean particle diameter
    return ParticleDiameter();
}

template <ThermoMap Thermo>
double BrookesMoss<Thermo>::SpecificSurface() const noexcept
{
    const double dp = ParticleDiameter();
    const double N  = ParticleNumberDensity();
    return this->pi_ * dp * dp * N;
}

template <ThermoMap Thermo>
double BrookesMoss<Thermo>::DiffusionCoefficient() const noexcept
{
    return this->mu_ / this->schmidt_number_;
}

// ============================================================================
// PrintSummary
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::PrintSummary() const
{
    std::cout << "\n"
              << "------------------------------------------------------------------------------------------\n"
              << "                        Brookes-Moss Soot Model Summary\n"
              << "------------------------------------------------------------------------------------------\n"
              << " * Soot density (kg/m3):      " << this->rho_particle_     << "\n"
              << " * Schmidt number (-):         " << this->schmidt_number_   << "\n"
              << " * Ns normalisation (#/m3):    " << Ns_norm_                << "\n"
              << " * Soot particle MW (kg/kmol): " << mwp_                    << "\n"
              << "\n"
              << " * Processes\n"
              << "    + Nucleation:     " << static_cast<int>(nucleation_variant_) << "\n"
              << "    + Surface growth: " << surface_growth_model_                 << "\n"
              << "    + Oxidation:      " << static_cast<int>(oxidation_variant_)  << "\n"
              << "    + Coagulation:    " << coagulation_model_                    << "\n"
              << "\n"
              << " * Precursor species:          " << prec_species_
              << "  (nC=" << prec_nc_ << ", nH=" << prec_nh_ << ")\n"
              << " * Surface growth species:     " << sg_species_
              << "  (nC=" << sg_nc_   << ", nH=" << sg_nh_   << ")\n"
              << "\n"
              << " * Nucleation constants\n"
              << "    + Calpha  = " << Calpha_  << "\n"
              << "    + Talpha  = " << Talpha_  << " [K]\n"
              << "\n"
              << " * Surface growth constants\n"
              << "    + Cgamma  = " << Cgamma_  << "\n"
              << "    + Tgamma  = " << Tgamma_  << " [K]\n"
              << "    + exp_m   = " << exp_m_   << "\n"
              << "    + exp_n   = " << exp_n_   << "\n"
              << "\n"
              << " * Coagulation constants\n"
              << "    + Cbeta   = " << Cbeta_   << "\n"
              << "\n"
              << " * Oxidation constants\n"
              << "    + Comega  = " << Comega_  << "\n"
              << "    + etaColl = " << etaColl_ << "\n"
              << "    + Coxid   = " << Coxid_   << "\n"
              << "\n"
              << " * BM-Hall extended constants\n"
              << "    + Calpha1_MBH = " << Calpha1_MBH_ << "\n"
              << "    + Calpha2_MBH = " << Calpha2_MBH_ << "\n"
              << "    + Talpha1_MBH = " << Talpha1_MBH_ << " [K]\n"
              << "    + Talpha2_MBH = " << Talpha2_MBH_ << " [K]\n"
              << "    + Comega2_MBH = " << Comega2_MBH_ << "\n"
              << "    + Tomega2_MBH = " << Tomega2_MBH_ << " [K]\n"
              << "\n"
              << " * Numerical floors\n"
              << "    + Ys_min (-):    " << Ys_min_ << "\n"
              << "    + bs_min (-):    " << bs_min_ << "\n"
              << "\n"
              << " * Gas consumption:  " << (this->gas_consumption_         ? "yes" : "no") << "\n"
              << " * Radiative HT:     " << (this->radiative_heat_transfer_ ? "yes" : "no") << "\n"
              << "------------------------------------------------------------------------------------------\n";
}



// ============================================================================
// Write on files
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::WriteHeaderLine(MOM::OutputFileColumns& fOutput, const unsigned int precision)
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
	fOutput.AddColumn("SG[kg/m3/s]", precision);
	fOutput.AddColumn("H2[kg/m3/s]", precision);
	fOutput.AddColumn("H[kg/m3/s]", precision);
	fOutput.AddColumn("OH[kg/m3/s]", precision);
	fOutput.AddColumn("O2[kg/m3/s]", precision);

	// Specific quantities
	fOutput.AddColumn("ss[m2/#]", precision);
	fOutput.AddColumn("vs[m3/#]", precision);

	// Diffusion coefficient
	fOutput.AddColumn("Gamma[m2/s]", precision);


	// Overall source terms
	{
		std::string title = "Sall[1/s]";
		fOutput.AddColumn(title, precision);
	}
	{
		std::string title = "Sall[m3/kg/s]";
		fOutput.AddColumn(title, precision);
	}

	// Nucleation terms
	{
		std::string title = "Snuc[1/s]";
		fOutput.AddColumn(title, precision);
	}
	{
		std::string title = "Snuc[m3/kg/s]";
		fOutput.AddColumn(title, precision);
	}

	// Surface growth source terms
	{
		std::string title = "Sgro[1/s]";
		fOutput.AddColumn(title, precision);
	}
	{
		std::string title = "Sgro[m3/kg/s]";
		fOutput.AddColumn(title, precision);
	}

	// Oxidation source terms
	{
		std::string title = "Soxi[1/s]";
		fOutput.AddColumn(title, precision);
	}
	{
		std::string title = "Soxi[m3/kg/s]";
		fOutput.AddColumn(title, precision);
	}

	// Condensation source terms (dummy)
	{
		std::string title = "Scon[1/s]";
		fOutput.AddColumn(title, precision);
	}
	{
		std::string title = "Scon[m3/kg/s]";
		fOutput.AddColumn(title, precision);
	}

	// Coagulation source terms
	{
		std::string title = "Scoa[1/s]";
		fOutput.AddColumn(title, precision);
	}
	{
		std::string title = "Scoa[m3/kg/s]";
		fOutput.AddColumn(title, precision);
	}	
}

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::WriteOutputLine( 	MOM::OutputFileColumns& fOutput,
												const double T, const double P_Pa, const double* Y, const double mu,
												const double* M)
{
	SetMoments(M[0], M[1]);
	SetStatus(T, P_Pa, Y);
	this->SetViscosity(mu);
	CalculateSourceMoments();

	// Soot properties
	fOutput << MassFraction();
	fOutput << ParticleNumberDensity();
	fOutput << 0.; // SootSpecificSurface(); TODO
	fOutput << VolumeFraction(); 		
	fOutput << ParticleDiameter()*1e9;
	fOutput << 0.; // dc*1e9;	TODO
	fOutput << 0.; // np;		TODO

	// Gas consumption (formation rates)
	fOutput << this->omega_gas_.sum();
	fOutput << this->omega_gas_[prec_index_];
	fOutput << this->omega_gas_[sg_index_];
	fOutput << this->omega_gas_[index_H2_];
	fOutput << this->omega_gas_[index_H_];
	fOutput << this->omega_gas_[index_OH_];
	fOutput << this->omega_gas_[index_O2_];
	
	// Specific quantities
	fOutput << 0.; // ss;	TODO
	fOutput << 0.; // vs;	TODO

	// Diffusion coefficient (m2/s)
	fOutput << DiffusionCoefficient()/this->rho_;

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
		fOutput << 0.;	// dummy

	// Source terms (coagulation)
	for (unsigned int j = 0; j < this->n_equations; j++)
		fOutput << this->source_coagulation_[j];		
}

// ============================================================================
// SetupFromDictionary 
// ============================================================================

#if defined(MOM_USE_DICTIONARY)

template <ThermoMap Thermo>
template <typename Dictionary>
std::expected<void, std::string>
BrookesMoss<Thermo>::SetupFromDictionary(Dictionary& dict)
{
	BrookesMoss_Grammar grammar;
	dict.SetGrammar(grammar);

	if (dict.CheckOption("@BrookesMoss") == true)
		dict.ReadBool("@BrookesMoss", this->is_active_);

	if (dict.CheckOption("@NucleationModel") == true)
	{
		std::string name;
		dict.ReadString("@NucleationModel", name);
		if (name == "0" || name == "none")
			this->SetNucleation(0);
		else if (name == "1" || name == "BrookesMoss")
			this->SetNucleation(1);
		else if (name == "2" || name == "BrookesMossHall")
			this->SetNucleation(2);
		else return std::unexpected("@NucleationModel: allowed options: 0=none, 1=BrookesMoss, 2=BrookesMossHall");
	}

	if (dict.CheckOption("@SurfaceGrowthModel") == true)
	{
		int flag;
		dict.ReadInt("@SurfaceGrowthModel", flag);
		this->SetSurfaceGrowth(flag);
	}

	if (dict.CheckOption("@OxidationModel") == true)
	{
		std::string name;
		dict.ReadString("@OxidationModel", name);
		if (name == "0" || name == "none")
			this->SetOxidation(0);
		else if (name == "1" || name == "BrookesMoss")
			this->SetOxidation(1);
		else if (name == "2" || name == "BrookesMossHall")
			this->SetOxidation(2);
		else return std::unexpected("@OxidationModel: allowed options: 0=none, 1=BrookesMoss, 2=BrookesMossHall");
	}

	if (dict.CheckOption("@CoagulationModel") == true)
	{
		int flag;
		dict.ReadInt("@CoagulationModel", flag);
		SetCoagulation(flag);
	}

	if (dict.CheckOption("@ThermophoreticModel") == true)
	{
		int flag;
		dict.ReadInt("@ThermophoreticModel", flag);
		this->SetThermophoreticModel(flag);
	}

	if (dict.CheckOption("@Precursors") == true)
	{
		dict.ReadString("@Precursors", prec_species_);
		this->SetPrecursors(prec_species_);
	}

	if (dict.CheckOption("@SurfaceGrowthSpecies") == true)
	{
		dict.ReadString("@SurfaceGrowthSpecies", sg_species_);
		this->SetSurfaceGrowthSpecies(sg_species_);
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

	if (dict.CheckOption("@SootDensity") == true)
	{
		double value;
		std::string units;
		dict.ReadMeasure("@SootDensity", value, units);

		if (units == "kg/m3")		value = value;
		else if (units == "g/cm3")	value *= 1000.;
		else return std::unexpected("@SootDensity: allowed units: kg/m3 | g/cm3");

		this->SetSootDensity(value);
	}

	if (dict.CheckOption("@SootParticleDiameter") == true)
	{
		double value;
		std::string units;
		dict.ReadMeasure("@SootParticleDiameter", value, units);

		if (units == "m")		value = value;
		else if (units == "mm")	value /= 1.e3;
		else if (units == "nm")	value /= 1.e9;
		else return std::unexpected("@SootParticleDiameter: allowed units: m | mm | nm");

		this->SetDp(value);
	}
	
	if (dict.CheckOption("@SootParticleMolecularWeight") == true)
	{
		double value;
		std::string units;
		dict.ReadMeasure("@SootParticleMolecularWeight", value, units);

		if (units == "kg/kmol")		value = value;
		else if (units == "g/mol")	value = value;
		else return std::unexpected("@SootParticleMolecularWeight: allowed units: kg/kmol | g/mol");

		this->SetMWp(value);
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

	// Normalization values for properties calculation 
	if ( dict.CheckOption("@NsNorm") == true )
	{
		double value;
		std::string units;

		dict.ReadMeasure("@NsNorm", value, units);
		if (units == "#/m3")		value = value;
		else if (units == "#/cm3")	value *= 1.e6;
		else return std::unexpected("Allowed units for @NsNorm: #/m3 | #/cm3");

		this->SetNsNorm(value);
	}

	if ( dict.CheckOption("@Calpha") == true )
	{
		double value; std::string units;
		dict.ReadMeasure("@Calpha", value, units);
		if (units == "1/s")		value = value;
		else return std::unexpected("Allowed units for @Calpha: 1/s");

		this->Calpha_ = value;
	}

	if ( dict.CheckOption("@Talpha") == true )
	{
		double value; std::string units;
		dict.ReadMeasure("@Talpha", value, units);
		if (units == "K")		value = value;
		else return std::unexpected("Allowed units for @Talpha: 1/s");

		this->Talpha_ = value;
	}

	if ( dict.CheckOption("@Cbeta") == true )
		dict.ReadDouble("@Cbeta", Cbeta_);

	if ( dict.CheckOption("@Cgamma") == true )
	{
		double value; std::string units;
		dict.ReadMeasure("@Cgamma", value, units);
		if (units == "kg.m/kmol/s")		value = value;
		else return std::unexpected("Allowed units for @Cgamma: kg.m/kmol/s");

		this->Cgamma_ = value;
	}

	if ( dict.CheckOption("@Tgamma") == true )
	{
		double value; std::string units;
		dict.ReadMeasure("@Tgamma", value, units);
		if (units == "K")		value = value;
		else return std::unexpected("Allowed units for @Tgamma: 1/s");

		this->Tgamma_ = value;
	}	

	if ( dict.CheckOption("@Comega") == true )
	{
		double value; std::string units;
		dict.ReadMeasure("@Comega", value, units);
		if (units == "kg.m/kmol/sqrt(K)/s")	value = value;
		else return std::unexpected("Allowed units for @Comega: kg.m/kmol/sqrt(K)/s");

		this->Comega_ = value;
	}		

	if ( dict.CheckOption("@EtaColl") == true )
		dict.ReadDouble("@EtaColl", this->etaColl_);

	if ( dict.CheckOption("@Coxid") == true )
		dict.ReadDouble("@Coxid", this->Coxid_);

	if ( dict.CheckOption("@NucleationExponent") == true )
		dict.ReadDouble("@NucleationExponent", this->exp_l_);

	if ( dict.CheckOption("@SurfaceGrowthExponent1") == true )
		dict.ReadDouble("@SurfaceGrowthExponent1", this->exp_m_);	
	
	if ( dict.CheckOption("@SurfaceGrowthExponent2") == true )
		dict.ReadDouble("@SurfaceGrowthExponent2", this->exp_n_);	

    PrintSummary();
    return {};
}
#endif  // MOM_USE_DICTIONARY expected

} // namespace MOM
