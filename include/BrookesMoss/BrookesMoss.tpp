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
namespace MOM
{

// ============================================================================
// Constructor
// ============================================================================

template <ThermoMap Thermo> BrookesMoss<Thermo>::BrookesMoss(const Thermo& thermo) : thermo_(thermo)
{
    // -- Species indices for SetStatus / gas-coupling ----------------------
    // These are mechanism lookups, not user-configurable parameters.
    // index_C6H5_ and index_C6H6_ stay at their in-class default of -1
    // until SetBenzeneSpecies / SetPhenylRadicalSpecies are called.
    index_H_    = thermo_.IndexOfSpecies("H");
    index_H2_   = thermo_.IndexOfSpecies("H2");
    index_CO_   = thermo_.IndexOfSpecies("CO");
    index_OH_   = thermo_.IndexOfSpecies("OH");
    index_O2_   = thermo_.IndexOfSpecies("O2");
    index_C2H2_ = thermo_.IndexOfSpecies("C2H2");

    // -- Internal numerical floors (not user-configurable via Config) ------
    Ys_min_ = 1.e-30;
    bs_min_ = 1.e-30;

    // -- Apply all tunable parameter defaults from Config{} ----------------
    // Config{} is the single source of truth for every numerical constant.
    ApplyConfig(Config{});

    // -- Memory allocation (size depends on thermo_.NumberOfSpecies()) -----
    MemoryAllocation();
}

// ============================================================================
// MemoryAllocation
// ============================================================================

template <ThermoMap Thermo> void BrookesMoss<Thermo>::MemoryAllocation()
{
    this->ZeroSources();          // zeros source_all_ (base class)
    source_nucleation_.setZero(); // owned by BrookesMoss — zeroed explicitly
    source_coagulation_.setZero();
    source_growth_.setZero();
    source_oxidation_.setZero();

    this->omega_gas_            = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(thermo_.NumberOfSpecies()));
    omega_gas_oxidation_        = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(thermo_.NumberOfSpecies()));

    // Default initial moment values
    initial_moments_cache_(0) = 1.e-21; // Ys_init [-]
    initial_moments_cache_(1) = 1.e-12; // bs_init [m3/kg]
}

// ============================================================================
// SetStatus
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::SetStatus(double T, double P_Pa, const double* Y) noexcept
{
    const double cTot = this->template UpdateMixtureState<>(T, P_Pa, Y, thermo_);

    // Precursor concentration [kmol/m3]
    conc_prec_ = this->SpeciesConcentrationKmolM3(prec_index_, Y, cTot, thermo_);

    // Surface growth species concentration [kmol/m3]
    conc_sg_ = this->SpeciesConcentrationKmolM3(sg_index_, Y, cTot, thermo_);

    // Key species concentrations [kmol/m3]
    // NOTE: conc_H2_ is required by the BM-Hall nucleation rate (denominator);
    //       index_H2_ is always set in the constructor (soft lookup, −1 if absent).
    conc_OH_   = this->SpeciesConcentrationKmolM3(index_OH_, Y, cTot, thermo_);
    conc_O2_   = this->SpeciesConcentrationKmolM3(index_O2_, Y, cTot, thermo_);
    conc_H2_   = this->SpeciesConcentrationKmolM3(index_H2_, Y, cTot, thermo_);
    conc_C2H2_ = this->SpeciesConcentrationKmolM3(index_C2H2_, Y, cTot, thermo_);
    conc_C6H5_ = this->SpeciesConcentrationKmolM3(index_C6H5_, Y, cTot, thermo_);
    conc_C6H6_ = this->SpeciesConcentrationKmolM3(index_C6H6_, Y, cTot, thermo_);
}

// ============================================================================
// SetMoments
// ============================================================================

template <ThermoMap Thermo> void BrookesMoss<Thermo>::SetMoments(std::span<const double> m) noexcept
{
    assert(m.size() == static_cast<std::size_t>(Base::n_equations) &&
           "[BrookesMoss::SetMoments] expected exactly 2 moment values.");
    SetMoments(m[0], m[1]);
}

template <ThermoMap Thermo> void BrookesMoss<Thermo>::SetMoments(double Ys, double bs) noexcept
{
    Ys_ = std::max(Ys, 0.);
    bs_ = std::max(bs, 0.);
}

// ============================================================================
// SetPrecursors
// ============================================================================

template <ThermoMap Thermo> void BrookesMoss<Thermo>::SetPrecursors(std::string_view name)
{
    prec_species_ = std::string(name);
    prec_index_   = thermo_.IndexOfSpecies(prec_species_);

    if (prec_index_ < 0)
        throw std::runtime_error("[BrookesMoss] Precursor species not found in mechanism: " +
                                 prec_species_);

    prec_nc_ = static_cast<double>(thermo_.NumberOfCarbonAtoms(static_cast<unsigned>(prec_index_)));
    prec_nh_ = static_cast<double>(thermo_.NumberOfHydrogenAtoms(static_cast<unsigned>(prec_index_)));

    if (prec_nc_ <= 0.)
        throw std::runtime_error("[BrookesMoss] Precursor species has no carbon atoms: " +
                                 prec_species_);
}

// ============================================================================
// SetSurfaceGrowthSpecies
// ============================================================================

template <ThermoMap Thermo> void BrookesMoss<Thermo>::SetSurfaceGrowthSpecies(std::string_view name)
{
    sg_species_ = std::string(name);
    sg_index_   = thermo_.IndexOfSpecies(sg_species_);

    if (sg_index_ < 0)
        throw std::runtime_error("[BrookesMoss] Surface growth species not found in mechanism: " +
                                 sg_species_);

    sg_nc_ = static_cast<double>(thermo_.NumberOfCarbonAtoms(static_cast<unsigned>(sg_index_)));
    sg_nh_ = static_cast<double>(thermo_.NumberOfHydrogenAtoms(static_cast<unsigned>(sg_index_)));

    if (sg_nc_ <= 0.)
        throw std::runtime_error("[BrookesMoss] Surface growth species has no carbon atoms: " +
                                 sg_species_);
}

// ============================================================================
// SetBenzene
// ============================================================================

template <ThermoMap Thermo> void BrookesMoss<Thermo>::SetBenzeneSpecies(std::string_view name)
{
    const std::string species_name = std::string(name);
    const int species_index = thermo_.IndexOfSpecies(name);

    if (species_index < 0)
        throw std::runtime_error("[BrookesMoss] Benzene species not found in mechanism: " +
                                 species_name);

    const double nc = static_cast<double>(thermo_.NumberOfCarbonAtoms(static_cast<unsigned>(species_index)));
    const double nh = static_cast<double>(thermo_.NumberOfHydrogenAtoms(static_cast<unsigned>(species_index)));

    if (nc != 6. || nh != 6.)
        throw std::runtime_error("[BrookesMoss] Benzene species has wrong atomic composition: " +
                                 species_name);

    benzene_species_ = species_name;
    index_C6H6_     = species_index;
}

// ============================================================================
// SetPhenylRadical
// ============================================================================

template <ThermoMap Thermo> void BrookesMoss<Thermo>::SetPhenylRadicalSpecies(std::string_view name)
{
    const std::string species_name = std::string(name);
    const int species_index = thermo_.IndexOfSpecies(name);

    if (species_index < 0)
        throw std::runtime_error("[BrookesMoss] Phenyl radical species not found in mechanism: " +
                                 species_name);

    const double nc = static_cast<double>(thermo_.NumberOfCarbonAtoms(static_cast<unsigned>(species_index)));
    const double nh = static_cast<double>(thermo_.NumberOfHydrogenAtoms(static_cast<unsigned>(species_index)));

    if (nc != 6. || nh != 5.)
        throw std::runtime_error("[BrookesMoss] Phenyl radical species has wrong atomic composition: " +
                                 species_name);

    phenylradical_species_ = species_name;
    index_C6H5_            = species_index;
}

// ============================================================================
// SetGasClosureDummySpecies
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::SetGasClosureDummySpecies(std::string_view name)
{
    this->closure_dummy_species_ = std::string(name);

    if (this->closure_dummy_species_ == "none")
    {
        this->closure_dummy_index_           = -1;
        this->is_closure_dummy_species_ = false;
        return;
    }

    this->closure_dummy_index_ = thermo_.IndexOfSpecies(this->closure_dummy_species_);

    if (this->closure_dummy_index_ < 0)
        throw std::runtime_error("[BrookesMoss] Dummy species not found in mechanism: " +
                                 this->closure_dummy_species_);

    this->is_closure_dummy_species_ = true;
}

// ============================================================================
// SetNucleation (string variant)
// ============================================================================

template <ThermoMap Thermo> void BrookesMoss<Thermo>::SetNucleation(std::string_view label)
{
    if (label == "none")
    {
        nucleation_variant_ = NucleationVariant::Off;
    }
    else if (label == "BrookesMoss")
    {
        nucleation_variant_ = NucleationVariant::BrookesMoss;
        mwp_                = 144.;
    }
    else if (label == "BrookesMossHall")
    {
        nucleation_variant_ = NucleationVariant::BrookesMossHall;
        mwp_                = 1200.;
        CheckBrookesMossHallSpecies();
    }
    else
    {
        throw std::runtime_error("[BrookesMoss] Unknown nucleation model: " + std::string(label) +
                                 ". Allowed: none | BrookesMoss | BrookesMossHall");
    }
}

// ============================================================================
// SetOxidation (string variant)
// ============================================================================

template <ThermoMap Thermo> void BrookesMoss<Thermo>::SetOxidation(std::string_view label)
{
    if (label == "none")
    {
        oxidation_variant_ = OxidationVariant::Off;
    }
    else if (label == "BrookesMoss")
    {
        oxidation_variant_ = OxidationVariant::BrookesMoss;
        etaColl_           = 0.04;
        Coxid_             = 0.015;
    }
    else if (label == "BrookesMossHall")
    {
        oxidation_variant_ = OxidationVariant::BrookesMossHall;
        etaColl_           = 0.13;
        Coxid_             = 0.015;
        CheckBrookesMossHallSpecies();
    }
    else
    {
        throw std::runtime_error("[BrookesMoss] Unknown oxidation model: " + std::string(label) +
                                 ". Allowed: none | BrookesMoss | BrookesMossHall");
    }
}

// ============================================================================
// CheckBrookesMossHallSpecies
// ============================================================================

template <ThermoMap Thermo> void BrookesMoss<Thermo>::CheckBrookesMossHallSpecies()
{
    SetBenzeneSpecies(benzene_species_);
    SetPhenylRadicalSpecies(phenylradical_species_);

    // Adjust surface growth activation temperature for BMH
    Cgamma_ = 9000.6;
}

// ============================================================================
// CalculateSourceMoments
// ============================================================================

template <ThermoMap Thermo> void BrookesMoss<Thermo>::CalculateSourceMoments() noexcept
{
    this->ZeroSources();          // zeros source_all_ (base class)
    source_nucleation_.setZero(); // owned by BrookesMoss — zeroed explicitly
    source_coagulation_.setZero();
    source_growth_.setZero();
    source_oxidation_.setZero();

    // Reset intermediate quantities
    dMdt_nucleation_       = 0.;
    dMdt_nucleation_BMH_1_ = 0.;
    dMdt_nucleation_BMH_2_ = 0.;
    dMdt_surface_growth_   = 0.;
    dMdt_oxidation_        = 0.;
    dMdt_oxidation_OH_     = 0.;
    dMdt_oxidation_O2_     = 0.;

    // Sub-process source terms
    if (nucleation_variant_ != NucleationVariant::Off)
        NucleationSourceTerms();
    if (surface_growth_model_ > 0)
        SurfaceGrowthSourceTerms();
    if (oxidation_variant_ != OxidationVariant::Off)
        OxidationSourceTerms();
    if (coagulation_model_ > 0)
        CoagulationSourceTerms();

    // Sanitise and accumulate total source
    for (unsigned i = 0; i < 2u; ++i)
    {
        if (!std::isfinite(this->source_nucleation_(i)))
            this->source_nucleation_(i) = 0.;
        if (!std::isfinite(this->source_growth_(i)))
            this->source_growth_(i) = 0.;
        if (!std::isfinite(this->source_oxidation_(i)))
            this->source_oxidation_(i) = 0.;
        if (!std::isfinite(this->source_coagulation_(i)))
            this->source_coagulation_(i) = 0.;

        this->source_all_(i) = this->source_nucleation_(i) + this->source_growth_(i) +
                               this->source_oxidation_(i) + this->source_coagulation_(i);
    }

    if (this->gas_consumption_)
        CalculateOmegaGas();
}

// ============================================================================
// NucleationSourceTerms  (dispatcher)
// ============================================================================

template <ThermoMap Thermo> void BrookesMoss<Thermo>::NucleationSourceTerms()
{
    if (nucleation_variant_ == NucleationVariant::BrookesMoss)
        NucleationSourceTerms_BM();
    else if (nucleation_variant_ == NucleationVariant::BrookesMossHall)
        NucleationSourceTerms_BMH();
}

// ============================================================================
// NucleationSourceTerms_BM  — Brookes & Moss (1999)
// ============================================================================

template <ThermoMap Thermo> void BrookesMoss<Thermo>::NucleationSourceTerms_BM()
{
    if (conc_prec_ <= 0.)
        return;

    // Particle number formation rate [#/m3/s]
    const double dNdt =
        Calpha_ * this->Nav_kmol_ * std::pow(conc_prec_, exp_l_) * std::exp(-Talpha_ / this->T_);

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

template <ThermoMap Thermo> void BrookesMoss<Thermo>::NucleationSourceTerms_BMH()
{
    dMdt_nucleation_BMH_1_ = 0.;
    dMdt_nucleation_BMH_2_ = 0.;

    // Channel 1: C2H2-based inception
    if (conc_C2H2_ > 0.)
    {
        // Inception rate (#/m3/s)
        const double dNdt1 = 8.* Calpha1_BMH_ * this->Nav_kmol_/mwp_ * ( conc_C2H2_*conc_C2H2_*conc_C6H5_ / (conc_H2_ + 1.e-12) ) * std::exp(-Talpha1_BMH_ / this->T_);

        // Inception rate (kg/m3/s)
        dMdt_nucleation_BMH_1_ = mwp_ / this->Nav_kmol_ * dNdt1 ;
    }

    // Channel 2: C6H6-based inception
    if (conc_C6H6_ >= 0)
    {
        // Inception rate (#/m3/s)
        const double dNdt2 = 8.* Calpha2_BMH_ * this->Nav_kmol_/mwp_ * ( conc_C2H2_*conc_C6H6_*conc_C6H5_ / (conc_H2_ + 1.e-12) ) * std::exp(-Talpha2_BMH_ / this->T_);

        // Inception rate (kg/m3/s)
        dMdt_nucleation_BMH_2_ = mwp_ / this->Nav_kmol_ * dNdt2 ;
    }

    // Total nucleation mass rate [kg/m3/s]
    const double dMdt = dMdt_nucleation_BMH_1_ + dMdt_nucleation_BMH_2_;
    dMdt_nucleation_  = dMdt;

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

template <ThermoMap Thermo> void BrookesMoss<Thermo>::SurfaceGrowthSourceTerms()
{
    // Soot mass and number concentrations
    const double M = std::max(Ys_, 0.) * this->rho_;            // [kg/m3]
    const double N = std::max(bs_, 0.) * this->rho_ * Ns_norm_; // [#/m3]

    if (M <= 0. || N <= 0. || conc_sg_ <= 0.)
    {
        dMdt_surface_growth_ = 0.;
        return;
    }

    // Particle surface area per unit volume [m2/m3]:
    //   A = (pi * N)^(1/3) * (6 * M / rho_p)^(2/3)
    const double A =
        std::pow(this->pi_ * N, 1. / 3.) * std::pow(6. * M / this->rho_particle_, 2. / 3.);

    // Surface growth mass rate [kg/m3/s]
    const double dMdt =
        Cgamma_ * std::pow(conc_sg_, exp_m_) * std::exp(-Tgamma_ / this->T_) * std::pow(A, exp_n_);

    dMdt_surface_growth_ = dMdt;

    // Source for Ys [1/s]
    this->source_growth_(0) = dMdt / this->rho_;

    // No change in number density from surface growth
    this->source_growth_(1) = 0.;
}

// ============================================================================
// CoagulationSourceTerms  — free-molecular regime
// ============================================================================

template <ThermoMap Thermo> void BrookesMoss<Thermo>::CoagulationSourceTerms()
{
    const double N  = std::max(bs_, 0.) * this->rho_ * Ns_norm_; // [#/m3]
    const double dp = particle_diameter();                        // [m]

    if (N <= 0. || dp <= 0.)
        return;

    // Free-molecular coagulation kernel:
    //   dNdt = Cbeta * sqrt(24 * R * T / rho_p / Nav) * sqrt(dp) * N^2
    const double dNdt =
        Cbeta_ * std::sqrt(24. * this->Rgas_ * this->T_ / this->rho_particle_ / this->Nav_kmol_) *
        std::sqrt(dp) * N * N;

    // No change in soot mass from coagulation
    this->source_coagulation_(0) = 0.;

    // Source for bs [m3/kg/s]:  -dNdt / (rho * Ns_norm)
    this->source_coagulation_(1) = -1. / this->rho_ * (dNdt / Ns_norm_);
}

// ============================================================================
// OxidationSourceTerms  (dispatcher)
// ============================================================================

template <ThermoMap Thermo> void BrookesMoss<Thermo>::OxidationSourceTerms()
{
    if (oxidation_variant_ == OxidationVariant::BrookesMoss)
        OxidationSourceTerms_BM();
    else if (oxidation_variant_ == OxidationVariant::BrookesMossHall)
        OxidationSourceTerms_BMH();
}

// ============================================================================
// OxidationSourceTerms_BM  — Brookes & Moss (1999), OH attack
// ============================================================================

template <ThermoMap Thermo> void BrookesMoss<Thermo>::OxidationSourceTerms_BM()
{
    const double M = std::max(Ys_, 0.) * this->rho_;            // [kg/m3]
    const double N = std::max(bs_, 0.) * this->rho_ * Ns_norm_; // [#/m3]

    if (M <= 0. || N <= 0. || conc_OH_ <= 0.)
        return;

    // Particle surface area per unit volume [m2/m3]
    const double A =
        std::pow(this->pi_ * N, 1. / 3.) * std::pow(6. * M / this->rho_particle_, 2. / 3.);

    // Oxidation mass rate [kg/m3/s]
    //   dMdt = Coxid * Comega * etaColl * [OH] * sqrt(T) * A
    const double dMdt_oxid = Coxid_ * Comega_ * etaColl_ * conc_OH_ * std::sqrt(this->T_) * A;

    dMdt_oxidation_OH_ = dMdt_oxid;  // entire rate is OH-driven
    dMdt_oxidation_O2_ = 0.;
    dMdt_oxidation_    = dMdt_oxid;

    // Source for Ys [1/s]  (negative: consumption)
    this->source_oxidation_(0) = -dMdt_oxid / this->rho_;

    // No change in number density from oxidation (mass loss only)
    this->source_oxidation_(1) = 0.;
}

// ============================================================================
// OxidationSourceTerms_BMH  — Hall et al. (2016), OH + O2 channels
// ============================================================================

template <ThermoMap Thermo> void BrookesMoss<Thermo>::OxidationSourceTerms_BMH()
{
    const double M = std::max(Ys_, 0.) * this->rho_;            // [kg/m3]
    const double N = std::max(bs_, 0.) * this->rho_ * Ns_norm_; // [#/m3]

    if (M <= 0. || N <= 0.)
        return;

    // Particle surface area per unit volume [m2/m3]
    const double A =
        std::pow(this->pi_ * N, 1. / 3.) * std::pow(6. * M / this->rho_particle_, 2. / 3.);

    // Channel 1: OH oxidation  →  C + OH → CO + ½H₂
    dMdt_oxidation_OH_ = 0.;
    if (conc_OH_ > 0.)
        dMdt_oxidation_OH_ = Coxid_ * Comega_ * etaColl_ * conc_OH_ * std::sqrt(this->T_) * A;

    // Channel 2: O2 oxidation  →  C + O₂ → CO + ½O₂  (net: C + ½O₂ → CO)
    dMdt_oxidation_O2_ = 0.;
    if (conc_O2_ > 0.)
        dMdt_oxidation_O2_ = Comega2_BMH_ * conc_O2_ * std::exp(-Tomega2_BMH_ / this->T_) * A;

    const double dMdt_oxid = dMdt_oxidation_OH_ + dMdt_oxidation_O2_;
    dMdt_oxidation_ = dMdt_oxid;

    // Source for Ys [1/s]  (negative: consumption)
    this->source_oxidation_(0) = -dMdt_oxid / this->rho_;

    // No change in number density from oxidation
    this->source_oxidation_(1) = 0.;
}

// ============================================================================
// CalculateOmegaGas
// ============================================================================

template <ThermoMap Thermo> void BrookesMoss<Thermo>::CalculateOmegaGas() noexcept
{
    this->omega_gas_.setZero();
    omega_gas_oxidation_.setZero();
    if (!this->gas_consumption_)
        return;

    const int nsp = static_cast<int>(thermo_.NumberOfSpecies());

    // Helper: add mass source [kg/m3/s] directly for species idx
    auto AddMass = [&](int idx, double omega_kg_m3_s)
    {
        if (idx < 0 || idx >= nsp || !std::isfinite(omega_kg_m3_s))
            return;
        this->omega_gas_[idx] += omega_kg_m3_s;
    };

    // -- Nucleation coupling -----------------------------------------------
    //
    // BrookesMoss variant
    // -------------------
    // Net reaction:  Precursor  →  nC × [C]_soot  +  (nH/2) × H₂
    //
    // Per kg of soot formed, the stoichiometric factor is
    //   prec_per_kg_soot = 1 / (nC_prec × W_C)   [kmol_prec / kg_soot]
    //
    // Mass balance (pure hydrocarbon CₙHₘ, MW = 12n+m):
    //   precursor consumed  = (12n+m)/(n·12) kg/kg_soot
    //   H₂ produced        =  m / (n·12)     kg/kg_soot
    //   net gas removed    = 1.0              kg/kg_soot  → absorbed by dummy closure  ✓

    if (nucleation_variant_ == NucleationVariant::BrookesMoss)
    {
        if (dMdt_nucleation_ > 0. && prec_index_ >= 0 && prec_nc_ > 0.)
        {
            const double MW_prec       = thermo_.MolecularWeight(static_cast<unsigned>(prec_index_));
            const double prec_per_kg_soot = 1. / (prec_nc_ * this->WC_);

            AddMass(prec_index_, -dMdt_nucleation_ * prec_per_kg_soot * MW_prec);

            if (index_H2_ >= 0 && prec_nh_ > 0.)
            {
                const double MW_H2 = thermo_.MolecularWeight(static_cast<unsigned>(index_H2_));
                AddMass(index_H2_, dMdt_nucleation_ * prec_per_kg_soot * (prec_nh_ / 2.) * MW_H2);
            }
        }
    }

    // BM-Hall variant (Hall et al. 2016) — two-channel inception
    // -----------------------------------------------------------
    // The stoichiometry is derived per inception event (one soot nucleus of mass mwp_):
    //   events_per_s = dMdt / mwp_   [kmol_events / m³ / s]
    //
    // Channel 1:  2 C₂H₂  +  C₆H₅•  →  C₁₀H₇(soot)  +  H₂
    //   Atom check — C: 4+6=10 ✓   H: 4+5=9=7+2 ✓   MW: 52+77=129=127+2 ✓
    //   Consumed: 2 mol C₂H₂, 1 mol C₆H₅
    //   Produced: 1 mol H₂  (C₁₀H₇ absorbed by dummy closure)
    //
    // Channel 2:  C₂H₂  +  C₆H₆  +  C₆H₅•  →  C₁₄H₁₀(soot)  +  H•  +  H₂
    //   Atom check — C: 2+6+6=14 ✓  H: 2+6+5=13=10+1+2 ✓  MW: 26+78+77=181=178+1+2 ✓
    //   Consumed: 1 mol C₂H₂, 1 mol C₆H₆, 1 mol C₆H₅
    //   Produced: 1 mol H•, 1 mol H₂  (C₁₄H₁₀ absorbed by dummy closure)

    else if (nucleation_variant_ == NucleationVariant::BrookesMossHall)
    {
        // Channel 1: 2 C₂H₂ + C₆H₅ → C₁₀H₇(soot) + H₂
        if (dMdt_nucleation_BMH_1_ > 0. && mwp_ > 0.)
        {
            // [kmol_events/m³/s] — each event consumes exactly 2 C₂H₂ + 1 C₆H₅
            const double ev1 = dMdt_nucleation_BMH_1_ / mwp_;

            if (prec_index_ >= 0)   // C₂H₂ (precursor species)
            {
                const double MW_C2H2 = thermo_.MolecularWeight(static_cast<unsigned>(prec_index_));
                AddMass(prec_index_, -ev1 * 2. * MW_C2H2);
            }
            if (index_C6H5_ >= 0)
            {
                const double MW_C6H5 = thermo_.MolecularWeight(static_cast<unsigned>(index_C6H5_));
                AddMass(index_C6H5_, -ev1 * MW_C6H5);
            }
            if (index_H2_ >= 0)
            {
                const double MW_H2 = thermo_.MolecularWeight(static_cast<unsigned>(index_H2_));
                AddMass(index_H2_, ev1 * MW_H2);
            }
            // C₁₀H₇ soot product (MW=127) is absorbed by the dummy closure
        }

        // Channel 2: C₂H₂ + C₆H₆ + C₆H₅ → C₁₄H₁₀(soot) + H• + H₂
        if (dMdt_nucleation_BMH_2_ > 0. && mwp_ > 0.)
        {
            const double ev2 = dMdt_nucleation_BMH_2_ / mwp_;

            if (prec_index_ >= 0)   // C₂H₂ (precursor species)
            {
                const double MW_C2H2 = thermo_.MolecularWeight(static_cast<unsigned>(prec_index_));
                AddMass(prec_index_, -ev2 * MW_C2H2);
            }
            if (index_C6H6_ >= 0)
            {
                const double MW_C6H6 = thermo_.MolecularWeight(static_cast<unsigned>(index_C6H6_));
                AddMass(index_C6H6_, -ev2 * MW_C6H6);
            }
            if (index_C6H5_ >= 0)
            {
                const double MW_C6H5 = thermo_.MolecularWeight(static_cast<unsigned>(index_C6H5_));
                AddMass(index_C6H5_, -ev2 * MW_C6H5);
            }
            if (index_H_ >= 0)
            {
                const double MW_H = thermo_.MolecularWeight(static_cast<unsigned>(index_H_));
                AddMass(index_H_, ev2 * MW_H);
            }
            if (index_H2_ >= 0)
            {
                const double MW_H2 = thermo_.MolecularWeight(static_cast<unsigned>(index_H2_));
                AddMass(index_H2_, ev2 * MW_H2);
            }
            // C₁₄H₁₀ soot product (MW=178) is absorbed by the dummy closure
        }
    }

    // -- Surface growth coupling -------------------------------------------
    //
    // Net reaction:  sg_species  →  nC_sg × [C]_soot  +  (nH_sg/2) × H₂
    //
    // Same stoichiometric logic as BM nucleation.  For C₂H₂ (nC=2, nH=2):
    //   C₂H₂ consumed = 26/24 kg/kg_soot,  H₂ produced = 2/24 kg/kg_soot
    //   net gas removed = 1.0 kg/kg_soot  → absorbed by dummy closure  ✓

    if (dMdt_surface_growth_ > 0. && sg_index_ >= 0 && sg_nc_ > 0.)
    {
        const double MW_sg          = thermo_.MolecularWeight(static_cast<unsigned>(sg_index_));
        const double sg_per_kg_soot = 1. / (sg_nc_ * this->WC_);

        AddMass(sg_index_, -dMdt_surface_growth_ * sg_per_kg_soot * MW_sg);

        if (index_H2_ >= 0 && sg_nh_ > 0.)
        {
            const double MW_H2 = thermo_.MolecularWeight(static_cast<unsigned>(index_H2_));
            AddMass(index_H2_, dMdt_surface_growth_ * sg_per_kg_soot * (sg_nh_ / 2.) * MW_H2);
        }
    }

    // -- Oxidation coupling ------------------------------------------------
    //
    // Snapshot omega_gas_ before adding any oxidation terms.  The diff taken
    // after the block gives the oxidation-only gas contribution, stored in
    // omega_gas_oxidation_ for operator-splitting (GetOmegaGasOxidation).
    const Eigen::VectorXd omega_snap_before_ox = this->omega_gas_;

    // OH channel (both BM and BM-Hall):  C_soot + OH → CO + ½H₂
    //   Per kg soot oxidized:
    //     OH consumed = MW_OH / W_C            (17/12 kg/kg_soot)
    //     CO produced = MW_CO / W_C            (28/12 kg/kg_soot)
    //     H₂ produced = ½ × MW_H₂ / W_C       ( 1/12 kg/kg_soot)
    //     Net gas produced = +1 kg/kg_soot  ✓  → dummy closure absorbs −dMdt_oxid
    //
    // O₂ channel (BM-Hall only):  C_soot + O₂ → CO + ½O₂  (net: C + ½O₂ → CO)
    //   Per kg soot oxidized:
    //     O₂ consumed (net) = ½ × MW_O₂ / W_C (16/12 kg/kg_soot)
    //     CO produced       = MW_CO / W_C       (28/12 kg/kg_soot)
    //     Net gas produced = +1 kg/kg_soot  ✓

    if (dMdt_oxidation_OH_ > 0.)
    {
        const double inv_WC = 1. / this->WC_;

        if (index_OH_ >= 0)
        {
            const double MW_OH = thermo_.MolecularWeight(static_cast<unsigned>(index_OH_));
            AddMass(index_OH_, -dMdt_oxidation_OH_ * inv_WC * MW_OH);
        }
        if (index_CO_ >= 0)
        {
            const double MW_CO = thermo_.MolecularWeight(static_cast<unsigned>(index_CO_));
            AddMass(index_CO_, dMdt_oxidation_OH_ * inv_WC * MW_CO);
        }
        if (index_H2_ >= 0)
        {
            const double MW_H2 = thermo_.MolecularWeight(static_cast<unsigned>(index_H2_));
            AddMass(index_H2_, dMdt_oxidation_OH_ * inv_WC * 0.5 * MW_H2);
        }
    }

    if (dMdt_oxidation_O2_ > 0.)
    {
        const double inv_WC = 1. / this->WC_;

        if (index_O2_ >= 0)
        {
            const double MW_O2 = thermo_.MolecularWeight(static_cast<unsigned>(index_O2_));
            AddMass(index_O2_, -dMdt_oxidation_O2_ * inv_WC * 0.5 * MW_O2);
        }
        if (index_CO_ >= 0)
        {
            const double MW_CO = thermo_.MolecularWeight(static_cast<unsigned>(index_CO_));
            AddMass(index_CO_, dMdt_oxidation_O2_ * inv_WC * MW_CO);
        }
    }

    // Capture oxidation-only gas contribution (before closure modifies the dummy species).
    omega_gas_oxidation_ = this->omega_gas_ - omega_snap_before_ox;

    // -- Dummy species closure (enforce mass conservation) -----------------
    if (this->is_closure_dummy_species_)
    {
        // dummy_index_ must be set during setup — a negative value is a programming error.
        assert(this->closure_dummy_index_ >= 0 && "[BrookesMoss::CalculateOmegaGas] dummy_index_ not set.");
        double sum = 0.;
        for (int i = 0; i < nsp; ++i)
            if (i != this->closure_dummy_index_)
                sum += this->omega_gas_[i];
        this->omega_gas_[this->closure_dummy_index_] = -sum;

        // Apply the same closure to the oxidation-only vector so that
        // omega_gas_[k] == omega_gas_oxidation_[k] + omega_gas_without_oxidation[k]
        // holds for ALL k, including the dummy species.
        double sum_ox = 0.;
        for (int i = 0; i < nsp; ++i)
            if (i != this->closure_dummy_index_)
                sum_ox += omega_gas_oxidation_[i];
        omega_gas_oxidation_[this->closure_dummy_index_] = -sum_ox;
    }
}

// ============================================================================
// Property methods
// ============================================================================

template <ThermoMap Thermo> double BrookesMoss<Thermo>::volume_fraction() const noexcept
{
    return this->rho_ / this->rho_particle_ * std::max(Ys_, 0.);
}

template <ThermoMap Thermo> double BrookesMoss<Thermo>::mass_fraction() const noexcept
{
    return std::max(Ys_, 0.);
}

template <ThermoMap Thermo> double BrookesMoss<Thermo>::particle_number_density() const noexcept
{
    return std::max(bs_, 0.) * this->rho_ * Ns_norm_;
}

template <ThermoMap Thermo> double BrookesMoss<Thermo>::particle_diameter() const noexcept
{
    const double M = std::max(Ys_, 0.) * this->rho_;            // [kg/m3]
    const double N = std::max(bs_, 0.) * this->rho_ * Ns_norm_; // [#/m3]

    if (N <= 0. || M <= 0.)
        return 0.;

    return std::pow(6. * M / (this->pi_ * this->rho_particle_ * N), 1. / 3.);
}

template <ThermoMap Thermo> double BrookesMoss<Thermo>::collision_diameter() const noexcept
{
    // For BrookesMoss the collision diameter equals the mean particle diameter
    return particle_diameter();
}

template <ThermoMap Thermo> double BrookesMoss<Thermo>::specific_surface() const noexcept
{
    const double dp = particle_diameter();
    const double N  = particle_number_density();
    return this->pi_ * dp * dp * N;
}

template <ThermoMap Thermo> double BrookesMoss<Thermo>::diffusion_coefficient() const noexcept
{
    return this->mu_ / this->schmidt_number_;
}

template <ThermoMap Thermo> double BrookesMoss<Thermo>::number_primary_particles() const noexcept
{
    return 0.;
}

// ============================================================================
// PrintSummary
// ============================================================================

template <ThermoMap Thermo> void BrookesMoss<Thermo>::PrintSummary() const
{
    // Helper lambdas — no heap allocation, resolved at compile time
    const auto nuc_str = [](NucleationVariant v) -> const char* {
        switch (v) {
            case NucleationVariant::Off:             return "off";
            case NucleationVariant::BrookesMoss:     return "BrookesMoss";
            case NucleationVariant::BrookesMossHall: return "BM-Hall (Hall 2016)";
            default:                                  return "unknown";
        }
    };
    const auto oxid_str = [](OxidationVariant v) -> const char* {
        switch (v) {
            case OxidationVariant::Off:             return "off";
            case OxidationVariant::BrookesMoss:     return "BrookesMoss";
            case OxidationVariant::BrookesMossHall: return "BM-Hall (Hall 2016)";
            default:                                  return "unknown";
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
        << "                        Brookes-Moss Soot Model Summary\n"
        << "------------------------------------------------------------------------------------------\n"
        << " * Active: " << (this->is_active_ ? "yes" : "no") << "\n"
        << "\n"
        << " [Physical properties]\n"
        << "    + Soot density (kg/m3):          " << this->rho_particle_ << "\n"
        << "    + Soot particle MW (kg/kmol):    " << mwp_ << "\n"
        << "    + Particle diameter (m):          " << dp_ << "\n"
        << "    + Ns normalisation (#/m3):        " << Ns_norm_ << "\n"
        << "\n"
        << " [Species]\n"
        << "    + Precursor:       " << prec_species_ << "  (nC=" << prec_nc_ << ", nH=" << prec_nh_ << ")\n"
        << "    + Surface growth:  " << sg_species_   << "  (nC=" << sg_nc_   << ", nH=" << sg_nh_   << ")\n"
        << "    + C6H6 index (-):  " << index_C6H6_ << "  (< 0 = not found; required for BM-Hall nucleation ch.2)\n"
        << "    + C6H5 index (-):  " << index_C6H5_ << "  (< 0 = not found; required for BM-Hall nucleation)\n"
        << "    + H2   index (-):  " << index_H2_   << "  (< 0 = not found; required for nucleation/SG/oxidation coupling)\n"
        << "    + H    index (-):  " << index_H_    << "  (< 0 = not found; required for BM-Hall nucleation ch.2)\n"
        << "    + CO   index (-):  " << index_CO_   << "  (< 0 = not found; required for oxidation gas coupling)\n"
        << "\n"
        << " [Processes]\n"
        << "    + Nucleation:      " << static_cast<int>(nucleation_variant_) << "  (" << nuc_str(nucleation_variant_) << ")\n"
        << "    + Surface growth:  " << surface_growth_model_ << "\n"
        << "    + Oxidation:       " << static_cast<int>(oxidation_variant_) << "  (" << oxid_str(oxidation_variant_) << ")\n"
        << "    + Coagulation:     " << coagulation_model_ << "\n"
        << "\n"
        << " [Standard kinetics]\n"
        << "    Nucleation:\n"
        << "      Calpha  = " << Calpha_ << " [1/s]\n"
        << "      Talpha  = " << Talpha_ << " [K]\n"
        << "      exp_l   = " << exp_l_ << " [-]\n"
        << "    Surface growth:\n"
        << "      Cgamma  = " << Cgamma_ << " [kg*m/kmol]\n"
        << "      Tgamma  = " << Tgamma_ << " [K]\n"
        << "      exp_m   = " << exp_m_ << " [-]\n"
        << "      exp_n   = " << exp_n_ << " [-]\n"
        << "    Coagulation:\n"
        << "      Cbeta   = " << Cbeta_ << " [-]\n"
        << "    Oxidation:\n"
        << "      Comega  = " << Comega_ << " [kg*m/kmol/s/sqrt(K)]\n"
        << "      etaColl = " << etaColl_ << " [-]\n"
        << "      Coxid   = " << Coxid_ << " [-]\n";

    if (nucleation_variant_ == NucleationVariant::BrookesMossHall ||
        oxidation_variant_  == OxidationVariant::BrookesMossHall)
    {
        std::cout
            << "\n"
            << " [BM-Hall extended kinetics]\n"
            << "    Nucleation (phenyl-based):\n"
            << "      Calpha1_BMH = " << Calpha1_BMH_ << " [kg*m3/kmol2/s]\n"
            << "      Talpha1_BMH = " << Talpha1_BMH_ << " [K]\n"
            << "      Calpha2_BMH = " << Calpha2_BMH_ << " [kg*m3/kmol2/s]\n"
            << "      Talpha2_BMH = " << Talpha2_BMH_ << " [K]\n"
            << "    Oxidation (OH-extended):\n"
            << "      Comega2_BMH = " << Comega2_BMH_ << " [kg*m/kmol/s/sqrt(K)]\n"
            << "      Tomega2_BMH = " << Tomega2_BMH_ << " [K]\n";
    }

    std::cout
        << "\n"
        << " [Transport & radiation]\n"
        << "    + Schmidt number (-):       " << this->schmidt_number_ << "\n"
        << "    + Thermophoretic model:     " << this->thermophoretic_model() << "  (" << thermo_str(this->thermophoretic_model_) << ")\n"
        << "    + Gas consumption:          " << (this->gas_consumption_ ? "yes" : "no") << "\n"
        << "    + Radiative heat transfer:  " << (this->radiative_heat_transfer_ ? "yes" : "no") << "\n"
        << "    + Planck coeff. model:      " << static_cast<int>(this->planck_model_) << "  (" << planck_str(this->planck_model_) << ")\n"
        << "    + Closure dummy species:    " << (this->is_closure_dummy_species_ ? this->closure_dummy_species_ : "none") << "\n"
        << "\n"
        << " [Numerical floors]\n"
        << "    + Ys_min (-):  " << Ys_min_ << "\n"
        << "    + bs_min (-):  " << bs_min_ << "\n"
        << "\n"
        << " [Debug]\n"
        << "    + Debug mode:  " << (is_debug_mode_ ? "yes" : "no") << "\n"
        << "------------------------------------------------------------------------------------------\n";
}

// ============================================================================
// ApplyConfig  (private — all parameter assignments, no I/O)
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::ApplyConfig(const Config& cfg)
{
    this->is_active_ = cfg.is_active;

    const Config defaults{};
    const bool is_bmh_nucleation = cfg.nucleation_model == static_cast<int>(NucleationVariant::BrookesMossHall);
    const bool is_bmh_oxidation  = cfg.oxidation_model  == static_cast<int>(OxidationVariant::BrookesMossHall);
    const bool is_bmh_active     = is_bmh_nucleation || is_bmh_oxidation;

    double soot_particle_mw = cfg.soot_particle_mw_kg_kmol;
    double cgamma           = cfg.cgamma;
    double eta_coll         = cfg.eta_coll;

    // Keep explicit user values, but make BM-Hall config setup match the string setters
    // when the Brookes-Moss defaults were left untouched.
    if (is_bmh_nucleation && soot_particle_mw == defaults.soot_particle_mw_kg_kmol)
        soot_particle_mw = 1200.;
    if (is_bmh_active && cgamma == defaults.cgamma)
        cgamma = 9000.6;
    if (is_bmh_oxidation && eta_coll == defaults.eta_coll)
        eta_coll = 0.13;

    // -- Gas species -------------------------------------------------------
    this->SetPrecursors(cfg.precursors_species);
    this->SetSurfaceGrowthSpecies(cfg.surface_growth_species);
    // BM-Hall species are only needed when the BM-Hall variant is active;
    // avoid spurious "species not found" warnings for basic BrookesMoss runs.
    if (cfg.nucleation_model == 2 || cfg.oxidation_model == 2)
    {
        this->SetBenzeneSpecies(cfg.benzene_species);
        this->SetPhenylRadicalSpecies(cfg.phenylradical_species);
    }
    this->SetGasClosureDummySpecies(cfg.gas_closure_dummy_species);
    this->SetGasConsumption(cfg.gas_consumption);

    // -- Process models ----------------------------------------------------
    this->SetNucleation(cfg.nucleation_model);
    this->SetSurfaceGrowth(cfg.surface_growth_model);
    this->SetOxidation(cfg.oxidation_model);
    SetCoagulation(cfg.coagulation_model);
    this->SetThermophoreticModel(cfg.thermophoretic_model);

    // -- Particle properties -----------------------------------------------
    this->SetParticleDensity(cfg.soot_density_kg_m3);
    this->SetDp(cfg.soot_particle_diameter_m);
    this->SetMWp(soot_particle_mw);
    this->SetNsNorm(cfg.ns_norm);

    // -- Kinetic constants (direct member assignment — no setters) ----------
    Calpha_      = cfg.calpha;
    Talpha_      = cfg.talpha;
    Cbeta_       = cfg.cbeta;
    Cgamma_      = cgamma;
    Tgamma_      = cfg.tgamma;
    Comega_      = cfg.comega;
    etaColl_     = eta_coll;
    Coxid_       = cfg.coxid;
    exp_l_       = cfg.nucleation_exponent;
    exp_m_       = cfg.sg_exponent1;
    exp_n_       = cfg.sg_exponent2;
    Calpha1_BMH_ = cfg.calpha1_bmh;
    Calpha2_BMH_ = cfg.calpha2_bmh;
    Comega2_BMH_ = cfg.comega2_bmh;
    Talpha1_BMH_ = cfg.talpha1_bmh;
    Talpha2_BMH_ = cfg.talpha2_bmh;
    Tomega2_BMH_ = cfg.tomega2_bmh;

    // -- Radiation / transport ---------------------------------------------
    this->SetRadiativeHeatTransfer(cfg.radiative_heat_transfer);
    this->SetPlanckAbsorptionCoefficient(cfg.planck_coefficient);
    this->SetSchmidtNumber(cfg.schmidt_number);

    // -- Debug -------------------------------------------------------------
    is_debug_mode_ = cfg.debug_mode;
}

// ============================================================================
// SetupFromConfig  (public — delegates to ApplyConfig, then prints summary)
// ============================================================================

template <ThermoMap Thermo>
void BrookesMoss<Thermo>::SetupFromConfig(const Config& cfg)
{
    ApplyConfig(cfg);
    PrintSummary();
}

#if defined(MOM_USE_DICTIONARY)
// ============================================================================
// ParseConfig — OpenSMOKE++ dictionary → BrookesMoss::Config
// ============================================================================

template <ThermoMap Thermo>
template <typename DictType>
std::expected<typename BrookesMoss<Thermo>::Config, std::string>
BrookesMoss<Thermo>::ParseConfig(DictType& dict)
{
    // SetGrammar validates the dictionary against the grammar rules defined in
    // BrookesMoss_Grammar::DefineRules().  On any violation (missing mandatory
    // keyword, unknown keyword, duplicate, conflicting keyword) it calls
    // Dictionary::ErrorMessage() which throws std::runtime_error.
    // That exception is intentionally NOT caught here so it propagates to the
    // caller and stops the simulation.
    BrookesMoss_Grammar grammar;
    dict.SetGrammar(grammar);

    Config cfg;

    if (dict.CheckOption("@BrookesMoss"))
        dict.ReadBool("@BrookesMoss", cfg.is_active);

    if (dict.CheckOption("@NucleationModel"))
    {
        std::string name;
        dict.ReadString("@NucleationModel", name);
        if      (name == "0" || name == "none")           cfg.nucleation_model = 0;
        else if (name == "1" || name == "BrookesMoss")    cfg.nucleation_model = 1;
        else if (name == "2" || name == "BrookesMossHall") cfg.nucleation_model = 2;
        else return std::unexpected(std::string{
            "@NucleationModel: allowed: 0=none, 1=BrookesMoss, 2=BrookesMossHall"});
    }

    if (dict.CheckOption("@SurfaceGrowthModel"))
        dict.ReadInt("@SurfaceGrowthModel", cfg.surface_growth_model);

    if (dict.CheckOption("@OxidationModel"))
    {
        std::string name;
        dict.ReadString("@OxidationModel", name);
        if      (name == "0" || name == "none")           cfg.oxidation_model = 0;
        else if (name == "1" || name == "BrookesMoss")    cfg.oxidation_model = 1;
        else if (name == "2" || name == "BrookesMossHall") cfg.oxidation_model = 2;
        else return std::unexpected(std::string{
            "@OxidationModel: allowed: 0=none, 1=BrookesMoss, 2=BrookesMossHall"});
    }

    if (dict.CheckOption("@CoagulationModel"))    dict.ReadInt("@CoagulationModel",    cfg.coagulation_model);
    if (dict.CheckOption("@ThermophoreticModel")) dict.ReadInt("@ThermophoreticModel", cfg.thermophoretic_model);

    if (dict.CheckOption("@Precursors"))            dict.ReadString("@Precursors",            cfg.precursors_species);
    if (dict.CheckOption("@SurfaceGrowthSpecies"))  dict.ReadString("@SurfaceGrowthSpecies",  cfg.surface_growth_species);
    if (dict.CheckOption("@GasClosureDummySpecies"))    dict.ReadString("@GasClosureDummySpecies", cfg.gas_closure_dummy_species);
    if (dict.CheckOption("@Benzene"))                   dict.ReadString("@Benzene",            cfg.benzene_species);
    if (dict.CheckOption("@PhenylRadical"))             dict.ReadString("@PhenylRadical",      cfg.phenylradical_species);

    if (dict.CheckOption("@GasConsumption"))
        dict.ReadBool("@GasConsumption", cfg.gas_consumption);

    if (dict.CheckOption("@SootDensity"))
    {
        double v; std::string u;
        dict.ReadMeasure("@SootDensity", v, u);
        if (u == "kg/m3")      cfg.soot_density_kg_m3 = v;
        else if (u == "g/cm3") cfg.soot_density_kg_m3 = v * 1000.;
        else return std::unexpected(std::string{"@SootDensity: allowed units: kg/m3 | g/cm3"});
    }

    if (dict.CheckOption("@SootParticleDiameter"))
    {
        double v; std::string u;
        dict.ReadMeasure("@SootParticleDiameter", v, u);
        if (u == "m")       cfg.soot_particle_diameter_m = v;
        else if (u == "mm") cfg.soot_particle_diameter_m = v * 1.e-3;
        else if (u == "nm") cfg.soot_particle_diameter_m = v * 1.e-9;
        else return std::unexpected(std::string{"@SootParticleDiameter: allowed units: m | mm | nm"});
    }

    if (dict.CheckOption("@SootParticleMolecularWeight"))
    {
        double v; std::string u;
        dict.ReadMeasure("@SootParticleMolecularWeight", v, u);
        if (u == "kg/kmol" || u == "g/mol") cfg.soot_particle_mw_kg_kmol = v;
        else return std::unexpected(std::string{
            "@SootParticleMolecularWeight: allowed units: kg/kmol | g/mol"});
    }

    if (dict.CheckOption("@RadiativeHeatTransfer"))
        dict.ReadBool("@RadiativeHeatTransfer", cfg.radiative_heat_transfer);

    if (dict.CheckOption("@PlanckCoefficient"))
        dict.ReadString("@PlanckCoefficient", cfg.planck_coefficient);

    if (dict.CheckOption("@SchmidtNumber"))
        dict.ReadDouble("@SchmidtNumber", cfg.schmidt_number);

    if (dict.CheckOption("@NsNorm"))
    {
        double v; std::string u;
        dict.ReadMeasure("@NsNorm", v, u);
        if (u == "#/m3")       cfg.ns_norm = v;
        else if (u == "#/cm3") cfg.ns_norm = v * 1.e6;
        else return std::unexpected(std::string{"@NsNorm: allowed units: #/m3 | #/cm3"});
    }

    if (dict.CheckOption("@Calpha"))
    {
        double v; std::string u;
        dict.ReadMeasure("@Calpha", v, u);
        if (u != "1/s") return std::unexpected(std::string{"@Calpha: allowed units: 1/s"});
        cfg.calpha = v;
    }

    if (dict.CheckOption("@Talpha"))
    {
        double v; std::string u;
        dict.ReadMeasure("@Talpha", v, u);
        if (u != "K") return std::unexpected(std::string{"@Talpha: allowed units: K"});
        cfg.talpha = v;
    }

    if (dict.CheckOption("@Cbeta")) dict.ReadDouble("@Cbeta", cfg.cbeta);

    if (dict.CheckOption("@Cgamma"))
    {
        double v; std::string u;
        dict.ReadMeasure("@Cgamma", v, u);
        if (u != "kg*m/kmol/s") return std::unexpected(std::string{"@Cgamma: allowed units: kg*m/kmol/s"});
        cfg.cgamma = v;
    }

    if (dict.CheckOption("@Tgamma"))
    {
        double v; std::string u;
        dict.ReadMeasure("@Tgamma", v, u);
        if (u != "K") return std::unexpected(std::string{"@Tgamma: allowed units: K"});
        cfg.tgamma = v;
    }

    if (dict.CheckOption("@Comega"))
    {
        double v; std::string u;
        dict.ReadMeasure("@Comega", v, u);
        if (u != "kg*m/kmol/sqrt(K)/s")
            return std::unexpected(std::string{"@Comega: allowed units: kg*m/kmol/sqrt(K)/s"});
        cfg.comega = v;
    }

    if (dict.CheckOption("@EtaColl"))  dict.ReadDouble("@EtaColl",  cfg.eta_coll);
    if (dict.CheckOption("@Coxid"))    dict.ReadDouble("@Coxid",    cfg.coxid);

    if (dict.CheckOption("@NucleationExponent"))     dict.ReadDouble("@NucleationExponent",     cfg.nucleation_exponent);
    if (dict.CheckOption("@SurfaceGrowthExponent1")) dict.ReadDouble("@SurfaceGrowthExponent1", cfg.sg_exponent1);
    if (dict.CheckOption("@SurfaceGrowthExponent2")) dict.ReadDouble("@SurfaceGrowthExponent2", cfg.sg_exponent2);
    
    if (dict.CheckOption("@Calpha1"))
    {
        double v; std::string u;
        dict.ReadMeasure("@Calpha1", v, u);
        if (u != "kg*m3/kmol2/s")
            return std::unexpected(std::string{"@Calpha1: allowed units: kg*m3/kmol2/s"});
        cfg.calpha1_bmh = v;
    }

    if (dict.CheckOption("@Talpha1"))
    {
        double v; std::string u;
        dict.ReadMeasure("@Talpha1", v, u);
        if (u != "K") return std::unexpected(std::string{"@Talpha1: allowed units: K"});
        cfg.talpha1_bmh = v;
    }      

    if (dict.CheckOption("@Calpha2"))
    {
        double v; std::string u;
        dict.ReadMeasure("@Calpha2", v, u);
        if (u != "kg*m3/kmol2/s")
            return std::unexpected(std::string{"@Calpha2: allowed units: kg*m3/kmol2/s"});
        cfg.calpha2_bmh = v;
    }    

    if (dict.CheckOption("@Talpha2"))
    {
        double v; std::string u;
        dict.ReadMeasure("@Talpha2", v, u);
        if (u != "K") return std::unexpected(std::string{"@Talpha2: allowed units: K"});
        cfg.talpha2_bmh = v;
    }    

    if (dict.CheckOption("@Comega2"))
    {
        double v; std::string u;
        dict.ReadMeasure("@Comega2", v, u);
        if (u != "kg*m/kmol/s/sqrt(K)")
            return std::unexpected(std::string{"@Comega2: allowed units: kg*m/kmol/s/sqrt(K)"});
        cfg.comega2_bmh = v;
    }        
    
    if (dict.CheckOption("@Tomega2"))
    {
        double v; std::string u;
        dict.ReadMeasure("@Tomega2", v, u);
        if (u != "K") return std::unexpected(std::string{"@Tomega2: allowed units: K"});
        cfg.tomega2_bmh = v;
    }

    if (dict.CheckOption("@DebugMode"))
        dict.ReadBool("@DebugMode", cfg.debug_mode);

    // Cross-field validation: BM-Hall requires benzene and phenyl radical species.
    if (cfg.nucleation_model == 2 || cfg.oxidation_model == 2)
    {
        if (!dict.CheckOption("@Benzene"))
            return std::unexpected(std::string{
                "@Benzene is required when @NucleationModel or @OxidationModel is BrookesMossHall"});
        if (!dict.CheckOption("@PhenylRadical"))
            return std::unexpected(std::string{
                "@PhenylRadical is required when @NucleationModel or @OxidationModel is BrookesMossHall"});
    }

    return cfg;
}
#endif // MOM_USE_DICTIONARY

} // namespace MOM
