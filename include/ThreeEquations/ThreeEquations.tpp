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

namespace MOM
{

// ============================================================================
// File-local helpers  (anonymous namespace — not exported)
// ============================================================================
namespace
{

inline double Heaviside(double x) noexcept
{
    return (x > 0.) ? 1. : 0.;
}

inline double SmoothHeaviside(double x, double eps) noexcept
{
    const double epsSafe = std::max(eps, 1.e-300);
    return 0.5 * (1.0 + std::tanh(x / epsSafe));
}

// Gas constant [J/mol/K] — equals MomentMethodBase::Rgas_mol_ (= Rgas_ / 1000).
// Used here by the non-member Arrhenius_cgs helper; member functions access this->Rgas_mol_.
static constexpr double kRgas_mol = 8.31446261815324;

/// Arrhenius rate in CGS: A in [cm3/mol/s] or [1/s], E in [J/mol].
inline double Arrhenius_cgs(double A, double n, double E_J_mol, double T) noexcept
{
    return A * std::pow(T, n) * std::exp(-E_J_mol / (kRgas_mol * T));
}

template <ThermoMap Thermo>
int RequireSpeciesIndexThreeEquations(const Thermo& thermo,
                                      std::string_view species,
                                      std::string_view context)
{
    const int index = thermo.IndexOfSpecies(species);
    if (index < 0)
        throw std::runtime_error("[ThreeEquations] " + std::string(context) +
                                 ": required species not found: " + std::string(species));
    return index;
}

} // anonymous namespace

// ============================================================================
// Constructor
// ============================================================================

template <ThermoMap Thermo>
ThreeEquations<Thermo>::ThreeEquations(const Thermo& thermo) : thermo_(thermo)
{
    // -- Species indices for SetState / HACA kinetics ---------------------
    // Required species: throws at construction if absent (ThreeEquations
    // cannot function without H, OH, O2, H2, H2O, C2H2).
    index_H_    = RequireSpeciesIndexThreeEquations(thermo_, "H",    "constructor");
    index_OH_   = RequireSpeciesIndexThreeEquations(thermo_, "OH",   "constructor");
    index_O2_   = RequireSpeciesIndexThreeEquations(thermo_, "O2",   "constructor");
    index_H2_   = RequireSpeciesIndexThreeEquations(thermo_, "H2",   "constructor");
    index_H2O_  = RequireSpeciesIndexThreeEquations(thermo_, "H2O",  "constructor");
    index_CO_   = thermo_.IndexOfSpecies("CO");   // optional — not present in all mechanisms
    index_C2H2_ = RequireSpeciesIndexThreeEquations(thermo_, "C2H2", "constructor");

    // -- Apply all tunable parameter defaults from Config{} ----------------
    // Config{} is the single source of truth for every numerical constant.
    // NOTE: ApplyConfig sets rho_particle_ via SetParticleDensity BEFORE
    //       calling SetPAH, so Precalculations() uses the correct density.
    ApplyConfig(Config{});

    // -- Memory allocation (size depends on thermo_.NumberOfSpecies()) -----
    MemoryAllocation();
}

// ============================================================================
// MemoryAllocation
// ============================================================================

template <ThermoMap Thermo> void ThreeEquations<Thermo>::MemoryAllocation()
{
    this->ZeroSources();          // zeros source_all_ (base class)
    source_nucleation_.setZero(); // owned by ThreeEquations — zeroed explicitly
    source_coagulation_.setZero();
    source_condensation_.setZero();
    source_growth_.setZero();
    source_oxidation_.setZero();

    // Gas source vector — dynamic size (depends on mechanism).
    this->omega_gas_     = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(thermo_.NumberOfSpecies()));
    omega_gas_oxidation_ = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(thermo_.NumberOfSpecies()));
}

// ============================================================================
// Precalculations  (call whenever rho_soot, PAH species, or Ns_min changes)
// ============================================================================

template <ThermoMap Thermo> void ThreeEquations<Thermo>::Precalculations()
{
    const double rho_soot = this->rho_particle_;

    // -- PAH identification ------------------------------------------------
    pah_index_ = RequireSpeciesIndexThreeEquations(thermo_, pah_species_, "Precalculations");
    const auto pah_index_u = static_cast<unsigned>(pah_index_);
    ncpah_     = static_cast<double>(thermo_.NumberOfCarbonAtoms(pah_index_u));
    nhpah_     = static_cast<double>(thermo_.NumberOfHydrogenAtoms(pah_index_u));

    // -- PAH mass and geometry ---------------------------------------------
    mwpah_ = thermo_.MolecularWeight(pah_index_u); // [kg/kmol]
    if (is_simplified_pah_mass_)
        mwpah_ = ncpah_ * this->WC_;

    vpah_ = mwpah_ / rho_soot / this->Nav_kmol_;       // [m3]
    dpah_ = this->SphereDiameter(vpah_);               // [m]
    spah_ = this->SphereSurface(dpah_);                // [m2]
    mpah_ = mwpah_ / this->Nav_kmol_;                  // [kg/#]

    // -- Dimer geometry ----------------------------------------------------
    vdim_ = 2. * vpah_;
    ddim_ = this->SphereDiameter(vdim_);
    sdim_ = this->SphereSurface(ddim_);

    // -- Nucleus geometry  (dimer + dimer) ---------------------------------
    vnucl_ = 2. * vdim_;
    dnucl_ = this->SphereDiameter(vnucl_);
    snucl_ = this->SphereSurface(dnucl_);

    // -- C2 pair geometry --------------------------------------------------
    vc2_ = (this->WC_ / rho_soot / this->Nav_kmol_) * 2.;
    dc2_ = this->SphereDiameter(vc2_);
    sc2_ = this->SphereSurface(dc2_);

    // -- Numerical floors --------------------------------------------------
    vs_min_ = vnucl_;
    Ss_min_ = Ns_min_ * snucl_;
    Ys_min_ = (Ns_min_ * vnucl_) * rho_soot / 1.0; // reference gas density 1 kg/m3

    // -- Initial moments cache ---------------------------------------------
    {
        const double Ns0          = Ns_min_;
        const double Ys0          = rho_soot / 1. * Ns0 * vnucl_;
        const double Ss0          = Ns0 * snucl_;
        initial_moments_cache_(0) = Ys0;
        initial_moments_cache_(1) = Ns0 / N0_scaling_;
        initial_moments_cache_(2) = Ss0;
    }
}

// ============================================================================
// Setters that trigger recalculation
// ============================================================================

template <ThermoMap Thermo> void ThreeEquations<Thermo>::SetNsMinimum(double v) noexcept
{
    Ns_min_ = v;
    Ss_min_ = Ns_min_ * snucl_;
    Ys_min_ = (Ns_min_ * vnucl_) * this->rho_particle_ / 1.0;
    vs_min_ = vnucl_;
    // Update initial moments cache to reflect new floor
    initial_moments_cache_(0) = Ys_min_;
    initial_moments_cache_(1) = Ns_min_ / N0_scaling_;
    initial_moments_cache_(2) = Ss_min_;
}

template <ThermoMap Thermo> void ThreeEquations<Thermo>::SetPAH(std::string_view name)
{
    pah_species_ = std::string(name);
    Precalculations();
}

template <ThermoMap Thermo>
void ThreeEquations<Thermo>::SetGasClosureDummySpecies(std::string_view name)
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
        throw std::runtime_error("[ThreeEquations] Dummy species not found: " + this->closure_dummy_species_);

    if (this->closure_dummy_index_ == pah_index_)
        throw std::runtime_error("[ThreeEquations] Dummy species cannot be the PAH precursor.");

    if (this->closure_dummy_index_ == index_H_ || this->closure_dummy_index_ == index_H2_ ||
        this->closure_dummy_index_ == index_O2_ || this->closure_dummy_index_ == index_OH_ ||
        this->closure_dummy_index_ == index_H2O_ || this->closure_dummy_index_ == index_C2H2_)
        throw std::runtime_error(
            "[ThreeEquations] Dummy species cannot be H, H2, O2, OH, H2O, or C2H2.");

    this->is_closure_dummy_species_ = true;
}

template <ThermoMap Thermo>
void ThreeEquations<Thermo>::SetSurfaceChemistryModel(std::string_view label)
{
    if (label == "RC-PAH" || label == "rc-pah" || label == "RCPAH" || label == "rcpah")
        surface_chem_model_ = SurfaceChemistryModel::RCPAH;
    else if (label == "HMOM" || label == "hmom")
        surface_chem_model_ = SurfaceChemistryModel::HMOM;
    else
        throw std::runtime_error("[ThreeEquations] @SurfaceChemistryModel: allowed: RC-PAH | HMOM");
}

template <ThermoMap Thermo>
void ThreeEquations<Thermo>::SetStickingCoefficientModel(std::string_view label)
{
    if (label == "constant")
        sticking_model_ = StickingModel::Constant;
    else if (label == "pah-dependent")
        sticking_model_ = StickingModel::PAH4;
    else
        throw std::runtime_error(
            "[ThreeEquations] @StickingCoefficientModel: allowed: constant | pah-dependent");
}

// ============================================================================
// SetState  — injects thermodynamic state from the CFD solver
// ============================================================================

template <ThermoMap Thermo>
void ThreeEquations<Thermo>::SetState(double T, double P_Pa, const double* Y) noexcept
{
    // Gas constant: R [J/kmol/K] = kB * Nav_kmol
    const double R_J_kmol = this->kB_ * this->Nav_kmol_;
    const double cTot = this->template UpdateMixtureState<>(T, P_Pa, Y, thermo_, R_J_kmol);

    // Mass fractions of HACA-relevant species (for active-site cutoff)
    mass_fraction_OH_ = std::max(Y[index_OH_], 0.);
    mass_fraction_H_  = std::max(Y[index_H_], 0.);

    conc_PAH_  = this->SpeciesConcentrationKmolM3(pah_index_, Y, cTot, thermo_);
    conc_H_    = this->SpeciesConcentrationKmolM3(index_H_, Y, cTot, thermo_);
    conc_OH_   = this->SpeciesConcentrationKmolM3(index_OH_, Y, cTot, thermo_);
    conc_O2_   = this->SpeciesConcentrationKmolM3(index_O2_, Y, cTot, thermo_);
    conc_H2_   = this->SpeciesConcentrationKmolM3(index_H2_, Y, cTot, thermo_);
    conc_H2O_  = this->SpeciesConcentrationKmolM3(index_H2O_, Y, cTot, thermo_);
    conc_C2H2_ = this->SpeciesConcentrationKmolM3(index_C2H2_, Y, cTot, thermo_);

    if (is_debug_mode_)
    {
        this->T_          = 1800.;
        this->P_Pa_       = 101325.;
        const double R_km = this->kB_ * this->Nav_kmol_;
        const double cD   = this->P_Pa_ / (R_km * this->T_);
        this->rho_        = 1.8956e-4 * 1000.;
        conc_PAH_         = cD * 1e-7;
        conc_H_           = cD * 1e-5;
        conc_OH_          = cD * 2e-4;
        conc_O2_          = cD * 1e-2;
        conc_H2_          = cD * 1e-3;
        conc_H2O_         = cD * 1e-1;
        conc_C2H2_        = cD * 1e-5;
        std::cout << "Ctot (kmol/m3): " << cD << "\n";
        std::cout << "CPAH (kmol/m3): " << conc_PAH_ << "\n";
        std::cout << "dPAH (cm):      " << dpah_ * 100. << "\n";
        std::cout << "ddim (cm):      " << ddim_ * 100. << "\n";
        std::cout << "dnucl (cm):     " << dnucl_ * 100. << "\n";
    }
}

// ============================================================================
// SetMoments
// ============================================================================

template <ThermoMap Thermo>
void ThreeEquations<Thermo>::SetMoments(std::span<const double> m) noexcept
{
    assert(m.size() == static_cast<std::size_t>(Base::n_equations) &&
           "[ThreeEquations::SetMoments] expected exactly 3 moment values.");
    SetMoments(m[0], m[1], m[2]);
}

template <ThermoMap Thermo>
void ThreeEquations<Thermo>::SetMoments(double Ys, double NsNorm, double Ss) noexcept
{
    Ys_     = std::max(Ys, 0.);
    NsNorm_ = std::max(NsNorm, 0.);
    Ss_     = std::max(Ss, 0.);
}

// ============================================================================
// Particle properties
// ============================================================================

template <ThermoMap Thermo>
void ThreeEquations<Thermo>::Properties(
    double& fv, double& dp, double& dc, double& np, double& ss, double& vs) const noexcept
{
    // Regularize to floor values for property calculation
    const double YsStar = std::max(Ys_, Ys_min_);
    const double NsStar = std::max(NsNorm_ * N0_scaling_, Ns_min_);
    const double SsStar = std::max(Ss_, Ss_min_);

    fv = this->rho_ / this->rho_particle_ * YsStar; // [-]
    vs = std::max(fv / NsStar, vs_min_);            // [m3]

    const double s_sphere = this->SphereSurfaceFromVolume(vs);                              // [m2]
    ss                    = std::max(SsStar / NsStar, s_sphere);                             // [m2]
    dp                    = 6. * vs / ss;                                                    // [m]
    np                    = this->NumberPrimaryParticles(ss, vs);                            // [-]
    dc = dp * std::pow(np, 1. / Df_);                                                        // [m]
}

template <ThermoMap Thermo> double ThreeEquations<Thermo>::volume_fraction() const noexcept
{
    return this->rho_ / this->rho_particle_ * std::max(Ys_, 0.);
}

template <ThermoMap Thermo> double ThreeEquations<Thermo>::mass_fraction() const noexcept
{
    return std::max(Ys_, 0.);
}

template <ThermoMap Thermo> double ThreeEquations<Thermo>::particle_number_density() const noexcept
{
    return std::max(NsNorm_ * N0_scaling_, 0.);
}

template <ThermoMap Thermo> double ThreeEquations<Thermo>::specific_surface_area() const noexcept
{
    return std::max(Ss_, 0.);
}

template <ThermoMap Thermo> double ThreeEquations<Thermo>::particle_diameter() const noexcept
{
    double fv, dp, dc, np, ss, vs;
    Properties(fv, dp, dc, np, ss, vs);
    return dp;
}

template <ThermoMap Thermo> double ThreeEquations<Thermo>::collision_diameter() const noexcept
{
    double fv, dp, dc, np, ss, vs;
    Properties(fv, dp, dc, np, ss, vs);
    return dc;
}

template <ThermoMap Thermo> double ThreeEquations<Thermo>::number_primary_particles() const noexcept
{
    double fv, dp, dc, np, ss, vs;
    Properties(fv, dp, dc, np, ss, vs);
    return np;
}

template <ThermoMap Thermo> double ThreeEquations<Thermo>::diffusion_coefficient() const noexcept
{
    double fv, dp, dc, np, ss, vs;
    Properties(fv, dp, dc, np, ss, vs);

    const double dc_safe       = std::max(dc, 1.e-12);
    const double GammaBrownian = this->CunninghamDiffusionCoeff(dc_safe); // [kg/m/s]
    const double GammaSc       = this->mu_ / this->schmidt_number_;
    return std::max(GammaBrownian, GammaSc);
}

// ============================================================================
// Planck absorption coefficient — ThreeEquations-specific constants
// ============================================================================

template <ThermoMap Thermo>
double ThreeEquations<Thermo>::planck_coefficient(double T, double fv) const noexcept
{
    switch (this->planck_model_)
    {
        case PlanckCoeffModel::Smooke:
            return 1307. * fv * T; // Smooke et al. C&F 2009
        case PlanckCoeffModel::Kent:
            return 2262. * fv * T; // Kent et al. C&F 1990
        case PlanckCoeffModel::Sazhin:
        {
            const double rhos = this->rho_particle_ * fv;      // [kg/m3]
            return 1232. * rhos * (1. + 4.8e-4 * (T - 2000.)); // Sazhin 1994
        }
        default:
            return 0.;
    }
}

// ============================================================================
// Surface geometry utilities
// ============================================================================

template <ThermoMap Thermo>
double ThreeEquations<Thermo>::DeltaSurfaceSpherical(double deltaV, double V, double S) noexcept
{
    if (V <= 0. || S <= 0. || deltaV <= 0.)
        return 0.;
    return (2. / 3.) * (deltaV / V) * S;
}

template <ThermoMap Thermo>
double ThreeEquations<Thermo>::DeltaSurfaceFractal(double deltaV, double V, double S, double np) noexcept
{
    if (V <= 0. || S <= 0. || deltaV <= 0.)
        return 0.;

    const double deltaSph = DeltaSurfaceSpherical(deltaV, V, S);

    if (np <= 1.)
    {
        const double deltaS = S * (std::pow(1. + deltaV / V, 2. / 3.) - 1.);
        return std::max(deltaS, deltaSph);
    }

    // Fractal scaling factor
    constexpr double Df      = 1.8; // same as Df_ — static helper, so hardcode
    const double numerator   = 1. - std::pow(np, -1. / Df) - 2. / Df;
    const double denominator = 1. - std::pow(np, -1. / Df) - 3. / Df;

    double deltaS = deltaSph;
    if (std::abs(denominator) > 1.e-12)
    {
        const double a = numerator / denominator;
        if (std::isfinite(a) && a > 0.)
            deltaS = a * (deltaV / V) * S;
    }
    return std::max(deltaS, deltaSph);
}

// ============================================================================
// Surface site density
// ============================================================================

template <ThermoMap Thermo> double ThreeEquations<Thermo>::SurfaceSiteDensity() const noexcept
{
    if (surface_chem_model_ == SurfaceChemistryModel::HMOM)
        return 1.7e19; // [#/m2], HMOM-consistent
    return 1. / sc2_;  // [#/m2], RC-PAH geometric default
}

// ============================================================================
// Surface kinetics
// ============================================================================

template <ThermoMap Thermo>
typename ThreeEquations<Thermo>::SurfaceKineticsRates ThreeEquations<Thermo>::SurfaceKinetics() const noexcept
{
    if (surface_chem_model_ == SurfaceChemistryModel::HMOM)
        return Kinetics_HMOM();
    return Kinetics_RCPAH();
}

// ----------------------------------------------------------------------------
// RC-PAH kinetics (Franzelli et al. 2019)
// ----------------------------------------------------------------------------
template <ThermoMap Thermo>
typename ThreeEquations<Thermo>::SurfaceKineticsRates ThreeEquations<Thermo>::Kinetics_RCPAH() const noexcept
{
    SurfaceKineticsRates r{}; // zero-initialised

    constexpr double small = 1.e-10;

    // Concentrations [mol/cm3]
    const double cH    = conc_H_ * 1e-3;
    const double cOH   = conc_OH_ * 1e-3;
    const double cO2   = conc_O2_ * 1e-3;
    const double cH2   = conc_H2_ * 1e-3;
    const double cH2O  = conc_H2O_ * 1e-3;
    const double cC2H2 = conc_C2H2_ * 1e-3;

    // HACA-RC Arrhenius constants  (A in [cm3/mol/s] or [1/s], E in [J/mol])
    const double k01f = Arrhenius_cgs(1.000e14, 0.0, 0.0, this->T_);
    const double k01b = Arrhenius_cgs(1.439e13, 0.0, -37.63e3, this->T_);
    const double k02f = Arrhenius_cgs(1.630e08, 1.4, 6.10e3, this->T_);
    const double k02b = Arrhenius_cgs(1.101e08, 1.4, 31.14e3, this->T_);
    const double k03f = Arrhenius_cgs(1.000e13, 0.0, 0.0, this->T_);
    const double k03b = 0.0;
    const double k04f = Arrhenius_cgs(3.500e13, 0.0, 0.0, this->T_);
    const double k04b = Arrhenius_cgs(3.225e14, 0.0, 181.69e3, this->T_);
    const double k05f = Arrhenius_cgs(1.000e10, 0.0, 20.00e3, this->T_);
    const double k05b = Arrhenius_cgs(8.770e11, 0.0, 74.44e3, this->T_);
    const double k06f = Arrhenius_cgs(1.000e12, 0.0, 8.40e3, this->T_);

    // Pseudo-first-order rates [1/s]
    const double r01f    = std::max(small, k01f * cH);
    const double r01b    = std::max(small, k01b * cH2);
    const double r02f    = std::max(small, k02f * cOH);
    const double r02b    = std::max(small, k02b * cH2O);
    const double r03f    = std::max(small, k03f * cH);
    const double r03b    = std::max(small, k03b);
    const double r04f    = std::max(small, k04f * cC2H2);
    const double r04b    = std::max(small, k04b);
    const double r05f    = std::max(small, k05f);
    const double r05b    = std::max(small, k05b * cH);
    const double r06f    = std::max(small, k06f * cO2);
    const double r06bisf = r06f;

    // OH oxidation collision rate [1/s]
    const double surface_c2_cm2 = sc2_ * 1.e4;
    const double MW_OH_kg_mol   = thermo_.MolecularWeight(static_cast<unsigned>(index_OH_)) / 1000.;
    const double mean_OH_m_s    = std::sqrt(8. * this->Rgas_mol_ * this->T_ / (this->pi_ * MW_OH_kg_mol));
    const double r07f =
        std::max(small, 0.25 * this->Nav_mol_ * surface_c2_cm2 * 0.13 * (mean_OH_m_s * 100.) * cOH);

    // RC-PAH closure
    const double fk10     = r05f / (r04b + r06bisf + r05f);
    const double denom_p  = r02b + r01b + r03f + r04f * fk10;
    const double denom_c1 = r04b + r06bisf + r05f;
    const double denom_c2 = r04b + r06f + r05f;
    const double fsootp   = (r02f + r01f + r03b + r07f + r05b * (1. - fk10)) / denom_p;
    const double fsootc   = fsootp * r04f / denom_c1 + r05b / denom_c2;

    if (!std::isfinite(fsootp) || !std::isfinite(fsootc) || fsootp < 0. || fsootc < 0.)
        return r;

    // Active-site fractions (chi = 1.0: no active-surface correction)
    const double chip = fsootp;
    const double chic = fsootc;

    const double kd   = r04f * chip;
    const double krev = r04b * chic;
    const double ko2  = r06f * (chip + chic);
    const double koh  = r07f;

    r.kd     = kd;
    r.krev   = krev;
    r.ko2    = ko2;
    r.koh    = koh;
    r.fsootp = fsootp;
    r.fsootc = fsootc;
    r.ksg    = kd - krev;
    r.kox    = ko2 + koh;

    if (!std::isfinite(r.ksg))
        r.ksg = 0.;
    if (!std::isfinite(r.kox) || r.kox < 0.)
        r.kox = 0.;
    return r;
}

// ----------------------------------------------------------------------------
// HMOM HACA kinetics
// ----------------------------------------------------------------------------
template <ThermoMap Thermo>
typename ThreeEquations<Thermo>::SurfaceKineticsRates ThreeEquations<Thermo>::Kinetics_HMOM() const noexcept
{
    SurfaceKineticsRates r{};

    if (!std::isfinite(this->T_) || this->T_ <= 0.)
        return r;
    if (!std::isfinite(sc2_) || sc2_ <= 0.)
        return r;

    // Concentrations [mol/cm3]
    const double cH    = std::max(conc_H_, 0.) * 1.e-3;
    const double cOH   = std::max(conc_OH_, 0.) * 1.e-3;
    const double cO2   = std::max(conc_O2_, 0.) * 1.e-3;
    const double cH2   = std::max(conc_H2_, 0.) * 1.e-3;
    const double cH2O  = std::max(conc_H2O_, 0.) * 1.e-3;
    const double cC2H2 = std::max(conc_C2H2_, 0.) * 1.e-3;

    const double k1f = Arrhenius_cgs(6.72e1, 3.33, 6.09e3, this->T_);
    const double k1b = Arrhenius_cgs(6.44e-1, 3.79, 27.96e3, this->T_);
    const double k2f = Arrhenius_cgs(1.00e8, 1.80, 68.42e3, this->T_);
    const double k2b = Arrhenius_cgs(8.68e4, 2.36, 25.46e3, this->T_);
    const double k3f = Arrhenius_cgs(1.13e16, -0.06, 476.05e3, this->T_);
    const double k3b = Arrhenius_cgs(4.17e13, 0.15, 0.00e3, this->T_);
    const double k4  = Arrhenius_cgs(2.52e9, 1.10, 17.13e3, this->T_);
    const double k5  = Arrhenius_cgs(2.20e12, 0.00, 31.38e3, this->T_);

    const double denominator = k1b * cH2O + k2b * cH2 + k3b * cH + k4 * cC2H2;
    double sootStar          = 0.;
    if (denominator > 0. && std::isfinite(denominator))
    {
        const double numerator = k1f * cOH + k2f * cH + k3f;
        if (numerator > 0. && std::isfinite(numerator))
            sootStar = numerator / denominator;
    }
    if (mass_fraction_H_ < 2.e-9 && mass_fraction_OH_ < 2.e-8)
        sootStar = 0.;
    if (!std::isfinite(sootStar) || sootStar < 0.)
        sootStar = 0.;
    sootStar = sootStar / (1. + sootStar);
    if (!std::isfinite(sootStar) || sootStar < 0.)
        sootStar = 0.;
    if (sootStar > 1.)
        sootStar = 1.;

    const double ksg      = k4 * cC2H2 * sootStar;
    const double kox_O2   = k5 * cO2 * sootStar;
    constexpr double eff6 = 0.13;
    const double k6       = 8.94 * eff6 * std::sqrt(this->T_) * this->Nav_mol_;
    const double lambda   = SurfaceSiteDensity();
    const double kox_OH   = 0.5 / lambda * k6 * cOH;

    r.ksg    = std::isfinite(ksg) ? ksg : 0.;
    r.ko2    = (std::isfinite(kox_O2) && kox_O2 > 0.) ? kox_O2 : 0.;
    r.koh    = (std::isfinite(kox_OH) && kox_OH > 0.) ? kox_OH : 0.;
    r.kox    = r.ko2 + r.koh;
    r.kd     = r.ksg;
    r.krev   = 0.;
    r.fsootp = sootStar;
    r.fsootc = 0.;
    if (!std::isfinite(r.kox) || r.kox < 0.)
        r.kox = 0.;
    return r;
}

// ============================================================================
// Dimer QSSA concentration
// ============================================================================

template <ThermoMap Thermo> void ThreeEquations<Thermo>::DimerConcentration()
{
    c_dimer_           = 0.;
    dimerization_rate_ = 0.;

    if (dimer_concentration_model_ != DimerModel::QSSA_Rodrigues)
        return;

    const double cPAH_kmol = std::max(conc_PAH_, 0.);
    const double cPAH_mol  = 1000. * cPAH_kmol;
    const double Npah      = cPAH_kmol * this->Nav_kmol_;

    if (Npah <= 0. || cPAH_mol <= 0.)
        return;

    // Sticking coefficient
    double alpha_s = 0.;
    if (sticking_model_ == StickingModel::Constant)
        alpha_s = sticking_coeff_constant_;
    else if (sticking_model_ == StickingModel::PAH4)
        alpha_s = sticking_coeff_constant_ * std::pow(mwpah_, 4.);

    // PAH thermal velocity [m/s]
    const double vtpah = std::sqrt(4. * this->pi_ * this->kB_ * this->T_ / mpah_);

    // PAH–PAH formation kernel [m3/s]
    double beta_pp = 0.5 * alpha_s * vtpah * dpah_ * dpah_;
    beta_pp *= correction_coeff_pah_pah_;

    double Jpah = beta_pp * Npah * Npah; // [#/m3/s]

    // Low-PAH cutoffs (FORTRAN-compatible)
    if (cPAH_mol < 2.e-7)
        Jpah = 0.;
    if (cPAH_mol < 1.e-10)
        Jpah = 0.;
    if (Jpah <= 0.)
        return;

    // Dimer–dimer kernel [m3/s]
    const double beta_dd = epsilon_nucleation_ *
                           std::sqrt(this->pi_ * this->kB_ * this->T_ / (2. * this->rho_particle_)) *
                           std::sqrt(2. / vdim_) * std::pow(2. * ddim_, 2.);

    if (beta_dd <= 0.)
        return;

    // Dimer loss by condensation on soot [1/s]
    double kcond = 0.;
    if (condensation_model_ > 0)
    {
        const double Ns = std::max(NsNorm_ * N0_scaling_, 0.);
        if (Ns > 0.)
        {
            double fv, dp, dc, np, ss, vs;
            Properties(fv, dp, dc, np, ss, vs);
            const double beta_ds =
                epsilon_condensation_ *
                std::sqrt(this->pi_ * this->kB_ * this->T_ / (2. * this->rho_particle_)) *
                std::sqrt(1. / vdim_ + 1. / vs) * std::pow(ddim_ + dc, 2.);
            kcond = beta_ds * Ns;
        }
    }

    // QSSA positive root: Ndim = 2*Jpah / (kcond + sqrt(kcond^2 + 4*beta_dd*Jpah))
    double Ndim       = 0.;
    const double disc = kcond * kcond + 4. * beta_dd * Jpah;
    if (disc > 0.)
        Ndim = 2. * Jpah / (kcond + std::sqrt(disc));

    if (cPAH_mol < 1.e-10)
        Ndim = 0.;

    c_dimer_           = Ndim / this->Nav_kmol_; // [kmol/m3]
    dimerization_rate_ = Jpah;                   // [#/m3/s]

    if (is_debug_mode_)
    {
        std::cout << "DEBUG DIMER QSSA\n"
                  << "  Npah          = " << Npah << " [#/m3]\n"
                  << "  beta_pp       = " << beta_pp << " [m3/s]\n"
                  << "  beta_dd       = " << beta_dd << " [m3/s]\n"
                  << "  Jpah          = " << Jpah << " [#/m3/s]\n"
                  << "  kcond         = " << kcond << " [1/s]\n"
                  << "  Ndim          = " << Ndim << " [#/m3]\n"
                  << "  c_dimer       = " << c_dimer_ << " [kmol/m3]\n";
    }
}

template <ThermoMap Thermo> double ThreeEquations<Thermo>::PAHDimerizationRate() const noexcept
{
    return 2. * dimerization_rate_ / this->Nav_kmol_; // [kmol/m3/s]
}

// ============================================================================
// Source term sub-routines
// ============================================================================

template <ThermoMap Thermo> void ThreeEquations<Thermo>::NucleationSourceTerms()
{
    const double Ndim   = c_dimer_ * this->Nav_kmol_;
    const double Betafm = epsilon_nucleation_ *
                          std::sqrt(this->pi_ * this->kB_ * this->T_ / 2. / this->rho_particle_) *
                          std::sqrt(2. / vdim_) * std::pow(2. * ddim_, 2.);
    const double Betafm_N2 = Betafm * Ndim * Ndim;

    const double Omegafv = vdim_ * Betafm_N2;
    const double OmegaNs = 0.5 * Betafm_N2;
    const double OmegaSs = 0.5 * snucl_ * Betafm_N2; // nucleus surface, FORTRAN-compatible

    this->source_nucleation_(0) = this->rho_particle_ / this->rho_ * Omegafv;
    this->source_nucleation_(1) = OmegaNs / N0_scaling_;
    this->source_nucleation_(2) = OmegaSs;

    if (is_debug_mode_)
    {
        std::cout << "Omegafv  (1/s):    " << Omegafv << "\n"
                  << "OmegaYs  (1/s):    " << this->rho_particle_ / this->rho_ * Omegafv << "\n"
                  << "OmegaNs  (#/m3/s): " << OmegaNs << "\n"
                  << "OmegaSs  (1/m/s):  " << OmegaSs << "\n";
    }
}

template <ThermoMap Thermo> void ThreeEquations<Thermo>::CondensationSourceTerms()
{
    const double Ns = std::max(NsNorm_ * N0_scaling_, 0.);

    double fv, dp, dc, np, ss, vs;
    Properties(fv, dp, dc, np, ss, vs);

    const double deltas = DeltaSurfaceFractal(vdim_, vs, ss, np);

    const double Ndim    = c_dimer_ * this->Nav_kmol_;
    const double Betafms = epsilon_condensation_ *
                           std::sqrt(this->pi_ * this->kB_ * this->T_ / 2. / this->rho_particle_) *
                           std::sqrt(1. / vdim_ + 1. / vs) * std::pow(ddim_ + dc, 2.);
    const double BetaNdimNs = Betafms * Ndim * Ns;

    this->source_condensation_(0) = this->rho_particle_ / this->rho_ * vdim_ * BetaNdimNs;
    this->source_condensation_(1) = 0.;
    this->source_condensation_(2) = deltas * BetaNdimNs;
}

template <ThermoMap Thermo> void ThreeEquations<Thermo>::CoagulationSourceTerms()
{
    const double Ns = std::max(NsNorm_ * N0_scaling_, 0.);

    double fv, dp, dc, np, ss, vs;
    Properties(fv, dp, dc, np, ss, vs);

    const double m_mol = this->rho_ * this->kB_ * this->T_ / this->P_Pa_;
    const double lambdaG =
        this->mu_ / this->rho_ * std::sqrt(this->pi_ * m_mol / (2. * this->kB_ * this->T_));
    const double dcSafe = std::max(dc, 1.e-12);
    const double Cu     = 1. + 2.154 * lambdaG / dcSafe;

    const double Beta1  = epsilon_coagulation_ *
                          std::sqrt(this->pi_ * this->kB_ * this->T_ / 2. / this->rho_particle_) *
                          std::sqrt(2. / vs) * std::pow(2. * dcSafe, 2.);
    const double Beta2  = 8. * this->kB_ * this->T_ / 3. / this->mu_ * Cu;
    const double Betavs = Beta1 * Beta2 / (Beta1 + Beta2);

    // Smooth activation at the FORTRAN-heritage threshold (100 #/cm3 = 1e8 #/m3).
    // The original hard step (Heaviside) created a Dirac-delta-like discontinuity
    // in the ODE Jacobian as Ns crossed the threshold, forcing the stiff solver to
    // bracket the crossing and shrink dt drastically.  Replacing it with a tanh
    // transition of width 20 % of the threshold makes the source C¹-continuous
    // while preserving the correct asymptotic behaviour (zero well below, full
    // -½βNs² well above).  This also prevents unbounded Ns accumulation in
    // nucleation-onset cells where Ns < threshold and no other sink is active.
    static constexpr double Ns_coag_threshold = 1.e8;   // #/m3  (100 #/cm3)
    static constexpr double Ns_coag_width     = 2.e7;   // #/m3  (20 % transition)
    const double smooth_coag =
        0.5 * (1. + std::tanh((Ns - Ns_coag_threshold) / Ns_coag_width));
    const double OmegaNs = -0.5 * Betavs * Ns * Ns * smooth_coag;

    this->source_coagulation_(0) = 0.;
    this->source_coagulation_(1) = OmegaNs / N0_scaling_;
    this->source_coagulation_(2) = 0.;
}

template <ThermoMap Thermo> void ThreeEquations<Thermo>::SurfaceGrowthSourceTerms()
{
    const double Ns = std::max(NsNorm_ * N0_scaling_, 0.);
    const double Ss = std::max(Ss_, 0.);

    if (Ss <= 0. || Ns <= 0.)
        return;

    double fv, dp, dc, np, ss, vs;
    Properties(fv, dp, dc, np, ss, vs);

    const double lambda     = SurfaceSiteDensity();
    const double deltas_fra = DeltaSurfaceFractal(vc2_, vs, ss, np);

    const SurfaceKineticsRates r = SurfaceKinetics();
    const double coeff_mass      = lambda * r.ksg * Ss;
    const double coeff_surface   = lambda * std::max(r.ksg, 0.) * Ss;

    this->source_growth_(0) = this->rho_particle_ / this->rho_ * vc2_ * coeff_mass;
    this->source_growth_(1) = 0.;
    this->source_growth_(2) = deltas_fra * coeff_surface;
}

template <ThermoMap Thermo> void ThreeEquations<Thermo>::OxidationSourceTerms()
{
    const double Ys = std::max(Ys_, 0.);
    const double Ns = std::max(NsNorm_ * N0_scaling_, 0.);
    const double Ss = std::max(Ss_, 0.);

    if (Ys <= 0. || Ss <= 0. || Ns <= 0.)
        return;

    const double fv_oxid = this->rho_ / this->rho_particle_ * Ys;
    const double vs_oxid = fv_oxid / Ns;
    const double ss_oxid = Ss / Ns;

    double fv, dp, dc, np, ss, vs;
    Properties(fv, dp, dc, np, ss, vs);

    const double lambda          = SurfaceSiteDensity();
    const double deltas_sph      = DeltaSurfaceSpherical(vc2_, vs, ss);
    const SurfaceKineticsRates r = SurfaceKinetics();
    const double kox             = std::max(r.kox, 0.);
    const double coeff           = lambda * kox * Ss;

    if (!std::isfinite(coeff) || coeff <= 0.)
        return;

    const double Omegafv = -vc2_ * coeff;
    const double OmegaSs = -deltas_sph * coeff;

    // Number-density sink: destroy particle if vs < vc2_ or ss < deltas_sph
    const bool destroy = (vs_oxid <= vc2_) || (ss_oxid <= deltas_sph);
    double OmegaNs     = destroy ? -coeff : 0.;

    if (smooth_heaviside_oxidation_)
    {
        // Wider bandwidth (50 % of vc2_) compared to the original 10 %.
        // The narrower band produced a near-discontinuous ∂OmegaNs/∂vs that
        // contributed to stiffness when the average particle volume hovered near vc2_.
        const double eps_v = 0.5 * vc2_;
        const double H     = SmoothHeaviside(vs_oxid - vc2_, eps_v);
        double destroy_sw  = 1. - H;
        if (ss_oxid <= deltas_sph)
            destroy_sw = 1.;
        OmegaNs = -destroy_sw * coeff;
    }

    this->source_oxidation_(0) = this->rho_particle_ / this->rho_ * Omegafv;
    this->source_oxidation_(1) = OmegaNs / N0_scaling_;
    this->source_oxidation_(2) = OmegaSs;
}

// ============================================================================
// ComputeSources  — master entry point
// ============================================================================

template <ThermoMap Thermo> void ThreeEquations<Thermo>::ComputeSources() noexcept
{
    this->ZeroSources();          // zeros source_all_ (base class)
    source_nucleation_.setZero(); // owned by ThreeEquations — zeroed explicitly
    source_coagulation_.setZero();
    source_condensation_.setZero();
    source_growth_.setZero();
    source_oxidation_.setZero();

    DimerConcentration();

    if (nucleation_model_ > 0)
        NucleationSourceTerms();
    if (surface_growth_model_ > 0)
        SurfaceGrowthSourceTerms();
    if (oxidation_model_ > 0)
        OxidationSourceTerms();
    if (condensation_model_ > 0)
        CondensationSourceTerms();
    if (coagulation_model_ > 0)
        CoagulationSourceTerms();

    // Sanitise and accumulate total source
    for (unsigned i = 0; i < 3u; ++i)
    {
        if (!std::isfinite(this->source_nucleation_(i)))
            this->source_nucleation_(i) = 0.;
        if (!std::isfinite(this->source_oxidation_(i)))
            this->source_oxidation_(i) = 0.;
        if (!std::isfinite(this->source_condensation_(i)))
            this->source_condensation_(i) = 0.;
        if (!std::isfinite(this->source_growth_(i)))
            this->source_growth_(i) = 0.;
        if (!std::isfinite(this->source_coagulation_(i)))
            this->source_coagulation_(i) = 0.;

        this->source_all_(i) = this->source_nucleation_(i) + this->source_growth_(i) +
                               this->source_oxidation_(i) + this->source_coagulation_(i) +
                               this->source_condensation_(i);
    }

    if (this->gas_consumption_)
        CalculateOmegaGas();
}

// ============================================================================
// CalculateOmegaGas  — gas-phase consumption (kg/m3/s)
// ============================================================================

template <ThermoMap Thermo> void ThreeEquations<Thermo>::CalculateOmegaGas() noexcept
{
    this->omega_gas_.setZero();
    omega_gas_oxidation_.setZero();
    if (!this->gas_consumption_)
        return;

    const int nsp = thermo_.NumberOfSpecies();

    // Helper: add molar source [kmol/m3/s] for species idx, convert to [kg/m3/s]
    auto AddMolar = [&](int idx, double omega_kmol_m3_s)
    {
        if (idx < 0 || idx >= nsp || !std::isfinite(omega_kmol_m3_s))
            return;
        this->omega_gas_[idx] +=
            omega_kmol_m3_s * thermo_.MolecularWeight(static_cast<unsigned>(idx));
    };

    // 1. PAH consumed by dimerization (2 PAH per event)
    if (nucleation_model_ > 0 && dimerization_rate_ > 0. && std::isfinite(dimerization_rate_))
    {
        AddMolar(pah_index_, -2. * dimerization_rate_ / this->Nav_kmol_);
    }

    // 2 & 3. Surface-related consumption (requires soot)
    const double Ns = std::max(NsNorm_ * N0_scaling_, 0.);
    const double Ss = std::max(Ss_, 0.);
    if (Ns > 0. && Ss > 0.)
    {
        const double lambda = SurfaceSiteDensity();
        if (std::isfinite(lambda) && lambda > 0.)
        {
            const SurfaceKineticsRates r = SurfaceKinetics();

            // C2H2 net consumption by surface growth.
            // Net reaction: C2H2 → 2 C_soot + H2  (one H2 released per C2H2 added).
            if (surface_growth_model_ > 0)
            {
                const double rate_c2h2 = lambda * r.ksg * Ss; // [#/m3/s]
                if (std::isfinite(rate_c2h2) && !(rate_c2h2 > 0. && conc_C2H2_ <= 0.))
                {
                    // C2H2 consumed
                    AddMolar(index_C2H2_, -rate_c2h2 / this->Nav_kmol_);
                    // H2 produced: 1 mol H2 per mol C2H2
                    if (index_H2_ >= 0)
                        AddMolar(index_H2_, +rate_c2h2 / this->Nav_kmol_);
                }
            }

            // O2 and OH consumption by oxidation, with gaseous products.
            // Conventions (same as BrookesMoss):
            //   O2 channel: C + ½O2 → CO  (net per C2 pair: C2 + O2 → 2CO)
            //   OH channel: C + OH  → CO + ½H2
            if (oxidation_model_ > 0 && Ys_ > 0.)
            {
                // Snapshot for oxidation-only extraction (operator splitting support).
                const Eigen::VectorXd omega_snap_before_ox = this->omega_gas_;
                const double R_O2 = lambda * std::max(r.ko2, 0.) * Ss; // [#/m3/s]
                const double R_OH = lambda * std::max(r.koh, 0.) * Ss; // [#/m3/s]

                // O2 channel: 1 O2 consumed per C2 pair, 2 CO produced
                if (conc_O2_ > 0. && R_O2 > 0. && std::isfinite(R_O2))
                {
                    AddMolar(index_O2_, -R_O2 / this->Nav_kmol_);
                    if (index_CO_ >= 0)
                        AddMolar(index_CO_, +2. * R_O2 / this->Nav_kmol_);
                }

                // OH channel: 1 OH consumed per C atom, 1 CO + ½ H2 produced
                if (conc_OH_ > 0. && R_OH > 0. && std::isfinite(R_OH))
                {
                    AddMolar(index_OH_, -R_OH / this->Nav_kmol_);
                    if (index_CO_ >= 0)
                        AddMolar(index_CO_, +R_OH / this->Nav_kmol_);
                    if (index_H2_ >= 0)
                        AddMolar(index_H2_, +0.5 * R_OH / this->Nav_kmol_);
                }

                // Capture oxidation-only gas contribution (before closure).
                omega_gas_oxidation_ = this->omega_gas_ - omega_snap_before_ox;
            }
        }
    }

    // 4. Dummy species closure (mass-fraction conservation)
    if (this->is_closure_dummy_species_)
    {
        // dummy_index_ must be set during setup — a negative value is a programming error.
        assert(this->closure_dummy_index_ >= 0 && "[ThreeEquations::CalculateOmegaGas] dummy_index_ not set.");
        double sum = 0.;
        for (int i = 0; i < nsp; ++i)
            if (i != this->closure_dummy_index_)
                sum += this->omega_gas_[i];
        this->omega_gas_[this->closure_dummy_index_] = -sum;

        // Apply the same closure to the oxidation-only vector.
        double sum_ox = 0.;
        for (int i = 0; i < nsp; ++i)
            if (i != this->closure_dummy_index_)
                sum_ox += omega_gas_oxidation_[i];
        omega_gas_oxidation_[this->closure_dummy_index_] = -sum_ox;
    }

    if (is_debug_mode_)
    {
        std::cout << "PAH dim rate: " << -2. * dimerization_rate_ / this->Nav_kmol_ << "\n";
        for (int j = 0; j < nsp; ++j)
            if (std::fabs(this->omega_gas_[j]) > 0.)
                std::cout << "j=" << j << " omega=" << this->omega_gas_[j] << "\n";
    }
}

// ============================================================================
// PrintSummary
// ============================================================================
template <ThermoMap Thermo> void ThreeEquations<Thermo>::PrintSummary() const
{
    // Helper lambdas — no heap allocation, resolved at compile time
    const auto surf_chem_str = [](SurfaceChemistryModel m) -> const char* {
        switch (m) {
            case SurfaceChemistryModel::RCPAH: return "RCPAH (Franzelli 2019)";
            case SurfaceChemistryModel::HMOM:  return "HMOM HACA kinetics";
            default:                            return "unknown";
        }
    };
    const auto sticking_str = [](StickingModel m) -> const char* {
        switch (m) {
            case StickingModel::Constant: return "constant";
            case StickingModel::PAH4:     return "PAH 4-ring";
            default:                       return "unknown";
        }
    };
    const auto dimer_str = [](DimerModel m) -> const char* {
        switch (m) {
            case DimerModel::QSSA_Rodrigues: return "QSSA Rodrigues";
            default:                          return "unknown";
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
        << "                        3 Equations Soot Model Summary\n"
        << "------------------------------------------------------------------------------------------\n"
        << " * Active: " << (this->is_active_ ? "yes" : "no") << "\n"
        << "\n"
        << " [Physical properties]\n"
        << "    + Soot density (kg/m3):    " << this->rho_particle_ << "\n"
        << "    + Fractal dimension (-):   " << Df_ << "\n"
        << "\n"
        << " [Species — PAH precursor]\n"
        << "    + Name:                    " << pah_species_ << "\n"
        << "    + Index (-):               " << pah_index_ << "\n"
        << "    + Diameter (m):            " << dpah_ << "\n"
        << "    + Surface (m2):            " << spah_ << "\n"
        << "    + Volume (m3):             " << vpah_ << "\n"
        << "    + MW (kg/kmol):            " << mwpah_ << "\n"
        << "    + Simplified PAH mass:     " << (is_simplified_pah_mass_ ? "yes" : "no") << "\n"
        << "\n"
        << " [Particle geometry — dimer]\n"
        << "    + Diameter (m):            " << ddim_ << "\n"
        << "    + Surface (m2):            " << sdim_ << "\n"
        << "    + Volume (m3):             " << vdim_ << "\n"
        << "    + MW (kg/kmol):            " << 2. * mwpah_ << "\n"
        << "\n"
        << " [Particle geometry — C2 pair]\n"
        << "    + Diameter (m):            " << dc2_ << "\n"
        << "    + Surface (m2):            " << sc2_ << "\n"
        << "    + Volume (m3):             " << vc2_ << "\n"
        << "\n"
        << " [Processes]\n"
        << "    + Nucleation:              " << nucleation_model_ << "\n"
        << "    + Condensation:            " << condensation_model_ << "\n"
        << "    + Surface growth:          " << surface_growth_model_ << "\n"
        << "    + Oxidation:               " << oxidation_model_ << "\n"
        << "    + Coagulation:             " << coagulation_model_ << "\n"
        << "    + Surface chemistry model: " << static_cast<int>(surface_chem_model_) << "  (" << surf_chem_str(surface_chem_model_) << ")\n"
        << "    + Dimer concentration:     " << static_cast<int>(dimer_concentration_model_) << "  (" << dimer_str(dimer_concentration_model_) << ")\n"
        << "    + Sticking model:          " << static_cast<int>(sticking_model_) << "  (" << sticking_str(sticking_model_) << ")\n"
        << "    + Sticking coeff. (-):     " << sticking_coeff_constant_ << "\n"
        << "\n"
        << " [Collision enhancement factors]\n"
        << "    + Nucleation (-):          " << epsilon_nucleation_ << "\n"
        << "    + Condensation (-):        " << epsilon_condensation_ << "\n"
        << "    + Coagulation (-):         " << epsilon_coagulation_ << "\n"
        << "\n"
        << " [Additional parameters]\n"
        << "    + PAH-PAH correction (-):  " << correction_coeff_pah_pah_ << "\n"
        << "    + Smooth Heaviside ox.:    " << (smooth_heaviside_oxidation_ ? "yes" : "no") << "\n"
        << "\n"
        << " [Transport & radiation]\n"
        << "    + Schmidt number (-):      " << this->schmidt_number_ << "\n"
        << "    + Thermophoretic model:    " << static_cast<int>(this->thermophoretic_model()) << "  (" << thermo_str(this->thermophoretic_model_) << ")\n"
        << "    + Gas consumption:         " << (this->gas_consumption_ ? "yes" : "no") << "\n"
        << "    + Radiative heat transfer: " << (this->radiative_heat_transfer_ ? "yes" : "no") << "\n"
        << "    + Planck coeff. model:     " << static_cast<int>(this->planck_model_) << "  (" << planck_str(this->planck_model_) << ")\n"
        << "    + Closure dummy species:   " << (this->is_closure_dummy_species_ ? this->closure_dummy_species_ : "none") << "\n"
        << "\n"
        << " [Numerical floors]\n"
        << "    + N0 scaling (#/m3):       " << N0_scaling_ << "\n"
        << "    + Ys_min (-):              " << Ys_min_ << "\n"
        << "    + Ns_min (#/m3):           " << Ns_min_ << "\n"
        << "    + Ss_min (m2/m3):          " << Ss_min_ << "\n"
        << "    + vs_min (m3):             " << vs_min_ << "\n"
        << "\n"
        << " [Debug]\n"
        << "    + Debug mode:              " << (is_debug_mode_ ? "yes" : "no") << "\n"
        << "------------------------------------------------------------------------------------------\n";
}

// ============================================================================
// ApplyConfig  (private — all parameter assignments, no I/O)
// ============================================================================

template <ThermoMap Thermo>
void ThreeEquations<Thermo>::ApplyConfig(const Config& cfg)
{
    this->is_active_ = cfg.is_active;

    // -- Particle properties (set BEFORE SetPAH so Precalculations() uses  --
    //    the correct density when computing PAH geometry)  ------------------
    this->SetParticleDensity(cfg.soot_density_kg_m3);

    // -- PAH / gas setup ---------------------------------------------------
    this->is_simplified_pah_mass_ = cfg.simplified_pah_mass;
    SetPAH(cfg.pah_species);  // calls Precalculations() — needs rho_particle_ set above
    this->SetGasConsumption(cfg.gas_consumption);
    this->SetGasClosureDummySpecies(cfg.gas_closure_dummy_species);

    // -- Process models ----------------------------------------------------
    SetNucleation(cfg.nucleation_model);
    SetSurfaceGrowth(cfg.surface_growth_model);
    SetOxidation(cfg.oxidation_model);
    SetCondensation(cfg.condensation_model);
    SetCoagulation(cfg.coagulation_model);
    this->SetThermophoreticModel(cfg.thermophoretic_model);
    SetSurfaceChemistryModel(cfg.surface_chemistry_model);

    // -- Collision enhancement factors -------------------------------------
    SetNucleationCollisionEnhancementFactor(cfg.eps_nucleation);
    SetCondensationCollisionEnhancementFactor(cfg.eps_condensation);
    SetCoagulationCollisionEnhancementFactor(cfg.eps_coagulation);
    SetCorrectionCoefficientPAHPAH(cfg.correction_coeff_pah_pah);

    // -- Dimer model -------------------------------------------------------
    if (cfg.dimer_model == "qssa-rodrigues")
        dimer_concentration_model_ = DimerModel::QSSA_Rodrigues;
    else
        throw std::invalid_argument(
            "ThreeEquations::ApplyConfig: unknown dimer_model \"" + cfg.dimer_model +
            "\" — allowed: qssa-rodrigues");

    // -- Numerical floor ---------------------------------------------------
    SetNsMinimum(cfg.ns_minimum_per_m3);

    // -- Sticking coefficient ----------------------------------------------
    SetStickingCoefficientModel(cfg.sticking_model);
    SetStickingCoefficientConstant(cfg.sticking_coeff_constant);

    // -- Radiation / transport ---------------------------------------------
    this->SetRadiativeHeatTransfer(cfg.radiative_heat_transfer);
    this->SetPlanckAbsorptionCoefficient(cfg.planck_coefficient);
    this->SetSchmidtNumber(cfg.schmidt_number);

    // -- Debug mode --------------------------------------------------------
    this->is_debug_mode_ = cfg.debug_mode;
}

// ============================================================================
// SetupFromConfig  (public — delegates to ApplyConfig, then prints summary)
// ============================================================================

template <ThermoMap Thermo>
void ThreeEquations<Thermo>::SetupFromConfig(const Config& cfg)
{
    ApplyConfig(cfg);
    PrintSummary();
}

// ============================================================================
// NDF reconstruction  (Franzelli et al. 2019: Pareto + log-normal)
// ============================================================================

template <ThermoMap Thermo>
typename ThreeEquations<Thermo>::NDFReconstructionData ThreeEquations<Thermo>::ReconstructedNDFData(
    bool use_reg) const
{
    NDFReconstructionData d{}; // all zero, valid = false

    const double Ys_raw = std::max(Ys_, 0.);
    const double Ns_raw = std::max(NsNorm_ * N0_scaling_, 0.);

    double Ys = Ys_raw;
    double Ns = Ns_raw;
    if (use_reg)
    {
        Ys = std::max(Ys, Ys_min_);
        Ns = std::max(Ns, Ns_min_);
    }

    if (Ys <= 0. || Ns <= 0.)
        return d;

    const double fv = this->rho_ / this->rho_particle_ * Ys;
    if (!std::isfinite(fv) || fv <= 0.)
        return d;

    double nus = fv / Ns;
    if (!std::isfinite(nus) || nus <= 0.)
        return d;

    const double nu_nucl = vnucl_;
    if (!std::isfinite(nu_nucl) || nu_nucl <= 0.)
        return d;
    if (nus < nu_nucl)
        nus = nu_nucl;

    // Eq. (8) Franzelli et al. (2019)
    double alpha = 1.0 - 0.18 * std::pow(nus / nu_nucl, 0.12);
    alpha        = std::clamp(alpha, 0., 1.);

    const double nbar0 = 8.0 * std::pow(1. - alpha, 2.) * 1.e27; // [1/m3]
    const double sigma = 1.0 + 0.65 * (1. - alpha);              // [-]

    constexpr double eps_k   = 1.01;
    constexpr double eps_min = 1.e-14;
    constexpr double tiny    = 1.e-300;

    double k = 0., nu1mean = 0., nu2mean = 0., mu = 0.;

    if (alpha > eps_min)
    {
        const double k_peak = nbar0 * nu_nucl / alpha;
        const double denom  = 1. - alpha * nu_nucl / nus;
        if (denom <= 0. || !std::isfinite(denom))
            return d;
        k       = std::max({k_peak, eps_k / denom, 1. + 1.e-12});
        nu1mean = k / (k - 1.) * nu_nucl;
    }

    if (alpha < (1. - eps_min))
    {
        nu2mean = (nus - alpha * nu1mean) / (1. - alpha);
        if (!std::isfinite(nu2mean) || nu2mean <= 0.)
            return d;
        mu = std::log(std::max(nu2mean, tiny)) - 0.5 * sigma * sigma;
    }

    d.valid   = true;
    d.Ns      = Ns;
    d.fv      = fv;
    d.nus     = nus;
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
double ThreeEquations<Thermo>::ReconstructedNormalizedNDF(double nu, bool use_reg) const
{
    if (!std::isfinite(nu) || nu <= 0.)
        return 0.;

    const auto d = ReconstructedNDFData(use_reg);
    if (!d.valid)
        return 0.;

    const double nu_nucl = vnucl_;
    double nbar_p = 0., nbar_ln = 0.;

    // Pareto contribution
    if (d.alpha > 0. && d.k > 1. && nu >= nu_nucl)
    {
        const double lg   = std::log(d.k) + d.k * std::log(nu_nucl) - (d.k + 1.) * std::log(nu);
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
double ThreeEquations<Thermo>::ReconstructedNDF(double nu, bool use_reg) const
{
    const auto d = ReconstructedNDFData(use_reg);
    if (!d.valid)
        return 0.;
    const double n = d.Ns * ReconstructedNormalizedNDF(nu, use_reg);
    return (std::isfinite(n) && n >= 0.) ? n : 0.;
}

template <ThermoMap Thermo>
void ThreeEquations<Thermo>::ReconstructedNDF(const Eigen::VectorXd& nu,
                                              Eigen::VectorXd& n,
                                              bool use_reg) const
{
    n.resize(nu.size());
    for (int i = 0; i < nu.size(); ++i)
        n(i) = ReconstructedNDF(nu(i), use_reg);
}

#if defined(MOM_USE_DICTIONARY)
// ============================================================================
// ParseConfig — OpenSMOKE++ dictionary → ThreeEquations::Config
// ============================================================================

template <ThermoMap Thermo>
template <typename DictType>
std::expected<typename ThreeEquations<Thermo>::Config, std::string>
ThreeEquations<Thermo>::ParseConfig(DictType& dict)
{
    // SetGrammar validates the dictionary against the grammar rules defined in
    // ThreeEquations_Grammar::DefineRules().  On any violation (missing
    // mandatory keyword, unknown keyword, duplicate, conflicting keyword) it
    // calls Dictionary::ErrorMessage() which throws std::runtime_error.
    // That exception is intentionally NOT caught here so it propagates to the
    // caller and stops the simulation — silently continuing with default values
    // when the input file is malformed is never acceptable.
    ThreeEquations_Grammar grammar;
    dict.SetGrammar(grammar);

    Config cfg;

    if (dict.CheckOption("@ThreeEquations"))
        dict.ReadBool("@ThreeEquations", cfg.is_active);

    if (dict.CheckOption("@NucleationModel"))    dict.ReadInt("@NucleationModel",    cfg.nucleation_model);
    if (dict.CheckOption("@SurfaceGrowthModel")) dict.ReadInt("@SurfaceGrowthModel", cfg.surface_growth_model);
    if (dict.CheckOption("@OxidationModel"))     dict.ReadInt("@OxidationModel",     cfg.oxidation_model);
    if (dict.CheckOption("@CondensationModel"))  dict.ReadInt("@CondensationModel",  cfg.condensation_model);
    if (dict.CheckOption("@CoagulationModel"))   dict.ReadInt("@CoagulationModel",   cfg.coagulation_model);
    if (dict.CheckOption("@ThermophoreticModel")) dict.ReadInt("@ThermophoreticModel", cfg.thermophoretic_model);

    if (dict.CheckOption("@SurfaceChemistryModel"))
    {
        std::string m;
        dict.ReadString("@SurfaceChemistryModel", m);
        if (m != "rcpah" && m != "hmom" &&
            m != "RCPAH" && m != "HMOM" &&
            m != "rc-pah" && m != "RC-PAH")
            return std::unexpected(std::string{
                "@SurfaceChemistryModel: allowed: rcpah | hmom (and their variants)"});
        cfg.surface_chemistry_model = m;
    }

    if (dict.CheckOption("@SimplifiedPAHMass"))
        dict.ReadBool("@SimplifiedPAHMass", cfg.simplified_pah_mass);

    if (dict.CheckOption("@PAH"))
        dict.ReadString("@PAH", cfg.pah_species);

    if (dict.CheckOption("@GasClosureDummySpecies"))
        dict.ReadString("@GasClosureDummySpecies", cfg.gas_closure_dummy_species);

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

    if (dict.CheckOption("@RadiativeHeatTransfer"))
        dict.ReadBool("@RadiativeHeatTransfer", cfg.radiative_heat_transfer);

    if (dict.CheckOption("@PlanckCoefficient"))
        dict.ReadString("@PlanckCoefficient", cfg.planck_coefficient);

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

    if (dict.CheckOption("@DimerModel"))
    {
        std::string m;
        dict.ReadString("@DimerModel", m);
        if (m != "qssa-rodrigues")
            return std::unexpected(std::string{"@DimerModel: allowed: qssa-rodrigues"});
        cfg.dimer_model = m;
    }

    if (dict.CheckOption("@epsNucleation"))   dict.ReadDouble("@epsNucleation",   cfg.eps_nucleation);
    if (dict.CheckOption("@epsCondensation")) dict.ReadDouble("@epsCondensation", cfg.eps_condensation);
    if (dict.CheckOption("@epsCoagulation"))  dict.ReadDouble("@epsCoagulation",  cfg.eps_coagulation);

    if (dict.CheckOption("@StickingCoefficientModel"))
        dict.ReadString("@StickingCoefficientModel", cfg.sticking_model);

    if (dict.CheckOption("@StickingCoefficientConstant"))
        dict.ReadDouble("@StickingCoefficientConstant", cfg.sticking_coeff_constant);

    if (dict.CheckOption("@CorrectionCoefficientPAHPAH"))
        dict.ReadDouble("@CorrectionCoefficientPAHPAH", cfg.correction_coeff_pah_pah);

    if (dict.CheckOption("@DebugMode"))
        dict.ReadBool("@DebugMode", cfg.debug_mode);

    return cfg;
}
#endif // MOM_USE_DICTIONARY

} // namespace MOM
