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

/**
 * @file HMOM.tpp
 * @brief Template implementation for the HMOM soot source-term model.
 */

namespace MOM
{

// Local geometry helpers.
namespace
{
// K_diam = (6/pi)^(1/3), K_spher = (36*pi)^(1/3)
inline double K_diam_HMOM(double pi) noexcept
{
    return std::pow(6. / pi, 1. / 3.);
}

inline double K_spher_HMOM(double pi) noexcept
{
    return std::pow(36. * pi, 1. / 3.);
}

// Returns fractal geometry coefficients {Av, As, K}.
inline std::array<double, 3> FractalGeometry(int model, double pi) noexcept
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

// Returns collision-diameter geometry coefficients {D, Av, As, K}.
inline std::array<double, 4> CollisionGeometry(int model, double pi) noexcept
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

template <ThermoMap Thermo> HMOM<Thermo>::HMOM(const Thermo& thermo) : thermo_(thermo)
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

template <ThermoMap Thermo> void HMOM<Thermo>::MemoryAllocation()
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

    NL_   = 0.;
    NLVL_ = 0.;
    NLSL_ = 0.;

    // Initial normalized moments for solver startup.
    const double Nseed = 1.e10;
    const double eta   = 0.90;
    const double gamma = 2.0;
    const double N0s   = eta * Nseed;
    const double NLs   = (1.0 - eta) * Nseed;
    const double gS    = std::pow(gamma, 2. / 3.);

    initial_moments_cache_(0) = Nseed / this->Nav_mol_;
    initial_moments_cache_(1) = (N0s + gamma * NLs) / this->Nav_mol_;
    initial_moments_cache_(2) = (N0s + gS * NLs) / this->Nav_mol_;
    initial_moments_cache_(3) = N0s / this->Nav_mol_;
}

template <ThermoMap Thermo> void HMOM<Thermo>::Precalculations()
{
    Cfm_                = std::sqrt(this->pi_ * this->kB_ / 2.0 / this->rho_particle_);
    const double K_diam = K_diam_HMOM(this->pi_);
    betaN_TV_           = 2.2 * 4. * this->sqrt2_ * K_diam * K_diam * Cfm_;

    // Update geometry coefficients from the selected sub-models.
    {
        auto [Av, As, K] = FractalGeometry(static_cast<int>(fractal_diameter_model_), this->pi_);
        Av_fractal_      = Av;
        As_fractal_      = As;
        K_fractal_       = K;
    }
    {
        auto [D, Av, As, K] =
            CollisionGeometry(static_cast<int>(collision_diameter_model_), this->pi_);
        D_collisional_  = D;
        Av_collisional_ = Av;
        As_collisional_ = As;
        K_collisional_  = K;
    }

    SetPAH(pah_species_);
}

template <ThermoMap Thermo> void HMOM<Thermo>::SetPAH(std::string_view name)
{
    const auto pah_index = thermo_.IndexOfSpecies(name);
    if (pah_index < 0)
        throw std::runtime_error("[HMOM] PAH species not found in mechanism: " + std::string(name));

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

template <ThermoMap Thermo> void HMOM<Thermo>::SetGasClosureDummySpecies(std::string_view name)
{
    this->closure_dummy_species_ = std::string(name);

    if (name == "none")
    {
        this->closure_dummy_index_           = -1;
        this->is_closure_dummy_species_ = false;
        return;
    }

    this->closure_dummy_index_ = thermo_.IndexOfSpecies(name);
    if (this->closure_dummy_index_ < 0)
        throw std::runtime_error("[HMOM] Dummy species not found: " + this->closure_dummy_species_);

    if (this->closure_dummy_index_ == pah_index_)
        throw std::runtime_error("[HMOM] Dummy species cannot be the same as the PAH precursor.");

    if (this->closure_dummy_index_ == index_H_ || this->closure_dummy_index_ == index_H2_ ||
        this->closure_dummy_index_ == index_O2_ || this->closure_dummy_index_ == index_OH_ ||
        this->closure_dummy_index_ == index_H2O_ || this->closure_dummy_index_ == index_C2H2_)
        throw std::runtime_error("[HMOM] Dummy species cannot be H, H2, O2, OH, H2O, or C2H2.");

    this->is_closure_dummy_species_ = true;
}

template <ThermoMap Thermo> void HMOM<Thermo>::SetStickingCoefficientModel(std::string_view label)
{
    if (label == "constant")
        sticking_model_ = StickingModel::Constant;
    else if (label == "pah-dependent")
        sticking_model_ = StickingModel::PAH4;
    else
        throw std::runtime_error("[HMOM] Unknown sticking coefficient model: " + std::string(label));
}

template <ThermoMap Thermo>
void HMOM<Thermo>::SetState(double T, double P_Pa, const double* Y) noexcept
{
    const double cTot = this->template UpdateMixtureState<>(
        T, P_Pa, Y, thermo_);

    // Mass fractions used by the smooth active-site damping factor.
    mass_fraction_H_  = (index_H_ >= 0) ? std::max(Y[index_H_], 0.) : 0.;
    mass_fraction_OH_ = (index_OH_ >= 0) ? std::max(Y[index_OH_], 0.) : 0.;

    conc_H_    = this->SpeciesConcentrationMolCm3(index_H_, Y, cTot, thermo_);
    conc_OH_   = this->SpeciesConcentrationMolCm3(index_OH_, Y, cTot, thermo_);
    conc_O2_   = this->SpeciesConcentrationMolCm3(index_O2_, Y, cTot, thermo_);
    conc_H2_   = this->SpeciesConcentrationMolCm3(index_H2_, Y, cTot, thermo_);
    conc_H2O_  = this->SpeciesConcentrationMolCm3(index_H2O_, Y, cTot, thermo_);
    conc_C2H2_ = this->SpeciesConcentrationMolCm3(index_C2H2_, Y, cTot, thermo_);

    // PAH concentration [mol/cm3].
    conc_PAH_ = this->SpeciesConcentrationMolCm3(pah_index_, Y, cTot, thermo_);
}

template <ThermoMap Thermo> void HMOM<Thermo>::SetMoments(std::span<const double> m) noexcept
{
    assert(m.size() == static_cast<std::size_t>(Base::n_equations) &&
           "[HMOM::SetMoments] expected exactly 4 moment values.");
    SetNormalizedMoments(m[0], m[1], m[2], m[3]);
}

template <ThermoMap Thermo>
void HMOM<Thermo>::SetNormalizedMoments(double M00_norm,
                                        double M10_norm,
                                        double M01_norm,
                                        double N0_norm) noexcept
{
    M00_normalized_ = M00_norm;
    M10_normalized_ = M10_norm;
    M01_normalized_ = M01_norm;
    N0_normalized_  = N0_norm;
    GetMoments();
}

template <ThermoMap Thermo> void HMOM<Thermo>::GetMoments()
{
    double M00_raw = M00_normalized_ * this->Nav_mol_;
    double M10_raw = M10_normalized_ * V0_ * this->Nav_mol_;
    double M01_raw = M01_normalized_ * S0_ * this->Nav_mol_;
    double N0_raw  = N0_normalized_ * this->Nav_mol_;

    if (!std::isfinite(M00_raw))
        M00_raw = 0.;
    if (!std::isfinite(M10_raw))
        M10_raw = 0.;
    if (!std::isfinite(M01_raw))
        M01_raw = 0.;
    if (!std::isfinite(N0_raw))
        N0_raw = 0.;

    M00_raw = std::max(M00_raw, 0.);
    M10_raw = std::max(M10_raw, 0.);
    M01_raw = std::max(M01_raw, 0.);
    N0_raw  = std::max(N0_raw, 0.);

    if (M00_raw < kSootNumberFloor || M10_raw < kSootVolumeFloor || M01_raw < kSootSurfaceFloor)
    {
        M00_ = M10_ = M01_ = N0_ = NL_ = NLVL_ = NLSL_ = 0.;
        return;
    }

    N0_raw = std::min(N0_raw, M00_raw);
    if (V0_ > 0.)
        N0_raw = std::min(N0_raw, M10_raw / V0_);
    if (S0_ > 0.)
        N0_raw = std::min(N0_raw, M01_raw / S0_);
    N0_raw = std::max(N0_raw, 0.);

    M00_ = M00_raw;
    M10_ = M10_raw;
    M01_ = M01_raw;
    N0_  = N0_raw;

    NL_   = M00_ - N0_;
    NLVL_ = M10_ - V0_ * N0_;
    NLSL_ = M01_ - S0_ * N0_;

    if (NL_ < 0. && std::fabs(NL_) < 1.e-12 * std::max(M00_, 1.))
        NL_ = 0.;
    if (NLVL_ < 0. && std::fabs(NLVL_) < 1.e-12 * std::max(M10_, 1.))
        NLVL_ = 0.;
    if (NLSL_ < 0. && std::fabs(NLSL_) < 1.e-12 * std::max(M01_, 1.))
        NLSL_ = 0.;

    NL_   = std::max(NL_, 0.);
    NLVL_ = std::max(NLVL_, 0.);
    NLSL_ = std::max(NLSL_, 0.);
}

template <ThermoMap Thermo> bool HMOM<Thermo>::HasSoot() const noexcept
{
    return (M00_ > kSootNumberFloor && M10_ > kSootVolumeFloor && M01_ > kSootSurfaceFloor &&
            std::isfinite(M00_) && std::isfinite(M10_) && std::isfinite(M01_));
}

template <ThermoMap Thermo> double HMOM<Thermo>::SafePowPositive(double x, double a) const noexcept
{
    if (!std::isfinite(x) || x <= 0.)
        return 0.;
    const double y = a * std::log(x);
    if (y > 700.)
        return std::exp(700.);
    if (y < -700.)
        return 0.;
    return std::exp(y);
}

template <ThermoMap Thermo> double HMOM<Thermo>::GetMoment(double i, double j) const noexcept
{
    if (!HasSoot())
        return 0.;

    double moment = 0.;
    if (N0_ > 0.)
    {
        const double small = N0_ * SafePowPositive(V0_, i) * SafePowPositive(S0_, j);
        if (std::isfinite(small))
            moment += small;
    }
    if (NL_ > kSootNumberFloor && NLVL_ > kSootVolumeFloor && NLSL_ > kSootSurfaceFloor)
    {
        const double VL    = NLVL_ / NL_;
        const double SL    = NLSL_ / NL_;
        const double large = NL_ * SafePowPositive(VL, i) * SafePowPositive(SL, j);
        if (std::isfinite(large))
            moment += large;
    }
    if (!std::isfinite(moment) || moment < 0.)
        return 0.;
    return moment;
}

template <ThermoMap Thermo> double HMOM<Thermo>::GetMissingMoment(double i, double j) const noexcept
{
    if (!HasSoot())
        return 0.;
    if (NL_ <= kSootNumberFloor || NLVL_ <= kSootVolumeFloor || NLSL_ <= kSootSurfaceFloor)
        return 0.;
    const double VL     = NLVL_ / NL_;
    const double SL     = NLSL_ / NL_;
    const double moment = NL_ * SafePowPositive(VL, i) * SafePowPositive(SL, j);
    if (!std::isfinite(moment) || moment < 0.)
        return 0.;
    return moment;
}

template <ThermoMap Thermo> double HMOM<Thermo>::GetBetaC() const noexcept
{
    const double K_diam = K_diam_HMOM(this->pi_);

    const double betaC =
        std::pow(K_collisional_, 2.) * std::pow(dimer_volume_, -3. / 6.) *
            GetMoment(2. * Av_collisional_, 2. * As_collisional_) +
        2. * K_diam * K_collisional_ * std::pow(dimer_volume_, -1. / 6.) *
            GetMoment(Av_collisional_, As_collisional_) +
        std::pow(K_diam, 2.) * std::pow(dimer_volume_, 1. / 6.) * GetMoment(0., 0.) +
        0.5 * std::pow(K_collisional_, 2.) * std::pow(dimer_volume_, 3. / 6.) *
            GetMoment(2. * Av_collisional_ - 1., 2. * As_collisional_) +
        2. * K_diam * K_collisional_ * std::pow(dimer_volume_, 5. / 6.) *
            GetMoment(Av_collisional_ - 1., As_collisional_) +
        0.5 * std::pow(K_diam, 2.) * std::pow(dimer_volume_, 7. / 6.) * GetMoment(-1., 0.);

    return Cfm_ * std::sqrt(this->T_) * betaC;
}

template <ThermoMap Thermo> void HMOM<Thermo>::DimerConcentration()
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

    const double beta_pah_pah = 0.5 * stickCoeff * KfmPAH;
    const double C_PAH        = std::max(conc_PAH_, 0.) * 1.e6; // mol/cm3 -> mol/m3
    const double reduced_rate = beta_pah_pah * C_PAH * C_PAH;

    dimerization_rate_ = reduced_rate;
    if (dimerization_rate_ <= 0. || !std::isfinite(dimerization_rate_))
    {
        dimerization_rate_ = 0.;
        return;
    }

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

template <ThermoMap Thermo> double HMOM<Thermo>::PAHDimerizationRate() const noexcept
{
    return 2. * dimerization_rate_ * this->Nav_mol_ / 1000.;
}

template <ThermoMap Thermo> void HMOM<Thermo>::CalculateAlphaCoefficient()
{
    alpha_ = 1.;
    if (!std::isfinite(this->T_) || this->T_ <= 0.)
        return;

    const double a = surf_dens_a1_ + surf_dens_a2_ * this->T_;
    const double b = surf_dens_b1_ + surf_dens_b2_ * this->T_;

    double VL = V0_;
    if (NL_ > kSootNumberFloor && NLVL_ > kSootVolumeFloor)
        VL = NLVL_ / NL_;
    if (!std::isfinite(VL) || VL <= 0.)
        VL = V0_;

    const double mC  = this->WC_ / this->Nav_kmol_;
    const double mu1 = VL * this->rho_particle_ / mC;
    if (!std::isfinite(mu1) || mu1 <= 1.)
        return;

    const double logMu1 = std::log(mu1);
    if (!std::isfinite(logMu1) || std::fabs(logMu1) < 1.e-12)
        return;

    // Appel et al. surface-density correction for mature particles.
    alpha_ = std::tanh(a / logMu1 + b);
    if (!std::isfinite(alpha_))
        alpha_ = 1.;
    alpha_ = std::max(0., std::min(alpha_, 1.));
}

template <ThermoMap Thermo> void HMOM<Thermo>::SootKineticConstants()
{
    const double T   = this->T_;
    const double k1f = A1f_ * std::pow(T, n1f_) * std::exp(-E1f_ / T);
    const double k1b = A1b_ * std::pow(T, n1b_) * std::exp(-E1b_ / T);
    const double k2f = A2f_ * std::pow(T, n2f_) * std::exp(-E2f_ / T);
    const double k2b = A2b_ * std::pow(T, n2b_) * std::exp(-E2b_ / T);
    const double k3f = A3f_ * std::pow(T, n3f_) * std::exp(-E3f_ / T);
    const double k3b = A3b_ * std::pow(T, n3b_) * std::exp(-E3b_ / T);
    const double k4  = A4_ * std::pow(T, n4_) * std::exp(-E4_ / T);

    const double ratio   = k1b * conc_H2O_ + k2b * conc_H2_ + k3b * conc_H_ + k4 * conc_C2H2_;
    double conc_sootStar = 0.;
    if (ratio > 0.)
        conc_sootStar = (k1f * conc_OH_ + k2f * conc_H_ + k3f) / ratio;

    // Smooth low-radical damping of the active-site concentration.
    {
        constexpr double YH_threshold  = 2.e-9;
        constexpr double YOH_threshold = 2.e-8;
        const double rampH  = 0.5 * (1. + std::tanh((mass_fraction_H_  - YH_threshold)
                                                     / (0.5 * YH_threshold)));
        const double rampOH = 0.5 * (1. + std::tanh((mass_fraction_OH_ - YOH_threshold)
                                                     / (0.5 * YOH_threshold)));
        // Product form damps the rate when both radical pools are depleted.
        conc_sootStar *= rampH * rampOH;
    }

    conc_sootStar = conc_sootStar / (1. + conc_sootStar);
    conc_sootStar = std::max(conc_sootStar, 0.);

    ksg_            = k4 * conc_C2H2_ * conc_sootStar;
    const double k6 = 8.94 * eff6_ * std::sqrt(T) * this->Nav_mol_;
    kox_O2_         = A5_ * std::pow(T, n5_) * std::exp(-E5_ / T) * conc_O2_ * conc_sootStar;
    kox_OH_         = (0.5 / (alpha_ * surface_density_)) * k6 * conc_OH_;
    kox_            = kox_O2_ + kox_OH_;
}

template <ThermoMap Thermo> void HMOM<Thermo>::SootNucleationM4()
{
    this->source_nucleation_.setZero();
    double nuc_N0 = 0.;
    if (conc_DIMER_ > 0. && betaN_ > 0. && std::isfinite(conc_DIMER_) && std::isfinite(betaN_))
        nuc_N0 = 0.5 * betaN_ * conc_DIMER_ * conc_DIMER_ * this->Nav_mol_;
    if (!std::isfinite(nuc_N0) || nuc_N0 < 0.)
        nuc_N0 = 0.;
    for (unsigned i = 0; i < 4u; ++i)
        this->source_nucleation_(i) = nuc_N0;
}

template <ThermoMap Thermo> void HMOM<Thermo>::SootSurfaceGrowthM4()
{
    if (!HasSoot())
        return;

    this->source_growth_(0) = 0.;
    this->source_growth_(1) =
        ksg_ * (alpha_ * surface_density_) * VC2_ * GetMoment(0., 1.) / this->Nav_mol_ / V0_;
    this->source_growth_(2) = ksg_ * (alpha_ * surface_density_) * VC2_ * K_fractal_ *
                              GetMoment(Av_fractal_, As_fractal_ + 2.) / this->Nav_mol_ / S0_;
    this->source_growth_(3) = -ksg_ * (alpha_ * surface_density_) * S0_ * N0_ / this->Nav_mol_;
}

template <ThermoMap Thermo> void HMOM<Thermo>::SootOxidationM4()
{
    if (!HasSoot())
        return;

    const double coeff = kox_ * (alpha_ * surface_density_) * VC2_;
    const double M01   = GetMissingMoment(0., 1.);
    const double M_12  = GetMissingMoment(-1., 2.);

    this->source_oxidation_(0) = -coeff * N0_ * S0_ / V0_ / this->Nav_mol_;
    this->source_oxidation_(1) = -coeff / V0_ / this->Nav_mol_ * M01;
    this->source_oxidation_(2) =
        -coeff / S0_ / this->Nav_mol_ * (S0_ * S0_ * N0_ / V0_ + 2. / 3. * (M_12 - S0_ * N0_));
    this->source_oxidation_(3) = this->source_oxidation_(0);
}

template <ThermoMap Thermo> void HMOM<Thermo>::SootCondensationM4()
{
    if (!HasSoot())
        return;

    const double K_diam = K_diam_HMOM(this->pi_);
    const double D_DIM  = K_diam * std::pow(dimer_volume_, 1. / 3.);
    const double D_NUCL = K_diam * std::pow(V0_, 1. / 3.);
    const double sqrtDv = std::sqrt(dimer_volume_);
    const double sqrtT  = std::sqrt(this->T_);
    const double cd     = conc_DIMER_;

    this->source_condensation_(0) = 0.;

    this->source_condensation_(1) =
        D_DIM * D_DIM * (GetMoment(0., 0.) + 0.5 * dimer_volume_ * GetMoment(-1., 0.)) +
        K_collisional_ * K_collisional_ *
            (GetMoment(2. * Av_collisional_, 2. * As_collisional_) +
             0.5 * dimer_volume_ * GetMoment(2. * Av_collisional_ - 1., 2. * As_collisional_)) +
        2. * D_DIM * K_collisional_ *
            (GetMoment(Av_collisional_, As_collisional_) +
             0.5 * dimer_volume_ * GetMoment(Av_collisional_ - 1., As_collisional_));
    this->source_condensation_(1) = Cfm_ * sqrtDv * cd * sqrtT * this->source_condensation_(1) / V0_;

    this->source_condensation_(2) =
        D_DIM * D_DIM *
            (GetMoment(Av_fractal_, As_fractal_ + 1.) +
             0.5 * dimer_volume_ * GetMoment(Av_fractal_ - 1., As_fractal_ + 1.)) +
        K_collisional_ * K_collisional_ *
            (GetMoment(Av_fractal_ + 2. * Av_collisional_, As_fractal_ + 1. + 2. * As_collisional_) +
             0.5 * dimer_volume_ *
                 GetMoment(Av_fractal_ + 2. * Av_collisional_ - 1.,
                           As_fractal_ + 1. + 2. * As_collisional_)) +
        2. * D_DIM * K_collisional_ *
            (GetMoment(Av_fractal_ + Av_collisional_, As_fractal_ + 1. + As_collisional_) +
             0.5 * dimer_volume_ *
                 GetMoment(Av_fractal_ + Av_collisional_ - 1., As_fractal_ + 1. + As_collisional_));
    this->source_condensation_(2) =
        Cfm_ * sqrtDv * cd * sqrtT * this->source_condensation_(2) / S0_ * K_fractal_;

    this->source_condensation_(3) = -Cfm_ / sqrtDv * cd * sqrtT * (1. + 0.5 * dimer_volume_ / V0_) *
                                    (D_DIM + D_NUCL) * (D_DIM + D_NUCL) * N0_;
}

template <ThermoMap Thermo> void HMOM<Thermo>::SootCoagulationM4()
{
    if (!HasSoot())
        return;
    SootCoagulationSmallSmallM4();
    SootCoagulationSmallLargeM4();
    SootCoagulationLargeLargeM4();
}

template <ThermoMap Thermo> void HMOM<Thermo>::SootCoagulationSmallSmallM4()
{
    const double K_diam = K_diam_HMOM(this->pi_);
    const double DcNUCL = K_diam * std::pow(V0_, 1. / 3.);
    const double S00    = this->SphereSurfaceFromVolume(2. * V0_);
    const double beta00 =
        2.20 * Cfm_ * std::sqrt(2. / V0_) * std::pow(2. * DcNUCL, 2.) * std::sqrt(this->T_);

    source_coagulation_ss_(0) = -0.5 * beta00 * N0_ * N0_ / this->Nav_mol_;
    source_coagulation_ss_(1) = 0.;
    source_coagulation_ss_(2) =
        0.5 * beta00 * S00 / S0_ * N0_ * N0_ / this->Nav_mol_ + 2. * source_coagulation_ss_(0);
    source_coagulation_ss_(3) = 2. * source_coagulation_ss_(0);
}

template <ThermoMap Thermo> void HMOM<Thermo>::SootCoagulationSmallLargeM4()
{
    const double K_diam = K_diam_HMOM(this->pi_);
    const double DcNUCL = K_diam * std::pow(V0_, 1. / 3.);
    const double sqrtT  = std::sqrt(this->T_);

    {
        const double psi0 =
            std::pow(V0_, -1. / 2.) *
            (K_collisional_ * K_collisional_ *
                 GetMissingMoment(2. * Av_collisional_ - 0.5, 2. * As_collisional_) +
             DcNUCL * DcNUCL * GetMissingMoment(-0.5, 0.) +
             2. * DcNUCL * K_collisional_ * GetMissingMoment(Av_collisional_ - 0.5, As_collisional_));

        double psi1 =
            std::pow(V0_, -1. / 2.) *
            (K_collisional_ * K_collisional_ *
                 GetMissingMoment(2. * Av_collisional_ + 0.5, 2. * As_collisional_) +
             DcNUCL * DcNUCL * GetMissingMoment(0.5, 0.) +
             2. * DcNUCL * K_collisional_ * GetMissingMoment(Av_collisional_ + 0.5, As_collisional_));
        psi1 += V0_ * psi0;

        source_coagulation_sl_(0) = 0.;
        if (psi0 * psi1 >= 0.)
            source_coagulation_sl_(0) =
                -2.2 * Cfm_ * sqrtT * N0_ / this->Nav_mol_ * std::sqrt(psi0 * psi1);
    }

    source_coagulation_sl_(1) = 0.;

    {
        const double psi0 =
            std::pow(V0_, -1. / 2.) *
            (K_collisional_ * K_collisional_ *
                 GetMissingMoment(2. * Av_collisional_ - 0.5 + Av_fractal_,
                                  2. * As_collisional_ + As_fractal_ + 1.) +
             DcNUCL * DcNUCL * GetMissingMoment(-0.5 + Av_fractal_, As_fractal_ + 1.) +
             2. * DcNUCL * K_collisional_ *
                 GetMissingMoment(Av_collisional_ - 0.5 + Av_fractal_,
                                  As_collisional_ + As_fractal_ + 1.));

        double psi1 = std::pow(V0_, -1. / 2.) *
                      (K_collisional_ * K_collisional_ *
                           GetMissingMoment(2. * Av_collisional_ + 0.5 + Av_fractal_,
                                            2. * As_collisional_ + As_fractal_ + 1.) +
                       DcNUCL * DcNUCL * GetMissingMoment(0.5 + Av_fractal_, As_fractal_ + 1.) +
                       2. * DcNUCL * K_collisional_ *
                           GetMissingMoment(Av_collisional_ + 0.5 + Av_fractal_,
                                            As_collisional_ + As_fractal_ + 1.));
        psi1 += V0_ * psi0;

        source_coagulation_sl_(2) = source_coagulation_sl_(0);
        if (psi0 * psi1 >= 0.)
            source_coagulation_sl_(2) += 2.2 * Cfm_ * sqrtT * N0_ * V0_ * K_fractal_ / S0_ /
                                         this->Nav_mol_ * std::sqrt(psi0 * psi1);
    }

    source_coagulation_sl_(3) = source_coagulation_sl_(0);
}

template <ThermoMap Thermo> void HMOM<Thermo>::SootCoagulationLargeLargeM4()
{
    source_coagulation_ll_.setZero();

    const double psi0 = 2. * K_collisional_ * K_collisional_ *
                        (GetMissingMoment(2. * Av_collisional_ - 0.5, 2. * As_collisional_) *
                             GetMissingMoment(-0.5, 0.) +
                         GetMissingMoment(Av_collisional_ - 0.5, As_collisional_) *
                             GetMissingMoment(Av_collisional_ - 0.5, As_collisional_));

    const double psi1 = 2. * K_collisional_ * K_collisional_ *
                        (GetMissingMoment(2. * Av_collisional_ + 0.5, 2. * As_collisional_) *
                             GetMissingMoment(-0.5, 0.) +
                         GetMissingMoment(Av_collisional_ + 0.5, As_collisional_) *
                             GetMissingMoment(Av_collisional_ - 0.5, As_collisional_) +
                         GetMissingMoment(2. * Av_collisional_ - 0.5, 2. * As_collisional_) *
                             GetMissingMoment(0.5, 0.));

    if (psi0 * psi1 >= 0.)
        source_coagulation_ll_(0) =
            -0.5 * 2.2 * Cfm_ * std::sqrt(this->T_) / this->Nav_mol_ * std::sqrt(psi0 * psi1);
}

template <ThermoMap Thermo> void HMOM<Thermo>::SootCoagulationContinuousM4()
{
    double lambda = 8.2057e-5 / this->sqrt2_ / std::pow(200.e-12, 2.) / this->Nav_mol_;
    lambda        = 1.257 * lambda * this->T_ / (this->P_Pa_ / 101325.);
    const double betai0 = 2. * this->kB_ * this->T_ / 3. / this->mu_;

    SootCoagulationContinuousSmallSmallM4(lambda, betai0);
    SootCoagulationContinuousSmallLargeM4(lambda, betai0);
    SootCoagulationContinuousLargeLargeM4(lambda, betai0);
}

template <ThermoMap Thermo> void HMOM<Thermo>::SootCoagulationContinuousSmallSmallM4(double lambda, double betai0)
{
    const double K_diam = K_diam_HMOM(this->pi_);
    const double DcNUCL = K_diam * std::pow(V0_, 1. / 3.);
    const double S00    = this->SphereSurfaceFromVolume(2. * V0_);
    const double CC0    = 1. + lambda / DcNUCL;
    const double beta00 = betai0 * (2. * CC0 / DcNUCL) * (2. * DcNUCL);

    source_coagulation_cont_ss_(0) = -0.5 * beta00 * N0_ * N0_ / this->Nav_mol_;
    source_coagulation_cont_ss_(1) = 0.;
    source_coagulation_cont_ss_(2) =
        0.5 * beta00 * S00 / S0_ * N0_ * N0_ / this->Nav_mol_ + 2. * source_coagulation_cont_ss_(0);
    source_coagulation_cont_ss_(3) = 2. * source_coagulation_cont_ss_(0);
}

template <ThermoMap Thermo> void HMOM<Thermo>::SootCoagulationContinuousSmallLargeM4(double lambda, double betai0)
{
    const double K_diam = K_diam_HMOM(this->pi_);
    const double DcNUCL = K_diam * std::pow(V0_, 1. / 3.);

    source_coagulation_cont_sl_(0) =
        (2. + lambda / DcNUCL) * GetMissingMoment(0., 0.) +
        (DcNUCL + lambda) / K_collisional_ * GetMissingMoment(-Av_collisional_, -As_collisional_) +
        (1. + lambda / DcNUCL) * K_collisional_ / DcNUCL *
            GetMissingMoment(Av_collisional_, As_collisional_) +
        lambda * DcNUCL * K_collisional_ * K_collisional_ *
            GetMissingMoment(-2. * Av_collisional_, -2. * As_collisional_);
    source_coagulation_cont_sl_(0) *= -N0_ / this->Nav_mol_ * betai0;

    source_coagulation_cont_sl_(1) = 0.;

    source_coagulation_cont_sl_(2) =
        (2. + lambda / DcNUCL) * GetMissingMoment(Av_fractal_, As_fractal_ + 1.) +
        (DcNUCL + lambda) / K_collisional_ *
            GetMissingMoment(-Av_collisional_ + Av_fractal_, -As_collisional_ + As_fractal_ + 1.) +
        (1. + lambda / DcNUCL) * K_collisional_ / DcNUCL *
            GetMissingMoment(Av_collisional_ + Av_fractal_, As_collisional_ + As_fractal_ + 1.) +
        lambda * DcNUCL * K_collisional_ * K_collisional_ *
            GetMissingMoment(-2. * Av_collisional_ + Av_fractal_,
                             -2. * As_collisional_ + As_fractal_ + 1.);
    source_coagulation_cont_sl_(2) =
        N0_ * V0_ * K_fractal_ / S0_ / this->Nav_mol_ * betai0 * source_coagulation_cont_sl_(2) +
        source_coagulation_cont_sl_(0);

    source_coagulation_cont_sl_(3) = source_coagulation_cont_sl_(0);
}

template <ThermoMap Thermo> void HMOM<Thermo>::SootCoagulationContinuousLargeLargeM4(double lambda, double betai0)
{

    const double val =
        GetMissingMoment(0., 0.) * GetMissingMoment(0., 0.) +
        lambda / K_collisional_ *
            (GetMissingMoment(0., 0.) * GetMissingMoment(-Av_collisional_, -As_collisional_) +
             GetMissingMoment(Av_collisional_, As_collisional_) *
                 GetMissingMoment(-2. * Av_collisional_, -2. * As_collisional_) +
             GetMissingMoment(Av_collisional_, As_collisional_) *
                 GetMissingMoment(-Av_collisional_, -As_collisional_));

    source_coagulation_cont_ll_(0) = -0.5 * betai0 / this->Nav_mol_ * val;
    source_coagulation_cont_ll_(1) = 0.;
    source_coagulation_cont_ll_(2) = 0.;
    source_coagulation_cont_ll_(3) = 0.;
}

template <ThermoMap Thermo> void HMOM<Thermo>::ComputeSources() noexcept
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

    if (!this->is_active_)
    {
        if (this->gas_consumption_)
            this->omega_gas_.setZero();
        return;
    }

    if (surface_density_correction_)
        CalculateAlphaCoefficient();

    DimerConcentration();

    if (nucleation_model_ > 0)
        SootNucleationM4();

    if (surface_growth_model_ > 0 || oxidation_model_ > 0)
    {
        SootKineticConstants();
        if (surface_growth_model_ > 0)
            SootSurfaceGrowthM4();
        if (oxidation_model_ > 0)
            SootOxidationM4();
    }

    if (condensation_model_ > 0)
        SootCondensationM4();
    if (coagulation_model_ > 0)
        SootCoagulationM4();
    if (coagulation_continuous_model_ > 0)
        SootCoagulationContinuousM4();

    // Sanitize per-process sources and combine the total source vector.
    for (unsigned i = 0; i < 4u; ++i)
    {
        auto nanZ = [](double& v)
        {
            if (std::isnan(v))
                v = 0.;
        };
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

        const double disc =
            source_coagulation_ss_(i) + source_coagulation_sl_(i) + source_coagulation_ll_(i);
        const double cont = source_coagulation_cont_ss_(i) + source_coagulation_cont_sl_(i) +
                            source_coagulation_cont_ll_(i);

        source_coagulation_discrete_(i)   = disc;
        source_coagulation_continuous_(i) = cont;

        if (disc == 0.)
            source_coagulation_all_(i) = cont;
        else if (cont == 0.)
            source_coagulation_all_(i) = disc;
        else
            source_coagulation_all_(i) = cont * disc / (cont + disc);

        this->source_coagulation_(i) = source_coagulation_all_(i);

        this->source_all_(i) = this->source_nucleation_(i) + this->source_growth_(i) +
                               this->source_oxidation_(i) + this->source_condensation_(i) +
                               source_coagulation_all_(i);
    }

    if (this->gas_consumption_)
        CalculateOmegaGas();
}

template <ThermoMap Thermo> void HMOM<Thermo>::CalculateOmegaGas() noexcept
{
    this->omega_gas_.setZero();
    omega_gas_oxidation_.setZero();
    if (!this->gas_consumption_)
        return;

    if (nucleation_model_ > 0 && pah_index_ >= 0)
    {
        const double R_PAH = PAHDimerizationRate(); // [kmol/m3/s]
        if (R_PAH > 0. && std::isfinite(R_PAH))
        {
            const double mw_pah = thermo_.MolecularWeight(static_cast<unsigned>(pah_index_));
            this->omega_gas_[static_cast<unsigned>(pah_index_)] -= R_PAH * mw_pah;
        }
    }

    if (surface_growth_model_ > 0 && VC2_ > 0. && index_C2H2_ >= 0)
    {
        // C2H2 consumption by HACA surface growth [mol/m3/s].
        const double R_C2H2 = this->source_growth_(1) * V0_ / VC2_;
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

    if (oxidation_model_ > 0 && VC2_ > 0.)
    {
        // Snapshot for oxidation-only extraction (operator splitting support).
        const Eigen::VectorXd omega_snap_before_ox = this->omega_gas_;

        // R_oxid_C2 [mol C2-pairs/m3/s] = rate of soot C2 pairs removed by oxidation.
        const double R_oxid_C2 = -this->source_oxidation_(1) * V0_ / VC2_;
        if (R_oxid_C2 > 0. && std::isfinite(R_oxid_C2))
        {
            const double kox_sum = kox_O2_ + kox_OH_;
            if (kox_sum > 0. && std::isfinite(kox_sum))
            {
                const double fO2 = kox_O2_ / kox_sum;
                const double fOH = kox_OH_ / kox_sum;

                // O2 channel: C2 + O2 -> 2CO.
                if (index_O2_ >= 0 && fO2 > 0.)
                {
                    const double R_O2 = fO2 * R_oxid_C2; // [mol C2/m3/s] via O2
                    const double MW_O2 = thermo_.MolecularWeight(static_cast<unsigned>(index_O2_));
                    this->omega_gas_[static_cast<unsigned>(index_O2_)] -= R_O2 * MW_O2 / 1000.;
                    if (index_CO_ >= 0)
                    {
                        const double MW_CO = thermo_.MolecularWeight(static_cast<unsigned>(index_CO_));
                        this->omega_gas_[static_cast<unsigned>(index_CO_)] += R_O2 * 2. * MW_CO / 1000.;
                    }
                }

                // OH channel gas coupling follows the model's C2-pair rate convention.
                if (index_OH_ >= 0 && fOH > 0.)
                {
                    const double R_OH = fOH * R_oxid_C2; // [mol OH consumed (per C2) / m3/s]
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

        // Capture oxidation-only gas contribution (before closure).
        omega_gas_oxidation_ = this->omega_gas_ - omega_snap_before_ox;
    }

    if (this->is_closure_dummy_species_ && this->closure_dummy_index_ >= 0)
    {
        double sum = 0.;
        for (Eigen::Index i = 0; i < this->omega_gas_.size(); ++i)
            if (i != static_cast<Eigen::Index>(this->closure_dummy_index_))
                sum += this->omega_gas_[i];
        this->omega_gas_[this->closure_dummy_index_] = -sum;

        // Apply the same closure to the oxidation-only vector.
        double sum_ox = 0.;
        for (Eigen::Index i = 0; i < omega_gas_oxidation_.size(); ++i)
            if (i != static_cast<Eigen::Index>(this->closure_dummy_index_))
                sum_ox += omega_gas_oxidation_[i];
        omega_gas_oxidation_[static_cast<Eigen::Index>(this->closure_dummy_index_)] = -sum_ox;
    }
}

template <ThermoMap Thermo> double HMOM<Thermo>::volume_fraction() const noexcept
{
    const double fv = GetMoment(1., 0.);
    return (!std::isfinite(fv) || fv < kSootVolumeFloor) ? 0. : fv;
}

template <ThermoMap Thermo> double HMOM<Thermo>::particle_diameter() const noexcept
{
    const double V = GetMoment(1., 0.);
    const double S = GetMoment(0., 1.);
    if (!std::isfinite(V) || !std::isfinite(S) || V <= kSootVolumeFloor || S <= kSootSurfaceFloor)
        return 0.;
    const double dp = 6. * V / S;
    return (std::isfinite(dp) && dp > 0.) ? dp : 0.;
}

template <ThermoMap Thermo> double HMOM<Thermo>::collision_diameter() const noexcept
{
    const double N = GetMoment(0., 0.);
    if (!std::isfinite(N) || N <= kSootNumberFloor)
        return 0.;
    const double dc = K_collisional_ * GetMoment(Av_collisional_, As_collisional_) / N;
    return (std::isfinite(dc) && dc > 0.) ? dc : 0.;
}

template <ThermoMap Thermo> double HMOM<Thermo>::particle_number_density() const noexcept
{
    const double n = GetMoment(0., 0.);
    return (!std::isfinite(n) || n < kSootNumberFloor) ? 0. : n;
}

template <ThermoMap Thermo> double HMOM<Thermo>::mass_fraction() const noexcept
{
    return this->rho_particle_ / this->rho_ * volume_fraction();
}

template <ThermoMap Thermo> double HMOM<Thermo>::specific_surface_area() const noexcept
{
    const double Ss = GetMoment(0., 1.);
    return (!std::isfinite(Ss) || Ss <= 0.) ? 0. : Ss;
}

template <ThermoMap Thermo> double HMOM<Thermo>::number_primary_particles() const noexcept
{
    const double N = GetMoment(0., 0.);
    if (!std::isfinite(N) || N <= kSootNumberFloor)
        return 0.;
    const double M = GetMoment(-2., 3.);
    if (!std::isfinite(M) || M <= 0.)
        return 0.;
    const double K_spher = K_spher_HMOM(this->pi_);
    const double np      = std::pow(K_spher, -3.) * M / N;
    return (std::isfinite(np) && np >= 1.) ? np : 1.;
}

template <ThermoMap Thermo> double HMOM<Thermo>::diffusion_coefficient() const noexcept
{
    const double dc_safe       = std::max(collision_diameter(), 1.e-12);
    const double GammaBrownian = this->CunninghamDiffusionCoeff(dc_safe);
    return std::max(GammaBrownian, this->mu_ / this->schmidt_number_);
}

template <ThermoMap Thermo>
void HMOM<Thermo>::Properties(
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

template <ThermoMap Thermo>
void HMOM<Thermo>::ApplyConfig(const Config& cfg)
{
    this->is_active_ = cfg.is_active;

    // Density and geometry must be set before PAH-derived geometry is computed.
    this->SetParticleDensity(cfg.soot_density_kg_m3);

    this->SetFractalDiameterModel(cfg.fractal_diameter_model);
    this->SetCollisionDiameterModel(cfg.collision_diameter_model);

    this->is_simplified_pah_mass_ = cfg.simplified_pah_mass;
    pah_species_ = std::string(cfg.pah_species);
    Precalculations();

    this->SetGasConsumption(cfg.gas_consumption);
    this->SetGasClosureDummySpecies(cfg.gas_closure_dummy_species);

    this->SetNucleation(cfg.nucleation_model);
    this->SetSurfaceGrowth(cfg.surface_growth_model);
    this->SetOxidation(cfg.oxidation_model);
    this->SetCondensation(cfg.condensation_model);
    this->SetCoagulation(cfg.coagulation_model);
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

    // HACA kinetics: A in cm3/mol/s; activation energies are converted to K.
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
void HMOM<Thermo>::SetupFromConfig(const Config& cfg)
{
    ApplyConfig(cfg);
    PrintSummary();
}

template <ThermoMap Thermo>
std::array<typename HMOM<Thermo>::NDFNode, 2> HMOM<Thermo>::NumberDensityFunctionNodes() const
{
    std::array<NDFNode, 2> nodes{};

    nodes[0].number_density    = N0_;
    nodes[0].volume            = V0_;
    nodes[0].surface           = S0_;
    nodes[0].primary_diameter  = 6. * V0_ / S0_;
    nodes[0].primary_particles = 1.;
    nodes[0].collision_diameter =
        K_collisional_ * std::pow(V0_, Av_collisional_) * std::pow(S0_, As_collisional_);

    nodes[1].number_density = NL_;
    if (NL_ > kSootNumberFloor && NLVL_ > kSootVolumeFloor && NLSL_ > kSootSurfaceFloor)
    {
        const double VL            = NLVL_ / NL_;
        const double SL            = NLSL_ / NL_;
        nodes[1].volume            = VL;
        nodes[1].surface           = SL;
        nodes[1].primary_diameter  = 6. * VL / SL;
        nodes[1].primary_particles = std::max(1., std::pow(SL, 3.) / (36. * this->pi_ * VL * VL));
        nodes[1].collision_diameter =
            K_collisional_ * std::pow(VL, Av_collisional_) * std::pow(SL, As_collisional_);
    }
    else
    {
        nodes[1]                = nodes[0];
        nodes[1].number_density = NL_;
    }
    return nodes;
}

template <ThermoMap Thermo> double HMOM<Thermo>::soot_small_number_density() const noexcept
{
    return N0_;
}

template <ThermoMap Thermo> double HMOM<Thermo>::soot_large_number_density() const noexcept
{
    return NL_;
}

template <ThermoMap Thermo> double HMOM<Thermo>::soot_large_fraction() const noexcept
{
    const double t = N0_ + NL_;
    return (t > 0.) ? NL_ / t : 0.;
}

template <ThermoMap Thermo> double HMOM<Thermo>::soot_small_fraction() const noexcept
{
    const double t = N0_ + NL_;
    return (t > 0.) ? N0_ / t : 1.;
}

template <ThermoMap Thermo> double HMOM<Thermo>::soot_mean_volume() const noexcept
{
    if (!std::isfinite(M00_) || !std::isfinite(M10_) || M00_ <= 0.0 || M10_ <= 0.0)
        return V0_;

    return M10_ / M00_; // [m3/#]
}

template <ThermoMap Thermo> double HMOM<Thermo>::soot_mean_surface() const noexcept
{
    if (!std::isfinite(M00_) || !std::isfinite(M01_) || M00_ <= 0.0 || M01_ <= 0.0)
        return S0_;

    return M01_ / M00_; // [m2/#]
}

template <ThermoMap Thermo> double HMOM<Thermo>::soot_large_mean_volume() const noexcept
{
    return (NL_ > kSootNumberFloor && NLVL_ > kSootVolumeFloor) ? NLVL_ / NL_ : V0_;
}

template <ThermoMap Thermo> double HMOM<Thermo>::soot_large_mean_surface() const noexcept
{
    return (NL_ > kSootNumberFloor && NLSL_ > kSootSurfaceFloor) ? NLSL_ / NL_ : S0_;
}

template <ThermoMap Thermo> double HMOM<Thermo>::soot_large_primary_particle_diameter() const noexcept
{
    const double VL = soot_large_mean_volume();
    const double SL = soot_large_mean_surface();
    return (SL > 0.) ? 6. * VL / SL : 0.;
}

template <ThermoMap Thermo> double HMOM<Thermo>::soot_large_primary_particle_number() const noexcept
{
    const double VL = soot_large_mean_volume();
    const double SL = soot_large_mean_surface();
    if (SL <= 0. || VL <= 0.)
        return 1.;
    return std::max(1., std::pow(SL, 3.) / (36. * this->pi_ * VL * VL));
}

template <ThermoMap Thermo>
double HMOM<Thermo>::LogGeomStdDevFromMoments(double M0, double M1, double M2) const noexcept
{
    if (M0 <= 0. || M1 <= 0. || M2 <= 0.)
        return 0.;
    const double ratio = M0 * M2 / (M1 * M1);
    return (ratio > 1.) ? std::sqrt(std::log(ratio)) : 0.;
}

template <ThermoMap Thermo>
double HMOM<Thermo>::soot_log_geom_std_dev_primary_particle_diameter() const noexcept
{
    return LogGeomStdDevFromMoments(GetMoment(0., 0.), GetMoment(-1., 1.5), GetMoment(-2., 3.));
}

template <ThermoMap Thermo>
double HMOM<Thermo>::soot_log_geom_std_dev_primary_particle_number() const noexcept
{
    return LogGeomStdDevFromMoments(GetMoment(0., 0.), GetMoment(1., 0.), GetMoment(2., 0.));
}

template <ThermoMap Thermo>
double HMOM<Thermo>::soot_large_log_geom_std_dev_primary_particle_diameter() const noexcept
{
    if (!HasSoot() || NL_ <= kSootNumberFloor)
        return 0.0;

    const double M0L = GetMissingMoment(0.0, 0.0);
    const double M1L = GetMissingMoment(1.0, -1.0);
    const double M2L = GetMissingMoment(2.0, -2.0);

    return LogGeomStdDevFromMoments(M0L, M1L, M2L);
}

template <ThermoMap Thermo>
double HMOM<Thermo>::soot_large_log_geom_std_dev_primary_particle_number() const noexcept
{
    if (!HasSoot() || NL_ <= kSootNumberFloor)
        return 0.0;

    const double M0L = GetMissingMoment(0.0, 0.0);
    const double M1L = GetMissingMoment(-2.0, 3.0);
    const double M2L = GetMissingMoment(-4.0, 6.0);

    return LogGeomStdDevFromMoments(M0L, M1L, M2L);
}

template <ThermoMap Thermo> double HMOM<Thermo>::soot_d63() const noexcept
{
    const double M10 = GetMoment(1., 0.);
    const double M43 = GetMoment(4. / 3., 0.);
    if (M10 <= 0. || !std::isfinite(M10))
        return 0.;
    return 6. * std::pow(M43 / M10, 1. / 3.);
}

// Diagnostic NDF reconstruction: two HMOM delta nodes are represented by
// fixed-width log-normal kernels for plotting and reporter output.

template <ThermoMap Thermo>
typename HMOM<Thermo>::NDFReconstructionData
HMOM<Thermo>::ReconstructedNDFData(bool use_regularized_moments) const
{
    NDFReconstructionData d{};
    d.valid = false;

    double N0 = N0_;   // small-mode number density [#/m3]
    double NL = NL_;   // large-mode number density [#/m3]

    if (use_regularized_moments)
    {
        // Optional floor for robust diagnostic output in particle-free cells.
        N0 = std::max(N0, kTinyNumberDensity);
        NL = std::max(NL, 0.);
    }

    if (N0 <= 0. && NL <= 0.)
        return d;

    const double V0 = V0_;
    if (V0 <= 0. || !std::isfinite(V0))
        return d;

    const double VL = (NL > kSootNumberFloor && NLVL_ > kSootVolumeFloor)
                          ? NLVL_ / NL
                          : V0;
    if (VL <= 0. || !std::isfinite(VL))
        return d;

    // Choose log-normal locations so each smeared kernel preserves the node mean.
    const double sigma = kNDFSmearSigmaLnV;
    const double mu0   = std::log(V0) - 0.5 * sigma * sigma;
    const double muL   = std::log(VL) - 0.5 * sigma * sigma;

    d.valid = true;
    d.N0    = std::max(N0, 0.);
    d.V0    = V0;
    d.mu0   = mu0;
    d.NL    = std::max(NL, 0.);
    d.VL    = VL;
    d.muL   = muL;
    d.sigma = sigma;
    return d;
}

template <ThermoMap Thermo>
double HMOM<Thermo>::ReconstructedNDF(double nu, bool use_regularized_moments) const
{
    if (!std::isfinite(nu) || nu <= 0.)
        return 0.;

    const auto d = ReconstructedNDFData(use_regularized_moments);
    if (!d.valid)
        return 0.;

    // Shared log-domain terms for stable PDF evaluation.
    const double log_nu       = std::log(nu);
    const double log_sigma    = std::log(d.sigma);
    const double half_log_2pi = 0.5 * std::log(2. * this->pi_);
    const double log_dbl_min  = std::log(std::numeric_limits<double>::min());
    const double log_dbl_max  = std::log(std::numeric_limits<double>::max());

    auto lognorm_pdf = [&](double mu) -> double
    {
        const double z  = (log_nu - mu) / d.sigma;
        const double lg = -log_nu - log_sigma - half_log_2pi - 0.5 * z * z;
        return (lg > log_dbl_min && lg < log_dbl_max) ? std::exp(lg) : 0.;
    };

    double n = 0.;

    if (d.N0 > 0.)
        n += d.N0 * lognorm_pdf(d.mu0);

    if (d.NL > 0.)
        n += d.NL * lognorm_pdf(d.muL);

    return (std::isfinite(n) && n >= 0.) ? n : 0.;
}

template <ThermoMap Thermo>
double HMOM<Thermo>::ReconstructedNormalizedNDF(double nu, bool use_regularized_moments) const
{
    if (!std::isfinite(nu) || nu <= 0.)
        return 0.;

    const auto d = ReconstructedNDFData(use_regularized_moments);
    if (!d.valid)
        return 0.;

    const double N_tot = d.N0 + d.NL;
    if (N_tot <= 0.)
        return 0.;

    const double n = ReconstructedNDF(nu, use_regularized_moments);
    return (std::isfinite(n) && n >= 0.) ? n / N_tot : 0.;
}

template <ThermoMap Thermo>
void HMOM<Thermo>::ReconstructedNDF(const Eigen::VectorXd& nu,
                                    Eigen::VectorXd&       n,
                                    bool use_regularized_moments) const
{
    n.resize(nu.size());
    for (int i = 0; i < nu.size(); ++i)
        n(i) = ReconstructedNDF(nu(i), use_regularized_moments);
}

template <ThermoMap Thermo>
std::span<const double> HMOM<Thermo>::sources_coagulation_discrete() const noexcept
{
    return {source_coagulation_discrete_.data(), 4u};
}

template <ThermoMap Thermo>
std::span<const double> HMOM<Thermo>::sources_coagulation_discrete_ss() const noexcept
{
    return {source_coagulation_ss_.data(), 4u};
}

template <ThermoMap Thermo>
std::span<const double> HMOM<Thermo>::sources_coagulation_discrete_sl() const noexcept
{
    return {source_coagulation_sl_.data(), 4u};
}

template <ThermoMap Thermo>
std::span<const double> HMOM<Thermo>::sources_coagulation_discrete_ll() const noexcept
{
    return {source_coagulation_ll_.data(), 4u};
}

template <ThermoMap Thermo>
std::span<const double> HMOM<Thermo>::sources_coagulation_continuous() const noexcept
{
    return {source_coagulation_continuous_.data(), 4u};
}

template <ThermoMap Thermo>
std::span<const double> HMOM<Thermo>::sources_coagulation_continuous_ss() const noexcept
{
    return {source_coagulation_cont_ss_.data(), 4u};
}

template <ThermoMap Thermo>
std::span<const double> HMOM<Thermo>::sources_coagulation_continuous_sl() const noexcept
{
    return {source_coagulation_cont_sl_.data(), 4u};
}

template <ThermoMap Thermo>
std::span<const double> HMOM<Thermo>::sources_coagulation_continuous_ll() const noexcept
{
    return {source_coagulation_cont_ll_.data(), 4u};
}

template <ThermoMap Thermo> void HMOM<Thermo>::SetA1f(double v) noexcept
{
    A1f_ = v;
}

template <ThermoMap Thermo> void HMOM<Thermo>::SetA1b(double v) noexcept
{
    A1b_ = v;
}

template <ThermoMap Thermo> void HMOM<Thermo>::SetA2f(double v) noexcept
{
    A2f_ = v;
}

template <ThermoMap Thermo> void HMOM<Thermo>::SetA2b(double v) noexcept
{
    A2b_ = v;
}

template <ThermoMap Thermo> void HMOM<Thermo>::SetA3f(double v) noexcept
{
    A3f_ = v;
}

template <ThermoMap Thermo> void HMOM<Thermo>::SetA3b(double v) noexcept
{
    A3b_ = v;
}

template <ThermoMap Thermo> void HMOM<Thermo>::SetA4(double v) noexcept
{
    A4_ = v;
}

template <ThermoMap Thermo> void HMOM<Thermo>::SetA5(double v) noexcept
{
    A5_ = v;
}

template <ThermoMap Thermo> void HMOM<Thermo>::SetE1f(double kJ) noexcept
{
    E1f_ = kJ * 1e3 / this->Rgas_mol_;
}

template <ThermoMap Thermo> void HMOM<Thermo>::SetE1b(double kJ) noexcept
{
    E1b_ = kJ * 1e3 / this->Rgas_mol_;
}

template <ThermoMap Thermo> void HMOM<Thermo>::SetE2f(double kJ) noexcept
{
    E2f_ = kJ * 1e3 / this->Rgas_mol_;
}

template <ThermoMap Thermo> void HMOM<Thermo>::SetE2b(double kJ) noexcept
{
    E2b_ = kJ * 1e3 / this->Rgas_mol_;
}

template <ThermoMap Thermo> void HMOM<Thermo>::SetE3f(double kJ) noexcept
{
    E3f_ = kJ * 1e3 / this->Rgas_mol_;
}

template <ThermoMap Thermo> void HMOM<Thermo>::SetE3b(double kJ) noexcept
{
    E3b_ = kJ * 1e3 / this->Rgas_mol_;
}

template <ThermoMap Thermo> void HMOM<Thermo>::SetE4(double kJ) noexcept
{
    E4_ = kJ * 1e3 / this->Rgas_mol_;
}

template <ThermoMap Thermo> void HMOM<Thermo>::SetE5(double kJ) noexcept
{
    E5_ = kJ * 1e3 / this->Rgas_mol_;
}

template <ThermoMap Thermo> void HMOM<Thermo>::Setn1f(double v) noexcept
{
    n1f_ = v;
}

template <ThermoMap Thermo> void HMOM<Thermo>::Setn1b(double v) noexcept
{
    n1b_ = v;
}

template <ThermoMap Thermo> void HMOM<Thermo>::Setn2f(double v) noexcept
{
    n2f_ = v;
}

template <ThermoMap Thermo> void HMOM<Thermo>::Setn2b(double v) noexcept
{
    n2b_ = v;
}

template <ThermoMap Thermo> void HMOM<Thermo>::Setn3f(double v) noexcept
{
    n3f_ = v;
}

template <ThermoMap Thermo> void HMOM<Thermo>::Setn3b(double v) noexcept
{
    n3b_ = v;
}

template <ThermoMap Thermo> void HMOM<Thermo>::Setn4(double v) noexcept
{
    n4_ = v;
}

template <ThermoMap Thermo> void HMOM<Thermo>::Setn5(double v) noexcept
{
    n5_ = v;
}

template <ThermoMap Thermo> void HMOM<Thermo>::PrintSummary() const
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
        << "                   HMOM — Hybrid Method of Moments Summary\n"
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
        << " [Processes]\n"
        << "    + Nucleation:                    " << nucleation_model_ << "\n"
        << "    + Condensation:                  " << condensation_model_ << "\n"
        << "    + Surface growth:                " << surface_growth_model_ << "\n"
        << "    + Oxidation:                     " << oxidation_model_ << "\n"
        << "    + Coagulation:                   " << coagulation_model_ << "\n"
        << "    + Coagulation (continuous):      " << coagulation_continuous_model_ << "\n"
        << "    + Fractal diameter model:        " << static_cast<int>(fractal_diameter_model_) << "  (" << frac_str(fractal_diameter_model_) << ")\n"
        << "    + Collision diameter model:      " << static_cast<int>(collision_diameter_model_) << "  (" << coll_str(collision_diameter_model_) << ")\n"
        << "    + Sticking model:                " << static_cast<int>(sticking_model_) << "  (" << sticking_str(sticking_model_) << ")\n"
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
        << "    + Thermophoretic model:          " << static_cast<int>(this->thermophoretic_model()) << "  (" << thermo_str(this->thermophoretic_model_) << ")\n"
        << "    + Gas consumption:               " << (this->gas_consumption_ ? "yes" : "no") << "\n"
        << "    + Radiative heat transfer:       " << (this->radiative_heat_transfer_ ? "yes" : "no") << "\n"
        << "    + Planck coeff. model:           " << static_cast<int>(this->planck_model_) << "  (" << planck_str(this->planck_model_) << ")\n"
        << "    + Closure dummy species:         " << (this->is_closure_dummy_species_ ? this->closure_dummy_species_ : "none") << "\n"
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

#if defined(MOM_USE_DICTIONARY)

template <ThermoMap Thermo>
template <typename DictType>
std::expected<typename HMOM<Thermo>::Config, std::string>
HMOM<Thermo>::ParseConfig(DictType& dict)
{
    HMOM_Grammar grammar;
    dict.SetGrammar(grammar);

    Config cfg; // start from library defaults

    if (dict.CheckOption("@HMOM"))
        dict.ReadBool("@HMOM", cfg.is_active);

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
        dict.ReadInt("@NucleationModel", cfg.nucleation_model);

    if (dict.CheckOption("@SurfaceGrowthModel"))
        dict.ReadInt("@SurfaceGrowthModel", cfg.surface_growth_model);

    if (dict.CheckOption("@OxidationModel"))
        dict.ReadInt("@OxidationModel", cfg.oxidation_model);

    if (dict.CheckOption("@CondensationModel"))
        dict.ReadInt("@CondensationModel", cfg.condensation_model);

    if (dict.CheckOption("@CoagulationModel"))
        dict.ReadInt("@CoagulationModel", cfg.coagulation_model);

    if (dict.CheckOption("@ContinuousCoagulationModel"))
        dict.ReadInt("@ContinuousCoagulationModel", cfg.continuous_coagulation_model);

    if (dict.CheckOption("@ThermophoreticModel"))
        dict.ReadInt("@ThermophoreticModel", cfg.thermophoretic_model);

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

    // Temperature exponents
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
