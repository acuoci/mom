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

#include "MOM/MomentMethodBase.hpp"
#include "MOM/ThermoProxy.hpp"
#include "Utilities/OutputFileColumns.h"
#include "Eigen/Dense"
#include <span>
#include <string>
#include <string_view>
#if defined(MOM_USE_DICTIONARY)
#  include <expected>
#endif

namespace MOM {

// ============================================================================
// TiO2<Thermo> — 3-equation method of moments for TiO2 nanoparticles
// ============================================================================
//
// Models the formation and evolution of TiO2 nanoparticles from gas-phase
// titanium precursors (e.g. Ti(OH)4 or TiCl4 families).
//
// Transported variables:
//   moments[0] = YTiO2   — TiO2 particle mass fraction [-]
//   moments[1] = NTiO2N  — scaled particle number density [-]
//                          (NTiO2N = N / N0_scaling, N0_scaling typically 1e15 #/m3)
//   moments[2] = STiO2   — total particle surface per gas volume [m2/m3]
//
// Physical processes modelled:
//   Nucleation    (binary or fixed-cluster collision variant)
//   Condensation  (precursor condensation on existing particles)
//   Coagulation   (free-molecular regime)
//   Sintering     (viscous-flow model: Kruis et al. 1993)
//   Thermophoresis (via effective diffusion coefficient)
//
// NOT modelled (and not present in base source vectors):
//   Surface growth by chemical vapour deposition (no HACA-equivalent for TiO2)
//   Oxidation
//   Radiative heat transfer (particles are dielectric — PlanckCoeffModel::None)
//
// @tparam Thermo  Must satisfy MOM::ThermoMap.
// ============================================================================

template <ThermoMap Thermo>
class TiO2 : public MomentMethodBase<TiO2<Thermo>, 3>
{
    using Base = MomentMethodBase<TiO2<Thermo>, 3>;

public:
    using typename Base::MomentVector;

    /// Labels accepted by MOM::MakeAnyMomentMethod for runtime variant selection.
    static constexpr std::array<std::string_view, 2> variant_labels { "TiO2", "tio2" };

    // ── Method-specific sub-model enums ─────────────────────────────────────

    enum class NucleationVariant : int
    {
        Off          = 0,
        Binary       = 1,        //!< Ti(OH)4 + Ti(OH)4 → (TiO2)_n + vapour
        FixedCluster = 2         //!< Nucleation via a fixed-size cluster
    };

    // ── NDF reconstruction (shared structure with ThreeEquations) ─────────────
    //
    // TiO2 uses the same Pareto + log-normal NDF reconstruction as
    // ThreeEquations (Franzelli et al. 2019), adapted for TiO2.

    struct NDFReconstructionData {
        bool   valid;        //!< true if reconstruction is physically meaningful
        double N;            //!< particle number density [#/m3]
        double fv;           //!< volume fraction [-]
        double nuMean;       //!< mean particle volume [m3/#]
        double nuNucl;       //!< nucleated particle volume [m3/#]
        double alpha;        //!< Pareto weight [-]
        double nbar0;        //!< nucleation-peak normalised NDF [1/m3]
        double sigma;        //!< log-normal standard deviation [-]
        double k;            //!< Pareto tail index [-]
        double nu1mean;      //!< mean volume of Pareto contribution [m3/#]
        double nu2mean;      //!< mean volume of log-normal contribution [m3/#]
        double mu;           //!< log-normal location parameter [log(m3)]
    };

    // ── Construction ─────────────────────────────────────────────────────────

    explicit TiO2(const Thermo& thermo);

    TiO2(const TiO2&)            = delete;
    TiO2& operator=(const TiO2&) = delete;
    TiO2(TiO2&&)                 = default;
    TiO2& operator=(TiO2&&)      = default;

#if defined(MOM_USE_DICTIONARY)
    template <typename Dictionary>
    [[nodiscard]] std::expected<void, std::string>
        SetupFromDictionary(Dictionary& dict);
#endif

    // ── MomentMethod concept — state injection ────────────────────────────────

    /// @param T    Temperature [K]
    /// @param P_Pa Pressure [Pa]
    /// @param Y    Mass fractions, size = n_species
    void SetStatus(double T, double P_Pa, const double* Y) noexcept;

    /// Generic span setter. Order: [YTiO2, NTiO2N, STiO2].
    /// YTiO2 [-], NTiO2N [-], STiO2 [m2/m3].
    void SetMoments(std::span<const double> m) noexcept;

    /// Named setter (preferred for TiO2-aware code).
    void SetMoments(double YTiO2, double NTiO2N, double STiO2) noexcept;

    // ── MomentMethod concept — core computation ───────────────────────────────

    /// Computes all source terms for the current cell state.
    void CalculateSourceMoments() noexcept;

    void CalculateOmegaGas() noexcept;

    // ── Deferred sintering (operator-splitting compatible) ────────────────────
    //
    // When sintering is stiff relative to the main transport time step,
    // it can be integrated separately using an ODE solver. Call this method
    // after CalculateSourceMoments to update surface area with a sub-step.
    //
    // Returns the sintering time scale [s].

    [[nodiscard]] double SinteringDeferredUpdate(double dt_ode);

    // ── MomentMethod concept — particle properties ────────────────────────────

    [[nodiscard]] double VolumeFraction()        const noexcept;
    [[nodiscard]] double ParticleDiameter()      const noexcept;  //!< primary particle diameter [m]
    [[nodiscard]] double CollisionDiameter()     const noexcept;  //!< aggregate collision diameter [m]
    [[nodiscard]] double AggregateDiameter()     const noexcept;  //!< mobility aggregate diameter [m]
    [[nodiscard]] double ParticleNumberDensity() const noexcept;  //!< [#/m3]
    [[nodiscard]] double MassFraction()          const noexcept;  //!< = YTiO2_
    [[nodiscard]] double SpecificSurface()       const noexcept;  //!< STiO2 [m2/m3]
    [[nodiscard]] double NumberOfPrimaryParticles()const noexcept; //!< np [-]
    [[nodiscard]] double DiffusionCoefficient()  const noexcept;  //!< [kg/m/s]

    // ── MomentMethod concept — initial conditions ─────────────────────────────

    [[nodiscard]] std::span<const double> initial_moments() const noexcept {
        return { initial_moments_cache_.data(), 3u };
    }

    // ── MomentMethod concept — precursor ──────────────────────────────────────

    [[nodiscard]] int         precursor_index()         const noexcept { return precursor_index_;  }
    [[nodiscard]] double      precursor_concentration() const noexcept { return c_precursor_;      }
    [[nodiscard]] const std::string& precursor_species()const noexcept { return precursor_species_; }

    // ── MomentMethod concept — diagnostics ────────────────────────────────────

    void PrintSummary() const;

    // ── Aggregated properties helper ──────────────────────────────────────────

    void Properties(double& fv,  double& dp,   double& dc, double& da,
                    double& np,  double& ss,   double& vs,
                    double& ssph,double& tauS) const noexcept;

    // ── NDF reconstruction (TiO2-specific) ───────────────────────────────────

    [[nodiscard]] NDFReconstructionData
        ReconstructedNDFData(bool use_regularized_moments = false) const;

    [[nodiscard]] double ReconstructedNormalizedNDF(
        double nu, bool use_regularized_moments = false) const;

    [[nodiscard]] double ReconstructedNDF(
        double nu, bool use_regularized_moments = false) const;

    void ReconstructedNDF(const Eigen::VectorXd& nu, Eigen::VectorXd& n,
                          bool use_regularized_moments = false) const;

    // ── Material / geometry accessors ─────────────────────────────────────────

    [[nodiscard]] double rhoTiO2()              const noexcept { return rhoTiO2_; }
    [[nodiscard]] double NucleationParticleVolume() const noexcept;
    [[nodiscard]] double s0()                   const noexcept { return s0_; }
    [[nodiscard]] double v0()                   const noexcept { return v0_; }
    [[nodiscard]] double ScalingFactorNs()      const noexcept { return N0_scaling_; }

    // ── Model switches ────────────────────────────────────────────────────────

    void SetNucleation(int flag) noexcept    { nucleation_variant_ = static_cast<NucleationVariant>(flag); }
    void SetCoagulation(int flag) noexcept   { coagulation_model_ = flag; }
    void SetCondensation(int flag) noexcept  { condensation_model_ = flag; }
    void SetSintering(int flag) noexcept     { sintering_model_ = flag; }

    void SetNumberOfTiO2UnitsPerNucleatedParticle(unsigned n) noexcept { n0_ = n; }
    void SetMinimumNumberOfTiO2Units(unsigned n) noexcept             { nTiO2_min_ = n; }

    void SetNucleationCollisionEnhancementFactor(double eps) noexcept  { epsilon_nuc_  = eps; }
    void SetCoagulationCollisionEnhancementFactor(double eps) noexcept { epsilon_coag_ = eps; }
    void SetCondensationCollisionEnhancementFactor(double eps) noexcept{ epsilon_cond_ = eps; }

    /// Sintering kinetics: τ_s = (1 / As) * T^(-ns) * exp(Ts / T) [s]
    void SetSinteringFrequencyFactor(double As) noexcept        { As_ = As; }
    void SetSinteringActivationTemperature(double Ts) noexcept  { Ts_ = Ts; }
    void SetSinteringTemperatureExponent(double ns) noexcept    { ns_ = ns; }

    void SetNMinimum(double v) noexcept     { N_min_ = v; }
    void SetFvMinimum(double v) noexcept    { fv_min_ = v; }

    void SetPrecursor(std::string_view name);
    void SetGasClosureDummySpecies(std::string_view name);

    /// Configure stoichiometry for gas-phase consumption from precursor formula.
    void SetupGasConsumptionStoichiometry();

    // ── Model state queries ────────────────────────────────────────────────────

    [[nodiscard]] int    nucleation_model()   const noexcept { return static_cast<int>(nucleation_variant_); }
    [[nodiscard]] int    coagulation_model()  const noexcept { return coagulation_model_; }
    [[nodiscard]] int    condensation_model() const noexcept { return condensation_model_; }
    [[nodiscard]] int    sintering_model()    const noexcept { return sintering_model_; }
    [[nodiscard]] bool   is_sintering_deferred() const noexcept { return is_sintering_deferred_; }

    // ── CRTP extension points — process source storage owned by TiO2 ─────────
    //
    // TiO2 models: nucleation, coagulation, condensation, sintering.
    // NOT modelled: growth, oxidation → base class returns zero span for both.

    [[nodiscard]] std::span<const double> sources_nucleation_impl()   const noexcept { return { source_nucleation_.data(),   this->n_equations }; }
    [[nodiscard]] std::span<const double> sources_coagulation_impl()  const noexcept { return { source_coagulation_.data(),  this->n_equations }; }
    [[nodiscard]] std::span<const double> sources_condensation_impl() const noexcept { return { source_condensation_.data(), this->n_equations }; }
    [[nodiscard]] std::span<const double> sources_sintering_impl()    const noexcept { return { source_sintering_.data(),    this->n_equations }; }

	void WriteHeaderLine(MOM::OutputFileColumns& fOutput, const unsigned int precision);

	void WriteOutputLine( MOM::OutputFileColumns& fOutput,
							const double T, const double P_Pa, const double* Y, const double mu,
							const double* M);

private:
    // ── Private computational methods ──────────────────────────────────────────

    void MemoryAllocation();
    void Precalculations();
    void NucleationSourceTerms();
    void NucleationSourceTerms_Binary();
    void NucleationSourceTerms_FixedCluster();
    void CoagulationSourceTerms();
    void CondensationSourceTerms();
    void SinteringSourceTerms();
    void CalculateOmegaGas_internal() noexcept;

private:
    // ── Thermodynamics reference ───────────────────────────────────────────────
    const Thermo& thermo_;

    // ── Transported variables ──────────────────────────────────────────────────
    double YTiO2_  = 0.;          //!< TiO2 mass fraction [-]
    double NTiO2N_ = 0.;          //!< N / N0_scaling [-]
    double STiO2_  = 0.;          //!< surface area concentration [m2/m3]
    double N0_scaling_ = 1.e15;   //!< [#/m3]

    // ── Material properties ────────────────────────────────────────────────────
    static constexpr double W_TiO2_  = 79.866;     //!< TiO2 molecular weight [kg/kmol]
    static constexpr double rhoTiO2_ = 3900.;      //!< solid anatase density [kg/m3]
    static constexpr double m_TiO2_  = W_TiO2_ / (6.02214076e26); //!< [kg/molecule]

    // ── Monomer/nucleus geometry ───────────────────────────────────────────────
    double v0_  = 0.;   //!< monomer volume [m3]
    double s0_  = 0.;   //!< monomer surface [m2]
    double d0_  = 0.;   //!< monomer diameter [m]

    // ── Precursor properties ───────────────────────────────────────────────────
    std::string precursor_species_;
    int    precursor_index_ = -1;
    double nti_precursor_   = 0.;   //!< Ti atoms per precursor molecule
    double nh_precursor_    = 0.;   //!< H atoms
    double no_precursor_    = 0.;   //!< O atoms
    double nc_precursor_    = 0.;   //!< C atoms
    double Y_precursor_     = 0.;   //!< mass fraction
    double c_precursor_     = 0.;   //!< molar concentration [kmol/m3]
    double W_precursor_     = 0.;   //!< molecular weight [kg/kmol]
    double m_precursor_     = 0.;   //!< molecular mass [kg]

    // Collision geometry of the gas precursor molecule
    double v_precursor_ = 0.;   //!< pseudo-molecular volume [m3]
    double d_precursor_ = 0.;   //!< collision diameter [m]

    // Effective precursor geometry used by nucleation/condensation kernels
    double vprec_ = 0.;   //!< solid TiO2 volume added per precursor molecule [m3]
    double dprec_ = 0.;   //!< collision diameter for beta_nuc / beta_cond [m]

    // ── Gas consumption stoichiometry ──────────────────────────────────────────
    // Derived from atom balance: Ti_a C_b H_c O_d + x O2 → TiO2(s) + y CO2 + z H2O
    int    H2O_index_ = -1, CO2_index_ = -1, O2_index_ = -1;
    double W_H2O_  = 0., W_CO2_  = 0., W_O2_  = 0.;
    double nu_H2O_from_prec_ = 0.;   //!< H2O stoichiometric coefficient
    double nu_CO2_from_prec_ = 0.;   //!< CO2 stoichiometric coefficient
    double nu_O2_from_prec_  = 0.;   //!< O2 stoichiometric coefficient (negative = consumed)

    // ── Nucleation parameters ──────────────────────────────────────────────────
    NucleationVariant nucleation_variant_ = NucleationVariant::Off;
    unsigned int      nTiO2_min_ = 1;    //!< minimum reference size for regularization
    unsigned int      n0_        = 5;    //!< TiO2 units per newly nucleated particle
    double            epsilon_nuc_= 2.5; //!< nucleation collision enhancement factor

    // ── Sintering parameters ───────────────────────────────────────────────────
    double As_ = 7.44e16;    //!< sintering frequency factor [1/s/K]
    double ns_ = 1.0;        //!< temperature exponent [-]
    double Ts_ = -31000.;    //!< activation temperature [K] (negative = Arrhenius)

    // Sintering regularisation parameters (set in Precalculations)
    double sintering_dp_min_              = 0.;
    double sintering_activation_np_       = 0.;
    double sintering_activation_width_np_ = 0.;
    double sintering_tau_min_             = 0.;
    double sintering_k_max_               = 0.;
    double sintering_relative_tolerance_  = 1.e-4;
    double sintering_tau_qss_             = 0.;
    bool   is_sintering_deferred_         = false;

    // ── Coagulation / condensation parameters ──────────────────────────────────
    double epsilon_coag_ = 2.2;    //!< coagulation enhancement factor
    double epsilon_cond_ = 1.3;    //!< condensation enhancement factor

    // Pre-computed kernel prefactors (set in Precalculations; temperature-independent)
    double alpha_nuc_  = 0.;   //!< [m3/s/K^0.5]
    double alpha_coag_ = 0.;   //!< [m^(5/2)/s/K^(1/2)]
    double alpha_cond_ = 0.;   //!< [m^(5/2)/s/K^(1/2)]

    // ── Model flags ────────────────────────────────────────────────────────────
    int coagulation_model_  = 0;
    int condensation_model_ = 0;
    int sintering_model_    = 0;

    // ── Numerical regularisation floors ───────────────────────────────────────
    double N_min_   = 1.;       //!< [#/m3]
    double fv_min_  = 1.e-15;   //!< [-]
    double v_min_   = 0.;       //!< [m3] set in Precalculations from nTiO2_min_
    double S_min_   = 0.;       //!< [m2/m3]

    // ── Per-process source storage (owned by TiO2, not by base) ──────────────
    //
    // TiO2 models: nucleation, coagulation, condensation, sintering.
    // growth and oxidation are absent; base class returns zero span for both.

    MomentVector source_nucleation_   = MomentVector::Zero();
    MomentVector source_coagulation_  = MomentVector::Zero();
    MomentVector source_condensation_ = MomentVector::Zero();
    MomentVector source_sintering_    = MomentVector::Zero();

    // ── Initial moments cache ──────────────────────────────────────────────────
    MomentVector initial_moments_cache_ = MomentVector::Zero();

	bool is_debug_mode_ = false;  //!< enable verbose diagnostic output
};

} // namespace MOM

#if !defined(MOM_COMPILED_LIBRARY)
#  include "TiO2.tpp"
#else
namespace MOM 
{
    extern template class TiO2<BasicThermoData>;
}
#endif
