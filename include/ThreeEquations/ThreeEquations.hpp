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
#include <expected>
#endif

namespace MOM {

// ============================================================================
// ThreeEquations<Thermo> — 3-equation soot model
// ============================================================================
//
// Reference:
//   Franzelli, Vie', Darabiha, Proc. Combust. Inst. 37 (2019) 5411–5419.
//
// Transported variables:
//   moments[0] = Ys      — soot mass fraction [-]
//   moments[1] = NsNorm  — scaled soot particle number density [-]
//                          (NsNorm = Ns / N0_scaling, N0_scaling typically 1e15 #/m3)
//   moments[2] = Ss      — soot specific surface area [m2/m3]
//
// Physical processes modelled:
//   Nucleation    (PAH dimerization, free-molecular regime)
//   Surface growth (HACA or RC-PAH surface chemistry model)
//   Condensation  (PAH adsorption on soot)
//   Oxidation     (O2 + OH)
//   Coagulation   (free-molecular + continuum)
//   Thermophoresis (via effective diffusion coefficient)
//
// @tparam Thermo  Must satisfy MOM::ThermoMap.
// ============================================================================

template <ThermoMap Thermo>
class ThreeEquations : public MomentMethodBase<ThreeEquations<Thermo>, 3>
{
    using Base = MomentMethodBase<ThreeEquations<Thermo>, 3>;

public:
    using typename Base::MomentVector;

    /// Labels accepted by MOM::MakeAnyMomentMethod for runtime variant selection.
    static constexpr std::array<std::string_view, 3> variant_labels { "ThreeEquations", "threeequations", "3Eq" };

    // ── Method-specific sub-model enums ─────────────────────────────────────

    /// Surface chemistry model for HACA (surface growth + oxidation).
    enum class SurfaceChemistryModel : int
    {
        RCPAH = 0,   //!< RC-PAH model (default): Franzelli et al. (2019)
        HMOM  = 1    //!< HMOM HACA kinetics (for cross-method consistency studies)
    };

    enum class StickingModel : int
    {
        Constant = 0,   //!< Fixed sticking coefficient (default: 2e-3)
        PAH4     = 1    //!< PAH 4-ring sticking model
    };

    // ── NDF reconstruction data structure ─────────────────────────────────────
    //
    // ThreeEquations uses the Franzelli et al. (2019) combined Pareto +
    // log-normal reconstruction for the marginal NDF n(ν).
    //
    //   n(ν) = Ns * nbar(ν)
    //   nbar(ν) = α * Pareto(ν; ν_nucl, k) + (1-α) * LogNormal(ν; μ, σ)
    //
    // where ν is the particle volume [m3/#].

    struct NDFReconstructionData {
        bool   valid;        //!< true if reconstruction is physically meaningful
        double Ns;           //!< soot number density [#/m3]
        double fv;           //!< soot volume fraction [-]
        double nus;          //!< mean particle volume [m3/#]
        double alpha;        //!< Pareto weight [-]
        double nbar0;        //!< nucleation-peak normalised NDF value [1/m3]
        double sigma;        //!< log-normal standard deviation [-]
        double k;            //!< Pareto tail index [-]
        double nu1mean;      //!< mean volume of Pareto contribution [m3/#]
        double nu2mean;      //!< mean volume of log-normal contribution [m3/#]
        double mu;           //!< log-normal location parameter [log(m3)]
    };

    // ── Construction ─────────────────────────────────────────────────────────

    explicit ThreeEquations(const Thermo& thermo);

    ThreeEquations(const ThreeEquations&)            = delete;
    ThreeEquations& operator=(const ThreeEquations&) = delete;
    ThreeEquations(ThreeEquations&&)                 = default;
    ThreeEquations& operator=(ThreeEquations&&)      = default;

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

    /// Generic span setter. Order: [Ys, NsNorm, Ss].
    /// Ys [-], NsNorm [-], Ss [m2/m3].
    void SetMoments(std::span<const double> m) noexcept;

    /// Named setter (preferred for ThreeEquations-aware code).
    void SetMoments(double Ys, double NsNorm, double Ss) noexcept;

    // ── MomentMethod concept — core computation ───────────────────────────────

    void CalculateSourceMoments() noexcept;
    void CalculateOmegaGas() noexcept;

    // ── MomentMethod concept — particle properties ────────────────────────────

    [[nodiscard]] double VolumeFraction()        const noexcept;
    [[nodiscard]] double ParticleDiameter()      const noexcept;  //!< primary particle diameter [m]
    [[nodiscard]] double CollisionDiameter()     const noexcept;  //!< aggregate collision diameter [m]
    [[nodiscard]] double ParticleNumberDensity() const noexcept;  //!< [#/m3]
    [[nodiscard]] double MassFraction()          const noexcept;  //!< = Ys_
    [[nodiscard]] double SpecificSurface()       const noexcept;  //!< Ss [m2/m3]
    [[nodiscard]] double NumberOfPrimaryParticles()const noexcept;
    [[nodiscard]] double DiffusionCoefficient()  const noexcept;  //!< [kg/m/s]

    // ── MomentMethod concept — initial conditions ─────────────────────────────

    [[nodiscard]] std::span<const double> initial_moments() const noexcept {
        return { initial_moments_cache_.data(), 3u };
    }

    // ── MomentMethod concept — precursor ──────────────────────────────────────

    [[nodiscard]] int         precursor_index()         const noexcept { return pah_index_;  }
    [[nodiscard]] double      precursor_concentration() const noexcept { return conc_PAH_;   }
    [[nodiscard]] const std::string& precursor_species()const noexcept { return pah_species_; }

    // ── MomentMethod concept — diagnostics ────────────────────────────────────

    void PrintSummary() const;

    // ── Aggregated properties helper ──────────────────────────────────────────

    /// Returns fv, dp, dc, np, ss, vs in one call (useful for post-processing).
    void Properties(double& fv, double& dp, double& dc,
                    double& np, double& ss, double& vs) const noexcept;

    // ── NDF reconstruction (ThreeEquations-specific) ──────────────────────────

    /// Computes the Franzelli et al. (2019) NDF reconstruction parameters.
    /// @param use_regularized_moments  If true, uses floor-clipped moments.
    [[nodiscard]] NDFReconstructionData
        ReconstructedNDFData(bool use_regularized_moments = false) const;

    /// Normalised NDF nbar(ν) [1/m3].
    [[nodiscard]] double ReconstructedNormalizedNDF(
        double nu, bool use_regularized_moments = false) const;

    /// Dimensional NDF n(ν) = Ns * nbar(ν) [#/m3/m3].
    [[nodiscard]] double ReconstructedNDF(
        double nu, bool use_regularized_moments = false) const;

    /// Vectorized dimensional NDF.
    void ReconstructedNDF(const Eigen::VectorXd& nu, Eigen::VectorXd& n,
                          bool use_regularized_moments = false) const;

    // ── Surface area geometry utilities ───────────────────────────────────────

    /// Change in surface area for a volume increment deltaV on a sphere.
    [[nodiscard]] static double DeltaSurfaceSpherical(
        double deltaV, double V, double S) noexcept;

    /// Change in surface area for a volume increment deltaV on a fractal aggregate.
    [[nodiscard]] static double DeltaSurfaceFractal(
        double deltaV, double V, double S, double np) noexcept;

    // ── Model switches ────────────────────────────────────────────────────────

    void SetNucleation(int flag) noexcept              { nucleation_model_ = flag; }
    void SetSurfaceGrowth(int flag) noexcept           { surface_growth_model_ = flag; }
    void SetCondensation(int flag) noexcept            { condensation_model_ = flag; }
    void SetOxidation(int flag) noexcept               { oxidation_model_ = flag; }
    void SetCoagulation(int flag) noexcept             { coagulation_model_ = flag; }
    void SetSurfaceChemistryModel(std::string_view label);
    void SetSurfaceChemistryModel(SurfaceChemistryModel m) noexcept { surface_chem_model_ = m; }

    void SetNucleationCollisionEnhancementFactor(double eps) noexcept  { epsilon_nucleation_  = eps; }
    void SetCondensationCollisionEnhancementFactor(double eps) noexcept{ epsilon_condensation_ = eps; }
    void SetCoagulationCollisionEnhancementFactor(double eps) noexcept { epsilon_coagulation_  = eps; }

    void SetCorrectionCoefficientPAHPAH(double v) noexcept { correction_coeff_pah_pah_ = v; }
    void SetStickingCoefficientModel(std::string_view label);
    void SetStickingCoefficientConstant(double v) noexcept { sticking_coeff_constant_ = v; }

    /// Particle density alias (triggers geometry recalculation).
    void SetSootDensity(double rhos) noexcept;
    void SetNsMinimum(double v) noexcept;
    void SetPAH(std::string_view name);
    void SetGasClosureDummySpecies(std::string_view name);

    /// Override with ThreeEquations-specific Planck constants.
    [[nodiscard]] double planck_coefficient(double T, double fv) const noexcept;

    /// Scaling factor for Ns transport variable [#/m3].
    [[nodiscard]] double ScalingFactorNs() const noexcept { return N0_scaling_; }

    // ── Model state queries ────────────────────────────────────────────────────

    [[nodiscard]] int nucleation_model()    const noexcept { return nucleation_model_; }
    [[nodiscard]] int surface_growth_model()const noexcept { return surface_growth_model_; }
    [[nodiscard]] int condensation_model()  const noexcept { return condensation_model_; }
    [[nodiscard]] int oxidation_model()     const noexcept { return oxidation_model_; }
    [[nodiscard]] int coagulation_model()   const noexcept { return coagulation_model_; }

    // ── CRTP extension points — process source storage owned by ThreeEquations ─
    //
    // ThreeEquations models: nucleation, coagulation, condensation, growth, oxidation.
    // NOT modelled: sintering → base class returns zero span automatically.

    [[nodiscard]] std::span<const double> sources_nucleation_impl()   const noexcept { return { source_nucleation_.data(),   this->n_equations }; }
    [[nodiscard]] std::span<const double> sources_coagulation_impl()  const noexcept { return { source_coagulation_.data(),  this->n_equations }; }
    [[nodiscard]] std::span<const double> sources_condensation_impl() const noexcept { return { source_condensation_.data(), this->n_equations }; }
    [[nodiscard]] std::span<const double> sources_growth_impl()       const noexcept { return { source_growth_.data(),       this->n_equations }; }
    [[nodiscard]] std::span<const double> sources_oxidation_impl()    const noexcept { return { source_oxidation_.data(),    this->n_equations }; }

	void WriteHeaderLine(MOM::OutputFileColumns& fOutput, const unsigned int precision);

	void WriteOutputLine( MOM::OutputFileColumns& fOutput,
							const double T, const double P_Pa, const double* Y, const double mu,
							const double* M);

private:
    // ── Private computational methods ──────────────────────────────────────────

    void MemoryAllocation();
    void Precalculations();
    void DimerConcentration();
    void NucleationSourceTerms();
    void SurfaceGrowthSourceTerms();
    void CondensationSourceTerms();
    void OxidationSourceTerms();
    void CoagulationSourceTerms();
    double SurfaceSiteDensity() const noexcept;

    struct SurfaceKineticsRates {
        double ksg;     //!< surface growth rate [1/s]
        double kox;     //!< oxidation rate [1/s]
        double kd;      //!< dehydrogenation rate [1/s]
        double krev;    //!< reverse rate [1/s]
        double ko2;     //!< O2 oxidation contribution [1/s]
        double koh;     //!< OH oxidation contribution [1/s]
        double fsootp;  //!< soot-plus fraction [-]
        double fsootc;  //!< soot-carbon fraction [-]
    };

    [[nodiscard]] SurfaceKineticsRates Kinetics_RCPAH() const noexcept;
    [[nodiscard]] SurfaceKineticsRates Kinetics_HMOM()  const noexcept;
    [[nodiscard]] SurfaceKineticsRates SurfaceKinetics()const noexcept;

    [[nodiscard]] double PAHDimerizationRate() const noexcept;

private:
    // ── Thermodynamics reference ───────────────────────────────────────────────
    const Thermo& thermo_;

    // ── Transported variables ──────────────────────────────────────────────────
    double Ys_     = 0.;   //!< soot mass fraction [-]
    double NsNorm_ = 0.;   //!< Ns / N0_scaling [-]
    double Ss_     = 0.;   //!< specific surface area [m2/m3]
    double N0_scaling_ = 1.e15;  //!< [#/m3]

    // ── Dimer properties ───────────────────────────────────────────────────────
    double vdim_  = 0., sdim_  = 0., ddim_  = 0.;
    double vnucl_ = 0., snucl_ = 0., dnucl_ = 0.;
    double vc2_   = 0., sc2_   = 0., dc2_   = 0.;
    double c_dimer_ = 0.;             //!< dimer concentration [kmol/m3]
    double dimerization_rate_ = 0.;   //!< [#/m3/s]

    // ── PAH properties ─────────────────────────────────────────────────────────
    std::string pah_species_;
    int    pah_index_ = -1;
    double vpah_ = 0., spah_ = 0., dpah_ = 0.;
    double mpah_ = 0., mwpah_= 0., ncpah_= 0., nhpah_= 0.;
    double conc_PAH_ = 0.;
    double correction_coeff_pah_pah_ = 4.4;

    // ── Species concentrations ─────────────────────────────────────────────────
    double conc_H_    = 0., conc_OH_   = 0., conc_O2_  = 0.;
    double conc_H2_   = 0., conc_H2O_  = 0., conc_C2H2_= 0.;

    int index_H_     = -1, index_OH_   = -1, index_O2_   = -1;
    int index_H2_    = -1, index_H2O_  = -1, index_C2H2_ = -1;

    double mass_fraction_H_  = 0.;
    double mass_fraction_OH_ = 0.;

    // ── Collision enhancement factors ──────────────────────────────────────────
    double epsilon_nucleation_  = 2.5;
    double epsilon_condensation_= 1.3;
    double epsilon_coagulation_ = 2.2;

    // ── Model flags ────────────────────────────────────────────────────────────
    int nucleation_model_     = 0;
    int condensation_model_   = 0;
    int coagulation_model_    = 0;
    int surface_growth_model_ = 0;
    int oxidation_model_      = 0;

    SurfaceChemistryModel surface_chem_model_     = SurfaceChemistryModel::RCPAH;
    StickingModel         sticking_model_         = StickingModel::Constant;
    double                sticking_coeff_constant_= 2.e-3;

    double Df_ = 1.8;    //!< fractal dimension [-]

    // ── Dimer concentration model ──────────────────────────────────────────────
    enum class DimerModel : int { QSSA_Rodrigues = 0 };
    DimerModel dimer_concentration_model_ = DimerModel::QSSA_Rodrigues;

    // ── Additional model flags ─────────────────────────────────────────────────
    bool smooth_heaviside_oxidation_ = true;  //!< use smooth Heaviside for oxidation particle destruction
    bool is_debug_mode_              = false;  //!< enable verbose diagnostic output
    bool is_simplified_pah_mass_     = false;  //!< use Nc*WC instead of full PAH MW

    // ── Numerical floors ───────────────────────────────────────────────────────
    double Ys_min_ = 1.e-15;    //!< [-]
    double Ns_min_ = 1.e6;      //!< [#/m3]
    double Ss_min_ = 1.e-15;    //!< [m2/m3]
    double vs_min_ = 1.e-30;    //!< [m3/#]

    // ── Per-process source storage (owned by ThreeEquations, not by base) ─────
    //
    // Only processes ThreeEquations models are declared here.
    // sintering is absent: base class returns zero span for sources_sintering().

    MomentVector source_nucleation_   = MomentVector::Zero();
    MomentVector source_coagulation_  = MomentVector::Zero();
    MomentVector source_condensation_ = MomentVector::Zero();
    MomentVector source_growth_       = MomentVector::Zero();
    MomentVector source_oxidation_    = MomentVector::Zero();

    // ── Initial moments cache ──────────────────────────────────────────────────
    MomentVector initial_moments_cache_ = MomentVector::Zero();
};

} // namespace MOM

// ── Header-only vs. pre-compiled library mode ─────────────────────────────────
//
// Default (MOM_COMPILED_LIBRARY not defined):
//   The template bodies are included here, making the class fully usable with
//   any ThermoMap type without linking against a pre-built library.
//
// When linking against libMOM (MOM_COMPILED_LIBRARY is defined by CMake via the
// INTERFACE compile definition on the MOM_lib target):
//   Template bodies are omitted from this header. The library provides pre-built
//   explicit instantiations for the types listed in the extern declarations below.
//   Custom Thermo types not in that list still require the .tpp to be visible.
//
#if !defined(MOM_COMPILED_LIBRARY)
#  include "ThreeEquations.tpp"
#else
namespace MOM 
{
    extern template class ThreeEquations<BasicThermoData>;
}
#endif
