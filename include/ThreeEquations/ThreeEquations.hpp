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

#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

#include "Eigen/Dense"

#include "MOM/MomentMethodBase.hpp"
#include "MOM/MOMConfig.hpp"
#include "MOM/ThermoProxy.hpp"

#if defined(MOM_USE_DICTIONARY)
#include <expected>
#endif

namespace MOM
{

/**
 * @class ThreeEquations
 * @brief Three-equation soot model of Franzelli, Vié & Darabiha (2019).
 *
 * Transports three conserved scalars that characterise the soot phase: mass
 * fraction, a scaled number density, and specific surface area.  Compared to
 * two-equation models, the explicit surface area equation enables more accurate
 * coupling between nucleation, condensation, and surface growth without
 * requiring an empirical surface density law.
 *
 * @par Reference
 * - Franzelli, Vié & Darabiha, *Proc. Combust. Inst.* **37** (2019) 5411–5419.
 *
 * @par Transported variables
 * | Index | Symbol | Physical meaning |
 * |---|---|---|
 * | 0 | Ys     | Soot mass fraction [-] |
 * | 1 | NsNorm | Scaled number density = Ns / N0_scaling [-] (default N0_scaling = 10¹⁵ #/m³) |
 * | 2 | Ss     | Soot specific surface area [m²/m³] |
 *
 * @par Physical processes modelled
 * - **Nucleation** — PAH dimerisation in the free-molecular regime.
 * - **Surface growth** — HACA mechanism or RC-PAH model (Franzelli 2019).
 * - **Condensation** — PAH adsorption on existing soot particles.
 * - **Oxidation** — O2 and OH attack.
 * - **Coagulation** — free-molecular + continuum kernel.
 * - **Thermophoresis** — encoded in the effective diffusion coefficient.
 *
 * @par NDF reconstruction
 * Uses a combined Pareto + log-normal reconstruction for the marginal NDF:
 * @code
 *   n(ν) = Ns · nbar(ν)
 *   nbar(ν) = α · Pareto(ν; ν_nucl, k) + (1−α) · LogNormal(ν; μ, σ)
 * @endcode
 * where ν is the particle volume per particle [m³/#].
 * Reconstruction parameters are available via `ReconstructedNDFData()`.
 *
 * @note Sintering is not modelled; `sources_sintering()` returns a zero span.
 *
 * @par Thread safety
 * Not thread-safe — one instance per OpenMP thread.
 * See `MomentMethodBase` for the complete thread-safety contract.
 *
 * @tparam Thermo  Must satisfy the MOM::ThermoMap concept.
 */

template <ThermoMap Thermo>
class ThreeEquations : public MomentMethodBase<ThreeEquations<Thermo>, 3>
{
    using Base = MomentMethodBase<ThreeEquations<Thermo>, 3>;

public:

    using typename Base::MomentVector;

    /// Labels accepted by MOM::MakeAnyMomentMethod for runtime variant selection.
    static constexpr std::array<std::string_view, 3> variant_labels{"ThreeEquations",
                                                                    "threeequations",
                                                                    "3Eq"};

    // -- Method-specific sub-model enums -------------------------------------

    /** @brief Surface chemistry model for HACA surface growth and oxidation. */
    enum class SurfaceChemistryModel : int
    {
        RCPAH = 0, //!< RC-PAH model (default): Franzelli et al. (2019).
        HMOM  = 1  //!< HMOM HACA kinetics (for cross-method consistency studies).
    };

    /** @brief PAH sticking coefficient model for nucleation and condensation. */
    enum class StickingModel : int
    {
        Constant = 0, //!< Fixed sticking coefficient (default: 2×10⁻³).
        PAH4     = 1  //!< PAH 4-ring sticking model.
    };

    /**
     * @struct NDFReconstructionData
     * @brief Parameters of the Franzelli et al. (2019) Pareto + log-normal NDF reconstruction.
     *
     * The marginal NDF n(ν) is reconstructed as:
     * @code
     *   n(ν) = Ns · nbar(ν)
     *   nbar(ν) = α · Pareto(ν; ν_nucl, k) + (1−α) · LogNormal(ν; μ, σ)
     * @endcode
     * where ν is the particle volume per particle [m³/#].
     *
     * Call `ReconstructedNDFData()` to compute and retrieve these parameters.
     */
    struct NDFReconstructionData
    {
        bool valid;     //!< True if reconstruction is physically meaningful.
        double Ns;      //!< Soot number density [#/m³].
        double fv;      //!< Soot volume fraction [-].
        double nus;     //!< Mean particle volume [m³/#].
        double alpha;   //!< Pareto weight α ∈ [0,1] [-].
        double nbar0;   //!< Nucleation-peak normalised NDF value [1/m³].
        double sigma;   //!< Log-normal standard deviation σ [-].
        double k;       //!< Pareto tail index k [-].
        double nu1mean; //!< Mean volume of the Pareto contribution [m³/#].
        double nu2mean; //!< Mean volume of the log-normal contribution [m³/#].
        double mu;      //!< Log-normal location parameter μ [log(m³)].
    };

    // -- Configuration struct ------------------------------------------------

    /**
     * @struct Config
     * @brief Plain configuration parameters for the ThreeEquations variant.
     * @note No external dependencies: only standard C++ types.
     */
    struct Config : CommonConfig<1>,
                    PAHConfig,
                    BinarySootProcessConfig,
                    CollisionEnhancementConfig,
                    StickingConfig,
                    GasConsumptionConfig<false>,
                    SootRadiationConfig
    {
        /// Surface chemistry model: "rcpah" | "hmom"
        std::string surface_chemistry_model = "rcpah";

        /// Dimer concentration model: "qssa-rodrigues" (currently only option)
        std::string dimer_model = "qssa-rodrigues";

        // ---- Collision correction ------------------------------------------
        double correction_coeff_pah_pah = 4.4; //!< PAH–PAH correction coefficient [-]

        // ---- Numerical floors ----------------------------------------------
        double ns_minimum_per_m3 = 1.e6; //!< Minimum soot number density [#/m³]
    };

    // -- Construction ---------------------------------------------------------

    /**
     * @brief Constructs ThreeEquations bound to the given thermodynamics map.
     * @param thermo  Const reference to the thermodynamics map (must outlive this object).
     */
    explicit ThreeEquations(const Thermo& thermo);
    explicit ThreeEquations(const Thermo&&) = delete; ///< Prevents binding a temporary as thermo (dangling ref).

    ThreeEquations(const ThreeEquations&)            = delete;
    ThreeEquations& operator=(const ThreeEquations&) = delete;
    ThreeEquations(ThreeEquations&&)            = default; ///< Move-constructible for placement in std::variant.
    ThreeEquations& operator=(ThreeEquations&&) = delete;  ///< Not move-assignable — const Thermo& member cannot be reseated.

    /**
     * @brief Configure all ThreeEquations parameters from a plain configuration struct.
     *
     * Applies every field of @p cfg by calling the corresponding `Set*()`
     * methods, followed by `PrintSummary()`.  This is the primary programmatic
     * setup path; it has no dependency on external parsing frameworks.
     *
     * @param cfg  Configuration struct.  Default-constructed @c Config
     *             reproduces the constructor defaults.
     */
    void SetupFromConfig(const Config& cfg);

#if defined(MOM_USE_DICTIONARY)
    /**
     * @brief Parse an OpenSMOKE++ dictionary into a ThreeEquations Config.
     * @tparam DictType  OpenSMOKE++ dictionary type — no include-time dependency.
     */
    template <typename DictType>
    [[nodiscard]] static std::expected<Config, std::string> ParseConfig(DictType& dict);
#endif

    // -- MomentMethod concept — state injection --------------------------------

    /// @param T    Temperature [K]
    /// @param P_Pa Pressure [Pa]
    /// @param Y    Mass fractions, size = n_species
    void SetStatus(double T, double P_Pa, const double* Y) noexcept;

    /// Generic span setter. Order: [Ys, NsNorm, Ss].
    /// Ys [-], NsNorm [-], Ss [m2/m3].
    void SetMoments(std::span<const double> m) noexcept;

    /// Named setter (preferred for ThreeEquations-aware code).
    void SetMoments(double Ys, double NsNorm, double Ss) noexcept;

    // -- MomentMethod concept — core computation -------------------------------

    void CalculateSourceMoments() noexcept;
    void CalculateOmegaGas() noexcept;

    // -- MomentMethod concept — particle properties ----------------------------

    [[nodiscard]] double volume_fraction() const noexcept;
    [[nodiscard]] double particle_diameter() const noexcept;            //!< primary particle diameter [m]
    [[nodiscard]] double collision_diameter() const noexcept;           //!< aggregate collision diameter [m]
    [[nodiscard]] double particle_number_density() const noexcept;      //!< particle number density [#/m3]
    [[nodiscard]] double mass_fraction() const noexcept;                //!< mass fraction (= Ys_) [-]
    [[nodiscard]] double specific_surface() const noexcept;             //!< specific surface area [m2/m3]
    [[nodiscard]] double number_primary_particles() const noexcept;     //!< number of primary particles [-]
    [[nodiscard]] double diffusion_coefficient() const noexcept;        //!< diffusion coefficient [kg/m/s]

    // -- MomentMethod concept — initial conditions -----------------------------

    [[nodiscard]] std::span<const double> initial_moments() const noexcept
    {
        return {initial_moments_cache_.data(), 3u};
    }

    // -- MomentMethod concept — precursor --------------------------------------

    [[nodiscard]] int precursor_index() const noexcept { return pah_index_; }

    [[nodiscard]] double precursor_concentration() const noexcept { return conc_PAH_; }

    [[nodiscard]] const std::string& precursor_species() const noexcept { return pah_species_; }

    // -- MomentMethod concept — diagnostics ------------------------------------

    void PrintSummary() const;

    // -- Aggregated properties helper ------------------------------------------

    /// Returns fv, dp, dc, np, ss, vs in one call (useful for post-processing).
    void Properties(double& fv, double& dp, double& dc, double& np, double& ss, double& vs) const noexcept;

    // -- Reporter output hook (MomentMethodReporter extensibility protocol) ------
    //
    // Makes ThreeEquations self-describing with respect to output.
    // MomentMethodReporter calls this with a lambda cb(label, value):
    //   • header mode — lambda uses label to register the column
    //   • row mode    — lambda uses value to write the data

    /// ThreeEquations-specific prefix columns: np, ss, vs, NDF parameters.
    template <typename CB> void variant_prefix_output(CB&& cb) const
    {
        double fv, dp, dc, np, ss, vs;
        Properties(fv, dp, dc, np, ss, vs);
        cb("np[-]", np);
        cb("ss[m2/#]", ss);
        cb("vs[m3/#]", vs);
        const auto ndf = ReconstructedNDFData();
        cb("alpha[-]", ndf.alpha);
        cb("nbar0[1/m3]", ndf.nbar0);
        cb("sigma[-]", ndf.sigma);
        cb("kPareto[-]", ndf.k);
        cb("nu1mean[m3/#]", ndf.nu1mean);
        cb("nu2mean[m3/#]", ndf.nu2mean);
        cb("mu[log(m3)]", ndf.mu);

        this->EmitStandardSootOmegaGas(
            cb, pah_index_, index_C2H2_, index_H2_, index_O2_, index_H2O_, index_OH_);
    }

    // -- NDF reconstruction (ThreeEquations-specific) --------------------------

    /// Computes the Franzelli et al. (2019) NDF reconstruction parameters.
    /// @param use_regularized_moments  If true, uses floor-clipped moments.
    [[nodiscard]] NDFReconstructionData ReconstructedNDFData(bool use_regularized_moments = false) const;

    /// Normalised NDF nbar(ν) [1/m3].
    [[nodiscard]] double ReconstructedNormalizedNDF(double nu,
                                                    bool use_regularized_moments = false) const;

    /// Dimensional NDF n(ν) = Ns * nbar(ν) [#/m3/m3].
    [[nodiscard]] double ReconstructedNDF(double nu, bool use_regularized_moments = false) const;

    /// Vectorized dimensional NDF.
    void ReconstructedNDF(const Eigen::VectorXd& nu,
                          Eigen::VectorXd& n,
                          bool use_regularized_moments = false) const;

    // -- Surface area geometry utilities ---------------------------------------

    /// Change in surface area for a volume increment deltaV on a sphere.
    [[nodiscard]] static double DeltaSurfaceSpherical(double deltaV, double V, double S) noexcept;

    /// Change in surface area for a volume increment deltaV on a fractal aggregate.
    [[nodiscard]] static double DeltaSurfaceFractal(double deltaV, double V, double S, double np) noexcept;

    // -- Model switches --------------------------------------------------------

    void SetNucleation(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument(
                "[ThreeEquations] Invalid nucleation model flag. Allowed values: 0, 1.");
        nucleation_model_ = flag;
    }

    void SetSurfaceGrowth(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument(
                "[ThreeEquations] Invalid surface-growth model flag. Allowed values: 0, 1.");
        surface_growth_model_ = flag;
    }

    void SetCondensation(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument(
                "[ThreeEquations] Invalid condensation model flag. Allowed values: 0, 1.");
        condensation_model_ = flag;
    }

    void SetOxidation(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument(
                "[ThreeEquations] Invalid oxidation model flag. Allowed values: 0, 1.");
        oxidation_model_ = flag;
    }

    void SetCoagulation(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument(
                "[ThreeEquations] Invalid coagulation model flag. Allowed values: 0, 1.");
        coagulation_model_ = flag;
    }

    void SetSurfaceChemistryModel(std::string_view label);

    void SetSurfaceChemistryModel(SurfaceChemistryModel m) noexcept { surface_chem_model_ = m; }

    void SetNucleationCollisionEnhancementFactor(double eps) noexcept { epsilon_nucleation_ = eps; }

    void SetCondensationCollisionEnhancementFactor(double eps) noexcept
    {
        epsilon_condensation_ = eps;
    }

    void SetCoagulationCollisionEnhancementFactor(double eps) noexcept
    {
        epsilon_coagulation_ = eps;
    }

    void SetCorrectionCoefficientPAHPAH(double v) noexcept { correction_coeff_pah_pah_ = v; }

    void SetStickingCoefficientModel(std::string_view label);

    void SetStickingCoefficientConstant(double v) noexcept { sticking_coeff_constant_ = v; }

    /// Particle density alias (triggers geometry recalculation).
    void SetNsMinimum(double v) noexcept;
    void SetPAH(std::string_view name);
    void SetGasClosureDummySpecies(std::string_view name);

    /// Override with ThreeEquations-specific Planck constants.
    [[nodiscard]] double planck_coefficient(double T, double fv) const noexcept;

    /// Scaling factor for Ns transport variable [#/m3].
    [[nodiscard]] double ScalingFactorNs() const noexcept { return N0_scaling_; }

    // -- Model state queries ----------------------------------------------------

    [[nodiscard]] int nucleation_model() const noexcept { return nucleation_model_; }

    [[nodiscard]] int surface_growth_model() const noexcept { return surface_growth_model_; }

    [[nodiscard]] int condensation_model() const noexcept { return condensation_model_; }

    [[nodiscard]] int oxidation_model() const noexcept { return oxidation_model_; }

    [[nodiscard]] int coagulation_model() const noexcept { return coagulation_model_; }

    /**
     * @name CRTP extension points — per-process source storage
     *
     * ThreeEquations models: nucleation, coagulation, condensation, growth, oxidation.
     * Sintering is **not** modelled; `sources_sintering()` returns a zero span.
     * @{
     */

    /** @brief Nucleation source terms [mol/m³/s], size = n_equations. */
    [[nodiscard, gnu::always_inline]] std::span<const double> sources_nucleation_impl() const noexcept
    {
        return {source_nucleation_.data(), this->n_equations};
    }

    [[nodiscard, gnu::always_inline]] std::span<const double> sources_coagulation_impl() const noexcept
    {
        return {source_coagulation_.data(), this->n_equations};
    }

    [[nodiscard, gnu::always_inline]] std::span<const double> sources_condensation_impl() const noexcept
    {
        return {source_condensation_.data(), this->n_equations};
    }

    [[nodiscard, gnu::always_inline]] std::span<const double> sources_growth_impl() const noexcept
    {
        return {source_growth_.data(), this->n_equations};
    }

    [[nodiscard, gnu::always_inline]] std::span<const double> sources_oxidation_impl() const noexcept
    {
        return {source_oxidation_.data(), this->n_equations};
    }

    /** @brief Oxidation-only gas-phase source terms [kg/m³/s] for operator splitting. */
    [[nodiscard, gnu::always_inline]] std::span<const double> omega_gas_oxidation_impl() const noexcept
    {
        return {omega_gas_oxidation_.data(),
                static_cast<std::size_t>(omega_gas_oxidation_.size())};
    }

    /** @} */

private:

    // -- Private computational methods ------------------------------------------

    void MemoryAllocation();
    void Precalculations();
    void ApplyConfig(const Config& cfg); //!< core of SetupFromConfig, without PrintSummary()
    void DimerConcentration();
    void NucleationSourceTerms();
    void SurfaceGrowthSourceTerms();
    void CondensationSourceTerms();
    void OxidationSourceTerms();
    void CoagulationSourceTerms();
    double SurfaceSiteDensity() const noexcept;

    struct SurfaceKineticsRates
    {
        double ksg;    //!< surface growth rate [1/s]
        double kox;    //!< oxidation rate [1/s]
        double kd;     //!< dehydrogenation rate [1/s]
        double krev;   //!< reverse rate [1/s]
        double ko2;    //!< O2 oxidation contribution [1/s]
        double koh;    //!< OH oxidation contribution [1/s]
        double fsootp; //!< soot-plus fraction [-]
        double fsootc; //!< soot-carbon fraction [-]
    };

    [[nodiscard]] SurfaceKineticsRates Kinetics_RCPAH() const noexcept;
    [[nodiscard]] SurfaceKineticsRates Kinetics_HMOM() const noexcept;
    [[nodiscard]] SurfaceKineticsRates SurfaceKinetics() const noexcept;

    [[nodiscard]] double PAHDimerizationRate() const noexcept;

private:

    // -- Thermodynamics reference -----------------------------------------------
    const Thermo& thermo_;

    // -- Transported variables --------------------------------------------------
    double Ys_         = 0.;    //!< soot mass fraction [-]
    double NsNorm_     = 0.;    //!< Ns / N0_scaling [-]
    double Ss_         = 0.;    //!< specific surface area [m2/m3]
    double N0_scaling_ = 1.e15; //!< [#/m3]

    // -- Dimer properties -------------------------------------------------------
    double vdim_ = 0.;
    double sdim_ = 0.;
    double ddim_ = 0.;
    double vnucl_ = 0.;
    double snucl_ = 0.;
    double dnucl_ = 0.;
    double vc2_ = 0.;
    double sc2_ = 0.;
    double dc2_ = 0.;
    double c_dimer_ = 0.; //!< dimer concentration [kmol/m3]
    double dimerization_rate_ = 0.; //!< [#/m3/s]

    // -- PAH properties ---------------------------------------------------------
    std::string pah_species_;
    int pah_index_ = -1;
    double vpah_ = 0.;
    double spah_ = 0.;
    double dpah_ = 0.;
    double mpah_ = 0.; 
    double mwpah_ = 0.;
    double ncpah_ = 0.;
    double nhpah_ = 0.;
    double conc_PAH_ = 0.;
    double correction_coeff_pah_pah_ = 4.4;

    // -- Species concentrations -------------------------------------------------
    double conc_H_ = 0.;
    double conc_OH_ = 0.; 
    double conc_O2_ = 0.;
    double conc_H2_ = 0.;
    double conc_H2O_ = 0.;
    double conc_C2H2_ = 0.;

    int index_H_ = -1;
    int index_OH_ = -1; 
    int index_O2_ = -1;
    int index_H2_   = -1;
    int index_H2O_  = -1;
    int index_CO_   = -1; //!< CO — oxidation product (both channels)
    int index_C2H2_ = -1;

    double mass_fraction_H_  = 0.;
    double mass_fraction_OH_ = 0.;

    // -- Collision enhancement factors ------------------------------------------
    double epsilon_nucleation_   = 2.5;
    double epsilon_condensation_ = 1.3;
    double epsilon_coagulation_  = 2.2;

    // -- Model flags ------------------------------------------------------------
    int nucleation_model_     = 0;
    int condensation_model_   = 0;
    int coagulation_model_    = 0;
    int surface_growth_model_ = 0;
    int oxidation_model_      = 0;

    SurfaceChemistryModel surface_chem_model_ = SurfaceChemistryModel::RCPAH;
    StickingModel sticking_model_             = StickingModel::Constant;
    double sticking_coeff_constant_           = 2.e-3;

    double Df_ = 1.8; //!< fractal dimension [-]

    // -- Dimer concentration model ----------------------------------------------
    enum class DimerModel : int
    {
        QSSA_Rodrigues = 0
    };
    DimerModel dimer_concentration_model_ = DimerModel::QSSA_Rodrigues;

    // -- Additional model flags -------------------------------------------------
    bool smooth_heaviside_oxidation_ = true; //!< Heaviside for oxidation particle destruction
    bool is_debug_mode_          = false; //!< enable verbose diagnostic output
    bool is_simplified_pah_mass_ = false; //!< use Nc*WC instead of full PAH MW

    // -- Numerical floors -------------------------------------------------------
    double Ys_min_ = 1.e-15; //!< [-]
    double Ns_min_ = 1.e6;   //!< [#/m3]
    double Ss_min_ = 1.e-15; //!< [m2/m3]
    double vs_min_ = 1.e-30; //!< [m3/#]

    // -- Per-process source storage (owned by ThreeEquations, not by base) -----
    //
    // Only processes ThreeEquations models are declared here.
    // sintering is absent: base class returns zero span for sources_sintering().

    MomentVector source_nucleation_   = MomentVector::Zero();
    MomentVector source_coagulation_  = MomentVector::Zero();
    MomentVector source_condensation_ = MomentVector::Zero();
    MomentVector source_growth_       = MomentVector::Zero();
    MomentVector source_oxidation_    = MomentVector::Zero();

    Eigen::VectorXd omega_gas_oxidation_; //!< Oxidation-only gas-phase sources [kg/m³/s].

    // -- Initial moments cache --------------------------------------------------
    MomentVector initial_moments_cache_ = MomentVector::Zero();
};

} // namespace MOM

// -- Header-only vs. pre-compiled library mode ---------------------------------
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
#if defined(MOM_USE_DICTIONARY)
#include "ThreeEquations_Grammar.h"
#endif

#if !defined(MOM_COMPILED_LIBRARY)
#include "ThreeEquations.tpp"
#else
namespace MOM
{
extern template class ThreeEquations<BasicThermoData>;
}
#endif
