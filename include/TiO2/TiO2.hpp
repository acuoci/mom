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
#include <string>
#include <string_view>

#include "Eigen/Dense"

#include "MOM/MomentMethodBase.hpp"
#include "MOM/ThermoProxy.hpp"

#if defined(MOM_USE_DICTIONARY)
#include <expected>
#endif

namespace MOM
{

/**
 * @class TiO2
 * @brief Three-equation method of moments for TiO2 nanoparticle formation and evolution.
 *
 * Models TiO2 (anatase) nanoparticle formation from gas-phase titanium precursors
 * (e.g. Ti(OH)4, TiCl4 families) via nucleation, condensation, coagulation, and
 * sintering.  Shares the three-scalar transport structure of `ThreeEquations` but
 * is adapted for an inorganic oxidic system without surface carbon chemistry.
 *
 * @par References
 * - Kruis, Kusters & Pratsinis, *Aerosol Sci. Technol.* **19** (1993) 514–526.
 * - Franzelli, Vié & Darabiha, *Proc. Combust. Inst.* **37** (2019) 5411–5419 (NDF reconstruction).
 *
 * @par Transported variables
 * | Index | Symbol | Physical meaning |
 * |---|---|---|
 * | 0 | YTiO2  | TiO2 particle mass fraction [-] |
 * | 1 | NTiO2N | Scaled number density = N / N0_scaling [-] (default N0_scaling = 10¹⁵ #/m³) |
 * | 2 | STiO2  | Total particle surface area per gas volume [m²/m³] |
 *
 * @par Physical processes modelled
 * - **Nucleation** — binary (precursor + precursor) or fixed-cluster variant.
 * - **Condensation** — precursor condensation on existing TiO2 particles.
 * - **Coagulation** — free-molecular kernel with enhancement factor.
 * - **Sintering** — viscous-flow model (Kruis et al. 1993):
 *   τ_s = (1/A_s)·T^{−n_s}·exp(T_s/T).
 * - **Thermophoresis** — encoded in the effective diffusion coefficient.
 *
 * @par NOT modelled
 * - Surface growth by CVD (no HACA-equivalent for TiO2).
 * - Oxidation (particles are already fully oxidised).
 * - Radiative heat transfer (dielectric — `PlanckCoeffModel::None`).
 *
 * @note `sources_growth()` and `sources_oxidation()` return zero spans (fallback
 *       from base class — no `_impl()` declared).
 * @note When sintering is stiff relative to the flow time step, use
 *       `SinteringDeferredUpdate()` to integrate it separately via an ODE sub-step.
 *
 * @tparam Thermo  Must satisfy the MOM::ThermoMap concept.
 */

template <ThermoMap Thermo> class TiO2 : public MomentMethodBase<TiO2<Thermo>, 3>
{
    using Base = MomentMethodBase<TiO2<Thermo>, 3>;

public:

    using typename Base::MomentVector;

    /// Labels accepted by MOM::MakeAnyMomentMethod for runtime variant selection.
    static constexpr std::array<std::string_view, 2> variant_labels{"TiO2", "tio2"};

    // -- Method-specific sub-model enums -------------------------------------

    /** @brief Nucleation mechanism selector for TiO2. */
    enum class NucleationVariant : int
    {
        Off          = 0, //!< Nucleation disabled.
        Binary       = 1, //!< Ti(OH)4 + Ti(OH)4 → (TiO2)_n + vapour (free-molecular collision).
        FixedCluster = 2  //!< Nucleation via a fixed-size cluster of n0 TiO2 units.
    };

    /**
     * @struct NDFReconstructionData
     * @brief Parameters of the Pareto + log-normal NDF reconstruction for TiO2.
     *
     * Uses the same reconstruction framework as `ThreeEquations` (Franzelli et al.
     * 2019), adapted for TiO2 nanoparticle size distributions.
     *
     * Call `ReconstructedNDFData()` to compute and retrieve these parameters.
     */
    struct NDFReconstructionData
    {
        bool valid;     //!< True if reconstruction is physically meaningful.
        double N;       //!< Particle number density [#/m³].
        double fv;      //!< Volume fraction [-].
        double nuMean;  //!< Mean particle volume [m³/#].
        double nuNucl;  //!< Volume of a newly nucleated particle [m³/#].
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
     * @brief Plain configuration parameters for the TiO2 variant.
     *
     * The @c nucleation_model field uses strings matching the grammar convention:
     * @c "none" | @c "binary" (default) | @c "fixed-cluster".
     * @note No external dependencies: only standard C++ types.
     */
    struct Config
    {
        // ---- Activation / precursor ----------------------------------------
        bool        is_active         = true;   //!< Enable this variant
        std::string precursor_species = "none"; //!< TiO2 precursor species

        // ---- Gas consumption / closure -------------------------------------
        bool        gas_consumption           = false;  //!< Consume gas-phase species
        std::string gas_closure_dummy_species = "none"; //!< Dummy mass-closure species

        // ---- Process model selection ---------------------------------------
        /// Nucleation model: "none" | "binary" (default) | "fixed-cluster"
        std::string nucleation_model = "binary";
        int sintering_model          = 1; //!< Sintering model index
        int coagulation_model        = 1; //!< Coagulation model index
        int condensation_model       = 1; //!< Condensation model index
        int thermophoretic_model     = 1; //!< Thermophoretic model index

        // ---- Particle cluster sizes ----------------------------------------
        int minimum_tio2_units            = 2; //!< Minimum TiO2 units per aggregate
        int nucleated_particle_tio2_units = 5; //!< TiO2 units per nucleated particle

        // ---- Sintering kinetics: τ_s = As · T^ns · d_p^4 · exp(Ts/T) -----
        double sintering_As_s_K_m = 7.44e16; //!< Pre-exponential [s,K,m]
        double sintering_Ts_K     = 31000.;  //!< Activation temperature [K] (positive; used as exp(Ts/T))
        double sintering_ns       =  1.0;    //!< Temperature/size exponent [-]

        // ---- Sintering numerical regularisation ----------------------------
        bool   sintering_deferred    = false;  //!< Defer sintering to operator-split step
        double sintering_dp_min_m    = 2.e-9;  //!< Diameter below which sintering inactive [m]
        double sintering_tau_min_s   = 1.e-10; //!< Minimum sintering time-scale [s]
        double sintering_k_max_per_s = 1.e6;   //!< Maximum sintering rate [1/s]

        // ---- Numerical floors ----------------------------------------------
        double ns_minimum_per_m3 = 1.e3;   //!< Minimum number density floor [#/m³]
        double fv_minimum        = 1.e-16; //!< Minimum volume fraction floor [-]

        // ---- Transport -----------------------------------------------------
        double schmidt_number = 50.; //!< Schmidt number for moment transport

        // ---- Debug ---------------------------------------------------------
        bool debug_mode = false; //!< Verbose diagnostic output
    };

    // -- Construction ---------------------------------------------------------

    explicit TiO2(const Thermo& thermo);

    TiO2(const TiO2&)            = delete;
    TiO2& operator=(const TiO2&) = delete;
    TiO2(TiO2&&)                 = default;
    TiO2& operator=(TiO2&&)      = default;

    /**
     * @brief Configure all TiO2 parameters from a plain configuration struct.
     *
     * Applies every field of @p cfg by calling the corresponding `Set*()`
     * methods or direct member assignment where no setter exists, followed
     * by `PrintSummary()`.  No dependency on external parsing frameworks.
     *
     * @param cfg  Configuration struct.  Default-constructed @c Config
     *             reproduces the constructor defaults.
     */
    void SetupFromConfig(const Config& cfg);

#if defined(MOM_USE_DICTIONARY)
    /**
     * @brief Parse an OpenSMOKE++ dictionary into a TiO2 Config.
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

    /// Generic span setter. Order: [YTiO2, NTiO2N, STiO2].
    /// YTiO2 [-], NTiO2N [-], STiO2 [m2/m3].
    void SetMoments(std::span<const double> m) noexcept;

    /// Named setter (preferred for TiO2-aware code).
    void SetMoments(double YTiO2, double NTiO2N, double STiO2) noexcept;

    // -- MomentMethod concept — core computation -------------------------------

    /// Computes all source terms for the current cell state.
    void CalculateSourceMoments() noexcept;

    void CalculateOmegaGas() noexcept;

    // -- Deferred sintering (operator-splitting compatible) --------------------
    //
    // When sintering is stiff relative to the main transport time step,
    // it can be integrated separately using an ODE solver. Call this method
    // after CalculateSourceMoments to update surface area with a sub-step.
    //
    // Returns the sintering time scale [s].

    [[nodiscard]] double SinteringDeferredUpdate(double dt_ode);

    // -- MomentMethod concept — particle properties ----------------------------

    [[nodiscard]] double volume_fraction() const noexcept;
    [[nodiscard]] double particle_diameter() const noexcept;  //!< primary particle diameter [m]
    [[nodiscard]] double collision_diameter() const noexcept; //!< aggregate collision diameter [m]
    [[nodiscard]] double AggregateDiameter() const noexcept; //!< mobility aggregate diameter [m]
    [[nodiscard]] double particle_number_density() const noexcept;    //!< [#/m3]
    [[nodiscard]] double mass_fraction() const noexcept;             //!< = YTiO2_
    [[nodiscard]] double specific_surface() const noexcept;          //!< STiO2 [m2/m3]
    [[nodiscard]] double number_primary_particles() const noexcept; //!< np [-]
    [[nodiscard]] double diffusion_coefficient() const noexcept;     //!< [kg/m/s]

    // -- MomentMethod concept — initial conditions -----------------------------

    [[nodiscard]] std::span<const double> initial_moments() const noexcept
    {
        return {initial_moments_cache_.data(), 3u};
    }

    // -- MomentMethod concept — precursor --------------------------------------

    [[nodiscard]] int precursor_index() const noexcept { return precursor_index_; }

    [[nodiscard]] double precursor_concentration() const noexcept { return c_precursor_; }

    [[nodiscard]] const std::string& precursor_species() const noexcept
    {
        return precursor_species_;
    }

    // -- MomentMethod concept — diagnostics ------------------------------------

    void PrintSummary() const;

    // -- Aggregated properties helper ------------------------------------------

    void Properties(double& fv,
                    double& dp,
                    double& dc,
                    double& da,
                    double& np,
                    double& ss,
                    double& vs,
                    double& ssph,
                    double& tauS) const noexcept;

    // -- Reporter output hook (MomentMethodReporter extensibility protocol) ------
    //
    // Makes TiO2 self-describing with respect to output.
    // MomentMethodReporter calls this with a lambda cb(label, value):
    //   • header mode — lambda uses label to register the column
    //   • row mode    — lambda uses value to write the data

    /// TiO2-specific prefix columns: da, np, ss, vs, tauS, NDF parameters.
    template <typename CB> void variant_prefix_output(CB&& cb) const
    {
        double fv, dp, dc, da, np, ss, vs, ssph, tauS;
        Properties(fv, dp, dc, da, np, ss, vs, ssph, tauS);
        cb("da[nm]", da * 1.e9);
        cb("np[-]", np);
        cb("ss[m2/#]", ss);
        cb("vs[m3/#]", vs);
        cb("tauS[s]", tauS);
        const auto ndf = ReconstructedNDFData();
        cb("alpha[-]", ndf.alpha);
        cb("nbar0[1/m3]", ndf.nbar0);
        cb("sigma[-]", ndf.sigma);
        cb("kPareto[-]", ndf.k);
        cb("nu1mean[m3/#]", ndf.nu1mean);
        cb("nu2mean[m3/#]", ndf.nu2mean);
        cb("mu[log(m3)]", ndf.mu);

        cb("omegaTot[kg/m3/s]", this->omega_gas_.sum());
        cb("omegaPrec[kg/m3/s]", this->omega_gas_[precursor_index_]);
        cb("omegaO2[kg/m3/s]", this->omega_gas_[O2_index_]);
        cb("omegaH2O[kg/m3/s]", this->omega_gas_[H2O_index_]);
        cb("omegaCO2[kg/m3/s]", this->omega_gas_[CO2_index_]);
    }

    // -- NDF reconstruction (TiO2-specific) -----------------------------------

    [[nodiscard]] NDFReconstructionData ReconstructedNDFData(bool use_regularized_moments = false) const;

    [[nodiscard]] double ReconstructedNormalizedNDF(double nu,
                                                    bool use_regularized_moments = false) const;

    [[nodiscard]] double ReconstructedNDF(double nu, bool use_regularized_moments = false) const;

    void ReconstructedNDF(const Eigen::VectorXd& nu,
                          Eigen::VectorXd& n,
                          bool use_regularized_moments = false) const;

    // -- Material / geometry accessors -----------------------------------------

    [[nodiscard]] double rhoTiO2() const noexcept { return rho_TiO2_; }

    [[nodiscard]] double NucleationParticleVolume() const noexcept;

    [[nodiscard]] double s0() const noexcept { return s0_; }

    [[nodiscard]] double v0() const noexcept { return v0_; }

    [[nodiscard]] double ScalingFactorNs() const noexcept { return N0_scaling_; }

    // -- Model switches --------------------------------------------------------

    void SetNucleation(int flag) noexcept
    {
        nucleation_variant_ = static_cast<NucleationVariant>(flag);
    }

    void SetCoagulation(int flag) noexcept { coagulation_model_ = flag; }

    void SetCondensation(int flag) noexcept { condensation_model_ = flag; }

    void SetSintering(int flag) noexcept { sintering_model_ = flag; }

    void SetNumberOfTiO2UnitsPerNucleatedParticle(unsigned n) noexcept { n0_ = n; }

    void SetMinimumNumberOfTiO2Units(unsigned n) noexcept { nTiO2_min_ = n; }

    void SetNucleationCollisionEnhancementFactor(double eps) noexcept { epsilon_nuc_ = eps; }

    void SetCoagulationCollisionEnhancementFactor(double eps) noexcept { epsilon_coag_ = eps; }

    void SetCondensationCollisionEnhancementFactor(double eps) noexcept { epsilon_cond_ = eps; }

    /// Sintering kinetics: τ_s = (1 / As) * T^(-ns) * exp(Ts / T) [s]
    void SetSinteringFrequencyFactor(double As) noexcept { As_ = As; }

    void SetSinteringActivationTemperature(double Ts) noexcept { Ts_ = Ts; }

    void SetSinteringTemperatureExponent(double ns) noexcept { ns_ = ns; }

    void SetNMinimum(double v) noexcept { N_min_ = v; }

    void SetFvMinimum(double v) noexcept { fv_min_ = v; }

    void SetPrecursor(std::string_view name);
    void SetGasClosureDummySpecies(std::string_view name);

    /// Configure stoichiometry for gas-phase consumption from precursor formula.
    void SetupGasConsumptionStoichiometry();

    // -- Model state queries ----------------------------------------------------

    [[nodiscard]] int nucleation_model() const noexcept
    {
        return static_cast<int>(nucleation_variant_);
    }

    [[nodiscard]] int coagulation_model() const noexcept { return coagulation_model_; }

    [[nodiscard]] int condensation_model() const noexcept { return condensation_model_; }

    [[nodiscard]] int sintering_model() const noexcept { return sintering_model_; }

    [[nodiscard]] bool is_sintering_deferred() const noexcept { return is_sintering_deferred_; }

    /**
     * @name CRTP extension points — per-process source storage
     *
     * TiO2 models: nucleation, coagulation, condensation, sintering.
     * Growth and oxidation are **not** modelled; `sources_growth()` and
     * `sources_oxidation()` return zero spans automatically.
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

    [[nodiscard, gnu::always_inline]] std::span<const double> sources_sintering_impl() const noexcept
    {
        return {source_sintering_.data(), this->n_equations};
    }

    /** @} */

private:

    // -- Private computational methods ------------------------------------------

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

    // -- Thermodynamics reference -----------------------------------------------
    const Thermo& thermo_;

    // -- Transported variables --------------------------------------------------
    double YTiO2_      = 0.;    //!< TiO2 mass fraction [-]
    double NTiO2N_     = 0.;    //!< N / N0_scaling [-]
    double STiO2_      = 0.;    //!< surface area concentration [m2/m3]
    double N0_scaling_ = 1.e15; //!< [#/m3]

    // -- Material properties ----------------------------------------------------
    static constexpr double W_TiO2_  = 79.866; //!< TiO2 molecular weight [kg/kmol]
    static constexpr double rho_TiO2_ = 3900.;  //!< solid anatase density [kg/m3]
    static constexpr double m_TiO2_  = W_TiO2_ / (6.02214076e26); //!< [kg/molecule]

    // -- Monomer/nucleus geometry -----------------------------------------------
    double v0_ = 0.; //!< monomer volume [m3]
    double s0_ = 0.; //!< monomer surface [m2]
    double d0_ = 0.; //!< monomer diameter [m]

    // -- Precursor properties ---------------------------------------------------
    std::string precursor_species_;
    int precursor_index_  = -1;
    double nti_precursor_ = 0.; //!< Ti atoms per precursor molecule
    double nh_precursor_  = 0.; //!< H atoms
    double no_precursor_  = 0.; //!< O atoms
    double nc_precursor_  = 0.; //!< C atoms
    double Y_precursor_   = 0.; //!< mass fraction
    double c_precursor_   = 0.; //!< molar concentration [kmol/m3]
    double W_precursor_   = 0.; //!< molecular weight [kg/kmol]
    double m_precursor_   = 0.; //!< molecular mass [kg]

    // Collision geometry of the gas precursor molecule
    double v_precursor_ = 0.; //!< pseudo-molecular volume [m3]
    double d_precursor_ = 0.; //!< collision diameter [m]

    // Effective precursor geometry used by nucleation/condensation kernels
    double vprec_ = 0.; //!< solid TiO2 volume added per precursor molecule [m3]
    double dprec_ = 0.; //!< collision diameter for beta_nuc / beta_cond [m]

    // -- Gas consumption stoichiometry ------------------------------------------
    // Derived from atom balance: Ti_a C_b H_c O_d + x O2 → TiO2(s) + y CO2 + z H2O
    int H2O_index_ = -1;
    int CO2_index_ = -1; 
    int O2_index_ = -1;
    double W_H2O_ = 0.;
    double W_CO2_ = 0.;
    double W_O2_ = 0.;
    double nu_H2O_from_prec_ = 0.; //!< H2O stoichiometric coefficient
    double nu_CO2_from_prec_ = 0.; //!< CO2 stoichiometric coefficient
    double nu_O2_from_prec_  = 0.; //!< O2 stoichiometric coefficient (negative = consumed)

    // -- Nucleation parameters --------------------------------------------------
    NucleationVariant nucleation_variant_ = NucleationVariant::Off;
    unsigned int nTiO2_min_               = 1;   //!< minimum reference size for regularization
    unsigned int n0_                      = 5;   //!< TiO2 units per newly nucleated particle
    double epsilon_nuc_                   = 2.5; //!< nucleation collision enhancement factor

    // -- Sintering parameters ---------------------------------------------------
    double As_ = 7.44e16; //!< sintering frequency factor [1/s/K]
    double ns_ = 1.0;     //!< temperature exponent [-]
    double Ts_ = 31000.;  //!< activation temperature [K]

    // Sintering regularisation parameters (set in Precalculations)
    double sintering_dp_min_              = 0.;
    double sintering_activation_np_       = 0.;
    double sintering_activation_width_np_ = 0.;
    double sintering_tau_min_             = 0.;
    double sintering_k_max_               = 0.;
    double sintering_relative_tolerance_  = 1.e-4;
    double sintering_tau_qss_             = 0.;
    bool is_sintering_deferred_           = false;

    // -- Coagulation / condensation parameters ----------------------------------
    double epsilon_coag_ = 2.2; //!< coagulation enhancement factor
    double epsilon_cond_ = 1.3; //!< condensation enhancement factor

    // Pre-computed kernel prefactors (set in Precalculations; temperature-independent)
    double alpha_nuc_  = 0.; //!< [m3/s/K^0.5]
    double alpha_coag_ = 0.; //!< [m^(5/2)/s/K^(1/2)]
    double alpha_cond_ = 0.; //!< [m^(5/2)/s/K^(1/2)]

    // -- Model flags ------------------------------------------------------------
    int coagulation_model_  = 0;
    int condensation_model_ = 0;
    int sintering_model_    = 0;

    // -- Numerical regularisation floors ---------------------------------------
    double N_min_  = 1.;     //!< [#/m3]
    double fv_min_ = 1.e-15; //!< [-]
    double v_min_  = 0.;     //!< [m3] set in Precalculations from nTiO2_min_
    double S_min_  = 0.;     //!< [m2/m3]

    // -- Per-process source storage (owned by TiO2, not by base) --------------
    //
    // TiO2 models: nucleation, coagulation, condensation, sintering.
    // growth and oxidation are absent; base class returns zero span for both.

    MomentVector source_nucleation_   = MomentVector::Zero();
    MomentVector source_coagulation_  = MomentVector::Zero();
    MomentVector source_condensation_ = MomentVector::Zero();
    MomentVector source_sintering_    = MomentVector::Zero();

    // -- Initial moments cache --------------------------------------------------
    MomentVector initial_moments_cache_ = MomentVector::Zero();

    bool is_debug_mode_ = false; //!< enable verbose diagnostic output
};

} // namespace MOM

#if defined(MOM_USE_DICTIONARY)
#include "TiO2_Grammar.h"
#endif

#if !defined(MOM_COMPILED_LIBRARY)
#include "TiO2.tpp"
#else
namespace MOM
{
extern template class TiO2<BasicThermoData>;
}
#endif
