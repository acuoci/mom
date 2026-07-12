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
#include <vector>

#include "Eigen/Dense"

#include "MOM/MomentMethodBase.hpp"
#include "MOM/MOMConfig.hpp"
#include "MOM/ThermoProxy.hpp"

#if defined(MOM_USE_DICTIONARY)
#include <expected>
#endif

/**
 * @file MetalOxide.hpp
 * @brief Three-equation moment model for solid oxide nanoparticle dynamics.
 */

namespace MOM
{

/**
 * @class MetalOxide
 * @brief Configurable solid oxide nanoparticle source-term model.
 *
 * Models particle formation and evolution from a gas-phase precursor through
 * nucleation, condensation, coagulation, and sintering. The solid material,
 * precursor, and gas stoichiometry are supplied explicitly through `Config` or
 * through the dictionary interface.
 *
 * @par References
 * - Kruis, Kusters & Pratsinis, *Aerosol Sci. Technol.* **19** (1993) 514–526.
 * - Franzelli, Vié & Darabiha, *Proc. Combust. Inst.* **37** (2019) 5411–5419 (NDF reconstruction).
 *
 * @par Transported variables
 * | Index | Symbol | Physical meaning |
 * |---|---|---|
 * | 0 | Ysolid | Solid particle mass fraction [-] |
 * | 1 | NsolidN | Scaled number density, N / N0_scaling [-] |
 * | 2 | Ssolid | Total particle surface area per gas volume [m2/m3] |
 *
 * @par Physical processes modelled
 * Nucleation, condensation, coagulation, and sintering are modelled. Surface
 * growth and oxidation are not modelled and therefore return zero spans through
 * the common source-accessor API.
 *
 * @note When sintering is stiff relative to the flow time step, use
 *       `SinteringDeferredUpdate()` to integrate it separately via an ODE sub-step.
 *
 * @par Thread safety
 * Instances are mutable work objects. Use one model instance per thread.
 *
 * @tparam Thermo  Must satisfy the MOM::ThermoMap concept.
 */

template <ThermoMap Thermo> class MetalOxide : public MomentMethodBase<MetalOxide<Thermo>, 3>
{
    using Base = MomentMethodBase<MetalOxide<Thermo>, 3>;

public:

    using typename Base::MomentVector;

    /** @brief Labels accepted by `MOM::MakeAnyMomentMethod()`. */
    static constexpr std::array<std::string_view, 2> variant_labels{"MetalOxide", "metaloxide"};

    // -- Method-specific sub-model enums -------------------------------------

    /** @brief Nucleation mechanism selector for the solid oxide model. */
    enum class NucleationVariant : int
    {
        Off          = 0, //!< Nucleation disabled.
        Binary       = 1, //!< Precursor + precursor free-molecular collision.
        FixedCluster = 2  //!< Nucleation via a fixed-size cluster of n0 formula units.
    };

    /**
     * @struct NDFReconstructionData
     * @brief Parameters of the Pareto + log-normal NDF reconstruction.
     *
     * Returned by `ReconstructedNDFData()` for reporting and NDF file output.
     */
    struct NDFReconstructionData
    {
        bool valid;     //!< True if reconstruction is physically meaningful.
        double N;       //!< Particle number density [#/m3].
        double fv;      //!< Volume fraction [-].
        double nuMean;  //!< Mean particle volume [m3/#].
        double nuNucl;  //!< Volume of a newly nucleated particle [m3/#].
        double alpha;   //!< Pareto weight [0,1] [-].
        double nbar0;   //!< Nucleation-peak normalized NDF value [1/m3].
        double sigma;   //!< Log-normal standard deviation [-].
        double k;       //!< Pareto tail index k [-].
        double nu1mean; //!< Mean volume of the Pareto contribution [m3/#].
        double nu2mean; //!< Mean volume of the log-normal contribution [m3/#].
        double mu;      //!< Log-normal location parameter [log(m3)].
    };

    // -- Configuration struct ------------------------------------------------

    /**
     * @struct GasStoichiometryTerm
     * @brief Gas-phase stoichiometric coefficient per precursor molecule.
     *
     * Coefficients are positive for products and negative for reactants. The
     * solid product is represented by `solid_formula_units_per_precursor`.
     */
    struct GasStoichiometryTerm
    {
        std::string species; //!< Gas species name.
        double coefficient = 0.; //!< Stoichiometric coefficient [kmol/kmol precursor]
    };

    /**
     * @struct Config
     * @brief Plain configuration parameters for the solid oxide variant.
     *
     * The `nucleation_model` field accepts `"none"`, `"binary"`, or
     * `"fixed-cluster"`.
     *
     * @note No external dependencies: only standard C++ types.
     */
    struct Config : CommonConfig<1>, GasConsumptionConfig<false>
    {
        // ---- Activation / precursor ----------------------------------------
        std::string precursor_species = "none"; //!< Gas precursor species.

        // ---- Solid material -------------------------------------------------
        std::string solid_name                       = "TiO2";   //!< Solid product label.
        double      solid_molecular_weight_kg_kmol   = 79.866;   //!< Solid formula-unit molecular weight [kg/kmol]
        double      solid_density_kg_m3              = 4230.;    //!< Solid density [kg/m3]
        double      solid_formula_units_per_precursor = 1.;      //!< Solid formula units per precursor molecule.

        // ---- Gas consumption / closure -------------------------------------
        std::vector<GasStoichiometryTerm> gas_stoichiometry; //!< Explicit gas reaction terms.
        double gas_stoichiometry_mass_tolerance = 1.e-3; //!< Relative mass-balance tolerance.

        // ---- Process model selection ---------------------------------------
        /// Nucleation model: "none" | "binary" | "fixed-cluster".
        std::string nucleation_model = "binary";
        int sintering_model          = 1; //!< Sintering model index
        int coagulation_model        = 1; //!< Coagulation model index
        int condensation_model       = 1; //!< Condensation model index

        // ---- Particle cluster sizes ----------------------------------------
        int minimum_formula_units            = 2; //!< Minimum solid formula units per aggregate.
        int nucleated_particle_formula_units = 5; //!< Solid formula units per nucleated particle.

        // ---- Sintering kinetics: tau_s = As * T^ns * d_p^4 * exp(Ts/T) ---
        double sintering_As_s_K_m = 7.44e16; //!< Pre-exponential [s,K,m]
        double sintering_Ts_K     = 31000.;  //!< Activation temperature [K] (positive; used as exp(Ts/T))
        double sintering_ns       =  1.0;    //!< Temperature/size exponent [-]

        // ---- Sintering numerical regularisation ----------------------------
        bool   sintering_deferred    = false;  //!< Defer sintering to operator-split step.
        double sintering_dp_min_m    = 2.e-9;  //!< Diameter below which sintering inactive [m]
        double sintering_tau_min_s   = 1.e-10; //!< Minimum sintering time-scale [s]
        double sintering_k_max_per_s = 1.e6;   //!< Maximum sintering rate [1/s]

        // ---- Numerical floors ----------------------------------------------
        double ns_minimum_per_m3 = 1.e3;   //!< Minimum number density floor [#/m3]
        double fv_minimum        = 1.e-16; //!< Minimum volume fraction floor [-]

    };

    // -- Construction ---------------------------------------------------------

    explicit MetalOxide(const Thermo& thermo);
    explicit MetalOxide(const Thermo&&) = delete; ///< Prevents binding a temporary as thermo (dangling ref).

    MetalOxide(const MetalOxide&)            = delete;
    MetalOxide& operator=(const MetalOxide&) = delete;
    MetalOxide(MetalOxide&&)            = default; ///< Move-constructible for placement in std::variant.
    MetalOxide& operator=(MetalOxide&&) = delete;  ///< Not move-assignable: const Thermo& member cannot be reseated.

    /**
     * @brief Configures the model from a plain configuration struct.
     * @param cfg Configuration values. A default-constructed `Config` applies
     *            the model defaults.
     */
    void SetupFromConfig(const Config& cfg);

#if defined(MOM_USE_DICTIONARY)
    /**
     * @brief Parses an OpenSMOKE++ dictionary into a MetalOxide configuration.
     * @tparam DictType OpenSMOKE++ dictionary type.
     * @param dict Input dictionary.
     * @return Parsed configuration, or an error string for invalid keyword values.
     */
    template <typename DictType>
    [[nodiscard]] static std::expected<Config, std::string> ParseConfig(DictType& dict);
#endif

    // -- MomentMethod concept: state injection ---------------------------------

    /**
     * @brief Sets the gas thermodynamic state for the current cell.
     * @param T Gas temperature [K].
     * @param P_Pa Gas pressure [Pa].
     * @param Y Species mass fractions, size equal to `thermo.NumberOfSpecies()`.
     */
    void SetState(double T, double P_Pa, const double* Y) noexcept;

    /**
     * @brief Sets the transported moment variables from a span.
     * @param m Span of size 3 ordered as `[Ysolid, NsolidN, Ssolid]`.
     */
    void SetMoments(std::span<const double> m) noexcept;

    /**
     * @brief Sets the transported moment variables by name.
     * @param solid_mass_fraction Solid particle mass fraction [-].
     * @param scaled_number_density Number density scaled by `N0_scaling_` [-].
     * @param surface_area_concentration Total particle surface area [m2/m3].
     */
    void SetMoments(double solid_mass_fraction,
                    double scaled_number_density,
                    double surface_area_concentration) noexcept;

    // -- MomentMethod concept: core computation --------------------------------

    /** @brief Computes all active moment source terms for the current state. */
    void ComputeSources() noexcept;

    /** @brief Computes gas-phase source terms from the current process rates. */
    void CalculateOmegaGas() noexcept;

    // -- Deferred sintering (operator-splitting compatible) --------------------

    /**
     * @brief Applies the analytical deferred-sintering update to surface area.
     * @param dt_ode Sub-step duration [s].
     * @return Updated surface-area concentration [m2/m3].
     */
    [[nodiscard]] double SinteringDeferredUpdate(double dt_ode);

    // -- MomentMethod concept: particle properties -----------------------------

    [[nodiscard]] double volume_fraction() const noexcept;
    [[nodiscard]] double particle_diameter() const noexcept;  //!< Primary particle diameter [m].
    [[nodiscard]] double collision_diameter() const noexcept; //!< Aggregate collision diameter [m].
    [[nodiscard]] double AggregateDiameter() const noexcept; //!< Mobility aggregate diameter [m].
    [[nodiscard]] double particle_number_density() const noexcept; //!< Particle number density [#/m3].
    [[nodiscard]] double mass_fraction() const noexcept; //!< Solid mass fraction [-].
    [[nodiscard]] double specific_surface_area() const noexcept; //!< Total surface area [m2/m3].
    [[nodiscard]] double number_primary_particles() const noexcept; //!< Mean primary particles per aggregate [-].
    [[nodiscard]] double diffusion_coefficient() const noexcept; //!< Effective diffusion coefficient [kg/m/s].

    // -- MomentMethod concept: initial conditions ------------------------------

    /** @brief Returns initial values for `[Ysolid, NsolidN, Ssolid]`. */
    [[nodiscard]] std::span<const double> initial_moments() const noexcept
    {
        return {initial_moments_cache_.data(), 3u};
    }

    // -- MomentMethod concept: precursor ---------------------------------------

    /** @brief Returns the 0-based precursor species index, or -1 if unset. */
    [[nodiscard]] int precursor_index() const noexcept { return precursor_index_; }

    /** @brief Returns the precursor molar concentration [kmol/m3]. */
    [[nodiscard]] double precursor_concentration() const noexcept { return c_precursor_; }

    /** @brief Returns the configured precursor species name. */
    [[nodiscard]] const std::string& precursor_species() const noexcept
    {
        return precursor_species_;
    }

    // -- MomentMethod concept: diagnostics -------------------------------------

    void PrintSummary() const;

    // -- Aggregated properties helper ------------------------------------------

    /**
     * @brief Computes regularized aggregate properties from the current moments.
     *
     * @param fv Particle volume fraction [-].
     * @param dp Primary particle diameter [m].
     * @param dc Collision diameter [m].
     * @param da Sphere-equivalent aggregate diameter [m].
     * @param np Mean number of primary particles per aggregate [-].
     * @param ss Surface area per particle [m2].
     * @param vs Volume per particle [m3].
     * @param ssph Spherical surface area for volume @p vs [m2].
     * @param tauS Sintering time scale [s].
     */
    void Properties(double& fv,
                    double& dp,
                    double& dc,
                    double& da,
                    double& np,
                    double& ss,
                    double& vs,
                    double& ssph,
                    double& tauS) const noexcept;

    /**
     * @brief Emits MetalOxide-specific reporter columns.
     *
     * The callback is called as `cb(label, value)` for aggregate properties,
     * NDF reconstruction parameters, and gas-source diagnostics.
     */
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
        this->EmitOmegaGas(cb, "omegaPrecursor[kg/m3/s]", precursor_index_);
        for (const auto& term : gas_stoichiometry_)
        {
            if (term.index == precursor_index_)
                continue;
            const std::string label = "omegaGas(" + term.species + ")[kg/m3/s]";
            this->EmitOmegaGas(cb, std::string_view{label}, term.index);
        }
    }

    // -- NDF reconstruction ----------------------------------------------------

    /** @brief Returns parameters of the current reconstructed NDF. */
    [[nodiscard]] NDFReconstructionData ReconstructedNDFData(bool use_regularized_moments = false) const;

    /**
     * @brief Returns the normalized NDF value at particle volume @p nu.
     * @param nu Particle volume [m3].
     * @param use_regularized_moments If true, applies numerical floors before reconstruction.
     * @return Normalized NDF [1/m3].
     */
    [[nodiscard]] double ReconstructedNormalizedNDF(double nu,
                                                    bool use_regularized_moments = false) const;

    /**
     * @brief Returns the dimensional NDF value at particle volume @p nu.
     * @param nu Particle volume [m3].
     * @param use_regularized_moments If true, applies numerical floors before reconstruction.
     * @return Dimensional NDF [#/m3 gas / m3 particle].
     */
    [[nodiscard]] double ReconstructedNDF(double nu, bool use_regularized_moments = false) const;

    /**
     * @brief Fills a vector with dimensional NDF values at the requested volumes.
     * @param nu Particle-volume grid [m3].
     * @param n Output dimensional NDF values.
     * @param use_regularized_moments If true, applies numerical floors before reconstruction.
     */
    void ReconstructedNDF(const Eigen::VectorXd& nu,
                          Eigen::VectorXd& n,
                          bool use_regularized_moments = false) const;

    // -- Material / geometry accessors -----------------------------------------

    /** @brief Returns the solid material label. */
    [[nodiscard]] const std::string& solid_name() const noexcept { return solid_name_; }

    /** @brief Returns the solid material density [kg/m3]. */
    [[nodiscard]] double solid_density() const noexcept { return solid_density_kg_m3_; }

    /** @brief Returns the solid formula-unit molecular weight [kg/kmol]. */
    [[nodiscard]] double solid_molecular_weight() const noexcept
    {
        return solid_molecular_weight_kg_kmol_;
    }

    /** @brief Returns solid formula units formed per precursor molecule. */
    [[nodiscard]] double solid_formula_units_per_precursor() const noexcept
    {
        return solid_formula_units_per_precursor_;
    }

    /** @brief Returns the volume of a nucleated particle [m3]. */
    [[nodiscard]] double NucleationParticleVolume() const noexcept;

    /** @brief Returns the minimum-particle surface area [m2]. */
    [[nodiscard]] double s0() const noexcept { return s0_; }

    /** @brief Returns the minimum-particle volume [m3]. */
    [[nodiscard]] double v0() const noexcept { return v0_; }

    /** @brief Returns the number-density scaling factor [#/m3]. */
    [[nodiscard]] double ScalingFactorNs() const noexcept { return N0_scaling_; }

    // -- Model switches --------------------------------------------------------

    /** @brief Sets the nucleation model by integer flag: 0=off, 1=binary, 2=fixed-cluster. */
    void SetNucleation(int flag)
    {
        if (flag != 0 && flag != 1 && flag != 2)
            throw std::invalid_argument(
                "[MetalOxide] Invalid nucleation model flag. Allowed values: 0, 1, 2.");
        nucleation_variant_ = static_cast<NucleationVariant>(flag);
    }

    /** @brief Sets the coagulation model by integer flag: 0=off, 1=on. */
    void SetCoagulation(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument(
                "[MetalOxide] Invalid coagulation model flag. Allowed values: 0, 1.");
        coagulation_model_ = flag;
    }

    /** @brief Sets the condensation model by integer flag: 0=off, 1=on. */
    void SetCondensation(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument(
                "[MetalOxide] Invalid condensation model flag. Allowed values: 0, 1.");
        condensation_model_ = flag;
    }

    /** @brief Sets the sintering model by integer flag: 0=off, 1=on. */
    void SetSintering(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument(
                "[MetalOxide] Invalid sintering model flag. Allowed values: 0, 1.");
        sintering_model_ = flag;
    }

    /** @brief Sets and validates solid material properties. */
    void SetSolidMaterial(std::string_view name, double molecular_weight_kg_kmol, double density_kg_m3);

    /** @brief Sets solid formula units formed per precursor molecule. */
    void SetSolidFormulaUnitsPerPrecursor(double n);

    /** @brief Sets formula units in a newly nucleated particle. */
    void SetNumberOfFormulaUnitsPerNucleatedParticle(unsigned n);

    /** @brief Sets the minimum formula-unit count used by regularized geometry. */
    void SetMinimumNumberOfFormulaUnits(unsigned n);

    void SetNucleationCollisionEnhancementFactor(double eps) noexcept { epsilon_nuc_ = eps; }

    void SetCoagulationCollisionEnhancementFactor(double eps) noexcept { epsilon_coag_ = eps; }

    void SetCondensationCollisionEnhancementFactor(double eps) noexcept { epsilon_cond_ = eps; }

    /// Sintering kinetics: tau_s = As * T^ns * dp^4 * exp(Ts/T) [s].
    void SetSinteringFrequencyFactor(double As) noexcept { As_ = As; }

    void SetSinteringActivationTemperature(double Ts) noexcept { Ts_ = Ts; }

    void SetSinteringTemperatureExponent(double ns) noexcept { ns_ = ns; }

    void SetNMinimum(double v) noexcept { N_min_ = v; }

    void SetFvMinimum(double v) noexcept { fv_min_ = v; }

    /** @brief Sets and validates the gas precursor species. */
    void SetPrecursor(std::string_view name);
    /** @brief Sets the optional gas-closure dummy species. */
    void SetGasClosureDummySpecies(std::string_view name);

    /** @brief Configures explicit gas-phase stoichiometry per precursor molecule. */
    void SetGasStoichiometry(std::span<const GasStoichiometryTerm> terms,
                             double relative_mass_tolerance = 1.e-3);

    // -- Model state queries ----------------------------------------------------

    [[nodiscard]] NucleationModel nucleation_model() const noexcept
    {
        return static_cast<NucleationModel>(static_cast<int>(nucleation_variant_));
    }

    [[nodiscard]] CoagulationModel  coagulation_model()  const noexcept { return static_cast<CoagulationModel>(coagulation_model_);   }

    [[nodiscard]] CondensationModel condensation_model() const noexcept { return static_cast<CondensationModel>(condensation_model_); }

    [[nodiscard]] SinteringModel    sintering_model()    const noexcept { return static_cast<SinteringModel>(sintering_model_);       }

    [[nodiscard]] bool is_sintering_deferred() const noexcept { return is_sintering_deferred_; }

    /**
     * @name Per-process source storage accessors
     *
     * Solid oxide models: nucleation, coagulation, condensation, sintering.
     * Growth and oxidation use the base-class zero fallback.
     * @{
     */

    /** @brief Nucleation source terms, size = n_equations. */
    [[nodiscard, gnu::always_inline]] std::span<const double> sources_nucleation_impl() const noexcept
    {
        return {source_nucleation_.data(), this->n_equations};
    }

    /** @brief Coagulation source terms, size = n_equations. */
    [[nodiscard, gnu::always_inline]] std::span<const double> sources_coagulation_impl() const noexcept
    {
        return {source_coagulation_.data(), this->n_equations};
    }

    /** @brief Condensation source terms, size = n_equations. */
    [[nodiscard, gnu::always_inline]] std::span<const double> sources_condensation_impl() const noexcept
    {
        return {source_condensation_.data(), this->n_equations};
    }

    /** @brief Sintering source terms, size = n_equations. */
    [[nodiscard, gnu::always_inline]] std::span<const double> sources_sintering_impl() const noexcept
    {
        return {source_sintering_.data(), this->n_equations};
    }

    /** @} */

private:

    // -- Private computational methods ------------------------------------------

    void MemoryAllocation();
    void Precalculations();
    void ApplyConfig(const Config& cfg);
    void NucleationSourceTerms();
    void NucleationSourceTerms_Binary();
    void NucleationSourceTerms_FixedCluster();
    void CoagulationSourceTerms();
    void CondensationSourceTerms();
    void SinteringSourceTerms();
    void CalculateOmegaGas_internal() noexcept;
    void ClearGasStoichiometry() noexcept;
    void AddGasStoichiometryTerm(std::string_view species, double coefficient);
    void ValidateGasStoichiometryMassBalance() const;

private:

    // -- Thermodynamics reference -----------------------------------------------
    const Thermo& thermo_;

    // -- Transported variables --------------------------------------------------
    double solid_mass_fraction_          = 0.; //!< solid particle mass fraction [-]
    double scaled_number_density_        = 0.; //!< N / N0_scaling [-]
    double surface_area_concentration_   = 0.; //!< surface area concentration [m2/m3]
    double N0_scaling_ = 1.e15; //!< [#/m3]

    // -- Material properties ----------------------------------------------------
    static constexpr double default_solid_molecular_weight_kg_kmol_ = 79.866;
    static constexpr double default_solid_density_kg_m3_            = 4230.;

    std::string solid_name_ = "TiO2";
    double solid_molecular_weight_kg_kmol_ = default_solid_molecular_weight_kg_kmol_;
    double solid_density_kg_m3_            = default_solid_density_kg_m3_;
    double solid_formula_unit_mass_kg_ =
        default_solid_molecular_weight_kg_kmol_ / (6.02214076e26);
    double solid_formula_unit_volume_m3_ =
        solid_formula_unit_mass_kg_ / default_solid_density_kg_m3_;
    double solid_formula_units_per_precursor_ = 1.;

    // -- Monomer/nucleus geometry -----------------------------------------------
    double v0_ = 0.; //!< monomer volume [m3]
    double s0_ = 0.; //!< monomer surface [m2]
    double d0_ = 0.; //!< monomer diameter [m]

    // -- Precursor properties ---------------------------------------------------
    std::string precursor_species_ = "none";
    int precursor_index_  = -1;
    double Y_precursor_   = 0.; //!< mass fraction
    double c_precursor_   = 0.; //!< molar concentration [kmol/m3]
    double W_precursor_   = 0.; //!< molecular weight [kg/kmol]
    double m_precursor_   = 0.; //!< molecular mass [kg]

    // Collision geometry of the gas precursor molecule
    double v_precursor_ = 0.; //!< pseudo-molecular volume [m3]
    double d_precursor_ = 0.; //!< collision diameter [m]

    // Effective precursor geometry used by nucleation/condensation kernels
    double vprec_ = 0.; //!< solid volume added per precursor molecule [m3]
    double dprec_ = 0.; //!< collision diameter for beta_nuc / beta_cond [m]

    // -- Gas consumption stoichiometry ------------------------------------------
    struct RuntimeGasStoichiometryTerm
    {
        int index = -1;
        double coefficient = 0.;
        double molecular_weight_kg_kmol = 0.;
        std::string species;
    };

    std::vector<RuntimeGasStoichiometryTerm> gas_stoichiometry_;
    double gas_stoichiometry_mass_tolerance_ = 1.e-3;

    // -- Nucleation parameters --------------------------------------------------
    NucleationVariant nucleation_variant_ = NucleationVariant::Off;
    unsigned int n_formula_units_min_     = 1;   //!< minimum reference size for regularization
    unsigned int n0_                      = 5;   //!< solid formula units per newly nucleated particle
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
    double v_min_  = 0.;     //!< [m3] set in Precalculations from n_formula_units_min_
    double S_min_  = 0.;     //!< [m2/m3]

    // -- Per-process source storage --------------------------------------------

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
#include "MetalOxide_Grammar.h"
#endif

#if !defined(MOM_COMPILED_LIBRARY)
#include "MetalOxide.tpp"
#else
namespace MOM
{
extern template class MetalOxide<BasicThermoData>;
}
#endif
