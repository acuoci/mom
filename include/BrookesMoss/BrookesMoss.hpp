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
 * @class BrookesMoss
 * @brief Two-equation soot model of Brookes & Moss (1999), with Hall et al. (2016) extension.
 *
 * A classical two-scalar soot model widely used in turbulent diffusion flames.
 * Transports soot mass fraction and a normalised radical nuclei concentration.
 * The BM-Hall extension (Hall et al. 2016) adds phenyl-based inception and an
 * improved oxidation correlation.
 *
 * @par References
 * - Brookes & Moss, *Combust. Flame* **116** (1999) 486–503.
 * - Hall, Smooke, Colket et al., *Combust. Flame* **169** (2016) 191–206.
 *
 * @par Transported variables
 * | Index | Symbol | Physical meaning |
 * |---|---|---|
 * | 0 | Ys | Soot mass fraction [-] |
 * | 1 | bs | Normalised radical nuclei concentration = Ns / (ρ·Ns_norm) [m³/kg] |
 *
 * @par Physical processes modelled
 * - **Nucleation** — Brookes-Moss (1999) or BM-Hall (2016) extended inception.
 * - **Surface growth** — C2H2 deposition with Arrhenius kinetics.
 * - **Oxidation** — O2 attack (Brookes-Moss) or O2+OH (BM-Hall).
 * - **Coagulation** — free-molecular regime.
 * - **Thermophoresis** — encoded in the effective diffusion coefficient.
 *
 * @par NOT modelled
 * - **Condensation**: no PAH surface adsorption term; `sources_condensation()`
 *   returns a zero span.
 * - **Sintering**: `sources_sintering()` returns a zero span.
 *
 * @par Thread safety
 * Not thread-safe — one instance per OpenMP thread.
 * See `MomentMethodBase` for the complete thread-safety contract.
 *
 * @tparam Thermo  Must satisfy the MOM::ThermoMap concept.
 */

template <ThermoMap Thermo> class BrookesMoss : public MomentMethodBase<BrookesMoss<Thermo>, 2>
{
    using Base = MomentMethodBase<BrookesMoss<Thermo>, 2>;

public:

    using typename Base::MomentVector;

    /// Labels accepted by MOM::MakeAnyMomentMethod for runtime variant selection.
    static constexpr std::array<std::string_view, 3> variant_labels{"BrookesMoss", "brookesmoss", "BM"};

    // -- Method-specific sub-model enums -------------------------------------

    /** @brief Nucleation kinetics variant. */
    enum class NucleationVariant : int
    {
        Off             = 0, //!< Nucleation disabled.
        BrookesMoss     = 1, //!< Original Brookes & Moss (1999) C2H2-based inception.
        BrookesMossHall = 2  //!< Hall et al. (2016) extended phenyl-based inception.
    };

    /** @brief Oxidation kinetics variant. */
    enum class OxidationVariant : int
    {
        Off             = 0, //!< Oxidation disabled.
        BrookesMoss     = 1, //!< Original Brookes & Moss (1999) O2 oxidation.
        BrookesMossHall = 2  //!< Hall et al. (2016) extended O2+OH oxidation.
    };

    // -- Configuration struct ------------------------------------------------

    /**
     * @struct Config
     * @brief Plain configuration parameters for the BrookesMoss variant.
     *
     * Nucleation and oxidation models are represented as integers:
     * 0 = off, 1 = BrookesMoss, 2 = BrookesMossHall.
     * @note No external dependencies: only standard C++ types.
     */
    struct Config : CommonConfig<0>,
                    BrookesMossProcessConfig,
                    GasConsumptionConfig<false>,
                    SootDensityConfig,
                    SootRadiationConfig
    {
        // ---- Gas species ---------------------------------------------------
        std::string precursors_species     = "C2H2"; //!< Nucleation precursor species
        std::string surface_growth_species = "C2H2"; //!< Surface growth species
        std::string benzene_species        = "C6H6";
        std::string phenylradical_species  = "C6H5";

        // ---- Soot/particle properties --------------------------------------
        double soot_particle_diameter_m = 1.e-9; //!< Mean particle diameter [m]
        double soot_particle_mw_kg_kmol = 144.;  //!< Particle MW            [kg/kmol]
        double ns_norm                  = 1.e15; //!< Ns normalisation value  [#/m3]

        // ---- Model kinetic constants ----------------------------------------
        double calpha   = 54.;      //!< Nucleation pre-exponential   [1/s]
        double talpha   = 21000.;   //!< Nucleation activation temp.  [K]
        double cbeta    = 1.0;      //!< Condensation coefficient     [-]
        double cgamma   = 11700.;   //!< Surface growth pre-exp.      [kg*m/kmol/s]
        double tgamma   = 12100.;   //!< Surface growth activation T  [K]
        double comega   = 105.8125; //!< Oxidation pre-exponential    [kg*m/kmol/s/sqrt(K)]
        double eta_coll = 0.04;     //!< Coagulation efficiency       [-]
        double coxid    = 0.015;    //!< Oxidation efficiency         [-]

        double nucleation_exponent = 1.; //!< Nucleation reaction order exponent [-]
        double sg_exponent1        = 1.; //!< Surface growth exponent 1          [-]
        double sg_exponent2        = 1.; //!< Surface growth exponent 2          [-]

        // ---- Model kinetic constants ----------------------------------------
        double calpha1_bmh   = 127.*std::pow(10.,8.88); //!< Nucleation pre-exponential   [kg*m3/kmol2/s]
        double talpha1_bmh   = 4378.;                  //!< Nucleation activation temp.  [K]
        double calpha2_bmh   = 178.*std::pow(10.,9.50);//!< Nucleation pre-exponential   [kg*m3/kmol2/s]
        double talpha2_bmh   = 6390.;                   //!< Nucleation activation temp.  [K]
        double comega2_bmh   = 8903.51;                 //!< Oxidation pre-exponential    [kg*m/kmol/s/sqrt(K)]
        double tomega2_bmh   = 19778;                   //!< Oxidation activation temp.   [K]
        
    };

    // -- Construction ---------------------------------------------------------

    explicit BrookesMoss(const Thermo& thermo);
    explicit BrookesMoss(const Thermo&&) = delete; ///< Prevents binding a temporary as thermo (dangling ref).

    BrookesMoss(const BrookesMoss&)            = delete;
    BrookesMoss& operator=(const BrookesMoss&) = delete;
    BrookesMoss(BrookesMoss&&)            = default; ///< Move-constructible for placement in std::variant.
    BrookesMoss& operator=(BrookesMoss&&) = delete;  ///< Not move-assignable — const Thermo& member cannot be reseated.

    /**
     * @brief Configure all BrookesMoss parameters from a plain configuration struct.
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
     * @brief Parse an OpenSMOKE++ dictionary into a BrookesMoss Config.
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

    /// Generic span setter. Order: [Ys, bs].
    /// Ys [-], bs [m3/kg].
    void SetMoments(std::span<const double> m) noexcept;

    /// Named setter (preferred for BrookesMoss-aware code).
    void SetMoments(double Ys, double bs) noexcept;

    // -- MomentMethod concept — core computation -------------------------------

    void CalculateSourceMoments() noexcept;
    void CalculateOmegaGas() noexcept;

    // -- MomentMethod concept — particle properties ----------------------------

    [[nodiscard]] double volume_fraction() const noexcept;
    [[nodiscard]] double particle_diameter() const noexcept;  //!< mean particle diameter [m]
    [[nodiscard]] double collision_diameter() const noexcept; //!< same as ParticleDiameter for BM
    [[nodiscard]] double particle_number_density() const noexcept; //!< [#/m3]
    [[nodiscard]] double mass_fraction() const noexcept;          //!< = Ys_
    [[nodiscard]] double specific_surface() const noexcept;       //!< [m2/m3]
    [[nodiscard]] double diffusion_coefficient() const noexcept;  //!< [kg/m/s]
    [[nodiscard]] double number_primary_particles() const noexcept;
    
    // -- MomentMethod concept — initial conditions -----------------------------

    [[nodiscard]] std::span<const double> initial_moments() const noexcept
    {
        return {initial_moments_cache_.data(), 2u};
    }

    // -- MomentMethod concept — precursor --------------------------------------

    [[nodiscard]] int precursor_index() const noexcept { return prec_index_; }

    [[nodiscard]] double precursor_concentration() const noexcept { return conc_prec_; }

    [[nodiscard]] const std::string& precursor_species() const noexcept { return prec_species_; }

    // -- MomentMethod concept — diagnostics ------------------------------------

    void PrintSummary() const;

    // -- Reporter output hook (MomentMethodReporter extensibility protocol) ------
    //
    // Makes BrookesMoss self-describing with respect to output.
    // MomentMethodReporter calls this with a lambda cb(label, value):
    //   • header mode — lambda uses label to register the column
    //   • row mode    — lambda uses value to write the data

    /// BrookesMoss-specific prefix columns: gas-phase source diagnostics.
    template <typename CB> void variant_prefix_output(CB&& cb) const
    {
        cb("omegaTot[kg/m3/s]", this->omega_gas_.sum());
        this->EmitOmegaGas(cb, "omegaPrec[kg/m3/s]", prec_index_);
        this->EmitOmegaGas(cb, "omegaSg[kg/m3/s]", sg_index_);
        this->EmitOmegaGas(cb, "omegaH2[kg/m3/s]", index_H2_);
        this->EmitOmegaGas(cb, "omegaC2H2[kg/m3/s]", index_C2H2_);
        this->EmitOmegaGas(cb, "omegaOH[kg/m3/s]", index_OH_);
        this->EmitOmegaGas(cb, "omegaO2[kg/m3/s]", index_O2_);

        if (nucleation_variant_ == NucleationVariant::BrookesMossHall)
        {
            this->EmitOmegaGas(cb, "omegaC6H5[kg/m3/s]", index_C6H5_);
            this->EmitOmegaGas(cb, "omegaC6H6[kg/m3/s]", index_C6H6_);
        }
        else
        {
            cb("omegaC6H5[kg/m3/s]", 0.);
            cb("omegaC6H6[kg/m3/s]", 0.);
        }
    }

    // -- Model switches --------------------------------------------------------

    void SetNucleation(int flag)
    {
        switch (flag)
        {
            case 0: SetNucleation("none"); break;
            case 1: SetNucleation("BrookesMoss"); break;
            case 2: SetNucleation("BrookesMossHall"); break;
            default:
                throw std::invalid_argument(
                    "[BrookesMoss] Invalid nucleation model flag. Allowed values: 0, 1, 2.");
        }
    }

    void SetNucleation(std::string_view label);

    void SetSurfaceGrowth(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument(
                "[BrookesMoss] Invalid surface-growth model flag. Allowed values: 0, 1.");
        surface_growth_model_ = flag;
    }

    void SetOxidation(int flag)
    {
        switch (flag)
        {
            case 0: SetOxidation("none"); break;
            case 1: SetOxidation("BrookesMoss"); break;
            case 2: SetOxidation("BrookesMossHall"); break;
            default:
                throw std::invalid_argument(
                    "[BrookesMoss] Invalid oxidation model flag. Allowed values: 0, 1, 2.");
        }
    }

    void SetOxidation(std::string_view label);

    void SetCoagulation(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument(
                "[BrookesMoss] Invalid coagulation model flag. Allowed values: 0, 1.");
        coagulation_model_ = flag;
    }

    void SetPrecursors(std::string_view name);
    void SetSurfaceGrowthSpecies(std::string_view name);
    void SetBenzeneSpecies(std::string_view name);
    void SetPhenylRadicalSpecies(std::string_view name);
    void SetGasClosureDummySpecies(std::string_view name);

    /// Sets normalisation factor for nuclei concentration [#/m3] (default: 1e15).
    void SetNsNorm(double value) noexcept { Ns_norm_ = value; }

    /// Sets fixed soot particle diameter used in the BM model [m].
    void SetDp(double dp) noexcept { dp_ = dp; }

    /// Sets soot particle molecular weight [kg/kmol] (used for diffusion).
    void SetMWp(double mwp) noexcept { mwp_ = mwp; }

    // -- BrookesMoss-Hall model constants --------------------------------------

    void SetCalpha1_BMH(double v) noexcept { Calpha1_BMH_ = v; }

    void SetCalpha2_BMH(double v) noexcept { Calpha2_BMH_ = v; }

    void SetTalpha1_BMH(double v) noexcept { Talpha1_BMH_ = v; }

    void SetTalpha2_BMH(double v) noexcept { Talpha2_BMH_ = v; }

    void SetComega2_BMH(double v) noexcept { Comega2_BMH_ = v; }

    void SetTomega2_BMH(double v) noexcept { Tomega2_BMH_ = v; }

    // -- Model state queries ----------------------------------------------------

    [[nodiscard]] int nucleation_model() const noexcept
    {
        return static_cast<int>(nucleation_variant_);
    }

    [[nodiscard]] int surface_growth_model() const noexcept { return surface_growth_model_; }

    [[nodiscard]] int oxidation_model() const noexcept
    {
        return static_cast<int>(oxidation_variant_);
    }

    [[nodiscard]] int coagulation_model() const noexcept { return coagulation_model_; }

    [[nodiscard]] const std::string& sg_species() const noexcept { return sg_species_; }

    /**
     * @name CRTP extension points — per-process source storage
     *
     * BrookesMoss models: nucleation, coagulation, growth, oxidation.
     * Condensation and sintering are **not** modelled; `sources_condensation()`
     * and `sources_sintering()` return zero spans automatically.
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
    void ApplyConfig(const Config& cfg); //!< core of SetupFromConfig, without PrintSummary()
    void NucleationSourceTerms();
    void NucleationSourceTerms_BM();
    void NucleationSourceTerms_BMH();
    void SurfaceGrowthSourceTerms();
    void OxidationSourceTerms();
    void OxidationSourceTerms_BM();
    void OxidationSourceTerms_BMH();
    void CoagulationSourceTerms();
    void CheckBrookesMossHallSpecies();

private:

    // -- Thermodynamics reference -----------------------------------------------
    const Thermo& thermo_;

    // -- Transported variables --------------------------------------------------
    double Ys_ = 0.; //!< soot mass fraction [-]
    double bs_ = 0.; //!< normalised nuclei concentration [m3/kg]

    // -- Precursor species ------------------------------------------------------
    std::string prec_species_;
    int prec_index_   = -1;
    double prec_nc_   = 0.; //!< C atoms per precursor molecule
    double prec_nh_   = 0.; //!< H atoms per precursor molecule
    double conc_prec_ = 0.; //!< precursor concentration [kmol/m3]

    // -- Surface growth species -------------------------------------------------
    std::string sg_species_;
    int sg_index_   = -1;
    double sg_nc_   = 0.;
    double sg_nh_   = 0.;
    double conc_sg_ = 0.;

    // -- Key species concentrations [kmol/m3] -----------------------------------
    double conc_H2_ = 0.;
    double conc_C2H2_ = 0.;
    double conc_C6H5_ = 0.;
    double conc_C6H6_ = 0.;
    double conc_OH_ = 0.;
    double conc_O2_ = 0.;    

    // 0-based species indices (−1 = not found in mechanism, safely ignored)
    int index_H_    = -1; //!< H radical  — BM-Hall nucleation ch.2 gas coupling
    int index_H2_   = -1; //!< H₂        — nucleation/surface-growth/oxidation (OH) coupling
    int index_CO_   = -1; //!< CO         — oxidation gas coupling (both variants)
    int index_C2H2_ = -1;
    std::string phenylradical_species_ = "C6H5";
    std::string benzene_species_       = "C6H6";
    int index_C6H5_ = -1;
    int index_C6H6_ = -1;
    int index_OH_   = -1;
    int index_O2_   = -1;

    // -- Particle properties ----------------------------------------------------
    double dp_  = 0.; //!< soot particle diameter [m]
    double mwp_ = 0.;   //!< soot particle molecular weight [kg/kmol]

    // -- Model constants --------------------------------------------------------
    double Calpha_  = 0.; //!< inception rate pre-exponential
    double Talpha_  = 0.; //!< inception activation temperature [K]
    double Cbeta_   = 0.; //!< coagulation rate constant
    double Cgamma_  = 0.; //!< surface growth pre-exponential
    double Tgamma_  = 0.; //!< surface growth activation temperature [K]
    double Comega_  = 0.; //!< oxidation pre-exponential
    double etaColl_ = 0.; //!< collision efficiency
    double Coxid_   = 0.; //!< oxidation rate constant
    double exp_l_   = 0.; //!< concentration exponent for nucleation rate
    double exp_m_   = 0.; //!< concentration exponent for surface growth rate
    double exp_n_   = 0.; //!< area exponent for surface growth rate

    // -- Debug flag -------------------------------------------------------------
    bool is_debug_mode_ = false; //!< enable verbose diagnostic output

    // -- BM-Hall extended constants ---------------------------------------------
    double Calpha1_BMH_ = 0.;
    double Calpha2_BMH_ = 0.;
    double Talpha1_BMH_ = 0.;
    double Talpha2_BMH_ = 0.;
    double Comega2_BMH_ = 0.;
    double Tomega2_BMH_ = 0.;

    // -- Model flags ------------------------------------------------------------
    NucleationVariant nucleation_variant_ = NucleationVariant::Off;
    OxidationVariant oxidation_variant_   = OxidationVariant::Off;
    int surface_growth_model_             = 0;
    int coagulation_model_                = 0;

    // -- Numerical parameters ---------------------------------------------------
    double Ys_min_  = 0.; //!< minimum Ys used in property calculations
    double bs_min_  = 0.;     //!< minimum bs used in property calculations
    double Ns_norm_ = 0.;  //!< normalisation factor for Ns [#/m3]

    // -- Gas consumption intermediate quantities --------------------------------
    double dMdt_nucleation_       = 0.; //!< total nucleation soot mass rate [kg/m3/s]
    double dMdt_nucleation_BMH_1_ = 0.; //!< BM-Hall channel 1 (C2H2-based) [kg/m3/s]
    double dMdt_nucleation_BMH_2_ = 0.; //!< BM-Hall channel 2 (C6H6-based) [kg/m3/s]
    double dMdt_surface_growth_   = 0.; //!< surface growth soot mass rate [kg/m3/s]
    double dMdt_oxidation_        = 0.; //!< total oxidation soot mass rate [kg/m3/s]
    double dMdt_oxidation_OH_     = 0.; //!< OH-channel contribution [kg/m3/s]
    double dMdt_oxidation_O2_     = 0.; //!< O2-channel contribution (BM-Hall only) [kg/m3/s]

    // -- Per-process source storage (owned by BrookesMoss, not by base) --------
    //
    // BrookesMoss models: nucleation, coagulation, growth, oxidation.
    // condensation and sintering are absent; base class returns zero span for both.

    MomentVector source_nucleation_  = MomentVector::Zero();
    MomentVector source_coagulation_ = MomentVector::Zero();
    MomentVector source_growth_      = MomentVector::Zero();
    MomentVector source_oxidation_   = MomentVector::Zero();

    Eigen::VectorXd omega_gas_oxidation_; //!< Oxidation-only gas-phase sources [kg/m³/s].

    // -- Initial moments cache --------------------------------------------------
    MomentVector initial_moments_cache_ = MomentVector::Zero();

    // -- Numerical floors -------------------------------------------------------
    static constexpr double kYsMin = 1.e-15; //!< [-]
    static constexpr double kBsMin = 0.;     //!< [m3/kg]
};

} // namespace MOM

#if defined(MOM_USE_DICTIONARY)
#include "BrookesMoss_Grammar.h"
#endif

#if !defined(MOM_COMPILED_LIBRARY)
#include "BrookesMoss.tpp"
#else
namespace MOM
{
extern template class BrookesMoss<BasicThermoData>;
}
#endif
