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
|   Copyright (C) 2026 Alberto Cuoci.                                     |
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

#include <array>
#include <cmath>
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

/**
 * @file HMOM6.hpp
 * @brief Hybrid Method of Moments soot model — 6+1 bivariate MOMIC closure.
 */

namespace MOM
{

/**
 * @class HMOM6
 * @brief Seven-equation Hybrid Method of Moments soot source-term model.
 *
 * HMOM6 extends the original 4-equation HMOM (Mueller et al., 2009) from a
 * two-delta DQMOM closure to a second-order MOMIC polynomial closure for the
 * large mode.  It transports six bivariate moments
 * {M_{0,0}, M_{1,0}, M_{0,1}, M_{2,0}, M_{1,1}, M_{0,2}} plus the small-mode
 * number density N_0, following the 6+1 formulation of Mueller et al. (2009).
 *
 * @par MOMIC closure (R=2)
 * The large-mode NDF is represented by the exponential polynomial:
 * @code
 *   ln N_L(V,S) ≈ Σ_{r=0}^{2} Σ_{k=0}^{r} a_{r,k} V^k S^{r-k}
 *                = a00 + a10 S + a11 V + a20 S² + a21 VS + a22 V²
 * @endcode
 * The six coefficients {a00, a10, a11, a20, a21, a22} are solved analytically
 * from the six large-mode moments {L_{0,0}, ..., L_{0,2}} at each time step.
 *
 * @par References
 * - Mueller, Pitsch & Raman, *Proc. Combust. Inst.* **32** (2009) 785-792.
 *
 * @par Transported variables (normalised, [mol/m3])
 * | Index | Symbol | Physical meaning |
 * |---|---|---|
 * | 0 | M00 | Zeroth-order moment (proportional to total number density) |
 * | 1 | M10 | First-order volume moment |
 * | 2 | M01 | First-order surface moment |
 * | 3 | M20 | Second-order volume moment |
 * | 4 | M11 | Cross volume-surface moment |
 * | 5 | M02 | Second-order surface moment |
 * | 6 | N0  | Small-particle (nucleation-mode) number density |
 *
 * @par Normalisation convention
 * M_{x,y,norm} = M_{x,y} / (V0^x × S0^y × Nav) [mol/m3]
 * so that a normalized value of 1/Nav corresponds to exactly one particle.
 *
 * @par Physical processes modelled
 * Identical to HMOM: nucleation, HACA surface growth, PAH condensation,
 * O2/OH oxidation, and coagulation (discrete + continuum correction).
 *
 * @par Thread safety
 * Instances are mutable work objects. Use one model instance per thread.
 *
 * @tparam Thermo  Must satisfy the MOM::ThermoMap concept.
 */
template <ThermoMap Thermo> class HMOM6 : public MomentMethodBase<HMOM6<Thermo>, 7>
{
    using Base = MomentMethodBase<HMOM6<Thermo>, 7>;

public:

    using typename Base::MomentVector;

    /** @brief Labels accepted by `MOM::MakeAnyMomentMethod()`. */
    static constexpr std::array<std::string_view, 2> variant_labels{"HMOM6", "hmom6"};

    // -- Method-specific sub-model enums --------------------------------------

    /** @brief Sticking coefficient model for PAH-soot collisions. */
    enum class StickingModel : int
    {
        Constant = 0, //!< Fixed sticking coefficient.
        PAH4     = 1  //!< PAH 4-ring collision cross-section sticking model.
    };

    /** @brief Model for the primary particle diameter of fractal aggregates. */
    enum class FractalDiameterModel : int
    {
        Model0 = 0, //!< Mueller et al. (2009) default.
        Model1 = 1  //!< Attili et al. (2014) extended model.
    };

    /** @brief Model for the aggregate collision diameter. */
    enum class CollisionDiameterModel : int
    {
        Model1 = 1, //!< Collision diameter proportional to N^{1/3} d_p (Mueller 2009).
        Model2 = 2  //!< Collision diameter from fractal geometry (Attili 2014).
    };

    // -- Configuration struct -------------------------------------------------

    /**
     * @struct Config
     * @brief Plain configuration parameters for the HMOM6 variant.
     *
     * Identical layout to HMOM::Config — the same process parameters apply;
     * only the number of transported moments changes.
     *
     * @note No external dependencies: only standard C++ types.
     */
    struct Config : CommonConfig<ThermophoreticModel::Standard>,
                    PAHConfig,
                    BinarySootProcessConfig,
                    GasConsumptionConfig<true>,
                    SootRadiationConfig,
                    StickingConfig
    {
        // ---- Geometry models -------------------------------------------------
        int fractal_diameter_model   = 1; //!< Fractal diameter model index   [1 = default]
        int collision_diameter_model = 2; //!< Collision diameter model index [2 = default]

        // ---- Surface density -------------------------------------------------
        double surface_density_per_m2     = 1.7e19;  //!< Active surface site density [#/m2]
        bool   surface_density_correction = false;    //!< Temperature-dependent correction.
        double surf_dens_a1 = 12.65;    //!< Correction coefficient A1 [-]
        double surf_dens_a2 = -0.00563; //!< Correction coefficient A2 [1/K]
        double surf_dens_b1 = -1.38;    //!< Correction coefficient B1 [-]
        double surf_dens_b2 =  0.00069; //!< Correction coefficient B2 [1/K]

        // ---- Additional process model selection ------------------------------
        int continuous_coagulation_model = 1; //!< Coagulation (continuum) model

        // ---- HACA kinetics (A in cm3/mol/s, E in kJ/mol) -------------------
        double A1f = 6.72e1;  double n1f =  3.33; double E1f =   6.09;
        double A1b = 6.44e-1; double n1b =  3.79; double E1b =  27.96;
        double A2f = 1.00e8;  double n2f =  1.80; double E2f =  68.42;
        double A2b = 8.68e4;  double n2b =  2.36; double E2b =  25.46;
        double A3f = 1.13e16; double n3f = -0.06; double E3f = 476.05;
        double A3b = 4.17e13; double n3b =  0.15; double E3b =   0.00;
        double A4  = 2.52e9;  double n4  =  1.10; double E4  =  17.13;
        double A5  = 2.20e12; double n5  =  0.00; double E5  =  31.38;
        double efficiency6 = 0.13; //!< R6 third-body efficiency [-]
    };

    // -- Construction ---------------------------------------------------------

    /**
     * @brief Constructs HMOM6 bound to the given thermodynamics map.
     *
     * @param thermo  Const reference to the thermodynamics map (must outlive this object).
     */
    explicit HMOM6(const Thermo& thermo);
    explicit HMOM6(const Thermo&&) = delete; ///< Prevents binding a temporary as thermo.

    HMOM6(const HMOM6&)            = delete;
    HMOM6& operator=(const HMOM6&) = delete;
    HMOM6(HMOM6&&)            = default;
    HMOM6& operator=(HMOM6&&) = delete;

    /**
     * @brief Configures the model from a plain configuration struct.
     */
    void SetupFromConfig(const Config& cfg);

#if defined(MOM_USE_DICTIONARY)
    /**
     * @brief Parse an OpenSMOKE++ dictionary into an HMOM6 Config.
     */
    template <typename DictType>
    [[nodiscard]] static std::expected<Config, std::string> ParseConfig(DictType& dict);
#endif

    // -- MomentMethod concept: state injection --------------------------------

    /**
     * @brief Inject the thermodynamic state for the current computational cell.
     *
     * @param T    Gas temperature [K].
     * @param P_Pa Gas pressure [Pa].
     * @param Y    Species mass fractions (pointer, size = `thermo.NumberOfSpecies()`).
     */
    void SetState(double T, double P_Pa, const double* Y) noexcept;

    /**
     * @param m  Span of size 7:
     *           `[M00_norm, M10_norm, M01_norm, M20_norm, M11_norm, M02_norm, N0_norm]`
     *           [mol/m3].
     */
    void SetMoments(std::span<const double> m) noexcept;

    /**
     * @brief Sets all 7 normalized moments by value.
     */
    void SetNormalizedMoments(double M00_norm, double M10_norm, double M01_norm,
                              double M20_norm, double M11_norm, double M02_norm,
                              double N0_norm) noexcept;

    // -- MomentMethod concept: core computation -------------------------------

    /** @brief Compute all moment source terms for the current cell state. */
    void ComputeSources() noexcept;

    /** @brief Compute only the gas-phase consumption terms (`omega_gas_`). */
    void CalculateOmegaGas() noexcept;

    // -- MomentMethod concept: particle properties ----------------------------

    /** @brief Soot volume fraction [-]. */
    [[nodiscard]] double volume_fraction() const noexcept;

    /** @brief Mean primary particle diameter dp [m]. */
    [[nodiscard]] double particle_diameter() const noexcept;

    /** @brief Aggregate collision diameter dc [m] (model-dependent). */
    [[nodiscard]] double collision_diameter() const noexcept;

    /** @brief Total soot number density N = N0 + NL [#/m3]. */
    [[nodiscard]] double particle_number_density() const noexcept;

    /** @brief Soot mass fraction [-]. */
    [[nodiscard]] double mass_fraction() const noexcept;

    /** @brief Soot specific surface area Ss = M01 [m2/m3]. */
    [[nodiscard]] double specific_surface_area() const noexcept;

    /** @brief Mean number of primary particles per aggregate np [-]. */
    [[nodiscard]] double number_primary_particles() const noexcept;

    /** @brief Effective particle diffusion coefficient D_p [kg/m/s]. */
    [[nodiscard]] double diffusion_coefficient() const noexcept;

    // -- MomentMethod concept: initial conditions -----------------------------

    /**
     * @brief Returns near-zero initial moment values for solver initialisation.
     *
     * @return Span of size 7: all seven normalized moments [mol/m3].
     */
    [[nodiscard]] std::span<const double> initial_moments() const noexcept
    {
        return {initial_moments_cache_.data(), 7u};
    }

    // -- MomentMethod concept: precursor --------------------------------------

    /** @brief 0-based index of the PAH precursor species, or -1 if unset. */
    [[nodiscard]] int precursor_index() const noexcept { return pah_index_; }

    /** @brief Molar concentration of the PAH precursor [kmol/m3]. */
    [[nodiscard]] double precursor_concentration() const noexcept { return conc_PAH_; }

    /** @brief Name of the PAH precursor species (e.g. "C16H10"). */
    [[nodiscard]] const std::string& precursor_species() const noexcept { return pah_species_; }

    // -- MomentMethod concept: diagnostics ------------------------------------

    /** @brief Print a human-readable summary of the HMOM6 configuration to stdout. */
    void PrintSummary() const;

    // -- Per-process source accessors -----------------------------------------

    /** @brief Nucleation source terms [mol/m3/s], size = n_equations. */
    [[nodiscard, gnu::always_inline]] std::span<const double> sources_nucleation_impl() const noexcept
    {
        return {source_nucleation_.data(), this->n_equations};
    }

    /** @brief Total coagulation source terms [mol/m3/s], size = n_equations. */
    [[nodiscard, gnu::always_inline]] std::span<const double> sources_coagulation_impl() const noexcept
    {
        return {source_coagulation_.data(), this->n_equations};
    }

    /** @brief Condensation source terms [mol/m3/s], size = n_equations. */
    [[nodiscard, gnu::always_inline]] std::span<const double> sources_condensation_impl() const noexcept
    {
        return {source_condensation_.data(), this->n_equations};
    }

    /** @brief Surface growth source terms [mol/m3/s], size = n_equations. */
    [[nodiscard, gnu::always_inline]] std::span<const double> sources_growth_impl() const noexcept
    {
        return {source_growth_.data(), this->n_equations};
    }

    /** @brief Oxidation source terms [mol/m3/s], size = n_equations. */
    [[nodiscard, gnu::always_inline]] std::span<const double> sources_oxidation_impl() const noexcept
    {
        return {source_oxidation_.data(), this->n_equations};
    }

    /**
     * @brief Splitting-safe oxidation rate coefficients with MOMIC realizability guard.
     *
     * Shadows `MomentMethodBase::kappa_oxidation`.  The base-class formula
     * κ_i = |source_i| / M_i is correct but can produce an anomalously large κ₅
     * for M02: the source term requires `GetMoment(Av_f, As_f+3)`, which evaluates
     * the quadratic MOMIC polynomial **outside** its calibrated y ∈ {0,1,2} domain
     * (y=3 for the spherical model, y≈2.4 for the fractal model).  The quadratic
     * `a₂₀·y²` term (a₂₀ = ½ σ_S² ≥ 0) can over-extrapolate by 10⁵–10⁷× for
     * broad soot distributions, making `κ₅·dt ≫ 1` and collapsing M₀₂ to zero in a
     * single splitting step.  Once M₀₂ drops below its small-mode floor N₀·S₀²,
     * L₀₂ < 0, the MOMIC system becomes invalid, and the ODE integrator diverges.
     *
     * This override applies the Cauchy-Schwarz moment inequalities as upper bounds
     * on the second-order kappas, preserving moment realizability after the split:
     *   - κ₃ ≤ max(0, 2·κ₁ − κ₀)   ensures M₂₀·M₀₀ ≥ M₁₀²
     *   - κ₅ ≤ max(0, 2·κ₂ − κ₀)   ensures M₀₂·M₀₀ ≥ M₀₁²
     *   - κ₄ ≤ max(0, (κ₃+κ₅)/2)   ensures M₁₁² ≤ M₂₀·M₀₂
     *
     * @note The source terms themselves (computed by `SootOxidationM7`) are unchanged.
     *       This guard only affects the operator-splitting path.
     *
     * @param current_moments  Transported moment values at the current time.
     * @return Span of size 7 into the internal κ cache; valid until the next
     *         `ComputeSources()` call.
     */
    [[nodiscard]] std::span<const double>
    kappa_oxidation(std::span<const double> current_moments) const noexcept;

    /** @brief Oxidation-only gas-phase source terms [kg/m3/s]. */
    [[nodiscard, gnu::always_inline]] std::span<const double> omega_gas_oxidation_impl() const noexcept
    {
        return {omega_gas_oxidation_.data(),
                static_cast<std::size_t>(omega_gas_oxidation_.size())};
    }

    // -- HMOM6-specific coagulation breakdown ---------------------------------

    /**
     * @struct CoagulationDetail
     * @brief Read-only bundle of all HMOM6 coagulation sub-vector spans.
     */
    struct CoagulationDetail
    {
        std::span<const double> all;        //!< Discrete + continuous total.
        std::span<const double> discrete;   //!< Discrete sub-total.
        std::span<const double> disc_ss;    //!< Discrete small-small.
        std::span<const double> disc_sl;    //!< Discrete small-large.
        std::span<const double> disc_ll;    //!< Discrete large-large.
        std::span<const double> continuous; //!< Continuous sub-total.
        std::span<const double> cont_ss;    //!< Continuous small-small.
        std::span<const double> cont_sl;    //!< Continuous small-large.
        std::span<const double> cont_ll;    //!< Continuous large-large.
    };

    /** @brief Returns zero-copy spans into all nine coagulation sub-vectors. */
    [[nodiscard]] CoagulationDetail coagulation_detail() const noexcept
    {
        return {
            {source_coagulation_all_.data(), this->n_equations},
            {source_coagulation_discrete_.data(), this->n_equations},
            {source_coagulation_ss_.data(), this->n_equations},
            {source_coagulation_sl_.data(), this->n_equations},
            {source_coagulation_ll_.data(), this->n_equations},
            {source_coagulation_continuous_.data(), this->n_equations},
            {source_coagulation_cont_ss_.data(), this->n_equations},
            {source_coagulation_cont_sl_.data(), this->n_equations},
            {source_coagulation_cont_ll_.data(), this->n_equations},
        };
    }

    // -- Reporter output hooks ------------------------------------------------

    /**
     * @brief HMOM6-specific prefix columns: bimodal NDF statistics.
     */
    template <typename CB> void variant_prefix_output(CB&& cb) const
    {
        cb("np[-]",        number_primary_particles());
        cb("ss[m2/#]",     soot_mean_surface());
        cb("vs[m3/#]",     soot_mean_volume());
        cb("N0[#/m3]",     N0_);
        cb("NL[#/m3]",     L00_);
        cb("alphaL[-]",    soot_large_fraction());
        cb("dpL[nm]",      soot_large_primary_particle_diameter() * 1.e9);
        cb("npL[-]",       soot_large_primary_particle_number());
        cb("d63[nm]",      soot_d63() * 1.e9);
        cb("sigma_dp[-]",  soot_log_geom_std_dev_primary_particle_diameter());
        cb("sigma_np[-]",  soot_log_geom_std_dev_primary_particle_number());
        cb("sigma_dp_L[-]",soot_large_log_geom_std_dev_primary_particle_diameter());
        cb("sigma_np_L[-]",soot_large_log_geom_std_dev_primary_particle_number());
        cb("gsd_dp[-]",    std::exp(soot_log_geom_std_dev_primary_particle_diameter()));
        cb("gsd_np[-]",    std::exp(soot_log_geom_std_dev_primary_particle_number()));
        cb("gsd_dp_L[-]",  std::exp(soot_large_log_geom_std_dev_primary_particle_diameter()));
        cb("gsd_np_L[-]",  std::exp(soot_large_log_geom_std_dev_primary_particle_number()));

        this->EmitStandardSootOmegaGas(
            cb, pah_index_, index_C2H2_, index_H2_, index_O2_, index_H2O_, index_OH_);
    }

    /**
     * @brief HMOM6-specific suffix columns: coagulation breakdown.
     */
    template <typename CB> void variant_suffix_output(CB&& cb) const
    {
        const auto cd    = coagulation_detail();
        const unsigned N = this->n_equations;
        for (unsigned j = 0; j < N; ++j)
            cb("ScoaTot(" + std::to_string(j) + ")[mol/m3/s]", cd.all[j]);
        for (unsigned j = 0; j < N; ++j)
            cb("ScoaDis(" + std::to_string(j) + ")[mol/m3/s]", cd.discrete[j]);
        for (unsigned j = 0; j < N; ++j)
            cb("ScoaDisSS(" + std::to_string(j) + ")[mol/m3/s]", cd.disc_ss[j]);
        for (unsigned j = 0; j < N; ++j)
            cb("ScoaDisSL(" + std::to_string(j) + ")[mol/m3/s]", cd.disc_sl[j]);
        for (unsigned j = 0; j < N; ++j)
            cb("ScoaDisLL(" + std::to_string(j) + ")[mol/m3/s]", cd.disc_ll[j]);
        for (unsigned j = 0; j < N; ++j)
            cb("ScoaCon(" + std::to_string(j) + ")[mol/m3/s]", cd.continuous[j]);
        for (unsigned j = 0; j < N; ++j)
            cb("ScoaConSS(" + std::to_string(j) + ")[mol/m3/s]", cd.cont_ss[j]);
        for (unsigned j = 0; j < N; ++j)
            cb("ScoaConSL(" + std::to_string(j) + ")[mol/m3/s]", cd.cont_sl[j]);
        for (unsigned j = 0; j < N; ++j)
            cb("ScoaConLL(" + std::to_string(j) + ")[mol/m3/s]", cd.cont_ll[j]);
    }

    // -- Model switches -------------------------------------------------------

    void SetNucleation(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument("[HMOM6] Invalid nucleation model flag.");
        nucleation_model_ = flag;
    }
    void SetSurfaceGrowth(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument("[HMOM6] Invalid surface-growth model flag.");
        surface_growth_model_ = flag;
    }
    void SetCondensation(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument("[HMOM6] Invalid condensation model flag.");
        condensation_model_ = flag;
    }
    void SetOxidation(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument("[HMOM6] Invalid oxidation model flag.");
        oxidation_model_ = flag;
    }
    void SetCoagulation(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument("[HMOM6] Invalid coagulation model flag.");
        coagulation_model_ = flag;
    }
    void SetCoagulationContinuous(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument("[HMOM6] Invalid continuous-coagulation model flag.");
        coagulation_continuous_model_ = flag;
    }
    // -- Model-state query accessors ------------------------------------------

    [[nodiscard]] NucleationModel    nucleation_model()    const noexcept { return static_cast<NucleationModel>(nucleation_model_);             }
    [[nodiscard]] SurfaceGrowthModel surface_growth_model() const noexcept { return static_cast<SurfaceGrowthModel>(surface_growth_model_);     }
    [[nodiscard]] CondensationModel  condensation_model()  const noexcept { return static_cast<CondensationModel>(condensation_model_);         }
    [[nodiscard]] OxidationModel     oxidation_model()     const noexcept { return static_cast<OxidationModel>(oxidation_model_);               }
    [[nodiscard]] CoagulationModel   coagulation_model()   const noexcept { return static_cast<CoagulationModel>(coagulation_model_);           }
    [[nodiscard]] int continuous_coagulation_model()       const noexcept { return coagulation_continuous_model_; }

    void SetFractalDiameterModel(int m)
    {
        if (m != 0 && m != 1)
            throw std::invalid_argument("[HMOM6] Invalid fractal-diameter model flag.");
        fractal_diameter_model_ = static_cast<FractalDiameterModel>(m);
    }
    void SetCollisionDiameterModel(int m)
    {
        if (m != 1 && m != 2)
            throw std::invalid_argument("[HMOM6] Invalid collision-diameter model flag.");
        collision_diameter_model_ = static_cast<CollisionDiameterModel>(m);
    }

    void SetStickingCoefficientModel(std::string_view label);
    void SetStickingCoefficientConstant(double value) noexcept { sticking_coeff_constant_ = value; }
    void SetSurfaceDensity(double value) noexcept { surface_density_ = value; }
    void SetSurfaceDensityCorrectionCoefficient(bool on) noexcept { surface_density_correction_ = on; }
    void SetSurfaceDensityCorrectionCoefficientA1(double v) noexcept { surf_dens_a1_ = v; }
    void SetSurfaceDensityCorrectionCoefficientA2(double v) noexcept { surf_dens_a2_ = v; }
    void SetSurfaceDensityCorrectionCoefficientB1(double v) noexcept { surf_dens_b1_ = v; }
    void SetSurfaceDensityCorrectionCoefficientB2(double v) noexcept { surf_dens_b2_ = v; }

    void SetPAH(std::string_view name);
    void SetGasClosureDummySpecies(std::string_view name);

    // -- HACA parameters -------------------------------------------------------

    void SetA1f(double v) noexcept;  void SetA1b(double v) noexcept;
    void SetA2f(double v) noexcept;  void SetA2b(double v) noexcept;
    void SetA3f(double v) noexcept;  void SetA3b(double v) noexcept;
    void SetA4(double v) noexcept;   void SetA5(double v) noexcept;

    void SetE1f(double kJ) noexcept; void SetE1b(double kJ) noexcept;
    void SetE2f(double kJ) noexcept; void SetE2b(double kJ) noexcept;
    void SetE3f(double kJ) noexcept; void SetE3b(double kJ) noexcept;
    void SetE4(double kJ) noexcept;  void SetE5(double kJ) noexcept;

    void Setn1f(double v) noexcept;  void Setn1b(double v) noexcept;
    void Setn2f(double v) noexcept;  void Setn2b(double v) noexcept;
    void Setn3f(double v) noexcept;  void Setn3b(double v) noexcept;
    void Setn4(double v) noexcept;   void Setn5(double v) noexcept;

    void SetEfficiency6(double v) noexcept { eff6_ = v; }

    // -- Accessors ------------------------------------------------------------

    [[nodiscard]] double V0() const noexcept { return V0_; }
    [[nodiscard]] double S0() const noexcept { return S0_; }
    [[nodiscard]] double dimerization_rate() const noexcept { return dimerization_rate_; }
    [[nodiscard]] double soot_small_number_density() const noexcept { return N0_; }
    [[nodiscard]] double soot_large_number_density() const noexcept
    {
        return std::max(M00_ - N0_, 0.);
    }
    [[nodiscard]] double soot_large_fraction() const noexcept
    {
        if (M00_ <= 0.) return 0.;
        const double NL = std::max(M00_ - N0_, 0.);
        return NL / M00_;
    }
    [[nodiscard]] double soot_mean_volume() const noexcept
    {
        return (M00_ > kSootNumberFloor && M10_ > 0.) ? M10_ / M00_ : V0_;
    }
    [[nodiscard]] double soot_mean_surface() const noexcept
    {
        return (M00_ > kSootNumberFloor && M01_ > 0.) ? M01_ / M00_ : S0_;
    }

    /** @brief Small-mode number fraction N0/(N0+NL) [-]. */
    [[nodiscard]] double soot_small_fraction() const noexcept
    {
        return (M00_ > 0.) ? N0_ / M00_ : 1.;
    }

    /** @brief Mean volume of large-mode particles VL [m3/#]. */
    [[nodiscard]] double soot_large_mean_volume() const noexcept;

    /** @brief Mean surface of large-mode particles SL [m2/#]. */
    [[nodiscard]] double soot_large_mean_surface() const noexcept;

    /** @brief Primary particle diameter of large-mode particles dp,L [m]. */
    [[nodiscard]] double soot_large_primary_particle_diameter() const noexcept;

    /** @brief Mean primary particle count per large-mode aggregate np,L [-]. */
    [[nodiscard]] double soot_large_primary_particle_number() const noexcept;

    /**
     * @brief Log geometric standard deviation of primary particle diameter (total NDF).
     * @return sigma on the natural-log scale [-].
     */
    [[nodiscard]] double soot_log_geom_std_dev_primary_particle_diameter() const noexcept;

    /**
     * @brief Log geometric standard deviation of primary particle count (total NDF).
     * @return sigma on the natural-log scale [-].
     */
    [[nodiscard]] double soot_log_geom_std_dev_primary_particle_number() const noexcept;

    /** @brief Log geometric std dev of primary diameter for the large mode only. */
    [[nodiscard]] double soot_large_log_geom_std_dev_primary_particle_diameter() const noexcept;

    /** @brief Log geometric std dev of primary count for the large mode only. */
    [[nodiscard]] double soot_large_log_geom_std_dev_primary_particle_number() const noexcept;

    /**
     * @brief Scattering effective diameter d63 [m].
     * @return d63 = 6*(M(4/3,0)/M(1,0))^(1/3) [m].
     */
    [[nodiscard]] double soot_d63() const noexcept;

    /**
     * @brief Fill common particle properties in one call.
     * @param[out] fv  Volume fraction [-].
     * @param[out] dp  Mean primary particle diameter [m].
     * @param[out] dc  Collision diameter [m].
     * @param[out] np  Mean primary particles per aggregate [-].
     * @param[out] ss  Mean surface per particle [m2/#].
     * @param[out] vs  Mean volume per particle [m3/#].
     */
    void Properties(double& fv, double& dp, double& dc,
                    double& np, double& ss, double& vs) const noexcept;

    /**
     * @brief Returns the MOMIC polynomial coefficients for the large mode.
     *
     * Order: {a00, a10, a11, a20, a21, a22} where the polynomial is:
     * p(V,S) = a00 + a10*S + a11*V + a20*S² + a21*V*S + a22*V²
     *
     * Valid after `ComputeSources()` when `momic_valid_ == true`.
     */
    [[nodiscard]] const std::array<double, 6>& momic_coefficients() const noexcept
    {
        return momic_coeffs_;
    }

    /** @brief True if the MOMIC system was successfully solved at the last call. */
    [[nodiscard]] bool momic_valid() const noexcept { return momic_valid_; }

private:

    // -- Private computational methods ----------------------------------------

    void MemoryAllocation();
    void Precalculations();
    void ApplyConfig(const Config& cfg);
    void GetMoments();
    void ComputeMOMICCoefficients();
    void DimerConcentration();
    void SootKineticConstants();
    void SootNucleationM7();
    void SootSurfaceGrowthM7();
    void SootOxidationM7();
    void SootCondensationM7();
    void SootCoagulationM7();
    void SootCoagulationSmallSmallM7();
    void SootCoagulationSmallLargeM7();
    void SootCoagulationLargeLargeM7();
    void SootCoagulationContinuousM7();
    void SootCoagulationContinuousSmallSmallM7(double lambda, double betai0);
    void SootCoagulationContinuousSmallLargeM7(double lambda, double betai0);
    void SootCoagulationContinuousLargeLargeM7(double lambda, double betai0);
    void CalculateAlphaCoefficient();

    /**
     * @brief Evaluate M_{x,y} = N0*V0^x*S0^y + exp(MOMIC(x,y)) for any real x,y.
     *
     * Returns the sum of the small-mode (delta) and large-mode (MOMIC) contributions.
     * Falls back to 0 if HasSoot() == false or momic_valid_ == false.
     */
    [[nodiscard]] double GetMoment(double x, double y) const noexcept;

    /**
     * @brief Evaluate M_{x,y}^large = exp(MOMIC(x,y)) — large mode only.
     *
     * This is the MOMIC closure for the large mode. Used in surface growth,
     * oxidation, coagulation source terms where small-mode contributions
     * are handled separately.
     *
     * Returns 0 if HasSoot() == false or momic_valid_ == false.
     */
    [[nodiscard]] double GetMissingMoment(double x, double y) const noexcept;

    /**
     * @brief Evaluate the MOMIC polynomial at (x,y).
     *
     * p(x,y) = a00 + a10*y + a11*x + a20*y² + a21*x*y + a22*x²
     */
    [[nodiscard]] double MOMICPolynomial(double x, double y) const noexcept;

    [[nodiscard]] double GetBetaC() const noexcept;
    [[nodiscard]] double PAHDimerizationRate() const noexcept;
    [[nodiscard]] bool   HasSoot() const noexcept;
    [[nodiscard]] double SafePowPositive(double x, double a) const noexcept;

    /**
     * @brief Compute log geometric standard deviation from three raw moments.
     *
     * Given M0 = ∫f, M1 = ∫x·f, M2 = ∫x²·f of a lognormal distribution,
     * returns sqrt(ln(M0·M2/M1²)).  Returns 0 if any moment is non-positive
     * or if the ratio ≤ 1 (perfectly monodisperse).
     */
    [[nodiscard]] double LogGeomStdDevFromMoments(double M0, double M1, double M2) const noexcept;

private:

    // -- Thermodynamics reference ---------------------------------------------
    const Thermo& thermo_;

    // -- Transported (normalised) moments -------------------------------------
    double M00_normalized_ = 0.; //!< [mol/m3]
    double M10_normalized_ = 0.; //!< [mol/m3]
    double M01_normalized_ = 0.; //!< [mol/m3]
    double M20_normalized_ = 0.; //!< [mol/m3]
    double M11_normalized_ = 0.; //!< [mol/m3]
    double M02_normalized_ = 0.; //!< [mol/m3]
    double N0_normalized_  = 0.; //!< [mol/m3]

    // -- Reconstructed physical moments ---------------------------------------
    //    Units: M_{x,y} has dimension [m3x × m2y × #/m3_gas]
    double M00_ = 0.; //!< [#/m3]
    double M10_ = 0.; //!< [m3/#/m3] = [−]   (volume fraction integral)
    double M01_ = 0.; //!< [m2/#/m3] = [1/m]
    double M20_ = 0.; //!< [m6/#/m3]
    double M11_ = 0.; //!< [m5/#/m3]
    double M02_ = 0.; //!< [m4/#/m3]
    double N0_  = 0.; //!< Small-particle number density [#/m3]

    // -- Large-mode moments for MOMIC -----------------------------------------
    //    L_{x,y} = M_{x,y} - N0 * V0^x * S0^y
    double L00_ = 0.;
    double L10_ = 0.;
    double L01_ = 0.;
    double L20_ = 0.;
    double L11_ = 0.;
    double L02_ = 0.;

    // -- MOMIC polynomial coefficients ----------------------------------------
    //    p(V,S) = a00 + a10*S + a11*V + a20*S² + a21*V*S + a22*V²
    //    Index mapping: [0]=a00, [1]=a10, [2]=a11, [3]=a20, [4]=a21, [5]=a22
    std::array<double, 6> momic_coeffs_{};
    bool momic_valid_ = false; //!< True when MOMIC system solved successfully.

    // -- Species concentrations [kmol/m3] (set by SetState) -------------------
    double conc_OH_   = 0.;
    double conc_H_    = 0.;
    double conc_H2O_  = 0.;
    double conc_H2_   = 0.;
    double conc_C2H2_ = 0.;
    double conc_O2_   = 0.;
    double conc_PAH_  = 0.;
    double conc_DIMER_= 0.;

    // 0-based species indices (-1 if absent in mechanism)
    int index_H_    = -1;
    int index_OH_   = -1;
    int index_O2_   = -1;
    int index_H2_   = -1;
    int index_H2O_  = -1;
    int index_CO_   = -1;
    int index_C2H2_ = -1;

    double mass_fraction_H_  = 0.;
    double mass_fraction_OH_ = 0.;

    // -- PAH (precursor) properties -------------------------------------------
    std::string pah_species_;
    int    pah_index_  = -1;
    double vpah_       = 0.;
    double spah_       = 0.;
    double dpah_       = 0.;
    double mpah_       = 0.;
    double mwpah_      = 0.;
    double ncpah_      = 0.;
    double nhpah_      = 0.;

    // -- Nucleated particle geometry ------------------------------------------
    double V0_  = 0.; //!< Nucleated particle volume [m3]
    double S0_  = 0.; //!< Nucleated particle surface [m2]
    double VC2_ = 0.; //!< Volume of 2 C atoms [m3]

    // -- Dimer properties -----------------------------------------------------
    double dimer_volume_      = 0.;
    double dimer_surface_     = 0.;
    double dimerization_rate_ = 0.; //!< [mol/m3/s]

    // -- Kinetic intermediate quantities --------------------------------------
    double kox_      = 0.;
    double kox_O2_   = 0.;
    double kox_OH_   = 0.;
    double ksg_      = 0.;
    double betaN_    = 0.;
    double Cfm_      = 0.;
    double betaN_TV_ = 0.;

    // -- Fractal / collision geometry pre-factors -----------------------------
    double Av_fractal_     = 0.;
    double As_fractal_     = 0.;
    double K_fractal_      = 0.;
    double D_collisional_  = 0.;
    double Av_collisional_ = 0.;
    double As_collisional_ = 0.;
    double K_collisional_  = 0.;

    // -- Surface density correction -------------------------------------------
    bool   surface_density_correction_ = false;
    double surface_density_            = 1.7e19; //!< [#/m2]
    double surf_dens_a1_ = 0., surf_dens_a2_ = 0.;
    double surf_dens_b1_ = 0., surf_dens_b2_ = 0.;
    double alpha_ = 1.; //!< Surface-density correction factor.

    // -- Model flags ----------------------------------------------------------
    int nucleation_model_             = 0;
    int condensation_model_           = 0;
    int surface_growth_model_         = 0;
    int oxidation_model_              = 0;
    int coagulation_model_            = 0;
    int coagulation_continuous_model_ = 0;

    FractalDiameterModel    fractal_diameter_model_    = FractalDiameterModel::Model1;
    CollisionDiameterModel  collision_diameter_model_  = CollisionDiameterModel::Model2;
    StickingModel           sticking_model_            = StickingModel::Constant;
    double                  sticking_coeff_constant_   = 2.e-3;

    bool is_debug_mode_          = false;
    bool is_simplified_pah_mass_ = false;

    // -- Per-process source storage -------------------------------------------

    MomentVector source_nucleation_   = MomentVector::Zero();
    MomentVector source_coagulation_  = MomentVector::Zero();
    MomentVector source_condensation_ = MomentVector::Zero();
    MomentVector source_growth_       = MomentVector::Zero();
    MomentVector source_oxidation_    = MomentVector::Zero();

    Eigen::VectorXd omega_gas_oxidation_;

    // -- Coagulation source breakdown -----------------------------------------

    MomentVector source_coagulation_discrete_   = MomentVector::Zero();
    MomentVector source_coagulation_ss_         = MomentVector::Zero();
    MomentVector source_coagulation_sl_         = MomentVector::Zero();
    MomentVector source_coagulation_ll_         = MomentVector::Zero();
    MomentVector source_coagulation_continuous_ = MomentVector::Zero();
    MomentVector source_coagulation_cont_ss_    = MomentVector::Zero();
    MomentVector source_coagulation_cont_sl_    = MomentVector::Zero();
    MomentVector source_coagulation_cont_ll_    = MomentVector::Zero();
    MomentVector source_coagulation_all_        = MomentVector::Zero();

    // -- Initial moments cache ------------------------------------------------
    MomentVector initial_moments_cache_ = MomentVector::Zero();

    // -- HACA kinetics parameters ---------------------------------------------
    double A1f_ = 0., n1f_ = 0., E1f_ = 0., A1b_ = 0., n1b_ = 0., E1b_ = 0.;
    double A2f_ = 0., n2f_ = 0., E2f_ = 0., A2b_ = 0., n2b_ = 0., E2b_ = 0.;
    double A3f_ = 0., n3f_ = 0., E3f_ = 0., A3b_ = 0., n3b_ = 0., E3b_ = 0.;
    double A4_ = 0., n4_ = 0., E4_ = 0.;
    double A5_ = 0., n5_ = 0., E5_ = 0.;
    double eff6_ = 0.;

    // -- Numerical floors -----------------------------------------------------
    static constexpr double kTinyNumberDensity = 1.e-30;
    static constexpr double kSootNumberFloor   = 1.e3;
    static constexpr double kSootVolumeFloor   = 1.e-40;
    static constexpr double kSootSurfaceFloor  = 1.e-30;
    static constexpr double kMOMICEps          = 1.e-300; //!< Guard against log(0)
};

} // namespace MOM

#if defined(MOM_USE_DICTIONARY)
#include "HMOM6_Grammar.h"
#endif

#if !defined(MOM_COMPILED_LIBRARY)
#include "HMOM6.tpp"
#else
namespace MOM
{
extern template class HMOM6<BasicThermoData>;
}
#endif
