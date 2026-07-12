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
#include <expected> // std::expected / std::unexpected — C++23, only with MOM_USE_DICTIONARY
#endif

namespace MOM
{

/**
 * @class HMOM
 * @brief Hybrid Method of Moments for soot with 4 transported equations.
 *
 * Implements the HMOM model of Mueller, Pitsch & Raman (2009) as extended by
 * Attili, Bisetti, Mueller & Pitsch (2014).  The method transports a set of
 * normalised moments that describe a bimodal number density function (NDF)
 * consisting of a small (nucleation) mode and a large (growth/coagulation) mode.
 *
 * @par References
 * - Mueller, Pitsch & Raman, *Proc. Combust. Inst.* **32** (2009) 785–792.
 * - Attili, Bisetti, Mueller & Pitsch, *Combust. Flame* **161** (2014) 1849–1865.
 *
 * @par Transported variables (normalised, [mol/m3])
 * | Index | Symbol | Physical meaning |
 * |---|---|---|
 * | 0 | M00 | Zeroth-order moment (proportional to total number density) |
 * | 1 | M10 | First-order volume moment |
 * | 2 | M01 | First-order surface moment |
 * | 3 | N0  | Small-particle (nucleation-mode) number density |
 *
 * @par Physical processes modelled
 * - **Nucleation** — PAH dimerisation in the free-molecular regime.
 * - **Surface growth** — HACA mechanism (H-abstraction / C2H2 addition),
 *   5-step kinetic scheme with surface-site efficiency.
 * - **Condensation** — PAH adsorption onto existing soot particles.
 * - **Oxidation** — O2 and OH attack (Lee et al. and Roper correlations).
 * - **Coagulation** — discrete (small–small, small–large, large–large collisions)
 *   plus a continuum correction term.
 * - **Thermophoresis** — encoded in the effective diffusion coefficient.
 *
 * @par NDF reconstruction
 * HMOM reconstructs the NDF as two delta-function nodes:
 * - Node 0 (small mode): nucleated particles at fixed volume V0, surface S0.
 * - Node 1 (large mode): aggregated particles with mean volume VL, surface SL.
 *
 * @note All five modelled processes have a corresponding `sources_X_impl()`
 *       extension point declared here, which MomentMethodBase detects at compile
 *       time via `if constexpr (requires(...))` to route calls with zero overhead.
 *       Sintering is NOT declared; `sources_sintering()` returns a zero span.
 *
 * @par Thread safety
 * Not thread-safe — one instance per OpenMP thread.
 * See `MomentMethodBase` for the complete thread-safety contract.
 *
 * @tparam Thermo  Must satisfy the MOM::ThermoMap concept.
 */

template <ThermoMap Thermo> class HMOM : public MomentMethodBase<HMOM<Thermo>, 4>
{
    using Base = MomentMethodBase<HMOM<Thermo>, 4>;

public:

    using typename Base::MomentVector;

    /// Labels accepted by MOM::MakeAnyMomentMethod for runtime variant selection.
    static constexpr std::array<std::string_view, 2> variant_labels{"HMOM", "hmom"};

    // -- Method-specific sub-model enums -------------------------------------

    /** @brief Sticking coefficient model for PAH-soot collisions. */
    enum class StickingModel : int
    {
        Constant = 0, //!< Fixed sticking coefficient (default: 2×10⁻³).
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
        Model1 = 1, //!< Collision diameter ∝ N^{1/3} d_p (Mueller 2009).
        Model2 = 2  //!< Collision diameter from fractal geometry (Attili 2014).
    };

    /**
     * @brief Properties of one node in the HMOM two-delta NDF reconstruction.
     *
     * HMOM reconstructs the NDF as two delta nodes (Mueller et al. 2009, §3.2):
     * - Node 0 (small mode): newly nucleated particles at fixed volume V₀, surface S₀.
     * - Node 1 (large mode): grown/coagulated particles with mean volume V_L, surface S_L.
     */
    struct NDFNode
    {
        double number_density;     //!< Number density N [#/m3].
        double volume;             //!< Mean particle volume V [m3/#].
        double surface;            //!< Mean particle surface S [m2/#].
        double primary_diameter;   //!< Primary particle diameter dp [m].
        double primary_particles;  //!< Mean primary particles per aggregate np [-].
        double collision_diameter; //!< Aggregate collision diameter dc [m].
    };

    /**
     * @struct NDFReconstructionData
     * @brief Parameters of the HMOM two-node NDF reconstruction, smeared to a
     *        bimodal log-normal for diagnostic visualization.
     *
     * @par Physical background
     * HMOM is a strict two-delta method (Mueller et al. 2009, §3.2).  The
     * exact NDF carried by the transported moments is:
     * @code
     *   n(v) = N0 · δ(v − V0)  +  NL · δ(v − VL)
     * @endcode
     * where @c V0 is the fixed nucleation volume (derived from PAH geometry)
     * and @c VL = NLVL/NL is the mean large-mode aggregate volume.  Both modes
     * are **point masses** — the four transported moments M00, M10, M01, N0
     * fully determine the node positions (V0, VL) and weights (N0, NL) but
     * carry **no information** about within-mode polydispersity.
     *
     * @par Smeared visualization
     * For output purposes each Dirac delta is replaced by a narrow log-normal
     * kernel with a fixed, purely cosmetic log-volume standard deviation
     * σ = kNDFSmearSigmaLnV:
     * @code
     *   n_vis(v) ≈ N0 · LN(v; μ₀, σ)  +  NL · LN(v; μL, σ)
     * @endcode
     * The location parameters preserve the original delta positions as means:
     * @code
     *   μ₀ = ln(V0) − σ²/2   so that  ⟨v⟩_small = V0
     *   μL = ln(VL) − σ²/2   so that  ⟨v⟩_large = VL
     * @endcode
     *
     * @warning The smearing width σ is **not** derived from any transported
     *          moment; it is a visualization convenience only.  Users should
     *          not interpret the width of the smeared peaks as physical
     *          polydispersity information.
     *
     * @note All length-containing fields are in SI units [m³, #/m³].
     *       MomentMethodReporter::WriteReconstructedNDF() applies nm³ conversion.
     */
    struct NDFReconstructionData
    {
        bool   valid;   //!< True if the reconstruction is physically meaningful.

        // -- Small (nucleation) mode — monodisperse delta at V₀ in HMOM ------
        double N0;      //!< Number density of the nucleation mode [#/m³].
        double V0;      //!< Nucleation-mode particle volume [m³/#].
        double mu0;     //!< Log-normal location: μ₀ = ln(V₀) − σ²/2  [ln(m³)].

        // -- Large (growth / coagulation) mode — monodisperse delta at VL ----
        double NL;      //!< Number density of the large (growth) mode [#/m³].
        double VL;      //!< Mean particle volume of the large mode [m³/#].
        double muL;     //!< Log-normal location: μL = ln(VL) − σ²/2  [ln(m³)].

        // -- Shared cosmetic smearing width -----------------------------------
        double sigma;   //!< Log-volume std dev used for both smeared modes [-].
                        //!< Equals kNDFSmearSigmaLnV; stored here for self-containedness.
    };

    // -- Configuration struct ------------------------------------------------

    /**
     * @struct Config
     * @brief Plain configuration parameters for the HMOM variant.
     *
     * All fields carry defaults that reproduce the post-constructor state.
     * Pass a default-constructed @c Config to `SetupFromConfig()` to apply
     * the library defaults without any input file.
     *
     * @note No external dependencies: only standard C++ types.
     */
    struct Config : CommonConfig<1>,
                    PAHConfig,
                    BinarySootProcessConfig,
                    GasConsumptionConfig<true>,
                    SootRadiationConfig,
                    StickingConfig
    {
        // ---- Geometry models -----------------------------------------------
        int fractal_diameter_model   = 1; //!< Fractal diameter model index   [1 = default]
        int collision_diameter_model = 2; //!< Collision diameter model index [2 = default]

        // ---- Surface density -----------------------------------------------
        double surface_density_per_m2   = 1.7e19;  //!< Active surface site density [#/m²]
        bool   surface_density_correction = false;  //!< Temperature-dependent χ correction
        double surf_dens_a1 = 12.65;    //!< Correction coefficient A1 [-]
        double surf_dens_a2 = -0.00563; //!< Correction coefficient A2 [1/K]
        double surf_dens_b1 = -1.38;   //!< Correction coefficient B1 [-]
        double surf_dens_b2 =  0.00069; //!< Correction coefficient B2 [1/K]

        // ---- Additional process model selection -----------------------------
        int continuous_coagulation_model = 1; //!< Coagulation (continuum) model

        // ---- HACA kinetics (A in cm³/mol/s, E in kJ/mol) ------------------
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

    // -- Construction --------------------------------------------------------

    /**
     * @brief Constructs HMOM bound to the given thermodynamics map.
     *
     * Does not allocate computational memory.  Call `SetupFromConfig()` or
     * the individual `Set*` methods, then call `CalculateSourceMoments()` each
     * cell iteration.
     *
     * @param thermo  Const reference to the thermodynamics map (must outlive this object).
     */
    explicit HMOM(const Thermo& thermo);
    explicit HMOM(const Thermo&&) = delete; ///< Prevents binding a temporary as thermo (dangling ref).

    HMOM(const HMOM&)            = delete; ///< Non-copyable — holds external thermo reference.
    HMOM& operator=(const HMOM&) = delete;
    HMOM(HMOM&&)            = default; ///< Move-constructible for placement in std::variant.
    HMOM& operator=(HMOM&&) = delete;  ///< Not move-assignable — const Thermo& member cannot be reseated.

    /**
     * @brief Configure all HMOM parameters from a plain configuration struct.
     *
     * Applies every field of @p cfg by calling the corresponding `Set*()`
     * methods.  This is the primary programmatic setup path; it has no
     * dependency on external parsing frameworks.
     *
     * Calling this method is equivalent to calling the individual `Set*()`
     * methods in the same order, followed by `PrintSummary()`.
     *
     * @param cfg  Configuration struct.  Default-constructed @c Config
     *             reproduces the constructor defaults exactly.
     */
    void SetupFromConfig(const Config& cfg);

#if defined(MOM_USE_DICTIONARY)
    /**
     * @brief Parse an OpenSMOKE++ dictionary into an HMOM Config.
     *
     * Reads every HMOM grammar key from @p dict, performs unit conversions
     * (kg/m³ ↔ g/cm³, kJ/mol → stored as kJ/mol, etc.) and returns the
     * populated struct.
     *
     * @tparam DictType  OpenSMOKE++ dictionary type.  The concrete type is
     *                   provided by the caller; this header does not include
     *                   any OpenSMOKE++ headers.
     * @param  dict      Mutable reference to the dictionary to parse.
     * @return           Populated @c Config on success; error string on failure.
     */
    template <typename DictType>
    [[nodiscard]] static std::expected<Config, std::string> ParseConfig(DictType& dict);
#endif // MOM_USE_DICTIONARY

    // -- MomentMethod concept — state injection -------------------------------

    /**
     * @brief Inject the thermodynamic state for the current computational cell.
     *
     * Extracts and caches species concentrations required by HMOM:
     * H, OH, O2, H2, H2O, C2H2, and the PAH precursor.
     *
     * @param T    Gas temperature [K].
     * @param P_Pa Gas pressure [Pa].
     * @param Y    Species mass fractions (pointer, size = `thermo.NumberOfSpecies()`).
     */
    void SetStatus(double T, double P_Pa, const double* Y) noexcept;

    /**
     * @brief Set moment values via a generic span (satisfies MomentMethod concept).
     *
     * @param m  Span of size 4: `[M00_norm, M10_norm, M01_norm, N0_norm]` [mol/m3].
     *           Indices match the transported variable order in the CFD solver.
     * @note Prefer `SetNormalizedMoments()` in HMOM-aware code for clarity.
     */
    void SetMoments(std::span<const double> m) noexcept;

    /**
     * @brief Set moment values by name (preferred in HMOM-aware code).
     *
     * @param M00_norm  Normalised zeroth-order moment [mol/m3].
     * @param M10_norm  Normalised first-order volume moment [mol/m3].
     * @param M01_norm  Normalised first-order surface moment [mol/m3].
     * @param N0_norm   Normalised small-particle number density [mol/m3].
     */
    void SetNormalizedMoments(double M00_norm, double M10_norm, double M01_norm, double N0_norm) noexcept;

    // -- MomentMethod concept — core computation ------------------------------

    /**
     * @brief Compute all moment source terms for the current cell state.
     *
     * Updates `source_all_`, `source_nucleation_`, `source_growth_`,
     * `source_oxidation_`, `source_condensation_`, the nine coagulation
     * sub-vectors, and `omega_gas_`.  Must be called after `SetStatus()`
     * and `SetMoments()` each cell iteration.
     */
    void CalculateSourceMoments() noexcept;

    /**
     * @brief Compute only the gas-phase consumption terms (`omega_gas_`).
     *
     * Called internally by `CalculateSourceMoments()`.  Exposed separately for
     * operator-split solvers where source terms are already known and only gas
     * coupling needs to be updated.
     */
    void CalculateOmegaGas() noexcept;

    // -- MomentMethod concept — particle properties ---------------------------

    /** @brief Soot volume fraction fv = M10 * WC / ρ_s [-]. */
    [[nodiscard]] double volume_fraction() const noexcept;

    /** @brief Mean primary particle diameter dp [m]. */
    [[nodiscard]] double particle_diameter() const noexcept;

    /** @brief Aggregate collision diameter dc [m] (model-dependent). */
    [[nodiscard]] double collision_diameter() const noexcept;

    /** @brief Total soot number density N = N0 + NL [#/m3]. */
    [[nodiscard]] double particle_number_density() const noexcept;

    /** @brief Soot mass fraction Ys = M10 * WC / (ρ_mix) [-]. */
    [[nodiscard]] double mass_fraction() const noexcept;

    /** @brief Soot specific surface area Ss = M01 [m2/m3]. */
    [[nodiscard]] double specific_surface() const noexcept;

    /** @brief Mean number of primary particles per aggregate np [-]. */
    [[nodiscard]] double number_primary_particles() const noexcept;

    /** @brief Effective particle diffusion coefficient D_p [kg/m/s]. */
    [[nodiscard]] double diffusion_coefficient() const noexcept;

    // -- MomentMethod concept — initial conditions ----------------------------

    /**
     * @brief Returns near-zero initial moment values for solver initialisation.
     *
     * Computed once during setup from the nucleated particle geometry and cached.
     * Repeated calls incur no allocation.
     *
     * @return Span of size 4: `[M00₀, M10₀, M01₀, N0₀]` [mol/m3].
     */
    [[nodiscard]] std::span<const double> initial_moments() const noexcept
    {
        return {initial_moments_cache_.data(), 4u};
    }

    // -- MomentMethod concept — precursor -------------------------------------

    /** @brief 0-based index of the PAH precursor species in the thermo map (−1 if unset). */
    [[nodiscard]] int precursor_index() const noexcept { return pah_index_; }

    /** @brief Molar concentration of the PAH precursor [kmol/m3]. */
    [[nodiscard]] double precursor_concentration() const noexcept { return conc_PAH_; }

    /** @brief Name of the PAH precursor species (e.g. "C16H10"). */
    [[nodiscard]] const std::string& precursor_species() const noexcept { return pah_species_; }

    // -- MomentMethod concept — diagnostics -----------------------------------

    /** @brief Print a human-readable summary of the HMOM configuration to stdout. */
    void PrintSummary() const;

    /**
     * @name CRTP extension points — per-process source storage
     *
     * Each `sources_X_impl()` method signals to MomentMethodBase that HMOM
     * models process X.  The base class `sources_X()` getter detects these at
     * compile time via `if constexpr (requires(...))` and forwards here with
     * zero overhead — no virtual dispatch, no runtime branch.
     *
     * `sources_sintering_impl()` is intentionally absent: HMOM does not model
     * sintering, so `sources_sintering()` returns a zero span automatically.
     *
     * @note These are `public` because MomentMethodBase accesses them through the
     *       CRTP down-cast from a base-class context.  They are not part of the
     *       intended user-facing API; prefer the base-class `sources_X()` wrappers.
     * @{
     */

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

    /** @brief Oxidation-only gas-phase source terms [kg/m³/s] for operator splitting. */
    [[nodiscard, gnu::always_inline]] std::span<const double> omega_gas_oxidation_impl() const noexcept
    {
        return {omega_gas_oxidation_.data(),
                static_cast<std::size_t>(omega_gas_oxidation_.size())};
    }

    /** @} */

    /**
     * @name HMOM coagulation sub-breakdown accessors
     *
     * Detailed decomposition of coagulation into discrete (small–small,
     * small–large, large–large) and continuous contributions.
     * All returned spans have size = n_equations and are valid after
     * `CalculateSourceMoments()`.
     * @{
     */

    /** @brief Discrete coagulation total [mol/m3/s]. */
    [[nodiscard]] std::span<const double> sources_coagulation_discrete() const noexcept;

    /** @brief Discrete small–small coagulation [mol/m3/s]. */
    [[nodiscard]] std::span<const double> sources_coagulation_discrete_ss() const noexcept;

    /** @brief Discrete small–large coagulation [mol/m3/s]. */
    [[nodiscard]] std::span<const double> sources_coagulation_discrete_sl() const noexcept;

    /** @brief Discrete large–large coagulation [mol/m3/s]. */
    [[nodiscard]] std::span<const double> sources_coagulation_discrete_ll() const noexcept;

    /** @brief Continuous coagulation total [mol/m3/s]. */
    [[nodiscard]] std::span<const double> sources_coagulation_continuous() const noexcept;

    /** @brief Continuous small–small coagulation [mol/m3/s]. */
    [[nodiscard]] std::span<const double> sources_coagulation_continuous_ss() const noexcept;

    /** @brief Continuous small–large coagulation [mol/m3/s]. */
    [[nodiscard]] std::span<const double> sources_coagulation_continuous_sl() const noexcept;

    /** @brief Continuous large–large coagulation [mol/m3/s]. */
    [[nodiscard]] std::span<const double> sources_coagulation_continuous_ll() const noexcept;

    /** @} */

    // -- NDF reconstruction ----------------------------------------------------

    /**
     * @brief Returns the two-node NDF reconstruction.
     *
     * Node 0 = small (nucleation) mode; Node 1 = large (growth/coagulation) mode.
     * See Mueller et al. (2009), §3.2.
     *
     * @return Array of two NDFNode structs, valid after `CalculateSourceMoments()`.
     */
    [[nodiscard]] std::array<NDFNode, 2> NumberDensityFunctionNodes() const;

    // -- NDF reconstruction (bimodal smeared log-normal) ----------------------

    /**
     * @brief Compute the bimodal smeared-log-normal NDF reconstruction parameters.
     *
     * Converts the two-delta HMOM representation (N0·δ(v−V₀) + NL·δ(v−VL))
     * into a pair of narrow log-normal kernels of equal width σ = kNDFSmearSigmaLnV
     * for visualization purposes.  See @c NDFReconstructionData for the full
     * mathematical specification.
     *
     * @par Precondition
     * Should be called after CalculateSourceMoments() so that N0_, NL_, NLVL_
     * reflect the current transported moments.
     *
     * @param use_regularized_moments  If true, applies a floor @c kTinyNumberDensity
     *   to N0 so that the reconstruction is well-defined even in particle-free
     *   cells (useful for diagnostic robustness).
     * @return  Populated @c NDFReconstructionData on success; an invalid struct
     *          (valid = false) if the moments are zero or non-finite.
     */
    [[nodiscard]] NDFReconstructionData
    ReconstructedNDFData(bool use_regularized_moments = false) const;

    /**
     * @brief Normalised smeared NDF  nbar(v) = n_vis(v) / (N0 + NL)  [1/m³].
     *
     * Satisfies the MOM::HasReconstructedNDF concept — same call signature as
     * ThreeEquations and MetalOxide, enabling transparent dispatch through
     * MomentMethodReporter::WriteReconstructedNDF().
     *
     * @par Formula
     * @code
     *   nbar(v) = (N0 · LN(v; μ₀, σ) + NL · LN(v; μL, σ)) / (N0 + NL)
     * @endcode
     * where LN denotes the log-normal PDF and σ = kNDFSmearSigmaLnV.
     *
     * @param nu                       Particle volume query point [m³].
     * @param use_regularized_moments  Forwarded to ReconstructedNDFData().
     * @return  nbar(v) [1/m³]; 0 if moments are invalid or v ≤ 0.
     */
    [[nodiscard]] double ReconstructedNormalizedNDF(
        double nu, bool use_regularized_moments = false) const;

    /**
     * @brief Dimensional smeared NDF  n_vis(v) [#/m³_gas / m³_particle].
     *
     * @par Formula
     * @code
     *   n_vis(v) = N0 · LN(v; μ₀, σ)  +  NL · LN(v; μL, σ)
     * @endcode
     *
     * @param nu                       Particle volume query point [m³].
     * @param use_regularized_moments  Forwarded to ReconstructedNDFData().
     * @return  n(v) [#/m³_gas / m³_particle]; 0 if moments are invalid or v ≤ 0.
     */
    [[nodiscard]] double ReconstructedNDF(
        double nu, bool use_regularized_moments = false) const;

    /**
     * @brief Vectorized form of ReconstructedNDF.
     *
     * @param nu                       Input volume grid [m³], size n.
     * @param n                        Output NDF values [#/m³/m³], resized to n.
     * @param use_regularized_moments  Forwarded to each scalar call.
     */
    void ReconstructedNDF(const Eigen::VectorXd& nu,
                          Eigen::VectorXd&       n,
                          bool use_regularized_moments = false) const;

    /** @brief Small-particle number density N0 [#/m3]. */
    [[nodiscard]] double soot_small_number_density() const noexcept;

    /** @brief Large-particle number density NL [#/m3]. */
    [[nodiscard]] double soot_large_number_density() const noexcept;

    /** @brief Large-mode number fraction αL = NL/(N0+NL) [-]. */
    [[nodiscard]] double soot_large_fraction() const noexcept;

    /** @brief Small-mode number fraction α0 = N0/(N0+NL) [-]. */
    [[nodiscard]] double soot_small_fraction() const noexcept;

    /** @brief Mean volume of large-mode particles VL [m3/#]. */
    [[nodiscard]] double soot_large_mean_volume() const noexcept;

    /** @brief Mean surface of large-mode particles SL [m2/#]. */
    [[nodiscard]] double soot_large_mean_surface() const noexcept;

    /** @brief Primary particle diameter of large-mode particles dp,L [m]. */
    [[nodiscard]] double soot_large_primary_particle_diameter() const noexcept;

    /** @brief Mean primary particle count per large-mode aggregate np,L [-]. */
    [[nodiscard]] double soot_large_primary_particle_number() const noexcept;

    /** @brief Mean particle volume over both modes vs = (M10·WC/Nav) / (M00·WC/Nav·ρ_s) [m3/#]. */
    [[nodiscard]] double soot_mean_volume() const noexcept;

    /** @brief Mean particle surface over both modes ss = M01/(M00) [m2/#]. */
    [[nodiscard]] double soot_mean_surface() const noexcept;

    /**
     * @brief Log geometric std dev of primary particle diameter σg,dp (Mueller 2009, Eq. 44).
     * @return σ_g [-] (natural log scale).
     */
    [[nodiscard]] double soot_log_geom_std_dev_primary_particle_diameter() const noexcept;

    /**
     * @brief Log geometric std dev of primary particle count σg,np (Mueller 2009, Eq. 45).
     * @return σ_g [-] (natural log scale).
     */
    [[nodiscard]] double soot_log_geom_std_dev_primary_particle_number() const noexcept;

    /**
     * @brief Scattering effective diameter d₆₃ = 6·(M₄,₋₃/M₁,₀)^{1/3} (Mueller 2009, Eq. 46).
     * @return d₆₃ [m].
     */
    [[nodiscard]] double soot_d63() const noexcept;

    /** @brief Log geometric std dev of primary diameter for the large mode only. */
    [[nodiscard]] double soot_large_log_geom_std_dev_primary_particle_diameter() const noexcept;

    /** @brief Log geometric std dev of primary count for the large mode only. */
    [[nodiscard]] double soot_large_log_geom_std_dev_primary_particle_number() const noexcept;

    /**
     * @brief Fill a set of common particle properties in one call.
     *
     * Convenience wrapper over the individual property accessors.
     *
     * @param[out] fv  Volume fraction [-].
     * @param[out] dp  Mean primary diameter [m].
     * @param[out] dc  Collision diameter [m].
     * @param[out] np  Mean primary particles per aggregate [-].
     * @param[out] ss  Mean surface per particle [m2/#].
     * @param[out] vs  Mean volume per particle [m3/#].
     */
    void Properties(double& fv, double& dp, double& dc, double& np, double& ss, double& vs) const noexcept;

    /**
     * @struct CoagulationDetail
     * @brief Read-only bundle of all HMOM coagulation sub-vector spans.
     *
     * Exposes the nine private coagulation sub-vectors as zero-copy spans.
     * This is the only HMOM-internal data not reachable through the common
     * `MomentMethod` concept interface.  The reporter accesses these to write
     * the detailed coagulation column block without requiring friendship or
     * private member access.
     *
     * All spans have size = `n_equations` and are valid after
     * `CalculateSourceMoments()` has been called.
     */
    struct CoagulationDetail
    {
        std::span<const double> all;        //!< Discrete + continuous total [mol/m3/s].
        std::span<const double> discrete;   //!< Discrete sub-total.
        std::span<const double> disc_ss;    //!< Discrete small–small.
        std::span<const double> disc_sl;    //!< Discrete small–large.
        std::span<const double> disc_ll;    //!< Discrete large–large.
        std::span<const double> continuous; //!< Continuous sub-total.
        std::span<const double> cont_ss;    //!< Continuous small–small.
        std::span<const double> cont_sl;    //!< Continuous small–large.
        std::span<const double> cont_ll;    //!< Continuous large–large.
    };

    /**
     * @brief Returns zero-copy spans into all nine HMOM coagulation sub-vectors.
     *
     * @note Valid only after `CalculateSourceMoments()` has been called.
     * @return CoagulationDetail bundle; all spans have size = `n_equations`.
     */
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

    /**
     * @name Reporter output hooks — MomentMethodReporter extensibility protocol
     *
     * These two template methods make HMOM self-describing with respect to output.
     * `MomentMethodReporter` detects them at compile time via
     * `if constexpr (requires(...))` and calls them with a callback:
     * @code
     *   cb(std::string_view label, double value)
     * @endcode
     * - In **header mode** the reporter supplies a lambda that uses the @p label
     *   to register a new output column.
     * - In **row mode** the reporter supplies a lambda that uses the @p value to
     *   write data.
     * The variant implementation is **identical** for both modes.
     *
     * - `variant_prefix_output` 
     * - `variant_suffix_output` 
     *
     * To add or remove columns: edit only these two methods.
     * `MomentMethodReporter` requires no modification.
     * @{
     */

    /**
     * @brief HMOM-specific prefix columns: bimodal NDF statistics.
     *
     * Emits: np, ss, vs, N0, NL, αL, dp,L, np,L, d63, σ(dp), σ(np),
     * σL(dp), σL(np), gsd(dp), gsd(np), gsdL(dp), gsdL(np).
     *
     * @tparam CB  Callable with signature `void(std::string_view, double)`.
     * @param  cb  Callback invoked once per column.
     */
    template <typename CB> void variant_prefix_output(CB&& cb) const
    {
        cb("np[-]", number_primary_particles());
        cb("ss[m2/#]", soot_mean_surface());
        cb("vs[m3/#]", soot_mean_volume());
        cb("N0[#/m3]", soot_small_number_density());
        cb("NL[#/m3]", soot_large_number_density());
        cb("alphaL[-]", soot_large_fraction());
        cb("dpL[nm]", soot_large_primary_particle_diameter() * 1.e9);
        cb("npL[-]", soot_large_primary_particle_number());
        cb("d63[nm]", soot_d63() * 1.e9);
        cb("sigma_dp[-]", soot_log_geom_std_dev_primary_particle_diameter());
        cb("sigma_np[-]", soot_log_geom_std_dev_primary_particle_number());
        cb("sigma_dp_L[-]", soot_large_log_geom_std_dev_primary_particle_diameter());
        cb("sigma_np_L[-]", soot_large_log_geom_std_dev_primary_particle_number());
        cb("gsd_dp[-]", std::exp(soot_log_geom_std_dev_primary_particle_diameter()));
        cb("gsd_np[-]", std::exp(soot_log_geom_std_dev_primary_particle_number()));
        cb("gsd_dp_L[-]", std::exp(soot_large_log_geom_std_dev_primary_particle_diameter()));
        cb("gsd_np_L[-]", std::exp(soot_large_log_geom_std_dev_primary_particle_number()));

        this->EmitStandardSootOmegaGas(
            cb, pah_index_, index_C2H2_, index_H2_, index_O2_, index_H2O_, index_OH_);
    }

    /**
     * @brief HMOM-specific NDF extra columns: two-node reconstruction parameters.
     *
     * Called by MomentMethodReporter::WriteReconstructedNDF() via
     * `if constexpr (requires(...))` detection — the same extensibility
     * protocol used by variant_prefix_output / variant_suffix_output.
     *
     * Emits seven scalar columns that characterise the HMOM two-node NDF:
     *   - N0, V0, dp0: nucleation-mode density, volume, and sphere-equivalent diameter.
     *   - NL, VL, dpL_mean: large-mode density, mean volume, and sphere-equivalent diameter.
     *   - sigma_ndf: the cosmetic smearing half-width (log-volume) used for visualization.
     *
     * These scalars are the **same for every nu grid point** in the NDF output
     * file — they are node properties, not functions of v.  They are repeated
     * in each row to make every row of the output file self-contained.
     *
     * @tparam CB  Callable with signature `void(std::string_view, double)`.
     * @param  cb  In header mode: uses the label to register the column.
     *             In row mode: uses the value to write data.
     */
    template <typename CB>
    void ndf_extra_output(CB&& cb) const
    {
        const auto   d       = ReconstructedNDFData();
        const double dp0_nm  = std::pow(6. * d.V0 / this->pi_, 1. / 3.) * 1.e9;
        const double dpL_nm  = (d.NL > kSootNumberFloor && d.VL > 0.)
                                   ? std::pow(6. * d.VL / this->pi_, 1. / 3.) * 1.e9
                                   : 0.;
        cb("N0[#/m3]",     d.N0);
        cb("V0[m3/#]",     d.V0);
        cb("dp0[nm]",      dp0_nm);
        cb("NL[#/m3]",     d.NL);
        cb("VL[m3/#]",     d.VL);
        cb("dpL_mean[nm]", dpL_nm);
        cb("sigma_ndf[-]", d.sigma);
    }

    /**
     * @brief HMOM-specific suffix columns: detailed coagulation sub-breakdown.
     *
     * Emits 9 groups × n_equations columns: ScoaTot, ScoaDis, ScoaDisSS,
     * ScoaDisSL, ScoaDisLL, ScoaCon, ScoaConSS, ScoaConSL, ScoaConLL.
     *
     * @tparam CB  Callable with signature `void(std::string_view, double)`.
     * @param  cb  Callback invoked once per column.
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

    /** @} */

    // -- Model switches --------------------------------------------------------

    /** @brief Enable/disable nucleation (0 = off, 1 = standard PAH dimerisation). */
    void SetNucleation(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument(
                "[HMOM] Invalid nucleation model flag. Allowed values: 0, 1.");
        nucleation_model_ = flag;
    }

    /** @brief Enable/disable HACA surface growth (0 = off, 1 = on). */
    void SetSurfaceGrowth(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument(
                "[HMOM] Invalid surface-growth model flag. Allowed values: 0, 1.");
        surface_growth_model_ = flag;
    }

    /** @brief Enable/disable PAH condensation (0 = off, 1 = on). */
    void SetCondensation(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument(
                "[HMOM] Invalid condensation model flag. Allowed values: 0, 1.");
        condensation_model_ = flag;
    }

    /** @brief Enable/disable O2/OH oxidation (0 = off, 1 = on). */
    void SetOxidation(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument(
                "[HMOM] Invalid oxidation model flag. Allowed values: 0, 1.");
        oxidation_model_ = flag;
    }

    /** @brief Enable/disable discrete coagulation (0 = off, 1 = on). */
    void SetCoagulation(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument(
                "[HMOM] Invalid coagulation model flag. Allowed values: 0, 1.");
        coagulation_model_ = flag;
    }

    /** @brief Enable/disable continuum coagulation correction (0 = off, 1 = on). */
    void SetCoagulationContinuous(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument(
                "[HMOM] Invalid continuous-coagulation model flag. Allowed values: 0, 1.");
        coagulation_continuous_model_ = flag;
    }

    /** @brief Select the fractal aggregate diameter model (0 or 1). */
    void SetFractalDiameterModel(int m)
    {
        if (m != 0 && m != 1)
            throw std::invalid_argument(
                "[HMOM] Invalid fractal-diameter model flag. Allowed values: 0, 1.");
        fractal_diameter_model_ = static_cast<FractalDiameterModel>(m);
    }

    /** @brief Select the collision diameter model (1 or 2). */
    void SetCollisionDiameterModel(int m)
    {
        if (m != 1 && m != 2)
            throw std::invalid_argument(
                "[HMOM] Invalid collision-diameter model flag. Allowed values: 1, 2.");
        collision_diameter_model_ = static_cast<CollisionDiameterModel>(m);
    }

    void SetStickingCoefficientModel(std::string_view label);

    void SetStickingCoefficientConstant(double value) noexcept { sticking_coeff_constant_ = value; }

    void SetSurfaceDensity(double value) noexcept { surface_density_ = value; }

    void SetSurfaceDensityCorrectionCoefficient(bool on) noexcept
    {
        surface_density_correction_ = on;
    }

    void SetSurfaceDensityCorrectionCoefficientA1(double v) noexcept { surf_dens_a1_ = v; }

    void SetSurfaceDensityCorrectionCoefficientA2(double v) noexcept { surf_dens_a2_ = v; }

    void SetSurfaceDensityCorrectionCoefficientB1(double v) noexcept { surf_dens_b1_ = v; }

    void SetSurfaceDensityCorrectionCoefficientB2(double v) noexcept { surf_dens_b2_ = v; }

    /// Sets the PAH precursor species by name (must be present in the thermo map).
    void SetPAH(std::string_view name);

    /// Sets the dummy gas-phase species used for mass-fraction closure.
    void SetGasClosureDummySpecies(std::string_view name);

    // -- HACA surface kinetics parameters -------------------------------------
    //
    // Arrhenius parameters for the 5-step HACA mechanism:
    //   R1f/b: H + Csoot-H <=> Csoot* + H2
    //   R2f/b: Csoot* + OH <=> Csoot-OH
    //   R3f/b: Csoot* + H  <=> Csoot-H
    //   R4:    Csoot* + C2H2 => Csoot-H + H
    //   R5:    Csoot* + O2  => 2CO + Csoot*
    //
    // Units: A [cm3, mol, s]; E [kJ/mol] (converted internally to [K]); n [-].

    void SetA1f(double v) noexcept;
    void SetA1b(double v) noexcept;
    void SetA2f(double v) noexcept;
    void SetA2b(double v) noexcept;
    void SetA3f(double v) noexcept;
    void SetA3b(double v) noexcept;
    void SetA4(double v) noexcept;
    void SetA5(double v) noexcept;

    void SetE1f(double kJ) noexcept;
    void SetE1b(double kJ) noexcept;
    void SetE2f(double kJ) noexcept;
    void SetE2b(double kJ) noexcept;
    void SetE3f(double kJ) noexcept;
    void SetE3b(double kJ) noexcept;
    void SetE4(double kJ) noexcept;
    void SetE5(double kJ) noexcept;

    void Setn1f(double v) noexcept;
    void Setn1b(double v) noexcept;
    void Setn2f(double v) noexcept;
    void Setn2b(double v) noexcept;
    void Setn3f(double v) noexcept;
    void Setn3b(double v) noexcept;
    void Setn4(double v) noexcept;
    void Setn5(double v) noexcept;

    void SetEfficiency6(double v) noexcept { eff6_ = v; }

    // -- Model state queries ---------------------------------------------------

    [[nodiscard]] int nucleation_model() const noexcept { return nucleation_model_; }

    [[nodiscard]] int surface_growth_model() const noexcept { return surface_growth_model_; }

    [[nodiscard]] int condensation_model() const noexcept { return condensation_model_; }

    [[nodiscard]] int oxidation_model() const noexcept { return oxidation_model_; }

    [[nodiscard]] int coagulation_model() const noexcept { return coagulation_model_; }

    [[nodiscard]] int continuous_coagulation_model() const noexcept
    {
        return coagulation_continuous_model_;
    }

    [[nodiscard]] double dimerization_rate() const noexcept { return dimerization_rate_; }

    [[nodiscard]] double V0() const noexcept { return V0_; }

    [[nodiscard]] double S0() const noexcept { return S0_; }

private:

    // -- Private computational methods -----------------------------------------

    void MemoryAllocation();
    void Precalculations();
    void ApplyConfig(const Config& cfg); //!< core of SetupFromConfig, without PrintSummary()
    void GetMoments();
    void DimerConcentration();
    void SootKineticConstants();
    void SootNucleationM4();
    void SootSurfaceGrowthM4();
    void SootOxidationM4();
    void SootCondensationM4();
    void SootCoagulationM4();
    void SootCoagulationSmallSmallM4();
    void SootCoagulationSmallLargeM4();
    void SootCoagulationLargeLargeM4();
    void SootCoagulationContinuousM4();
    void SootCoagulationContinuousSmallSmallM4(double lambda);
    void SootCoagulationContinuousSmallLargeM4(double lambda);
    void SootCoagulationContinuousLargeLargeM4(double lambda);
    void CalculateAlphaCoefficient();

    [[nodiscard]] double GetMoment(double i, double j) const noexcept;
    [[nodiscard]] double GetMissingMoment(double i, double j) const noexcept;
    [[nodiscard]] double GetBetaC() const noexcept;
    [[nodiscard]] double PAHDimerizationRate() const noexcept;
    [[nodiscard]] bool HasSoot() const noexcept;
    [[nodiscard]] double SafePowPositive(double x, double a) const noexcept;
    [[nodiscard]] double LogGeomStdDevFromMoments(double M0, double M1, double M2) const noexcept;

private:

    // -- Thermodynamics reference ----------------------------------------------
    const Thermo& thermo_;

    // -- Transported (normalised) moments -------------------------------------
    double M00_normalized_ = 0.; //!< [mol/m3]
    double M10_normalized_ = 0.; //!< [mol/m3]
    double M01_normalized_ = 0.; //!< [mol/m3]
    double N0_normalized_  = 0.; //!< [mol/m3]

    // -- Reconstructed physical moments ---------------------------------------
    double M00_ = 0.; //!< [#/m3]
    double M10_ = 0.; //!< [#]
    double M01_ = 0.; //!< [#/m]
    double N0_  = 0.; //!< [#/m3]   small-particle number density

    // Large-particle mode quantities (reconstructed from M00, M10, M01, N0)
    double NL_   = 0.; //!< large-particle number density [#/m3]
    double NLVL_ = 0.; //!< NL * mean volume of large particles [#]
    double NLSL_ = 0.; //!< NL * mean surface of large particles [#/m]

    // -- Species concentrations [kmol/m3] --------------------------------------
    double conc_OH_ = 0.;
    double conc_H_ = 0.;
    double conc_H2O_ = 0.;
    double conc_H2_ = 0.;
    double conc_C2H2_ = 0.;
    double conc_O2_ = 0.;
    double conc_PAH_ = 0.;
    double conc_DIMER_ = 0.;

    // 0-based species indices (-1 if absent in mechanism)
    int index_H_ = -1;
    int index_OH_ = -1;
    int index_O2_ = -1;
    int index_H2_   = -1;
    int index_H2O_  = -1;
    int index_CO_   = -1; //!< CO — oxidation product (both channels)
    int index_C2H2_ = -1;

    // Mass fractions (needed for some surface rate expressions)
    double mass_fraction_H_  = 0.;
    double mass_fraction_OH_ = 0.;

    // -- PAH (precursor) properties ---------------------------------------------
    std::string pah_species_;
    int pah_index_ = -1;
    double vpah_   = 0.; //!< PAH molecule volume [m3]
    double spah_   = 0.; //!< PAH molecule surface [m2]
    double dpah_   = 0.; //!< PAH molecule diameter [m]
    double mpah_   = 0.; //!< PAH molecule mass [kg]
    double mwpah_  = 0.; //!< PAH molecular weight [kg/kmol]
    double ncpah_  = 0.; //!< number of C atoms per PAH molecule
    double nhpah_  = 0.; //!< number of H atoms per PAH molecule

    // -- Nucleated particle geometry --------------------------------------------
    double V0_  = 0.; //!< nucleated particle volume [m3]
    double S0_  = 0.; //!< nucleated particle surface [m2]
    double VC2_ = 0.; //!< volume of 2 C atoms [m3]

    // -- Dimer properties -------------------------------------------------------
    double dimer_volume_      = 0.;
    double dimer_surface_     = 0.;
    double dimerization_rate_ = 0.; //!< [mol/m3/s]

    // -- Kinetic intermediate quantities ---------------------------------------
    double kox_      = 0.; //!< total oxidation rate constant [1/s]
    double kox_O2_   = 0.; //!< O2 contribution [1/s]
    double kox_OH_   = 0.; //!< OH contribution [1/s]
    double ksg_      = 0.; //!< surface growth rate constant
    double betaN_    = 0.;
    double Cfm_      = 0.;
    double betaN_TV_ = 0.;

    // -- Fractal/collision geometry pre-factors ---------------------------------
    double Av_fractal_ = 0.;
    double As_fractal_ = 0.;
    double K_fractal_ = 0.;
    double D_collisional_ = 0.;
    double Av_collisional_ = 0.;
    double As_collisional_ = 0.;
    double K_collisional_ = 0.;

    // -- Surface density correction ---------------------------------------------
    bool surface_density_correction_ = false;
    double surface_density_          = 1.7e19; //!< [#/m2]
    double surf_dens_a1_ = 0., surf_dens_a2_ = 0.;
    double surf_dens_b1_ = 0., surf_dens_b2_ = 0.;
    double alpha_ = 1.; //!< correction factor α

    // -- Model flags ------------------------------------------------------------
    int nucleation_model_             = 0;
    int condensation_model_           = 0;
    int surface_growth_model_         = 0;
    int oxidation_model_              = 0;
    int coagulation_model_            = 0;
    int coagulation_continuous_model_ = 0;

    FractalDiameterModel fractal_diameter_model_     = FractalDiameterModel::Model1;
    CollisionDiameterModel collision_diameter_model_ = CollisionDiameterModel::Model2;
    StickingModel sticking_model_                    = StickingModel::Constant;
    double sticking_coeff_constant_                  = 2.e-3;

    bool is_debug_mode_          = false; //!< enable verbose diagnostic output
    bool is_simplified_pah_mass_ = false; //!< use Nc*WC instead of full PAH MW

    // -- Per-process source storage (owned by HMOM, not inherited from base) ---
    //
    // Only processes HMOM actually models are declared here.  The compile-time
    // CRTP dispatch in MomentMethodBase::sources_X() detects which _impl()
    // methods exist via `if constexpr (requires ...)` and selects the right
    // implementation (or a zero-span fallback) with no runtime overhead.
    //
    // HMOM models: nucleation, coagulation, condensation, growth, oxidation.
    // HMOM does NOT model: sintering → base class returns zero span automatically.

    MomentVector source_nucleation_   = MomentVector::Zero();
    MomentVector source_coagulation_  = MomentVector::Zero();
    MomentVector source_condensation_ = MomentVector::Zero();
    MomentVector source_growth_       = MomentVector::Zero();
    MomentVector source_oxidation_    = MomentVector::Zero();

    Eigen::VectorXd omega_gas_oxidation_; //!< Oxidation-only gas-phase sources [kg/m³/s].

    // -- HMOM-specific coagulation source breakdown -----------------------------
    //
    // Coagulation in HMOM has discrete and continuous parts.
    // These are HMOM-specific vectors not present in other variants.

    MomentVector source_coagulation_discrete_   = MomentVector::Zero();
    MomentVector source_coagulation_ss_         = MomentVector::Zero();
    MomentVector source_coagulation_sl_         = MomentVector::Zero();
    MomentVector source_coagulation_ll_         = MomentVector::Zero();
    MomentVector source_coagulation_continuous_ = MomentVector::Zero();
    MomentVector source_coagulation_cont_ss_    = MomentVector::Zero();
    MomentVector source_coagulation_cont_sl_    = MomentVector::Zero();
    MomentVector source_coagulation_cont_ll_    = MomentVector::Zero();
    MomentVector source_coagulation_all_        = MomentVector::Zero();

    // -- Initial moments cache --------------------------------------------------
    MomentVector initial_moments_cache_ = MomentVector::Zero();

    // -- HACA kinetics parameters -----------------------------------------------
    // Stored in [1/s], [cm3/mol/s] as appropriate; conversions in Set* methods.
    double A1f_ = 0., n1f_ = 0., E1f_ = 0., A1b_ = 0., n1b_ = 0., E1b_ = 0.;
    double A2f_ = 0., n2f_ = 0., E2f_ = 0., A2b_ = 0., n2b_ = 0., E2b_ = 0.;
    double A3f_ = 0., n3f_ = 0., E3f_ = 0., A3b_ = 0., n3b_ = 0., E3b_ = 0.;
    double A4_ = 0., n4_ = 0., E4_ = 0.;
    double A5_ = 0., n5_ = 0., E5_ = 0.;
    double eff6_ = 0.;

    // -- Numerical floors (constexpr — zero cost) -------------------------------
    static constexpr double kTinyNumberDensity = 1.e-30; //!< [#/m3]
    static constexpr double kSootNumberFloor   = 1.e3;   //!< [#/m3]
    static constexpr double kSootVolumeFloor   = 1.e-40; //!< [-]
    static constexpr double kSootSurfaceFloor  = 1.e-30; //!< [m2/m3]
    static constexpr double kMomentEps         = 1.e-300;

    // -- NDF visualization constant --------------------------------------------
    //
    /// Log-volume standard deviation used to smear the two HMOM delta-function
    /// nodes into narrow log-normal kernels for diagnostic visualization.
    ///
    /// @par Value choice
    /// σ_ln(v) = 0.5 corresponds to a geometric standard deviation in primary-
    /// particle diameter of σ_g,dp = exp(0.5/3) ≈ 1.18, i.e. roughly ±18%
    /// variation in dp.  On a six-decade log-volume axis this produces a peak
    /// FWHM of ~0.51 decades (~8 % of the axis), which is clearly visible but
    /// easily identifiable as a spike rather than a broad mode.
    ///
    /// @par Important caveat
    /// This constant has **no physical meaning**.  HMOM carries no information
    /// about within-mode polydispersity.  The only physically meaningful
    /// quantities are the node weights (N0, NL) and positions (V0, VL).
    /// The smearing is a pure visualization aid to prevent the two delta masses
    /// from being invisible on a continuous NDF plot.
    static constexpr double kNDFSmearSigmaLnV = 0.5;
};

} // namespace MOM

#if defined(MOM_USE_DICTIONARY)
// Grammar header — pulls in OpenSMOKE++ internals.  Included only when
// MOM_USE_DICTIONARY is defined so that the core library stays dependency-free.
// Must appear before HMOM.tpp, where ParseConfig<> is defined.
#include "HMOM_Grammar.h"
#endif

#if !defined(MOM_COMPILED_LIBRARY)
#include "HMOM.tpp"
#else
namespace MOM
{
extern template class HMOM<BasicThermoData>;
}
#endif
