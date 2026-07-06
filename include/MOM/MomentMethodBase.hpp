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
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "Eigen/Dense"

#include "ProcessFlags.hpp"

namespace MOM
{

// ============================================================================
// MomentMethodBase<Derived, NEq>
// ============================================================================
//
// CRTP base class providing shared state and common implementations for all
// moment method classes. Zero virtual functions — all dispatch is resolved
// statically at compile time via the Curiously Recurring Template Pattern.
//
// Design rationale
// ----------------
// * NEq is a compile-time constant, so all source vectors are fixed-size
//   Eigen::Matrix<double, NEq, 1> objects. These live on the stack (for
//   NEq <= ~16), are fully unrolled by the compiler, and are SIMD-vectorised
//   without any heap allocation in the computational hot loop.
//
// * Physical sub-process source vectors (nucleation, coagulation, …) are
//   owned by the Derived class for each process it models, NOT stored here.
//   The base class provides a compile-time zero-span fallback via CRTP
//   detection: `if constexpr (requires(const Derived& d) { d.sources_X_impl(); })`
//   forwards to the derived implementation; otherwise it returns a view of
//   a static constexpr zero array. Zero overhead — the branch is resolved
//   at compile time and the non-taken arm is never instantiated.
//
// * The omega_gas_ vector (gas-phase source terms) is an Eigen::VectorXd
//   sized once during setup (n_species elements). It is NOT re-allocated
//   in the hot loop. Eigen provides cache-line alignment and the same
//   setZero() / .data() API as the fixed-size source vectors.
//
// * The Planck absorption coefficient implementation is shared here; MetalOxide
//   uses PlanckCoeffModel::None and gets 0.0 from planck_coefficient().
//
// Template parameters
// -------------------
//   Derived  — The concrete subclass (HMOM, BrookesMoss, ThreeEquations, MetalOxide)
//   NEq      — Number of transported moment equations (compile-time constant)
// ============================================================================

/**
 * @class MomentMethodBase
 * @brief CRTP base class providing shared state and common implementations for all
 *        Method of Moments particle transport models.
 *
 * @tparam Derived  The concrete subclass (HMOM, BrookesMoss, ThreeEquations, MetalOxide).
 *                  The CRTP pattern gives the base class access to the derived
 *                  implementation at compile time with zero overhead — no virtual
 *                  functions, no vtable, no indirect branches in the hot loop.
 * @tparam NEq      Number of transported moment equations (compile-time constant).
 *                  Typical values: 2 (BrookesMoss), 3 (ThreeEquations, MetalOxide), 4 (HMOM).
 *
 * @par Design rationale
 *
 * - **Zero virtual functions.**  All dispatch is resolved statically at compile
 *   time via CRTP.  The compiler can inline and SIMD-vectorise the full per-cell
 *   update path without any indirect branches.
 *
 * - **Fixed-size source vectors.**  Because `NEq` is a compile-time constant, all
 *   source vectors are `Eigen::Matrix<double, NEq, 1>` objects.  For `NEq <= ~16`
 *   these live entirely on the stack and are fully unrolled by the compiler.  No
 *   heap allocation occurs in the computational hot loop.
 *
 * - **Per-process source vectors owned by Derived.**  Only the total source vector
 *   (`source_all_`) is stored in the base.  Nucleation, coagulation, condensation,
 *   growth, oxidation, and sintering source vectors are owned by the concrete
 *   subclass, which only declares the ones it actually models.  The CRTP dispatch
 *   methods `sources_X()` detect at compile time whether `Derived` has declared
 *   `sources_X_impl()`.  If it has, the call is forwarded at zero cost.  If it
 *   has not, a static `constexpr` zero array is returned — no storage allocated,
 *   no runtime branch.
 *
 * - **Gas-phase source terms.**  `omega_gas_` is an `Eigen::VectorXd` sized once
 *   during setup (`n_species` elements).  It is not re-allocated in the hot loop.
 *
 * - **Shared physical constants.**  Boltzmann constant, Avogadro number, ideal gas
 *   constant, carbon atomic weight, π, and √2 are declared as `static constexpr`
 *   members carrying full double precision (CODATA 2018 / IUPAC 2021 values).
 *
 * @par CRTP extension points for per-process sources
 *
 * A derived class signals support for physical process X by declaring:
 * @code
 *   [[nodiscard, gnu::always_inline]]
 *   std::span<const double> sources_X_impl() const noexcept;
 * @endcode
 * The `_impl()` suffix avoids name hiding and makes the extension point explicit.
 * The `[[gnu::always_inline]]` attribute guarantees the two-level call chain
 * `sources_X() → derived().sources_X_impl()` is collapsed to a direct memory
 * access at **all** optimisation levels, including debug (`-O0`) and profiling
 * (`-Og`) builds.
 *
 * @par Thread safety
 * **One instance per thread — never shared across threads.**
 *
 * Each concrete instance holds mutable internal state that is overwritten on
 * every call to `CalculateSourceMoments()` (moment values, intermediate species
 * concentrations, source vectors).  Concurrent access to the same instance from
 * multiple threads is undefined behaviour.
 *
 * For MPI+OpenMP or pure-OpenMP CFD solvers:
 * - Allocate **one model instance per OpenMP thread** (e.g. in a `#pragma omp threadprivate`
 *   variable or via an `std::vector<Model>` of size `omp_get_max_threads()`).
 * - The shared `Thermo` reference may be safely read from multiple threads
 *   simultaneously, provided no thread writes to it during the parallel region.
 * - `MomentMethodReporter` and `OutputFileColumns` carry the same constraint —
 *   use one reporter instance per thread, or serialise output behind a mutex.
 */
template <class Derived, unsigned NEq> class MomentMethodBase
{
public:

    // -- Compile-time interface ---------------------------------------------

    /** @brief Number of transported equations. Satisfies the MomentMethod concept. */
    static constexpr unsigned n_equations = NEq;

    /** @brief Eigen column-vector type for a moment vector; used by derived classes. */
    using MomentVector = Eigen::Matrix<double, static_cast<int>(NEq), 1>;

    // -- Common setters -----------------------------------------------------

    /**
     * @brief Sets the particle Schmidt number used for diffusion coefficient scaling.
     * @param value Schmidt number [-] (default: 0.7).
     */
    void SetSchmidtNumber(double value) noexcept { schmidt_number_ = value; }

    /**
     * @brief Sets the particle material density.
     * @param value Density [kg/m3] (default: 1800 for soot; MetalOxide overrides to 3900).
     */
    void SetParticleDensity(double value) noexcept { rho_particle_ = value; }

    /**
     * @brief Sets the mixture dynamic viscosity.
     *
     * Must be called before each `CalculateSourceMoments()` call if viscosity
     * changes between time steps (e.g. in a variable-property flow).
     * @param mu Dynamic viscosity [kg/m/s].
     */
    void SetViscosity(double mu) noexcept { mu_ = mu; }

    /**
     * @brief Enables or disables gas-phase precursor consumption.
     * @param flag If `true` (default), `omega_gas_` is computed and the precursor
     *             source term is added to the gas-phase chemistry residual.  If
     *             `false`, the gas source vector is cleared once and then left
     *             untouched by the per-cell moment-source reset.
     */
    void SetGasConsumption(bool flag) noexcept
    {
        gas_consumption_ = flag;
        if (!gas_consumption_)
            omega_gas_.setZero();
    }

    /**
     * @brief Enables or disables particle contribution to radiative heat transfer.
     * @param flag If `true`, the particle Planck absorption coefficient is non-zero
     *             and should be passed to the radiation solver.
     */
    void SetRadiativeHeatTransfer(bool flag) noexcept { radiative_heat_transfer_ = flag; }

    /**
     * @brief Sets the thermophoretic drift model by integer flag.
     * @param flag 0 = off, 1 = standard (drift encoded in effective diffusion coefficient).
     */
    void SetThermophoreticModel(int flag)
    {
        if (flag != 0 && flag != 1)
            throw std::invalid_argument(
                "[MomentMethodBase] Invalid thermophoretic model flag. Allowed values: 0, 1.");
        thermophoretic_model_ = static_cast<ThermophoreticModel>(flag);
    }

    /** @brief Sets the thermophoretic drift model by strongly-typed enum. */
    void SetThermophoreticModel(ThermophoreticModel m) noexcept { thermophoretic_model_ = m; }

    /**
     * @brief Selects the Planck mean absorption coefficient correlation.
     * @param model One of `PlanckCoeffModel::Smooke`, `Kent`, `Sazhin`, or `None`.
     */
    void SetPlanckAbsorptionCoefficient(PlanckCoeffModel model) noexcept { planck_model_ = model; }

    /**
     * @brief Selects the Planck mean absorption coefficient correlation by label string.
     * @param label Case-insensitive label: `"Smooke"`, `"Kent"`, `"Sazhin"`, or `"None"`.
     */
    void SetPlanckAbsorptionCoefficient(std::string_view label) noexcept
    {
        planck_model_ = PlanckCoeffModelFromString(label);
    }

    // -- Common getters -----------------------------------------------------

    /** @brief Returns the number of transported moment equations (= NEq). */
    [[nodiscard]] constexpr unsigned n_moments() const noexcept { return NEq; }

    /** @brief Returns `true` if the model has been fully configured and is active. */
    [[nodiscard]] bool is_active() const noexcept { return is_active_; }

    /** @brief Returns `true` if gas-phase precursor consumption is enabled. */
    [[nodiscard]] bool gas_consumption() const noexcept { return gas_consumption_; }

    /** @brief Returns the particle Schmidt number [-]. */
    [[nodiscard]] double schmidt_number() const noexcept { return schmidt_number_; }

    /** @brief Returns the particle material density [kg/m3]. */
    [[nodiscard]] double particle_density() const noexcept { return rho_particle_; }

    /** @brief Returns `true` if the particle Planck coefficient is non-zero. */
    [[nodiscard]] bool radiative_heat_transfer() const noexcept { return radiative_heat_transfer_; }

    /**
     * @brief Returns the thermophoretic model flag as `int` (for CFD solver interop).
     * @return 0 = off, 1 = standard.
     */
    [[nodiscard]] int thermophoretic_model() const noexcept
    {
        return static_cast<int>(thermophoretic_model_);
    }

    /** @brief Returns `true` if a dummy closure species has been configured. */
    [[nodiscard]] bool is_closure_dummy_species() const noexcept
    {
        return is_closure_dummy_species_;
    }

    /** @brief Returns the name of the dummy closure species. */
    [[nodiscard]] const std::string& closure_dummy_species() const noexcept
    {
        return closure_dummy_species_;
    }

    /** @brief Returns the 0-based index of the dummy closure species (−1 if inactive). */
    [[nodiscard]] int closure_dummy_index() const noexcept { return closure_dummy_index_; }

    // -- Source term access — span-based for zero-copy CFD interop ----------

    /**
     * @brief Total source vector summed over all active processes.
     *
     * Returns a non-owning, zero-copy view into the internal fixed-size storage.
     * The span remains valid as long as the model object is alive and
     * `CalculateSourceMoments()` has not been called again.
     *
     * @return Span of size `n_equations` [mol/m3/s or variant-specific units].
     */
    [[nodiscard]] std::span<const double> sources() const noexcept
    {
        return {source_all_.data(), NEq};
    }

    /**
     * @name Per-process source span getters — CRTP dispatch with zero fallback
     *
     * Each method checks at compile time (via `if constexpr (requires(...))`) whether
     * the concrete `Derived` class has declared a `sources_X_impl()` extension point.
     *
     * - If **yes**: the call is forwarded to `derived().sources_X_impl()` at zero
     *   overhead (the compiler sees through the two-level call chain completely).
     * - If **no**: the base returns `{kZeroData, NEq}` — a view of a `static constexpr`
     *   zero array — with no storage allocation and no runtime branch.
     *
     * @note `[[gnu::always_inline]]` guarantees this inlining even at `-O0`/`-Og`,
     *       making profiling and debug builds faithfully represent the release-build
     *       call graph.
     * @{
     */

    /** @brief Nucleation source terms [mol/m3/s]. Zero span if Derived does not model nucleation. */
    [[nodiscard, gnu::always_inline]] std::span<const double> sources_nucleation() const noexcept
    {
        if constexpr (requires(const Derived& d) { d.sources_nucleation_impl(); })
            return derived().sources_nucleation_impl();
        else
            return {kZeroData, NEq};
    }

    /** @brief Coagulation source terms. Zero span if Derived does not model coagulation. */
    [[nodiscard, gnu::always_inline]] std::span<const double> sources_coagulation() const noexcept
    {
        if constexpr (requires(const Derived& d) { d.sources_coagulation_impl(); })
            return derived().sources_coagulation_impl();
        else
            return {kZeroData, NEq};
    }

    /** @brief Condensation source terms. Zero span if Derived does not model condensation. */
    [[nodiscard, gnu::always_inline]] std::span<const double> sources_condensation() const noexcept
    {
        if constexpr (requires(const Derived& d) { d.sources_condensation_impl(); })
            return derived().sources_condensation_impl();
        else
            return {kZeroData, NEq};
    }

    /** @brief Surface growth source terms. Zero span if Derived does not model surface growth. */
    [[nodiscard, gnu::always_inline]] std::span<const double> sources_growth() const noexcept
    {
        if constexpr (requires(const Derived& d) { d.sources_growth_impl(); })
            return derived().sources_growth_impl();
        else
            return {kZeroData, NEq};
    }

    /** @brief Oxidation source terms. Zero span if Derived does not model oxidation. */
    [[nodiscard, gnu::always_inline]] std::span<const double> sources_oxidation() const noexcept
    {
        if constexpr (requires(const Derived& d) { d.sources_oxidation_impl(); })
            return derived().sources_oxidation_impl();
        else
            return {kZeroData, NEq};
    }

    /** @brief Sintering source terms. Zero span if Derived does not model sintering. */
    [[nodiscard, gnu::always_inline]] std::span<const double> sources_sintering() const noexcept
    {
        if constexpr (requires(const Derived& d) { d.sources_sintering_impl(); })
            return derived().sources_sintering_impl();
        else
            return {kZeroData, NEq};
    }

    /** @} */

    /**
     * @brief Gas-phase species consumption source terms.
     *
     * Non-owning view into `omega_gas_` (sized to `n_species` during setup).
     * Non-zero only if `GasConsumption()` is `true` and the model is active.
     *
     * @return Span of size `n_species` [kg/m3/s].
     */
    [[nodiscard]] std::span<const double> omega_gas() const noexcept
    {
        return {omega_gas_.data(), static_cast<std::size_t>(omega_gas_.size())};
    }

    // -- Radiative properties -----------------------------------------------

    /**
     * @brief Planck mean absorption coefficient of the particle phase.
     *
     * Returns 0.0 when `planck_model_ == PlanckCoeffModel::None` (e.g. MetalOxide).
     * The three empirical correlations (Smooke, Kent, Sazhin) are implemented in
     * `MomentMethodBase.tpp` and selected at runtime via a `switch`.
     *
     * @param T  Gas temperature [K].
     * @param fv Particle volume fraction [-].
     * @return   Planck mean absorption coefficient [1/m].
     */
    [[nodiscard]] double planck_coefficient(double T, double fv) const noexcept;

protected:

    // -- Protected constructor — only accessible through derived classes -----

    MomentMethodBase()  = default;
    ~MomentMethodBase() = default;

    /**
     * @brief Non-copyable: moment method objects hold external thermo references
     *        and large internal state — copy semantics would be expensive and fragile.
     */
    MomentMethodBase(const MomentMethodBase&)            = delete;
    MomentMethodBase& operator=(const MomentMethodBase&) = delete;

    /** @brief Move construction is allowed (e.g. for placement in std::variant). */
    MomentMethodBase(MomentMethodBase&&)            = default;
    MomentMethodBase& operator=(MomentMethodBase&&) = default;

    // -- Shared thermodynamic state -----------------------------------------
    // Updated by each concrete class's SetStatus() implementation.

    double T_    = 0.; //!< Gas temperature [K].
    double P_Pa_ = 0.; //!< Gas pressure [Pa].
    double rho_  = 0.; //!< Gas mixture density [kg/m3].
    double MW_   = 0.; //!< Gas mixture molecular weight [kg/kmol].
    double mu_   = 0.; //!< Gas dynamic viscosity [kg/m/s].

    // -- Common control flags -----------------------------------------------

    bool is_active_                = false; //!< True after successful SetupFromConfig() or Set*() setup.
    bool gas_consumption_          = true;  //!< True if omega_gas_ is computed.
    bool radiative_heat_transfer_  = false; //!< True if planck_coefficient() > 0.
    bool is_closure_dummy_species_ = false; //!< True if a closure dummy species is set.

    double schmidt_number_ = 0.7;    //!< Particle Schmidt number [-].
    double rho_particle_   = 1800.;  //!< Particle material density [kg/m3]; MetalOxide overrides to 3900.

    ThermophoreticModel thermophoretic_model_ = ThermophoreticModel::Off;
    PlanckCoeffModel planck_model_            = PlanckCoeffModel::Smooke;

    std::string closure_dummy_species_;      //!< Name of the dummy closure species (empty if inactive).
    int closure_dummy_index_ = -1;           //!< 0-based thermo-map index of the dummy species (−1 if inactive).

    // -- Fixed-size source storage (stack-allocated) ------------------------
    //
    // Only the TOTAL source vector is stored here; per-process vectors are
    // owned by the Derived class (see class-level @par Design rationale).

    MomentVector source_all_ = MomentVector::Zero(); //!< Sum of all active process source terms.

    /**
     * @brief Compile-time zero array of length `NEq`.
     *
     * Used by the `sources_X()` CRTP dispatch methods as the fallback span when
     * `Derived` does not declare `sources_X_impl()`.  Being `constexpr`, the array
     * is placed in read-only data — no stack or heap allocation occurs.
     */
    static constexpr double kZeroData[NEq] = {};

    /**
     * @brief Gas-phase species source terms [kg/m3/s].
     *
     * Sized to `n_species` during `MemoryAllocation()` (called once at setup).
     * Never re-allocated in the hot loop.  `Eigen::VectorXd` provides aligned
     * storage and a uniform API (`setZero()`, `.data()`, `.sum()`) consistent
     * with the fixed-size `MomentVector` members.
     */
    Eigen::VectorXd omega_gas_;

    // -- Helpers for zero-initialising source vectors between time steps -----

    /**
     * @brief Zeroes the base-class-owned source accumulators.
     *
     * Each `Derived::CalculateSourceMoments()` implementation must call this
     * first, then zero its own per-process vectors (`source_nucleation_`, etc.)
     * before accumulating new source terms.  This deliberately does not clear
     * `omega_gas_`; gas source vectors can be large and are reset only when gas
     * consumption is enabled.
     */
    void ZeroSources() noexcept
    {
        source_all_.setZero();
    }

    // -- Safe gas-source accessor (for use in variant output hooks) ----------

    /**
     * @brief Safe accessor for a single gas-phase source term by species index.
     *
     * Returns `omega_gas_[idx]` when @p idx is a valid (non-negative,
     * in-range) species index, and 0.0 otherwise.  Specifically:
     * - Returns 0.0 if @p idx < 0.  The convention throughout the library is
     *   that species absent in the user's mechanism are assigned index −1 by
     *   `ThermoMap::IndexOfSpecies`.  Treating absent-species output as zero
     *   is correct: no contribution means no source term.
     * - Returns 0.0 if `idx >= omega_gas_.size()`.  This cannot occur after a
     *   correct `MemoryAllocation()` call (which sizes `omega_gas_` to
     *   `n_species`), but the guard makes the accessor unconditionally safe
     *   even in partially-initialised states.
     * - Returns 0.0 if `omega_gas_` is empty (not yet allocated).
     *
     * @par Intended use
     * Call this inside `variant_prefix_output` / `variant_suffix_output` hooks
     * instead of direct `omega_gas_[idx]` subscripting.  This prevents undefined
     * behaviour if a species listed in the output hook (e.g. OH, H2) is absent
     * from the user's reaction mechanism, which is a perfectly valid configuration.
     *
     * BrookesMoss and MetalOxide previously defined equivalent local lambdas
     * inside their output hooks; this method is the canonical, documented
     * replacement for that pattern.
     *
     * @param idx  Species index as returned by `ThermoMap::IndexOfSpecies`.
     *             −1 signals "species not present in mechanism".
     * @return     Gas-phase source term [kg/m³/s], or 0.0 if unavailable.
     */
    [[nodiscard]] double safe_omega_gas(int idx) const noexcept
    {
        if (idx < 0 || static_cast<Eigen::Index>(idx) >= omega_gas_.size())
            return 0.;
        return omega_gas_[idx];
    }

    // -- CRTP down-cast helpers ---------------------------------------------

    /** @brief Mutable CRTP down-cast — used internally by setters in derived classes. */
    [[nodiscard]] Derived& derived() noexcept { return static_cast<Derived&>(*this); }

    /** @brief Const CRTP down-cast — used by `sources_X()` dispatch and getters. */
    [[nodiscard]] const Derived& derived() const noexcept
    {
        return static_cast<const Derived&>(*this);
    }

    // -- Mathematical constants (from <numbers>, full double precision) --------
    //
    // std::numbers::pi_v<double> and sqrt2_v<double> carry the maximum
    // representable precision for double — no manual digit counting required.

    static constexpr double pi_    = std::numbers::pi_v<double>;    //!< π (full double precision).
    static constexpr double sqrt2_ = std::numbers::sqrt2_v<double>; //!< √2 (full double precision).

    // -- Physical constants (CODATA 2018 / IUPAC 2021 exact values, SI) ----
    //
    // kB and Nav are exact by the SI 2019 redefinition.
    // Rgas = kB * Nav (exact). WC is the IUPAC 2021 standard atomic weight.

    static constexpr double kB_       = 1.380649e-23;     //!< Boltzmann constant [J/K].
    static constexpr double Nav_mol_  = 6.02214076e23;    //!< Avogadro number [#/mol].
    static constexpr double Nav_kmol_ = 6.02214076e26;    //!< Avogadro number [#/kmol].
    static constexpr double Rgas_     = 8314.46261815324; //!< Ideal gas constant [J/kmol/K].
    static constexpr double WC_       = 12.011;           //!< Carbon standard atomic weight [kg/kmol].

private:

    // -- Planck coefficient model implementations ---------------------------
    // Implemented in MomentMethodBase.tpp; inlined via the planck_coefficient()
    // switch-dispatch in the header so that the compiler can eliminate the call
    // entirely when PlanckCoeffModel::None is used (MetalOxide).

    /** @brief Smooke et al. (1988) Planck mean absorption coefficient [1/m]. */
    [[nodiscard]] double PlanckSmooke(double T, double fv) const noexcept;

    /** @brief Kent & Honnery (1990) Planck mean absorption coefficient [1/m]. */
    [[nodiscard]] double PlanckKent(double T, double fv) const noexcept;

    /** @brief Sazhin (1994) Planck mean absorption coefficient [1/m]. */
    [[nodiscard]] double PlanckSazhin(double T, double fv) const noexcept;
};

/**
 * @brief `planck_coefficient` is kept in-header to allow full inlining.
 *
 * The `switch` on `planck_model_` is a compile-time-deducible constant in most
 * CFD configurations (set once at setup, never changed per cell), so the
 * compiler can eliminate the branch entirely and inline the chosen correlation.
 * The three correlation bodies are in `MomentMethodBase.tpp`.
 */

template <class Derived, unsigned NEq>
inline double MomentMethodBase<Derived, NEq>::planck_coefficient(double T, double fv) const noexcept
{
    switch (planck_model_)
    {
        case PlanckCoeffModel::Smooke:
            return PlanckSmooke(T, fv);
        case PlanckCoeffModel::Kent:
            return PlanckKent(T, fv);
        case PlanckCoeffModel::Sazhin:
            return PlanckSazhin(T, fv);
        case PlanckCoeffModel::None:
            return 0.;
        default:
            return 0.;
    }
}

} // namespace MOM

// MomentMethodBase is only instantiated via its Derived classes.
// The explicit template instantiations in src/*.cpp implicitly cover the base.
// No extern template declarations needed here.
#if !defined(MOM_COMPILED_LIBRARY)
#include "MomentMethodBase.tpp"
#endif
