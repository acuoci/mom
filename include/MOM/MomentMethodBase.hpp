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

#include <algorithm>
#include <array>
#include <cmath>
#include <numbers>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
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
 * every call to `ComputeSources()` (moment values, intermediate species
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
     * Must be called before each `ComputeSources()` call if viscosity
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
     * @brief Returns the thermophoretic model flag as a strongly-typed enum.
     * @return `ThermophoreticModel::Off` (0) or `ThermophoreticModel::Standard` (1).
     */
    [[nodiscard]] ThermophoreticModel thermophoretic_model() const noexcept
    {
        return thermophoretic_model_;
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
     * `ComputeSources()` has not been called again.
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

    /**
     * @name Per-process instance activation flags — CRTP dispatch with zero fallback
     *
     * Each method returns the strongly-typed process enum class (e.g. `NucleationModel`,
     * `OxidationModel`) configured by the user for the corresponding physical process
     * IN THIS INSTANCE.  The query detects at compile time whether `Derived` exposes
     * the matching getter; if not, `XxxModel::Off` is returned.
     *
     * @par This answers the INSTANCE question: "is process X currently enabled?"
     * The user might set `nucleation_model = 0` in the input file, causing
     * `model_nucleation()` to return `NucleationModel::Off` even for a type that is
     * structurally capable of nucleation.  To ask the TYPE question ("can this model
     * type EVER produce nucleation sources?"), use `capability_nucleation()` instead
     * — see below.
     *
     * @par Naming convention
     * - `model_X()` (base dispatcher, this group) → reads instance flag, returns enum class
     * - `nucleation_model()` / `oxidation_model()` / … (derived getter) → same enum class
     * - `IsActive(model_X())` → convenience bool predicate (see `ProcessFlags.hpp`)
     * @{
     */

    /** @brief Returns the nucleation model flag (`NucleationModel::Off` if not modelled). */
    [[nodiscard, gnu::always_inline]] NucleationModel model_nucleation() const noexcept
    {
        if constexpr (requires(const Derived& d) { d.nucleation_model(); })
            return derived().nucleation_model();
        else
            return NucleationModel::Off;
    }

    /** @brief Returns the surface-growth model flag (`SurfaceGrowthModel::Off` if not modelled). */
    [[nodiscard, gnu::always_inline]] SurfaceGrowthModel model_growth() const noexcept
    {
        if constexpr (requires(const Derived& d) { d.surface_growth_model(); })
            return derived().surface_growth_model();
        else
            return SurfaceGrowthModel::Off;
    }

    /** @brief Returns the coagulation model flag (`CoagulationModel::Off` if not modelled). */
    [[nodiscard, gnu::always_inline]] CoagulationModel model_coagulation() const noexcept
    {
        if constexpr (requires(const Derived& d) { d.coagulation_model(); })
            return derived().coagulation_model();
        else
            return CoagulationModel::Off;
    }

    /** @brief Returns the condensation model flag (`CondensationModel::Off` if not modelled). */
    [[nodiscard, gnu::always_inline]] CondensationModel model_condensation() const noexcept
    {
        if constexpr (requires(const Derived& d) { d.condensation_model(); })
            return derived().condensation_model();
        else
            return CondensationModel::Off;
    }

    /** @brief Returns the oxidation model flag (`OxidationModel::Off` if not modelled). */
    [[nodiscard, gnu::always_inline]] OxidationModel model_oxidation() const noexcept
    {
        if constexpr (requires(const Derived& d) { d.oxidation_model(); })
            return derived().oxidation_model();
        else
            return OxidationModel::Off;
    }

    /** @brief Returns the sintering model flag (`SinteringModel::Off`; non-Off for MetalOxide only). */
    [[nodiscard, gnu::always_inline]] SinteringModel model_sintering() const noexcept
    {
        if constexpr (requires(const Derived& d) { d.sintering_model(); })
            return derived().sintering_model();
        else
            return SinteringModel::Off;
    }

    /** @} */

    /**
     * @name Per-process TYPE capability accessors — always `constexpr`, never reads instance state
     *
     * These methods answer the STRUCTURAL question: "does this model TYPE implement
     * process X?" — not "is process X currently enabled in this instance?".
     *
     * @par This answers the TYPE question: "can this model type EVER produce X sources?"
     * The value is a `constexpr bool` derived from the same `requires(...)` check
     * used by `sources_X()`.  It is resolved entirely at compile time for each
     * concrete variant; the `static` keyword makes the type-level nature explicit.
     *
     * @par Comparison with instance activation (model_X())
     * @code
     *   // Type capability — always the same for a given model type:
     *   constexpr bool can_oxidise = mm.capability_oxidation(); // true for HMOM/3Eq/BM
     *
     *   // Instance activation — depends on the user's input configuration:
     *   OxidationModel oxi = mm.model_oxidation(); // Off if user wrote oxidation_model 0
     *
     *   // Use capability for structural decisions (output columns, static_assert):
     *   if constexpr (mm.capability_oxidation())
     *       register_oxidation_columns();           // compile-time branch
     *
     *   // Use activation for runtime decisions (should we do operator splitting?):
     *   if (IsActive(mm.model_oxidation()))
     *       apply_operator_splitting();             // runtime branch
     * @endcode
     *
     * @par Capability matrix (as of 2026-07)
     * | Method                    | HMOM | BrookesMoss | ThreeEq | MetalOxide |
     * |---------------------------|------|-------------|---------|------------|
     * | `capability_nucleation()` |  Y   |      Y      |    Y    |     Y      |
     * | `capability_growth()`     |  Y   |      Y      |    Y    |     N      |
     * | `capability_coagulation()`|  Y   |      Y      |    Y    |     Y      |
     * | `capability_condensation()`|  Y  |      N      |    Y    |     Y      |
     * | `capability_oxidation()`  |  Y   |      Y      |    Y    |     N      |
     * | `capability_sintering()`  |  N   |      N      |    N    |     Y      |
     *
     * @note This mirrors the `ModelsNucleation<Derived>` / `ModelsOxidation<Derived>`
     *       process-capability concepts in `MomentMethodConcept.hpp`.  Those concepts
     *       are the recommended way to ask the TYPE question when the concrete type is
     *       statically known; `capability_X()` is the way to ask it when you only have
     *       an instance (or an `AnyMomentMethod` wrapper that hides the type).
     * @{
     */

    /** @brief `true` if this model TYPE can compute nucleation source terms. */
    [[nodiscard]] static constexpr bool capability_nucleation() noexcept
    {
        return requires(const Derived& d) { d.sources_nucleation_impl(); };
    }

    /** @brief `true` if this model TYPE can compute surface-growth source terms. */
    [[nodiscard]] static constexpr bool capability_growth() noexcept
    {
        return requires(const Derived& d) { d.sources_growth_impl(); };
    }

    /** @brief `true` if this model TYPE can compute coagulation source terms. */
    [[nodiscard]] static constexpr bool capability_coagulation() noexcept
    {
        return requires(const Derived& d) { d.sources_coagulation_impl(); };
    }

    /** @brief `true` if this model TYPE can compute condensation source terms. */
    [[nodiscard]] static constexpr bool capability_condensation() noexcept
    {
        return requires(const Derived& d) { d.sources_condensation_impl(); };
    }

    /** @brief `true` if this model TYPE can compute oxidation source terms. */
    [[nodiscard]] static constexpr bool capability_oxidation() noexcept
    {
        return requires(const Derived& d) { d.sources_oxidation_impl(); };
    }

    /** @brief `true` if this model TYPE can compute sintering source terms (MetalOxide only). */
    [[nodiscard]] static constexpr bool capability_sintering() noexcept
    {
        return requires(const Derived& d) { d.sources_sintering_impl(); };
    }

    /** @} */

    /**
     * @brief Oxidation-only gas-phase source terms [kg/m³/s].
     *
     * Returns the subset of `omega_gas_` due exclusively to soot oxidation
     * (O2/OH surface attack).  Non-oxidation contributions (nucleation, surface
     * growth, condensation) are excluded.  Used by the operator-splitting API
     * (GetOmegaGasOxidation / FillOmegaGasWithoutOxidation) to remove the stiff
     * oxidation eigenvalue from the ODE system and integrate it analytically.
     *
     * Returns an empty span for models without oxidation gas coupling (e.g. TiO2).
     */
    [[nodiscard]] std::span<const double> omega_gas_oxidation() const noexcept
    {
        if constexpr (requires(const Derived& d) { d.omega_gas_oxidation_impl(); })
            return derived().omega_gas_oxidation_impl();
        else
            return {};   // empty span — model has no oxidation gas coupling
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

    // -- Operator-splitting helpers (used by Splitting.hpp free functions) -----

    /**
     * @brief Fills the internal `source_no_oxidation_` cache and returns a span.
     *
     * Computes `source_all_[i] − source_oxidation_[i]` for each moment equation
     * and stores the result in the `mutable source_no_oxidation_` cache.  Returns
     * a zero-copy span into that cache.
     *
     * Called by `MOM::GetSourcesWithoutOxidation()` in `Splitting.hpp`.  The free
     * function wraps this via `std::visit`; access to the `protected` cache from a
     * non-member lambda would violate C++ access control, so the computation lives
     * here instead.
     *
     * @pre  `ComputeSources()` must have been called at the current cell state.
     * @return Span of size `n_equations`; valid until next `ComputeSources()` or
     *         next call to this function — whichever comes first.
     */
    [[nodiscard]] std::span<const double>
    sources_without_oxidation() const noexcept
    {
        const auto total    = derived().sources();
        const auto ox       = derived().sources_oxidation();
        const std::size_t N = total.size();
        for (std::size_t i = 0; i < N; ++i)
            source_no_oxidation_[i] = total[i] - ox[i];
        return {source_no_oxidation_.data(), N};
    }

    /**
     * @brief Fills the internal `kappa_oxidation_` cache and returns a span.
     *
     * Linearises oxidation as a first-order decay:
     *
     *   κ_i = max(−source_oxidation[i], 0) / max(|M_i|, ε)
     *
     * and stores the result in the `mutable kappa_oxidation_` cache.  Returns a
     * zero-copy span into that cache.
     *
     * Called by `MOM::GetOxidationRateCoefficients()` in `Splitting.hpp`.  The
     * same access-control rationale as `sources_without_oxidation()` applies.
     *
     * @param current_moments  Transported moment values M_i at the current time.
     *                         Must have size ≥ `n_equations`.
     * @return Span of size `n_equations`; valid until the next call to this
     *         function or `ComputeSources()` — whichever comes first.
     */
    [[nodiscard]] std::span<const double>
    kappa_oxidation(std::span<const double> current_moments) const noexcept
    {
        const auto ox       = derived().sources_oxidation();
        const std::size_t N = std::min(ox.size(), current_moments.size());
        constexpr double eps = 1.e-300;
        for (std::size_t i = 0; i < N; ++i)
        {
            const double neg_src   = -ox[i];
            const double M         = std::max(std::abs(current_moments[i]), eps);
            kappa_oxidation_[i]    = (neg_src > 0.) ? neg_src / M : 0.;
        }
        return {kappa_oxidation_.data(), N};
    }

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
    // Updated by each concrete class's SetState() implementation.

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
     * @brief Cached: `source_all_ − source_oxidation_` [same units as source_all_].
     *
     * Populated on demand by `sources_without_oxidation()` (called via `GetSourcesWithoutOxidation()`).
     * Declared `mutable` so that the logically-const getter can update the cache
     * without requiring a non-const model reference.  This is the standard C++
     * justification for `mutable`: the cache is derived state, not primary state.
     *
     * Lifetime: valid until the next `ComputeSources()` call (same as all source spans).
     */
    mutable MomentVector source_no_oxidation_ = MomentVector::Zero();

    /**
     * @brief Cached: first-order oxidation rate coefficients κ_i [1/s].
     *
     *   κ_i = max(−source_oxidation[i], 0) / max(|M_i|, ε)
     *
     * Populated on demand by `kappa_oxidation()` (called via `GetOxidationRateCoefficients()`).
     * Declared `mutable` for the same reason as `source_no_oxidation_`.
     *
     * Lifetime: valid until the next call to `kappa_oxidation()`
     * or `ComputeSources()` — whichever comes first.
     */
    mutable MomentVector kappa_oxidation_ = MomentVector::Zero();

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

    // -- Shared thermodynamic-state helpers --------------------------------

    /**
     * @brief Update common gas state from temperature, pressure, and mass fractions.
     *
     * Computes the mixture molecular weight, total molar concentration, and
     * density using the same convention as the concrete variants:
     * `1/MW = sum_k Y_k/MW_k`, `cTot = P/(R T)`, `rho = cTot * MW`.
     * @return Total molar concentration [kmol/m3].
     */
    template <typename Thermo>
    [[nodiscard]] double UpdateMixtureState(double T,
                                            double P_Pa,
                                            const double* Y,
                                            const Thermo& thermo,
                                            double gas_constant = Rgas_) noexcept
    {
        T_    = T;
        P_Pa_ = P_Pa;

        double invMW = 0.;
        for (unsigned k = 0; k < thermo.NumberOfSpecies(); ++k)
            invMW += Y[k] / thermo.MolecularWeight(k);
        MW_ = 1. / invMW;

        const double cTot = P_Pa_ / (gas_constant * T_);
        rho_              = cTot * MW_;
        return cTot;
    }

    /**
     * @brief Species molar concentration from a mass fraction vector.
     * @return Concentration [kmol/m3], or zero when @p idx is absent.
     */
    template <typename Thermo>
    [[nodiscard]] double SpeciesConcentrationKmolM3(int idx,
                                                    const double* Y,
                                                    double cTot,
                                                    const Thermo& thermo) const noexcept
    {
        if (idx < 0)
            return 0.;
        return cTot * std::max(Y[idx], 0.) * MW_ /
               thermo.MolecularWeight(static_cast<unsigned>(idx));
    }

    /**
     * @brief Species molar concentration in [mol/cm3], used by HMOM HACA rates.
     */
    template <typename Thermo>
    [[nodiscard]] double SpeciesConcentrationMolCm3(int idx,
                                                    const double* Y,
                                                    double cTot,
                                                    const Thermo& thermo) const noexcept
    {
        return SpeciesConcentrationKmolM3(idx, Y, cTot, thermo) / 1.e3;
    }

    // -- Shared transport helper -------------------------------------------

    /**
     * @brief Cunningham-corrected Brownian effective diffusion coefficient [kg/m/s].
     *
     * Shared kernel for HMOM, ThreeEquations, and MetalOxide.  Implements the
     * Stokes–Einstein–Cunningham model in the continuum–slip transition regime:
     *
     *   @f[
     *     C_u = 1 + A \frac{\lambda}{d_c}, \quad
     *     D   = \frac{k_B T C_u}{3 \pi \mu d_c}, \quad
     *     \Gamma = \rho D
     *   @f]
     *
     * where @f$\lambda = \mu/\rho \cdot \sqrt{\pi m_\text{gas} / (2 k_B T)}@f$
     * is the mean free path of the carrier gas, @f$m_\text{gas} = \rho k_B T / P@f$
     * is the mean gas-molecule mass, and @f$A = @f$ `kCunningham_` = 2.154
     * (Hutchins et al. 1995, linearised Knudsen-number limit).
     *
     * @pre `dc_safe` **must be strictly positive** (> 0).  Each caller applies its
     *      own minimum guard before invoking this method:
     *      - HMOM and ThreeEquations: `std::max(dc, 1e-12)`
     *      - MetalOxide:              `std::max(dc, d0_)` (nucleation cluster diameter)
     *
     * @param dc_safe  Collision diameter [m], already clamped to a positive minimum.
     * @return @f$\rho D@f$ [kg/m/s]; the caller adds the Schmidt-number fallback
     *         `std::max(result, mu_ / schmidt_number_)` when desired.
     */
    [[nodiscard]] [[gnu::always_inline]]
    double CunninghamDiffusionCoeff(double dc_safe) const noexcept
    {
        const double m_gas  = rho_ * kB_ * T_ / P_Pa_;
        const double lambda = mu_ / rho_ * std::sqrt(pi_ * m_gas / (2. * kB_ * T_));
        const double Cu     = 1. + kCunningham_ * lambda / dc_safe;
        const double D      = kB_ * T_ * Cu / (3. * pi_ * mu_ * dc_safe);
        return rho_ * D;
    }

    // -- Sphere-geometry primitives ------------------------------------------
    //
    // These small helpers centralise the sphere-geometry identities used in
    // the per-particle Properties() functions and initialisation code of every
    // variant.  All four are `const noexcept` (they read the static pi_ only)
    // and are marked always_inline so they fold away completely in hot loops.
    //
    // Naming convention:
    //   SphereDiameter(v)          d = (6V/π)^{1/3}          [m]
    //   SphereSurface(d)           S = π d²                   [m²]
    //   SphereSurfaceFromVolume(v) S = (36π)^{1/3} V^{2/3}   [m²]
    //   NumberPrimaryParticles(ss,vs)  np = ss³/(36π vs²) ≥ 1 [-]
    //
    // ⚠ FP note for SphereDiameter: the operand order "6./pi_ * v" (pre-divide
    //   then multiply) is empirically more accurate than the alternative
    //   "6.*v/pi_" (pre-multiply then divide): in the ~10% of inputs where the
    //   two forms produce different IEEE-754 doubles, the pre-divide form gives
    //   the correctly-rounded result 83.7% of the time vs 16.3% for pre-multiply.
    //   Reason: fl(6/π) is a fixed, correctly-rounded constant; multiplying it
    //   by v incurs only one additional rounding, whereas fl(6*v) introduces a
    //   fresh, v-dependent rounding error before the division by π.

    //! Diameter of the sphere with volume @p v [m³].  d = (6V/π)^{1/3}.
    [[nodiscard]] [[gnu::always_inline]]
    double SphereDiameter(double v) const noexcept
    { return std::pow(6. / pi_ * v, 1. / 3.); }

    //! Surface area of the sphere with diameter @p d [m].  S = π d².
    [[nodiscard]] [[gnu::always_inline]]
    double SphereSurface(double d) const noexcept
    { return pi_ * d * d; }

    //! Surface area of the sphere with volume @p v [m³].
    //! Avoids the intermediate diameter: S = (36π)^{1/3} V^{2/3}.
    [[nodiscard]] [[gnu::always_inline]]
    double SphereSurfaceFromVolume(double v) const noexcept
    { return std::pow(36. * pi_, 1. / 3.) * std::pow(v, 2. / 3.); }

    //! Number of primary spherical particles from per-particle surface @p ss [m²]
    //! and volume @p vs [m³].  From the identity np = ss³/(36π vs²).
    //! Returns max(np, 1): a fully sintered aggregate counts as one particle.
    [[nodiscard]] [[gnu::always_inline]]
    double NumberPrimaryParticles(double ss, double vs) const noexcept
    { return std::max(std::pow(ss, 3.) / std::pow(vs, 2.) / (36. * pi_), 1.); }

    // -- Helpers for zero-initialising source vectors between time steps -----

    /**
     * @brief Zeroes the base-class-owned source accumulators.
     *
     * Each `Derived::ComputeSources()` implementation must call this
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

    /** @brief Emit a single labelled gas-source diagnostic column safely. */
    template <typename CB, typename Label>
    void EmitOmegaGas(CB&& cb, Label&& label, int idx) const
    {
        cb(std::forward<Label>(label), safe_omega_gas(idx));
    }

    /** @brief Emit gas-source diagnostics common to soot variants. */
    template <typename CB>
    void EmitStandardSootOmegaGas(CB&& cb,
                                  int precursor_index,
                                  int c2h2_index,
                                  int h2_index,
                                  int o2_index,
                                  int h2o_index,
                                  int oh_index) const
    {
        cb("omegaTot[kg/m3/s]", omega_gas_.sum());
        EmitOmegaGas(cb, "omegaPrec[kg/m3/s]", precursor_index);
        EmitOmegaGas(cb, "omegaC2H2[kg/m3/s]", c2h2_index);
        EmitOmegaGas(cb, "omegaH2[kg/m3/s]", h2_index);
        EmitOmegaGas(cb, "omegaO2[kg/m3/s]", o2_index);
        EmitOmegaGas(cb, "omegaH2O[kg/m3/s]", h2o_index);
        EmitOmegaGas(cb, "omegaOH[kg/m3/s]", oh_index);
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

    static constexpr double kB_          = 1.380649e-23;     //!< Boltzmann constant [J/K].
    static constexpr double Nav_mol_     = 6.02214076e23;    //!< Avogadro number [#/mol].
    static constexpr double Nav_kmol_    = 6.02214076e26;    //!< Avogadro number [#/kmol].
    static constexpr double Rgas_        = 8314.46261815324; //!< Ideal gas constant [J/kmol/K].
    static constexpr double Rgas_mol_    = Rgas_ / 1000.;   //!< Ideal gas constant [J/mol/K].
    static constexpr double WC_          = 12.011;           //!< Carbon standard atomic weight [kg/kmol].

    //! Cunningham slip-correction coefficient A (linearised Kn limit, Hutchins et al. 1995).
    //! Enters the correction as Cu = 1 + kCunningham_ * λ/dc where λ is the mean free path.
    static constexpr double kCunningham_ = 2.154;

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
