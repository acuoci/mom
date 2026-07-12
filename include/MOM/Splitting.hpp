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

/**
 * @file Splitting.hpp
 * @brief Operator-splitting support functions for AnyMomentMethod.
 *
 * Provides the complete set of functions required to implement Lie–Trotter
 * (or Strang) operator splitting that removes the stiff oxidation eigenvalue
 * from the moment ODE system.
 *
 * Included automatically by `MOM/MOM.hpp`.
 *
 * @par Motivation
 * Oxidation is a first-order sink on all moment variables.  Its linearised
 * rate coefficient κ_i = |source_oxidation[i]| / |M_i| can reach 10⁵–10⁶ s⁻¹
 * in a flame — orders of magnitude larger than the CFD time step allows in an
 * implicit BDF solver.  The Patankar limiter bounds values but not eigenvalues,
 * so the BDF solver still brackets the C⁰ kink and produces tiny sub-steps.
 * Operator splitting removes the eigenvalue entirely:
 *
 * **Step A** — ODE sub-step with non-oxidation sources only:
 * @code
 *   MOM::Compute(mom);
 *   auto src_no_ox = MOM::GetSourcesWithoutOxidation(mom); // zero-copy span
 *   // pass src_no_ox to the stiff ODE solver — no large eigenvalue
 * @endcode
 *
 * **Step B** — analytical oxidation sub-step after the ODE completes:
 * @code
 *   auto kappa = MOM::GetOxidationRateCoefficients(mom, {M, n_mom});
 *   for (int i = 0; i < n_mom; ++i)
 *       M[i] *= std::exp(-kappa[i] * dt);
 * @endcode
 *
 * **Step C** — gas-phase correction with saturation factor φ:
 * @code
 *   // Inside Equations(), pass only the non-oxidation gas sources:
 *   auto gas_no_ox = MOM::GetOmegaGasOxidation(mom);  // zero-copy, oxidation-only
 *   double gas_buf[MAX_SPECIES];
 *   MOM::FillOmegaGasWithoutOxidation(mom, {gas_buf, n_species});
 *
 *   // After Step B, apply the integrated oxidation contribution:
 *   const double x   = kappa[MASS_IDX] * dt;   // use the mass-fraction moment index
 *   const double phi = (x > 1e-8) ? (1. - std::exp(-x)) / x : 1.;
 *   auto ox_gas = MOM::GetOmegaGasOxidation(mom);
 *   for (int k = 0; k < n_species; ++k)
 *       Y[k] += ox_gas[k] / rho * dt * phi;
 * @endcode
 *
 * @par API convention
 *
 * All functions in this file follow the same convention as `Sources.hpp`:
 *
 * | Pattern | When used | Example |
 * |---|---|---|
 * | `[[nodiscard]] std::span<const double> GetXxx(m)` | Zero-copy into internal storage | `GetOxidationSources`, `GetSourcesWithoutOxidation`, `GetOxidationRateCoefficients` |
 * | `[[nodiscard]] std::span<const double> GetXxx(m, inputs...)` | Derived value, needs external input, written to internal cache | `GetOxidationRateCoefficients` |
 * | `void FillXxx(m, out)` | Output too large to buffer internally (gas-phase, n_species) | `FillOmegaGasWithoutOxidation` |
 *
 * The `Fill` prefix signals an output-parameter function.  It is used only for
 * `FillOmegaGasWithoutOxidation` where caching would require a second n_species
 * `Eigen::VectorXd` per model instance — cost not justified by the marginal
 * ergonomic gain.  For the moment-space functions the cache is a `MomentVector`
 * (≤ 32 bytes) and is unconditionally worth the uniformity.
 *
 * @par Functions provided
 * - `GetOxidationSources`           — zero-copy span of oxidation-only moment sources
 * - `GetSourcesWithoutOxidation`    — zero-copy span: total − oxidation (internal cache)
 * - `GetOxidationRateCoefficients`  — zero-copy span: κ_i [1/s] (internal cache)
 * - `GetOmegaGasOxidation`          — zero-copy span of oxidation-only gas-phase sources
 * - `FillOmegaGasWithoutOxidation`  — writes total gas − oxidation into caller buffer
 */

#include <algorithm>   // std::min
#include <cmath>       // std::abs

#include "AnyMomentMethod.hpp"

namespace MOM
{

/**
 * @name Operator-splitting — moment-space functions
 * @{
 */

/**
 * @brief Returns a zero-copy span over the **oxidation-only** moment source vector.
 *
 * Points directly into the model's internal `source_oxidation_` storage —
 * zero-overhead.  All entries are ≤ 0 (oxidation is always a sink for soot).
 *
 * @pre  `ComputeSources()` must have been called at the current state.
 * @return Span of size `n_equations`; valid until next `ComputeSources()`.
 */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double>
GetOxidationSources(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.sources_oxidation(); }, m);
}

/**
 * @brief Returns a zero-copy span over `source_all[i] − source_oxidation[i]`.
 *
 * For operator splitting of the stiff oxidation terms: pass these reduced
 * sources to the stiff ODE solver instead of the full `GetSources()` vector,
 * then apply the oxidation sub-step analytically with
 * `GetOxidationRateCoefficients()`.
 *
 * The result is written into `MomentMethodBase::source_no_oxidation_` (a
 * `mutable MomentVector`) and a span into that buffer is returned.  The span
 * is valid until the next `ComputeSources()` call or the next call to this
 * function — whichever comes first.
 *
 * @pre  `ComputeSources()` must have been called at the current state.
 * @return Span of size `n_equations`.
 */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double>
GetSourcesWithoutOxidation(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit(
        [](const auto& mm) -> std::span<const double>
        {
            return mm.sources_without_oxidation();
        },
        m);
}

/**
 * @brief Returns a zero-copy span of first-order oxidation rate coefficients κ_i [1/s].
 *
 * Linearises the (generally nonlinear) oxidation depletion as a first-order decay:
 *
 *   κ_i = max(−source_oxidation[i], 0) / max(|M_i|, ε)
 *
 * The exact analytical solution for the oxidation sub-step then reads:
 *
 *   M_i(t + Δt) = M_i(t) · exp(−κ_i · Δt)
 *
 * This is unconditionally stable regardless of how large κ_i is — it removes
 * the stiff oxidation eigenvalue from the ODE system entirely.
 *
 * The result is written into `MomentMethodBase::kappa_oxidation_` (a
 * `mutable MomentVector`) and a span into that buffer is returned.
 *
 * @note  κ_i is evaluated at the **current** stored oxidation sources (the
 *        last `ComputeSources()` call) and the @p current_moments passed by
 *        the caller.  For Lie–Trotter splitting, call this after the ODE step.
 *        For symmetric Strang splitting, evaluate at t + Δt/2.
 *
 * @pre   `ComputeSources()` must have been called at the current state.
 * @param m                Model variant.
 * @param current_moments  Transported moment values M_i at current time.
 *                         Size must be ≥ n_equations.
 * @return Span of size `n_equations`; valid until the next call to this
 *         function or the next `ComputeSources()` — whichever comes first.
 */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double>
GetOxidationRateCoefficients(const AnyMomentMethod<Thermo>& m,
                              std::span<const double> current_moments) noexcept
{
    return std::visit(
        [&current_moments](const auto& mm) -> std::span<const double>
        {
            return mm.kappa_oxidation(current_moments);
        },
        m);
}

/** @} */

/**
 * @name Operator-splitting — gas-phase functions
 * @{
 */

/**
 * @brief Returns a zero-copy span over the **oxidation-only** gas-phase source
 *        vector [kg/m³/s].
 *
 * Points directly into internal storage — zero-overhead.
 * Returns an empty span for models without oxidation gas coupling (e.g. MetalOxide/TiO₂).
 *
 * @pre `ComputeSources()` must have been called at the current state.
 */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double>
GetOmegaGasOxidation(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.omega_gas_oxidation(); }, m);
}

/**
 * @brief Writes `omega_gas[k] − omega_gas_oxidation[k]` into @p out for each species.
 *
 * For operator splitting: pass these reduced gas sources inside `Equations()`,
 * then apply the oxidation contribution analytically after the ODE step completes.
 *
 * @par Why this function uses an output parameter
 * Gas-phase source vectors are sized to `n_species` at runtime (typically 50–300
 * elements).  Caching a second copy inside the model would double the per-instance
 * `omega_gas_` storage, which is significant in multi-cell CFD loops.  The `Fill`
 * prefix signals intentionally that this function writes into a caller-owned buffer,
 * unlike the moment-space functions which return zero-copy spans into internal caches.
 *
 * @par Post-step gas-phase correction with saturation factor φ
 * After the outer CFD timestep @p dt has completed and the soot moments have been
 * updated with the exponential decay (Step B above), apply the integrated oxidation
 * effect on gas species:
 *
 * @code
 *   // kappa[MASS_IDX]: from GetOxidationRateCoefficients() for the mass-fraction moment
 *   //   MASS_IDX = 0 for ThreeEquations / BrookesMoss; = 1 for HMOM
 *   const double x   = kappa[MASS_IDX] * dt;
 *   const double phi = (x > 1e-8) ? (1. - std::exp(-x)) / x : 1.;
 *   auto ox_gas = MOM::GetOmegaGasOxidation(mom);
 *   for (int k = 0; k < n_species; ++k)
 *       Y[k] += ox_gas[k] / rho * dt * phi;
 * @endcode
 *
 * The factor φ ∈ (0, 1] collapses to 1 for small `κ·Δt` (first-order approximation)
 * and falls toward `1/(κ·Δt)` for fast oxidation, bounding the integrated gas
 * consumption to at most the available reactant mass.
 *
 * @pre  `ComputeSources()` must have been called at the current state.
 * @param m    Model variant.
 * @param out  Caller-allocated buffer; size must be ≥ n_species.
 */
template <ThermoMap Thermo>
inline void FillOmegaGasWithoutOxidation(const AnyMomentMethod<Thermo>& m,
                                          std::span<double> out) noexcept
{
    std::visit(
        [&out](const auto& mm)
        {
            const auto total = mm.omega_gas();           // full omega_gas_   — zero-copy
            const auto ox    = mm.omega_gas_oxidation(); // oxidation-only    — zero-copy
            const std::size_t N = std::min(total.size(), out.size());
            for (std::size_t k = 0; k < N; ++k)
                out[k] = total[k] - (k < ox.size() ? ox[k] : 0.);
        },
        m);
}

/** @} */

} // namespace MOM
