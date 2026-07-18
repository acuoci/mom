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
 * Provides source accessors and rate coefficients for treating oxidation as an
 * analytical sub-step outside the main moment ODE solve.
 *
 * Included automatically by `MOM/MOM.hpp`.
 *
 * @par Usage
 * Oxidation is treated as a first-order sink with rate coefficient
 * `kappa_i = max(-source_oxidation[i], 0) / max(abs(M_i), eps)`.
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
 * **Step C** — gas-phase correction with saturation factor phi:
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
 * @par Functions provided
 * - `GetOxidationSources`           — zero-copy span of oxidation-only moment sources
 * - `GetSourcesWithoutOxidation`    — zero-copy span: total − oxidation (internal cache)
 * - `GetOxidationRateCoefficients`  — zero-copy span: κ_i [1/s] (internal cache)
 * - `GetOmegaGasOxidation`          — zero-copy span of oxidation-only gas-phase sources
 * - `FillOmegaGasWithoutOxidation`  — writes total gas − oxidation into caller buffer
 */

#include <algorithm>   // std::copy, std::min
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
 * Points directly into the model's internal oxidation source storage. For
 * models without oxidation, the returned span is the standard zero fallback.
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
 * @brief Returns a zero-copy span over `source_all[i] - source_oxidation[i]`.
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
 * @brief Returns a zero-copy span of first-order oxidation rate coefficients [1/s].
 *
 * Linearises the (generally nonlinear) oxidation depletion as a first-order decay:
 *
 *   kappa_i = max(-source_oxidation[i], 0) / max(abs(M_i), eps)
 *
 * The exact analytical solution for the oxidation sub-step then reads:
 *
 *   M_i(t + dt) = M_i(t) * exp(-kappa_i * dt)
 *
 * This gives the analytical oxidation update used by the splitting step.
 *
 * The result is written into `MomentMethodBase::kappa_oxidation_` (a
 * `mutable MomentVector`) and a span into that buffer is returned.
 *
 * @note  `kappa_i` is evaluated at the current stored oxidation sources (the
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
 * Returns an empty span for models without oxidation gas coupling.
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
 * @brief Writes `omega_gas[k] - omega_gas_oxidation[k]` into @p out for each species.
 *
 * For operator splitting: pass these reduced gas sources inside `Equations()`,
 * then apply the oxidation contribution analytically after the ODE step completes.
 *
 * The output buffer is caller-owned because gas-phase vectors are sized by the
 * number of species at runtime.
 *
 * @par Post-step gas-phase correction with saturation factor phi
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
 * The factor `phi` tends to 1 for small `kappa*dt` and limits integrated gas
 * consumption for fast oxidation.
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
            // ox is either empty (model has no oxidation) or size == N (always).
            // Hoist the invariant check out of the loop to avoid a branch per species.
            if (ox.empty())
                std::copy(total.begin(), total.begin() + N, out.begin());
            else
                for (std::size_t k = 0; k < N; ++k)
                    out[k] = total[k] - ox[k];
        },
        m);
}

/** @} */

} // namespace MOM
