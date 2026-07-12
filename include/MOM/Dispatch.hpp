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
 * @file Dispatch.hpp
 * @brief Core per-cell dispatch free functions for AnyMomentMethod.
 *
 * Contains state injection, computation, configuration, and diagnostic
 * functions for the runtime-selected API.
 *
 * Included automatically by `MOM/MOM.hpp`.  Can also be included directly
 * when only dispatch functionality is needed (e.g. in a performance-critical
 * translation unit that does not use operator splitting or per-process sources).
 *
 * @par Functions provided
 * - `GetNEquations`       — number of transported equations
 * - `SetState`            — inject thermodynamic state (T, P, Y)
 * - `SetViscosity`        — inject mixture dynamic viscosity
 * - `SetMoments`          — inject current moment values
 * - `Compute`             — evaluate all source terms
 * - `ComputeCell`         — single-call per-cell entry point (preferred)
 * - `SetupFromDictionary` — configure from an OpenSMOKE++ dictionary (MOM_USE_DICTIONARY only)
 * - `PrintSummary`        — print model configuration to stdout
 */

#include "AnyMomentMethod.hpp"

namespace MOM
{

/**
 * @name Core dispatch: state injection and computation
 *
 * These functions forward the compile-time `MomentMethod` API through
 * `AnyMomentMethod<Thermo>`. Prefer `ComputeCell` in cell loops to perform state
 * injection and source evaluation with a single visit.
 * @{
 */

/** @brief Returns the number of transported equations for the active method. */
template <ThermoMap Thermo>
[[nodiscard]] inline unsigned GetNEquations(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) -> unsigned { return mm.n_equations; }, m);
}

/**
 * @brief Injects thermodynamic state into the active method.
 * @param T    Temperature [K].
 * @param P_Pa Pressure [Pa].
 * @param Y    Species mass fractions (pointer, size = n_species).
 */
template <ThermoMap Thermo>
inline void SetState(AnyMomentMethod<Thermo>& m, double T, double P_Pa, const double* Y) noexcept
{
    std::visit([&](auto& mm) { mm.SetState(T, P_Pa, Y); }, m);
}

/** @brief Sets the mixture dynamic viscosity [kg/m/s]. */
template <ThermoMap Thermo>
inline void SetViscosity(AnyMomentMethod<Thermo>& m, double mu) noexcept
{
    std::visit([mu](auto& mm) { mm.SetViscosity(mu); }, m);
}

/** @brief Sets current moment values from a contiguous span of size `n_equations`. */
template <ThermoMap Thermo>
inline void SetMoments(AnyMomentMethod<Thermo>& m, std::span<const double> moments) noexcept
{
    std::visit([moments](auto& mm) { mm.SetMoments(moments); }, m);
}

/** @brief Computes all source terms and gas-phase consumption. */
template <ThermoMap Thermo>
inline void Compute(AnyMomentMethod<Thermo>& m)
{
    std::visit([](auto& mm) { mm.ComputeSources(); }, m);
}

/**
 * @brief Single-call per-cell entry point for the runtime path.
 *
 * Collapses `SetState` / `SetMoments` / `SetViscosity` / `Compute` into a
 * single `std::visit`, reducing jump-table dispatches from four to one.
 *
 * @note For the compile-time path (type known statically), use the
 *       `MomentMethod`-constrained `ComputeCell` overload in
 *       `MomentMethodConcept.hpp` — it allows full inlining.
 */
template <ThermoMap Thermo>
inline void ComputeCell(AnyMomentMethod<Thermo>& m,
                        double T,
                        double P_Pa,
                        const double* Y,
                        double mu,
                        std::span<const double> moments) noexcept
{
    std::visit(
        [T, P_Pa, Y, mu, moments](auto& mm) noexcept
        {
            mm.SetState(T, P_Pa, Y);
            mm.SetMoments(moments);
            mm.SetViscosity(mu);
            mm.ComputeSources();
        },
        m);
}

/** @brief Prints the active variant's model configuration to stdout. */
template <ThermoMap Thermo>
inline void PrintSummary(const AnyMomentMethod<Thermo>& m)
{
    std::visit([](const auto& mm) { mm.PrintSummary(); }, m);
}

/** @} */

#if defined(MOM_USE_DICTIONARY)
/**
 * @brief Configure the active variant by parsing the given dictionary.
 *
 * Dispatches to the concrete variant's `ParseConfig(dict)` static member
 * template, then calls `SetupFromConfig()` on success.
 *
 * @tparam Dictionary  OpenSMOKE++ dictionary type (deduced).
 *
 * @throws std::runtime_error if the dictionary grammar check fails (missing
 *         mandatory keyword, unknown keyword, duplicate keyword), if a keyword
 *         value is invalid, or if `SetupFromConfig` raises an exception.
 *         Dictionary errors are always fatal — the solver must not continue
 *         with silently-defaulted parameters.
 */
template <ThermoMap Thermo, typename Dictionary>
inline void SetupFromDictionary(AnyMomentMethod<Thermo>& m, Dictionary& dict)
{
    std::visit(
        [&dict](auto& mm)
        {
            auto cfg = mm.ParseConfig(dict);
            if (!cfg)
                throw std::runtime_error(cfg.error());
            mm.SetupFromConfig(*cfg);
        },
        m);
}
#endif // MOM_USE_DICTIONARY

} // namespace MOM
