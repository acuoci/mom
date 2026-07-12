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
 * @file Sources.hpp
 * @brief Source term accessors and process activation queries for AnyMomentMethod.
 *
 * Provides zero-copy access to total, gas-phase, and per-process source terms.
 * Per-process getters return a static zero span of size `n_equations` when the
 * active variant does not implement the requested process.
 *
 * Included automatically by `MOM/MOM.hpp`.
 *
 * @note Capability queries answer whether a variant type implements a process.
 *       Activation queries answer whether that process is enabled in the current
 *       configured instance. Use `IsActive(GetXxxModel(m))` in runtime logic.
 */

#include "AnyMomentMethod.hpp"

namespace MOM
{

/**
 * @name Total source term accessors
 * @{
 */

/**
 * @brief Returns a zero-copy span over the total source vector [model-specific units].
 * @return Span of size `n_equations`; valid until next `ComputeSources()`.
 */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double> GetSources(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.sources(); }, m);
}

/** @brief Returns a zero-copy span over gas-phase consumption terms [kg/m³/s]. */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double> GetOmegaGas(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.omega_gas(); }, m);
}

/** @} */

/**
 * @name Per-process source term accessors
 *
 * Zero-copy spans into the model's internal per-process source storage.
 * For models that do not implement a given process, the CRTP base-class
 * fallback returns a span over a `static constexpr` zero array — no runtime
 * branch, no allocation, span size equals `n_equations`.
 *
 * Check `IsActive(GetXxxModel())` when the caller must distinguish a disabled
 * process from an active process whose instantaneous rate is zero.
 * @{
 */

/** @brief Zero-copy span over nucleation source terms. Zero span if not modelled. */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double>
GetNucleationSources(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.sources_nucleation(); }, m);
}

/** @brief Zero-copy span over surface-growth source terms. Zero span if not modelled. */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double>
GetGrowthSources(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.sources_growth(); }, m);
}

/** @brief Zero-copy span over coagulation source terms. Zero span if not modelled. */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double>
GetCoagulationSources(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.sources_coagulation(); }, m);
}

/** @brief Zero-copy span over condensation source terms. Zero span if not modelled (e.g. BrookesMoss). */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double>
GetCondensationSources(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.sources_condensation(); }, m);
}

/** @brief Zero-copy span over sintering source terms. Zero span if not modelled (MetalOxide only). */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double>
GetSinteringSources(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.sources_sintering(); }, m);
}

/** @} */

/**
 * @name Process activation flag queries
 *
 * Return the strongly-typed process enum class for each physical process.
 * These route through `model_X()` CRTP dispatchers in `MomentMethodBase`,
 * which return `XxxModel::Off` for models that do not implement that process
 * (e.g. `GetSinteringModel` always returns `SinteringModel::Off` for soot models).
 *
 * @par Checking whether a process is active
 * Use `MOM::IsActive()` from `ProcessFlags.hpp` for readable boolean tests:
 * @code
 *   if (MOM::IsActive(MOM::GetOxidationModel(mom_))) {
 *       // oxidation is active — use operator splitting (see Splitting.hpp)
 *       auto src_no_ox = MOM::GetSourcesWithoutOxidation(mom_);  // zero-copy span
 *   }
 * @endcode
 *
 * @par Comparing to a specific variant
 * @code
 *   if (MOM::GetOxidationModel(mom_) == MOM::OxidationModel::Extended) {
 *       // BrookesMoss-Hall extended oxidation variant
 *   }
 * @endcode
 * @{
 */

/** @brief Returns the nucleation model flag (`NucleationModel::Off` if not active). */
template <ThermoMap Thermo>
[[nodiscard]] inline NucleationModel GetNucleationModel(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.model_nucleation(); }, m);
}

/** @brief Returns the surface-growth model flag (`SurfaceGrowthModel::Off` if not active). */
template <ThermoMap Thermo>
[[nodiscard]] inline SurfaceGrowthModel GetGrowthModel(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.model_growth(); }, m);
}

/** @brief Returns the coagulation model flag (`CoagulationModel::Off` if not active). */
template <ThermoMap Thermo>
[[nodiscard]] inline CoagulationModel GetCoagulationModel(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.model_coagulation(); }, m);
}

/** @brief Returns the condensation model flag (`CondensationModel::Off` if not active). */
template <ThermoMap Thermo>
[[nodiscard]] inline CondensationModel GetCondensationModel(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.model_condensation(); }, m);
}

/** @brief Returns the oxidation model flag (`OxidationModel::Off` if not active). */
template <ThermoMap Thermo>
[[nodiscard]] inline OxidationModel GetOxidationModel(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.model_oxidation(); }, m);
}

/** @brief Returns the sintering model flag (`SinteringModel::Off` unless MetalOxide). */
template <ThermoMap Thermo>
[[nodiscard]] inline SinteringModel GetSinteringModel(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.model_sintering(); }, m);
}

/** @} */

/**
 * @name Process capability queries
 *
 * These functions return whether the active variant type implements each
 * process, independent of the instance's enabled/disabled model flag.
 * @{
 */

/** @brief `true` if the active variant TYPE can compute nucleation sources. */
template <ThermoMap Thermo>
[[nodiscard]] inline bool GetNucleationCapability(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.capability_nucleation(); }, m);
}

/** @brief `true` if the active variant TYPE can compute surface-growth sources. */
template <ThermoMap Thermo>
[[nodiscard]] inline bool GetGrowthCapability(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.capability_growth(); }, m);
}

/** @brief `true` if the active variant TYPE can compute coagulation sources. */
template <ThermoMap Thermo>
[[nodiscard]] inline bool GetCoagulationCapability(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.capability_coagulation(); }, m);
}

/** @brief `true` if the active variant TYPE can compute condensation sources. */
template <ThermoMap Thermo>
[[nodiscard]] inline bool GetCondensationCapability(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.capability_condensation(); }, m);
}

/** @brief `true` if the active variant TYPE can compute oxidation sources. */
template <ThermoMap Thermo>
[[nodiscard]] inline bool GetOxidationCapability(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.capability_oxidation(); }, m);
}

/** @brief `true` if the active variant TYPE can compute sintering sources (MetalOxide only). */
template <ThermoMap Thermo>
[[nodiscard]] inline bool GetSinteringCapability(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.capability_sintering(); }, m);
}

/** @} */

} // namespace MOM
