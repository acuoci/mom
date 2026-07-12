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
 * @file Properties.hpp
 * @brief Particle property and model-status free functions for AnyMomentMethod.
 *
 * Contains all physical property queries (geometry, transport, radiation) and
 * model-status queries for the runtime-selected API.
 *
 * Included automatically by `MOM/MOM.hpp`.
 *
 * @par Functions provided
 *
 * **Status / control**
 * - `IsActive`                     — true if model is configured and active
 * - `HasGasConsumption`            — true if gas-phase consumption is enabled
 * - `GetThermophoreticModel`       — ThermophoreticModel enum (Off or Standard)
 * - `HasRadiativeHeatTransfer`     — true if particles contribute to radiation
 * - `HasClosureDummySpecies`       — true if a dummy closure species is set
 * - `GetClosureDummyIndex`         — 0-based index of the dummy closure species
 * - `GetPrecursorIndex`            — 0-based index of the precursor species
 * - `GetInitialMoments`            — initialisation values for moment transport
 *
 * **Particle geometry**
 * - `GetVolumeFraction`            — particle volume fraction [-]
 * - `GetParticleDiameter`          — primary particle diameter [m]
 * - `GetParticleNumberDensity`     — total number density [#/m³]
 * - `GetMassFraction`              — particle mass fraction [-]
 * - `GetParticleDensity`           — particle material density [kg/m³]
 * - `GetSpecificSurfaceArea`       — surface area per unit volume [m²/m³]
 * - `GetCollisionDiameter`         — aggregate collision diameter [m]
 * - `GetNumberPrimaryParticles`    — mean primary particles per aggregate [-]
 *
 * **Transport**
 * - `GetSchmidtNumber`             — particle Schmidt number [-]
 * - `GetDiffusionCoefficient`      — effective diffusion coefficient [kg/m/s]
 *
 * **Radiation**
 * - `GetPlanckCoefficient`         — Planck mean absorption coefficient [1/m]
 */

#include "AnyMomentMethod.hpp"

namespace MOM
{

/**
 * @name Status and control queries
 * @{
 */

/** @brief Returns `true` if the object is active. */
template <ThermoMap Thermo>
[[nodiscard]] inline bool IsActive(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.is_active(); }, m);
}

/** @brief Returns `true` if gas-phase precursor consumption is enabled. */
template <ThermoMap Thermo>
[[nodiscard]] inline bool HasGasConsumption(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.gas_consumption(); }, m);
}

/** @brief Returns the thermophoretic model flag (ThermophoreticModel::Off or ::Standard). */
template <ThermoMap Thermo>
[[nodiscard]] inline ThermophoreticModel GetThermophoreticModel(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.thermophoretic_model(); }, m);
}

/** @brief Returns `true` if a gas-closure dummy species has been configured. */
template <ThermoMap Thermo>
[[nodiscard]] inline bool HasClosureDummySpecies(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.is_closure_dummy_species(); }, m);
}

/** @brief Returns the 0-based index of the gas-closure dummy species (−1 if inactive). */
template <ThermoMap Thermo>
[[nodiscard]] inline int GetClosureDummyIndex(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.closure_dummy_index(); }, m);
}

/** @brief Returns the 0-based precursor species index in the thermo map (−1 if unset). */
template <ThermoMap Thermo>
[[nodiscard]] inline int GetPrecursorIndex(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.precursor_index(); }, m);
}

/**
 * @brief Returns the initial moment values for solver initialisation.
 * @return Span of size `n_equations` with near-zero seed values.
 */
template <ThermoMap Thermo>
[[nodiscard]] inline std::span<const double> GetInitialMoments(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.initial_moments(); }, m);
}

/** @} */

/**
 * @name Particle geometry properties
 * @{
 */

/** @brief Returns the particle volume fraction [-]. */
template <ThermoMap Thermo>
[[nodiscard]] inline double GetVolumeFraction(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.volume_fraction(); }, m);
}

/** @brief Returns the mean primary particle diameter [m]. */
template <ThermoMap Thermo>
[[nodiscard]] inline double GetParticleDiameter(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.particle_diameter(); }, m);
}

/** @brief Returns the total particle number density [#/m³]. */
template <ThermoMap Thermo>
[[nodiscard]] inline double GetParticleNumberDensity(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.particle_number_density(); }, m);
}

/** @brief Returns the particle mass fraction [-]. */
template <ThermoMap Thermo>
[[nodiscard]] inline double GetMassFraction(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.mass_fraction(); }, m);
}

/** @brief Returns the particle material density [kg/m³]. */
template <ThermoMap Thermo>
[[nodiscard]] inline double GetParticleDensity(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.particle_density(); }, m);
}

/** @brief Returns the specific surface area [m²/m³]. */
template <ThermoMap Thermo>
[[nodiscard]] inline double GetSpecificSurfaceArea(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.specific_surface_area(); }, m);
}

/** @brief Returns the aggregate collision diameter [m]. */
template <ThermoMap Thermo>
[[nodiscard]] inline double GetCollisionDiameter(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.collision_diameter(); }, m);
}

/** @brief Returns the mean number of primary particles per aggregate [-]. */
template <ThermoMap Thermo>
[[nodiscard]] inline double GetNumberPrimaryParticles(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.number_primary_particles(); }, m);
}

/** @} */

/**
 * @name Transport properties
 * @{
 */

/** @brief Returns the particle Schmidt number [-]. */
template <ThermoMap Thermo>
[[nodiscard]] inline double GetSchmidtNumber(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.schmidt_number(); }, m);
}

/** @brief Returns the effective particle diffusion coefficient [kg/m/s]. */
template <ThermoMap Thermo>
[[nodiscard]] inline double GetDiffusionCoefficient(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.diffusion_coefficient(); }, m);
}

/** @} */

/**
 * @name Radiative heat transfer
 * @{
 */

/** @brief Returns `true` if the Planck absorption coefficient should be included in radiation. */
template <ThermoMap Thermo>
[[nodiscard]] inline bool HasRadiativeHeatTransfer(const AnyMomentMethod<Thermo>& m) noexcept
{
    return std::visit([](const auto& mm) { return mm.radiative_heat_transfer(); }, m);
}

/**
 * @brief Returns the Planck mean absorption coefficient of the particle phase [1/m].
 * @param T  Gas temperature [K].
 * @param fv Particle volume fraction [-].
 */
template <ThermoMap Thermo>
[[nodiscard]] inline double GetPlanckCoefficient(const AnyMomentMethod<Thermo>& m,
                                                 double T,
                                                 double fv) noexcept
{
    return std::visit([T, fv](const auto& mm) { return mm.planck_coefficient(T, fv); }, m);
}

/** @} */

} // namespace MOM
