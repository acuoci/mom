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

#include <optional>
#include <string_view>

namespace MOM
{

/**
 * @file ProcessFlags.hpp
 * @brief Strongly typed process-model selectors and parsing helpers.
 */

/**
 * @brief Strongly-typed flags for all physical sub-process models.
 *
 * Concrete variants expose these flags through their getters and accept integer
 * values in configuration paths where input files use numeric model selectors.
 *
 * @par Convention
 * - `Off = 0` — process disabled; corresponding source terms are zero.
 * - `Standard = 1` — primary model variant (method-specific implementation).
 * - `Extended = 2` — alternative or extended model variant where applicable.
 */

/** @brief Nucleation sub-model selector. */
enum class NucleationModel : int
{
    Off      = 0, //!< Nucleation disabled.
    Standard = 1, //!< Default nucleation model for each variant (PAH dimerisation for soot).
    Extended = 2  //!< Alternative nucleation (e.g. Brookes-Moss-Hall extended inception).
};

/** @brief Coagulation sub-model selector. */
enum class CoagulationModel : int
{
    Off      = 0, //!< Coagulation disabled.
    Standard = 1  //!< Free-molecular + continuum coagulation kernel.
};

/** @brief Surface growth sub-model selector. */
enum class SurfaceGrowthModel : int
{
    Off      = 0, //!< Surface growth disabled.
    Standard = 1  //!< HACA (H-abstraction–C2H2-addition) surface growth.
};

/** @brief Oxidation sub-model selector. */
enum class OxidationModel : int
{
    Off      = 0, //!< Oxidation disabled.
    Standard = 1, //!< OH + O2 oxidation (Lee et al. correlation).
    Extended = 2  //!< Alternative oxidation (e.g. BrookesMoss-Hall variant).
};

/** @brief PAH condensation sub-model selector. */
enum class CondensationModel : int
{
    Off      = 0, //!< Condensation disabled.
    Standard = 1  //!< PAH adsorption on existing soot particles.
};

/** @brief Sintering sub-model selector (used by MetalOxide). */
enum class SinteringModel : int
{
    Off      = 0, //!< Sintering disabled.
    Standard = 1  //!< Viscous-flow sintering model (Kruis et al. 1993).
};

/** @brief Thermophoretic drift sub-model selector. */
enum class ThermophoreticModel : int
{
    Off      = 0, //!< No thermophoretic correction to diffusion.
    Standard = 1  //!< Thermophoretic drift encoded in effective diffusion coefficient.
};

/**
 * @brief Planck mean absorption coefficient model for radiative heat transfer.
 *
 * Selects the empirical correlation used to compute the Planck mean absorption
 * coefficient κ_P [1/m] of the particle phase, needed by radiation solvers.
 * MetalOxide particles are dielectric and should use `None`.
 */
enum class PlanckCoeffModel : int
{
    None   = 0, //!< No radiative contribution from particles (κ_P = 0).
    Smooke = 1, //!< Smooke et al. (1988) correlation (default for soot).
    Kent   = 2, //!< Kent & Honnery (1990) correlation.
    Sazhin = 3  //!< Sazhin (1994) correlation.
};

/**
 * @brief Parse a Planck absorption coefficient model label from a string.
 *
 * Case-insensitive. Returns `std::nullopt` for unrecognised labels so that
 * callers can distinguish between an explicit `"None"` request and a typo.
 * Used by `SetPlanckAbsorptionCoefficient(std::string_view)` and
 * `ParseConfig()` implementations.
 *
 * @param s  Label string (e.g. `"Smooke"`, `"Kent"`, `"Sazhin"`, `"None"`).
 * @return   Matching `PlanckCoeffModel` enumerator wrapped in `std::optional`,
 *           or `std::nullopt` if the label is not recognised.
 */
[[nodiscard]] constexpr std::optional<PlanckCoeffModel>
PlanckCoeffModelFromString(std::string_view s) noexcept
{
    if (s == "Smooke" || s == "smooke" || s == "SMOOKE")
        return PlanckCoeffModel::Smooke;
    if (s == "Kent" || s == "kent" || s == "KENT")
        return PlanckCoeffModel::Kent;
    if (s == "Sazhin" || s == "sazhin" || s == "SAZHIN")
        return PlanckCoeffModel::Sazhin;
    if (s == "None" || s == "none" || s == "NONE")
        return PlanckCoeffModel::None;
    return std::nullopt;
}

/**
 * @brief Parse a nucleation model label from a string.
 *
 * Returns `NucleationModel::Off` for unrecognised labels.
 *
 * @param s  Label string (`"Standard"`, `"Extended"`, `"1"`, `"2"`, or `"Off"`).
 * @return   Matching `NucleationModel` enumerator.
 */
[[nodiscard]] constexpr NucleationModel NucleationModelFromString(std::string_view s) noexcept
{
    if (s == "Standard" || s == "standard" || s == "1")
        return NucleationModel::Standard;
    if (s == "Extended" || s == "extended" || s == "2")
        return NucleationModel::Extended;
    return NucleationModel::Off;
}

/**
 * @brief Parse an oxidation model label from a string.
 *
 * Returns `OxidationModel::Off` for unrecognised labels.
 *
 * @param s  Label string (`"Standard"`, `"Extended"`, `"1"`, `"2"`, or `"Off"`).
 * @return   Matching `OxidationModel` enumerator.
 */
[[nodiscard]] constexpr OxidationModel OxidationModelFromString(std::string_view s) noexcept
{
    if (s == "Standard" || s == "standard" || s == "1")
        return OxidationModel::Standard;
    if (s == "Extended" || s == "extended" || s == "2")
        return OxidationModel::Extended;
    return OxidationModel::Off;
}

/**
 * @name Process activation tests
 *
 * Convenience predicates that avoid comparing an `enum class` to the integer
 * literal `0`.  Prefer these over `static_cast<int>(m) != 0` or `m != XxxModel::Off`
 * for more readable call sites:
 * @code
 *   if (MOM::IsActive(MOM::GetOxidationModel(mom_)))
 *       apply_operator_splitting();
 * @endcode
 * @{
 */

[[nodiscard]] constexpr bool IsActive(NucleationModel    m) noexcept { return m != NucleationModel::Off;    }
[[nodiscard]] constexpr bool IsActive(CoagulationModel   m) noexcept { return m != CoagulationModel::Off;   }
[[nodiscard]] constexpr bool IsActive(SurfaceGrowthModel m) noexcept { return m != SurfaceGrowthModel::Off;  }
[[nodiscard]] constexpr bool IsActive(OxidationModel     m) noexcept { return m != OxidationModel::Off;     }
[[nodiscard]] constexpr bool IsActive(CondensationModel  m) noexcept { return m != CondensationModel::Off;  }
[[nodiscard]] constexpr bool IsActive(SinteringModel     m) noexcept { return m != SinteringModel::Off;     }
[[nodiscard]] constexpr bool IsActive(ThermophoreticModel m) noexcept { return m != ThermophoreticModel::Off; }

/** @} */

} // namespace MOM
