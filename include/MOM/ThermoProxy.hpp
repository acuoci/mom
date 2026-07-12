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

#include <concepts>
#include <string>
#include <string_view>
#include <vector>

namespace MOM
{

/**
 * @file ThermoProxy.hpp
 * @brief Thermodynamics backend concept and lightweight test implementation.
 */

/**
 * @concept ThermoMap
 * @brief Minimal thermodynamics interface required by any moment method variant.
 *
 * Any CFD or chemistry backend can be used by providing an adapter satisfying
 * this concept.
 *
 * @par Conventions
 * - All species indices are **0-based**.
 * - `IndexOfSpecies` returns **-1** if the species is absent in the mechanism.
 * - Molecular weights are in **[kg/kmol]**.
 * - Atom-count methods return **0.0** for species that do not contain that element
 *   (e.g. `NumberOfTitaniumAtoms` on a carbon species returns 0.0, not NaN).
 *
 * @par Requirement summary
 * | Expression | Return type | Description |
 * |---|---|---|
 * | `t.NumberOfSpecies()` | convertible to `unsigned` | Total species count |
 * | `t.IndexOfSpecies(name)` | convertible to `int` | 0-based index, −1 if absent |
 * | `t.MolecularWeight(i)` | `double` | MW of species *i* [kg/kmol] |
 * | `t.NumberOfCarbonAtoms(i)` | `double` | C atoms per molecule of species *i* |
 * | `t.NumberOfHydrogenAtoms(i)` | `double` | H atoms per molecule |
 * | `t.NumberOfOxygenAtoms(i)` | `double` | O atoms per molecule |
 * | `t.NumberOfNitrogenAtoms(i)` | `double` | N atoms per molecule |
 * | `t.NumberOfTitaniumAtoms(i)` | `double` | Ti atoms per molecule (0 if none) |
 *
 * @tparam T  Thermodynamics backend type to check.
 */
template <typename T>
concept ThermoMap = requires(const T& t, unsigned i, std::string_view name) {
    { t.NumberOfSpecies() } -> std::convertible_to<unsigned>;
    { t.IndexOfSpecies(name) } -> std::convertible_to<int>;
    { t.MolecularWeight(i) } -> std::same_as<double>;
    { t.NumberOfCarbonAtoms(i) } -> std::same_as<double>;
    { t.NumberOfHydrogenAtoms(i) } -> std::same_as<double>;
    { t.NumberOfOxygenAtoms(i) } -> std::same_as<double>;
    { t.NumberOfNitrogenAtoms(i) } -> std::same_as<double>;
    { t.NumberOfTitaniumAtoms(i) } -> std::same_as<double>;
};

/**
 * @struct BasicThermoData
 * @brief Lightweight, self-contained thermodynamics struct satisfying ThermoMap.
 *
 * Intended for unit tests, standalone executables, and CI builds where
 * OpenSMOKE++ is not available.  Populate the public vectors directly or parse
 * them from a minimal input file before constructing any moment method object.
 *
 * @code
 *   MOM::BasicThermoData thermo;
 *   thermo.names = {"C2H2", "H2", "O2", "OH", "H2O"};
 *   thermo.mw    = { 26.04, 2.016, 32.0, 17.008, 18.016 };
 *   // ... fill nc, nh, no, nn, nti ...
 *   MOM::HMOM<MOM::BasicThermoData> model(thermo);
 * @endcode
 */
struct BasicThermoData
{
    std::vector<std::string> names; //!< Species names in mechanism order.
    std::vector<double> mw;         //!< Molecular weights [kg/kmol], same order as names.
    std::vector<double> nc;         //!< Carbon atom counts per molecule.
    std::vector<double> nh;         //!< Hydrogen atom counts per molecule.
    std::vector<double> no;         //!< Oxygen atom counts per molecule.
    std::vector<double> nn;         //!< Nitrogen atom counts per molecule.
    std::vector<double> nti;        //!< Titanium atom counts per molecule (0 for most species).

    /**
     * @brief Returns the total number of species in the mechanism.
     * @return Species count as unsigned.
     */
    [[nodiscard]] unsigned NumberOfSpecies() const noexcept
    {
        return static_cast<unsigned>(names.size());
    }

    /**
     * @brief Returns the 0-based index of a species by name.
     * @param name Species name to look up.
     * @return 0-based index, or -1 if the species is not present.
     */
    [[nodiscard]] int IndexOfSpecies(std::string_view name) const noexcept
    {
        for (unsigned i = 0; i < names.size(); ++i)
            if (names[i] == name)
                return static_cast<int>(i);
        return -1;
    }

    /** @brief Molecular weight of species @p i [kg/kmol]. */
    [[nodiscard]] double MolecularWeight(unsigned i) const noexcept { return mw[i]; }

    /** @brief Number of carbon atoms per molecule of species @p i. */
    [[nodiscard]] double NumberOfCarbonAtoms(unsigned i) const noexcept { return nc[i]; }

    /** @brief Number of hydrogen atoms per molecule of species @p i. */
    [[nodiscard]] double NumberOfHydrogenAtoms(unsigned i) const noexcept { return nh[i]; }

    /** @brief Number of oxygen atoms per molecule of species @p i. */
    [[nodiscard]] double NumberOfOxygenAtoms(unsigned i) const noexcept { return no[i]; }

    /** @brief Number of nitrogen atoms per molecule of species @p i. */
    [[nodiscard]] double NumberOfNitrogenAtoms(unsigned i) const noexcept { return nn[i]; }

    /** @brief Number of titanium atoms per molecule of species @p i (0 for non-Ti species). */
    [[nodiscard]] double NumberOfTitaniumAtoms(unsigned i) const noexcept { return nti[i]; }
};

static_assert(ThermoMap<BasicThermoData>, "BasicThermoData must satisfy ThermoMap");

/**
 * @note Additional thermodynamics adapters live in compatibility layers and are
 *       intentionally not included by this header.
 *
 * @code
 *   #include "MOM/ThermoProxy.hpp"
 *   OpenSMOKE::ThermodynamicsMap_CHEMKIN osThermo(...);
 *   MOM::OpenSMOKEThermo thermo(osThermo);
 *   MOM::HMOM model(thermo);
 * @endcode
 */

} // namespace MOM
