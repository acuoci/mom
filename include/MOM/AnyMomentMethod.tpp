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

/**
 * @file AnyMomentMethod.tpp
 * @brief Template implementation of `MakeAnyMomentMethod`.
 *
 * This file is included by `AnyMomentMethod.hpp`; do not compile it as a
 * standalone translation unit.
 */

#pragma once

namespace MOM
{

/**
 * @tparam Thermo  Thermodynamics backend satisfying the ThermoMap concept.
 * @param  thermo  Constructed thermo object; stored by reference inside the
 *                 returned variant's active alternative.
 * @param  label   Case-sensitive string key for the desired variant, e.g.
 *                 `"HMOM"`, `"BrookesMoss"`, `"ThreeEquations"`, `"MetalOxide"`.
 * @return         An `AnyMomentMethod<Thermo>` whose active alternative is the
 *                 variant whose `variant_labels` array contains @p label.
 * @throws std::invalid_argument  If @p label is not found in any registered
 *                                variant's `variant_labels`.
 */
template <ThermoMap Thermo>
AnyMomentMethod<Thermo> MakeAnyMomentMethod(const Thermo& thermo, std::string_view label)
{
    return detail::make_from_type_list(AllVariants{}, thermo, label);
}

} // namespace MOM
