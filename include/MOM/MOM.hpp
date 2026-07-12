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
 * @file MOM.hpp
 * @brief Umbrella header for the Method of Moments particle source-term library.
 *
 * Include this header when a translation unit needs the complete public API:
 * concrete variants, compile-time concepts, runtime factory, per-cell dispatch,
 * source accessors, particle properties, splitting utilities, and reporting.
 *
 * @par Compile-time selection
 * @code
 *   #include "MOM/MOM.hpp"
 *
 *   using ParticleModel = MOM::ThreeEquations<MyThermo>;
 *   static_assert(MOM::MomentMethod<ParticleModel>);
 * @endcode
 *
 * @par Runtime selection
 * @code
 *   auto model = MOM::MakeAnyMomentMethod(thermo, "HMOM");
 *   MOM::ComputeCell(model, T, P, Y, mu, moments);
 *   auto sources = MOM::GetSources(model);
 * @endcode
 */

#pragma once

// -- Core infrastructure -------------------------------------------------------
#include "ThermoProxy.hpp"
#include "ProcessFlags.hpp"
#include "MomentMethodBase.hpp"
#include "MomentMethodConcept.hpp"

// -- Variant registry, runtime wrapper, and reporter ---------------------------
#include "AnyMomentMethod.hpp"
#include "MomentMethodReporter.hpp"

// -- Free-function dispatch API -----------------------------------------------
#include "Dispatch.hpp"
#include "Properties.hpp"
#include "Sources.hpp"
#include "Splitting.hpp"

/** @brief Checks every registered variant against the public MomentMethod concept. */
template struct MOM::AllVariants::ConceptCheck<MOM::BasicThermoData>;

static_assert( MOM::HasReconstructedNDF<MOM::HMOM<MOM::BasicThermoData>>,
               "[MOM] HMOM must satisfy HasReconstructedNDF.");
static_assert( MOM::HasReconstructedNDF<MOM::ThreeEquations<MOM::BasicThermoData>>,
               "[MOM] ThreeEquations must satisfy HasReconstructedNDF.");
static_assert( MOM::HasReconstructedNDF<MOM::MetalOxide<MOM::BasicThermoData>>,
               "[MOM] MetalOxide must satisfy HasReconstructedNDF.");
static_assert(!MOM::HasReconstructedNDF<MOM::BrookesMoss<MOM::BasicThermoData>>,
               "[MOM] BrookesMoss must NOT satisfy HasReconstructedNDF.");
