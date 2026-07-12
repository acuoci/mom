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

/*-----------------------------------------------------------------------*\
|   MOM Library — Method of Moments for particle population dynamics      |
|   CRECK Modeling Group, Politecnico di Milano                           |
|                                                                         |
|   Master include. In most cases, including only this header is enough.  |
|                                                                         |
|   USAGE (compile-time method selection — zero overhead):                |
|                                                                         |
|     #include "MOM/MOM.hpp"                                              |
|                                                                         |
|     using ParticleModel = MOM::ThreeEquations<MOM::Thermo>;             |
|     // ↑ change this one line to switch to HMOM, BrookesMoss, or MetalOxide  |
|     static_assert(MOM::MomentMethod<ParticleModel>);                   |
|                                                                         |
|   USAGE (runtime method selection — one indirect branch per call):      |
|                                                                         |
|     auto model = MOM::MakeAnyMomentMethod(thermo, "HMOM");             |
|     MOM::ComputeCell(model, T, P, Y, mu, moments_span);  // preferred  |
|     auto src = MOM::GetSources(model);                                  |
|                                                                         |
\*-----------------------------------------------------------------------*/

#pragma once

// -- Core infrastructure -------------------------------------------------------
#include "ThermoProxy.hpp"         // ThermoMap concept + Thermo adapter
#include "ProcessFlags.hpp"        // Shared process model enum classes
#include "MomentMethodBase.hpp"    // CRTP base: shared state, Planck, zero sources
#include "MomentMethodConcept.hpp" // MomentMethod C++20 concept (the contract)

// -- Variant registry + concrete implementations -------------------------------
//
// AnyMomentMethod.hpp includes MomVariantList.hpp, which is the single
// authoritative registry of all concrete variants.  Adding a new variant
// requires only editing MomVariantList.hpp — no changes here.
#include "AnyMomentMethod.hpp"      // AnyMomentMethod<T>, MakeAnyMomentMethod, ForEachCell
#include "MomentMethodReporter.hpp" // Observer: formats output via concept interface only

// -- Free-function dispatch API -----------------------------------------------
//
// Each header below has a single responsibility and depends only on
// AnyMomentMethod.hpp (no cross-dependencies between them).  They are all
// gathered here so that including "MOM/MOM.hpp" is sufficient for all callers.
//
//  Dispatch.hpp   — SetState, Compute, ComputeCell, PrintSummary, …
//  Properties.hpp — GetVolumeFraction, GetParticleDiameter, GetSchmidtNumber, …
//  Sources.hpp    — GetSources, GetNucleationSources, GetOxidationModel, …
//  Splitting.hpp  — GetSourcesWithoutOxidation, GetOxidationRateCoefficients, …
#include "Dispatch.hpp"
#include "Properties.hpp"
#include "Sources.hpp"
#include "Splitting.hpp"

// ============================================================================
// Compile-time concept satisfaction check — auto-covers all registered variants
// ============================================================================
//
// This explicit instantiation forces MOM::AllVariants::ConceptCheck to be
// instantiated with BasicThermoData, triggering a static_assert fold over
// every type registered in AllVariants (MomVariantList.hpp).
//
// Effect: if any registered variant fails the MomentMethod concept, the build
// fails here with a clear error pointing at the offending type.
//
// To see which variant broke the concept: the fold expression fires
// individually for each type, so the compiler error includes the concrete
// type name in the template instantiation backtrace.
// ============================================================================

template struct MOM::AllVariants::ConceptCheck<MOM::BasicThermoData>;

// ============================================================================
// [T1] Compile-time HasReconstructedNDF concept matrix — fires in every TU
// ============================================================================
//
// Mirrors the matrix documented in MomentMethodConcept.hpp.  If a variant's
// NDF reconstruction status changes (added or removed), the build will fail
// here with a clear message pointing at the offending type.
// ============================================================================

static_assert( MOM::HasReconstructedNDF<MOM::HMOM<MOM::BasicThermoData>>,
               "[MOM] HMOM must satisfy HasReconstructedNDF.");
static_assert( MOM::HasReconstructedNDF<MOM::ThreeEquations<MOM::BasicThermoData>>,
               "[MOM] ThreeEquations must satisfy HasReconstructedNDF.");
static_assert( MOM::HasReconstructedNDF<MOM::MetalOxide<MOM::BasicThermoData>>,
               "[MOM] MetalOxide must satisfy HasReconstructedNDF.");
static_assert(!MOM::HasReconstructedNDF<MOM::BrookesMoss<MOM::BasicThermoData>>,
               "[MOM] BrookesMoss must NOT satisfy HasReconstructedNDF.");
