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
|   Copyright (C) 2026 Alberto Cuoci, Benedetta Franzelli                 |
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

#include "HMOM/HMOM_Grammar.h"

/**
 * @file HMOM_Grammar.cpp
 * @brief Dictionary keyword registration for HMOM configuration.
 */

namespace MOM
{
void HMOM_Grammar::DefineRules()
{
    // Process and transport model switches.
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@HMOM",
                                              OpenSMOKEpp::SINGLE_BOOL,
                                              "Hybrid Method of Moments: on/off (default: true)",
                                              true));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@NucleationModel",
                                              OpenSMOKEpp::SINGLE_INT,
                                              "Nucleation model: 0=off, 1=on (default: 1)",
                                              false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@SurfaceGrowthModel",
                                              OpenSMOKEpp::SINGLE_INT,
                                              "Surface growth model: 0=off, 1=on (default: 1)",
                                              false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@OxidationModel",
                                              OpenSMOKEpp::SINGLE_INT,
                                              "Oxidation model: 0=off, 1=on (default: 1)",
                                              false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@CondensationModel",
                                              OpenSMOKEpp::SINGLE_INT,
                                              "Condensation model: 0=off, 1=on (default: 1)",
                                              false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@CoagulationModel",
                                              OpenSMOKEpp::SINGLE_INT,
                                              "Coagulation model: 0=off, 1=on (default: 1)",
                                              false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@ContinuousCoagulationModel",
                                              OpenSMOKEpp::SINGLE_INT,
                                              "Continuous coagulation model: 0=off, 1=on (default: 1)",
                                              false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@ThermophoreticModel",
                                              OpenSMOKEpp::SINGLE_INT,
                                              "Thermophoretic model: 0=off, 1=on (default: 1)",
                                              false));

    // Geometry and PAH precursor configuration.
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@FractalDiameterModel",
                                              OpenSMOKEpp::SINGLE_INT,
                                              "Fractal diameter model: 0=off, 1=on (default: 1)",
                                              false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@CollisionDiameterModel",
                                              OpenSMOKEpp::SINGLE_INT,
                                              "Collision diameter model: 1 or 2 (default: 2)",
                                              false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@PAH",
                                              OpenSMOKEpp::SINGLE_STRING,
                                              "Species to be assumed as PAH (example: @PAH C10H8;)",
                                              false));

    // Gas-phase coupling and radiative/transport options.
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@GasConsumption",
        OpenSMOKEpp::SINGLE_BOOL,
        "Consumption (and formation) of gaseous species is accounted for (default: true)",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@GasClosureDummySpecies",
        OpenSMOKEpp::SINGLE_STRING,
        "Species to be assumed as gaseous dummy species (example: @GasClosureDummySpecies N2;)",
        true));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@RadiativeHeatTransfer",
                                              OpenSMOKEpp::SINGLE_BOOL,
                                              "Radiative heat transfer (default: true)",
                                              false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@PlanckCoefficient",
        OpenSMOKEpp::SINGLE_STRING,
        "Calculation of Planck Coefficient: Smooke (default) | Kent | Sazhin | none",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@SchmidtNumber", OpenSMOKEpp::SINGLE_DOUBLE, "Schmidt number (default: 50)", false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@SimplifiedPAHMass",
        OpenSMOKEpp::SINGLE_BOOL,
        "Simplified calculation of PAH mass, based on C atoms only (default: false)",
        false));

    // Particle material and surface-density parameters.
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@SootDensity",
                                              OpenSMOKEpp::SINGLE_MEASURE,
                                              "Density of soot particles (default: 1800 kg/m3)",
                                              false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@SurfaceDensity",
                                              OpenSMOKEpp::SINGLE_MEASURE,
                                              "Surface density (default: 1.7e19 #/m2)",
                                              false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@SurfaceDensityCorrectionCoefficient",
                                              OpenSMOKEpp::SINGLE_BOOL,
                                              "SurfaceDensityCorrectionCoefficient (default: false)",
                                              false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@SurfaceDensityCorrectionCoefficientA1",
                                              OpenSMOKEpp::SINGLE_DOUBLE,
                                              "SurfaceDensityCorrectionCoefficientA1 (default: 12.65)",
                                              false));

    AddKeyWord(
        OpenSMOKEpp::DictionaryKeyWord("@SurfaceDensityCorrectionCoefficientA2",
                                       OpenSMOKEpp::SINGLE_DOUBLE,
                                       "SurfaceDensityCorrectionCoefficientA2 (default: -0.00563 1/K)",
                                       false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@SurfaceDensityCorrectionCoefficientB1",
                                              OpenSMOKEpp::SINGLE_DOUBLE,
                                              "SurfaceDensityCorrectionCoefficientB1 (default: -1.38 [-])",
                                              false));

    AddKeyWord(
        OpenSMOKEpp::DictionaryKeyWord("@SurfaceDensityCorrectionCoefficientB2",
                                       OpenSMOKEpp::SINGLE_DOUBLE,
                                       "SurfaceDensityCorrectionCoefficientB2 (default: 0.00068)",
                                       false));

    // Reaction 1: Soot-H + OH = Soot* + H2O
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@A1f",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Frequency factor reaction 1 (forward): Soot-H + OH = Soot* + H2O (default: 6.72e1 cm3,mol,s)",
        false));
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@A1b",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Frequency factor reaction 1 (backward): Soot-H + OH = Soot* + H2O (default: 6.44e-1 cm3,mol,s)",
        false));
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@n1f",
        OpenSMOKEpp::SINGLE_DOUBLE,
        "Temperature exponent reaction 1 (forward): Soot-H + OH = Soot* + H2O (default: 3.33)",
        false));
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@n1b",
        OpenSMOKEpp::SINGLE_DOUBLE,
        "Temperature exponent reaction 1 (backward): Soot-H + OH = Soot* + H2O (default: 3.79)",
        false));
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@E1f",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Activation energy reaction 1 (forward): Soot-H + OH = Soot* + H2O (default: 6.09 kJ/mol)",
        false));
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@E1b",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Activation energy reaction 1 (backward): Soot-H + OH = Soot* + H2O (default: 27.96 kJ/mol)",
        false));

    // Reaction 2: Soot-H + H = Soot* + H2
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@A2f",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Frequency factor reaction 2 (forward): Soot-H + H = Soot* + H2 (default: 1e8 cm3,mol,s)",
        false));
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@A2b",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Frequency factor reaction 2 (backward): Soot-H + H = Soot* + H2 (default: 8.68e4 cm3,mol,s)",
        false));
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@n2f",
        OpenSMOKEpp::SINGLE_DOUBLE,
        "Temperature exponent reaction 2 (forward): Soot-H + H = Soot* + H2 (default: 1.80)",
        false));
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@n2b",
        OpenSMOKEpp::SINGLE_DOUBLE,
        "Temperature exponent reaction 2 (backward): Soot-H + H = Soot* + H2 (default: 2.36)",
        false));
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@E2f",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Activation energy reaction 2 (forward): Soot-H + H = Soot* + H2 (default: 68.42 kJ/mol)",
        false));
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@E2b",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Activation energy reaction 2 (backward): Soot-H + H = Soot* + H2 (default: 25.46 kJ/mol)",
        false));

    // Reaction 3: Soot + H = Soot* + H
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@A3f",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Frequency factor reaction 3 (forward): Soot + H = Soot* + H (default: 1.13e16 cm3,mol,s)",
        false));
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@A3b",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Frequency factor reaction 3 (backward): Soot + H = Soot* + H (default: 4.17e13 cm3,mol,s)",
        false));
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@n3f",
        OpenSMOKEpp::SINGLE_DOUBLE,
        "Temperature exponent reaction 3 (forward): Soot + H = Soot* + H (default: -0.06)",
        false));
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@n3b",
        OpenSMOKEpp::SINGLE_DOUBLE,
        "Temperature exponent reaction 3 (backward): Soot + H = Soot* + H (default: 0.15)",
        false));
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@E3f",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Activation energy reaction 3 (forward): Soot + H = Soot* + H (default: 476.05 kJ/mol)",
        false));
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@E3b",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Activation energy reaction 3 (backward): Soot + H = Soot* + H (default: 0 kJ/mol)",
        false));

    // Reaction 4: Soot* + C2H2 => Soot-H
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@A4",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Frequency factor reaction 4: Soot* + C2H2 => Soot-H (default: 2.52e9 cm3,mol,s)",
        false));
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@n4",
        OpenSMOKEpp::SINGLE_DOUBLE,
        "Temperature exponent reaction 4: Soot* + C2H2 => Soot-H (default: 1.10)",
        false));
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@E4",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Activation energy reaction 4: Soot* + C2H2 => Soot-H (default: 17.13 kJ/mol)",
        false));

    // Reaction 5: Soot* + O2 => Soot-H + 2CO
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@A5",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Frequency factor reaction 5: Soot* + O2 => Soot-H + 2CO (default: 2.20e12 cm3,mol,s)",
        false));
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@n5",
        OpenSMOKEpp::SINGLE_DOUBLE,
        "Temperature exponent reaction 5: Soot* + O2 => Soot-H + 2CO (default: 0)",
        false));
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@E5",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Activation energy reaction 5: Soot* + O2 => Soot-H + 2CO (default: 31.38 kJ/mol)",
        false));

    // Reaction 6: Soot-H + OH => Soot-H + CO
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@Efficiency6",
        OpenSMOKEpp::SINGLE_DOUBLE,
        "Efficiency reaction 6: Soot-H + OH => Soot-H + CO (default: 0.13)",
        false));

    // Sticking coefficient.
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@StickingCoefficientModel",
        OpenSMOKEpp::SINGLE_STRING,
        "Sticking coefficient model: constant (default) | pah-dependent",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@StickingCoefficientConstant",
        OpenSMOKEpp::SINGLE_DOUBLE,
        "Sticking coefficient constant [-] (default: 2e-3 for constant model)",
        false));

    // Diagnostics.
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@DebugMode", OpenSMOKEpp::SINGLE_BOOL, "Debug mode (default: false)", false));
}
} // namespace MOM
