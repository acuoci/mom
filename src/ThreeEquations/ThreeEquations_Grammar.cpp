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

#include "ThreeEquations/ThreeEquations_Grammar.h"

namespace MOM
{
void ThreeEquations_Grammar::DefineRules()
{

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@ThreeEquations",
                                              OpenSMOKEpp::SINGLE_BOOL,
                                              "Three equations model: on/off (default: true)",
                                              true));

    // ----------------------------------------------------------------------------------------------------------- //
    // Soot models
    // ----------------------------------------------------------------------------------------------------------- //

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

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@ThermophoreticModel",
                                              OpenSMOKEpp::SINGLE_INT,
                                              "Thermophoretic model: 0=off, 1=on (default: 1)",
                                              false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@SurfaceChemistryModel",
                                              OpenSMOKEpp::SINGLE_STRING,
                                              "Surface chemistry model (RC-PAH (default) | HMOM)",
                                              false));

    // ----------------------------------------------------------------------------------------------------------- //
    // PAHs
    // ----------------------------------------------------------------------------------------------------------- //

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@PAH",
                                              OpenSMOKEpp::SINGLE_STRING,
                                              "Species to be assumed as PAH (example: @PAH C10H8;)",
                                              true));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@GasClosureDummySpecies",
        OpenSMOKEpp::SINGLE_STRING,
        "Species to be assumed as gaseous dummy species (example: @GasClosureDummySpecies TIO2RU;)",
        true));

    AddKeyWord(
        OpenSMOKEpp::DictionaryKeyWord("@GasConsumption",
                                       OpenSMOKEpp::SINGLE_BOOL,
                                       "Consumption of gaseous species is accounted for (default: true)",
                                       true));

    // ----------------------------------------------------------------------------------------------------------- //
    // Additional sub-models
    // ----------------------------------------------------------------------------------------------------------- //

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

    // ----------------------------------------------------------------------------------------------------------- //
    // Soot properties
    // ----------------------------------------------------------------------------------------------------------- //

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@SootDensity",
                                              OpenSMOKEpp::SINGLE_MEASURE,
                                              "Density of soot particles (default: 1800 kg/m3)",
                                              false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@SimplifiedPAHMass",
        OpenSMOKEpp::SINGLE_BOOL,
        "Simplified calculation of PAH mass, based on C atoms only (default: false)",
        false));

    // ----------------------------------------------------------------------------------------------------------- //
    // Collision enhancement factors
    // ----------------------------------------------------------------------------------------------------------- //

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@epsNucleation",
                                              OpenSMOKEpp::SINGLE_DOUBLE,
                                              "Collision enhancement factor for nucleation",
                                              false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@epsCondensation",
                                              OpenSMOKEpp::SINGLE_DOUBLE,
                                              "Collision enhancement factor for condensation",
                                              false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@epsCoagulation",
                                              OpenSMOKEpp::SINGLE_DOUBLE,
                                              "Collision enhancement factor for coagulation",
                                              false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@CorrectionCoefficientPAHPAH",
        OpenSMOKEpp::SINGLE_DOUBLE,
        "Correction coefficient for PAH-PAH nucleation kernel (from HMOM, default: 4.4)",
        false));

    // ----------------------------------------------------------------------------------------------------------- //
    // Minimum values for properties calculation
    // ----------------------------------------------------------------------------------------------------------- //

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@MinimumNs",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Minimum Ns for the calculation of soot properties (default: 1e6 #/m3)",
        false));

    // ----------------------------------------------------------------------------------------------------------- //
    // Dimer model
    // ----------------------------------------------------------------------------------------------------------- //

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord("@DimerModel",
                                              OpenSMOKEpp::SINGLE_STRING,
                                              "Dimer concentration model (qssa-rodrigues)",
                                              false));

    // ----------------------------------------------------------------------------------------------------------- //
    // Sticking coefficient
    // ----------------------------------------------------------------------------------------------------------- //

    AddKeyWord(
        OpenSMOKEpp::DictionaryKeyWord("@StickingCoefficientModel",
                                       OpenSMOKEpp::SINGLE_STRING,
                                       "Sticking coefficient model (constant (default) | pah-dependent)",
                                       false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@StickingCoefficientConstant",
        OpenSMOKEpp::SINGLE_DOUBLE,
        "Sticking coefficient constant, dimensionless in case of constant (default 2e-3), in kmol4/kg4 in case of pah-dependent (default 1.5e-11)",
        false));

    // ----------------------------------------------------------------------------------------------------------- //
    // Debug mode
    // ----------------------------------------------------------------------------------------------------------- //

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@DebugMode", OpenSMOKEpp::SINGLE_BOOL, "Debug mode (default: false)", false));
}
} // namespace MOM
