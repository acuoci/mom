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

#include "TiO2/TiO2_Grammar.h"

namespace MOM
{

void TiO2_Grammar::DefineRules()
{
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@TiO2", 
        OpenSMOKEpp::SINGLE_BOOL, 
        "TiO2 model: on/off (default: true)", 
        true));

    // ----------------------------------------------------------------------------------------------------------- //
    // Soot models
    // ----------------------------------------------------------------------------------------------------------- //

    AddKeyWord(
        OpenSMOKEpp::DictionaryKeyWord(
            "@NucleationModel",
            OpenSMOKEpp::SINGLE_STRING,
            "Nucleation model: 0=off, 1=binary, 2=fixed-cluster (default: 1)",
            false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@SinteringModel",
        OpenSMOKEpp::SINGLE_INT,
        "Sintering model: 0=off, 1=on (default: 1)",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@CoagulationModel",
        OpenSMOKEpp::SINGLE_INT,
        "Coagulation model: 0=off, 1=on (default: 1)",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@CondensationModel",
        OpenSMOKEpp::SINGLE_INT,
        "Condensation model: 0=off, 1=on (default: 1)",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@ThermophoreticModel",
        OpenSMOKEpp::SINGLE_INT,
        "Thermophoretic model: 0=off, 1=on (default: 1)",
        false));

    // ----------------------------------------------------------------------------------------------------------- //
    // Precursor
    // ----------------------------------------------------------------------------------------------------------- //

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@Precursor",
        OpenSMOKEpp::SINGLE_STRING,
        "Species to be assumed as precursor (example: @Precursor H4O4TI;)",
        true));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@GasClosureDummySpecies",
        OpenSMOKEpp::SINGLE_STRING,
        "Species to be assumed as gaseous dummy species (example: @GasClosureDummySpecies TIO2RU;)",
        true));

    AddKeyWord(
        OpenSMOKEpp::DictionaryKeyWord(
            "@GasConsumption",
            OpenSMOKEpp::SINGLE_BOOL,
            "Consumption of gaseous species is accounted for (default: false)",
            true));

    // ----------------------------------------------------------------------------------------------------------- //
    // Sintering kinetics
    // ----------------------------------------------------------------------------------------------------------- //

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@As",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Sintering frequency factor (s,K,m). Default: 7.44e16 1/s/K/m^(-4)",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@ns", OpenSMOKEpp::SINGLE_DOUBLE, 
        "Sintering temperature exponent. Default: 1", 
        false));
    
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@Ts",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Sintering activation temperature (K). Default: 31000. K",
        false));

    // ----------------------------------------------------------------------------------------------------------- //
    // Sintering options
    // ----------------------------------------------------------------------------------------------------------- //

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@SinteringDeferred",
        OpenSMOKEpp::SINGLE_BOOL,
        "Sintering source term is deferred. Default: false",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@SinteringDpMinimum",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Sintering minimum diameter. Default: 2e-9 m",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@SinteringTauMinimum",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Sintering minimum time. Default: 1e-10 s",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@SinteringKMaximum",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Sintering maximum relative rate. Default: 1e6 1/s",
        false));

    // ----------------------------------------------------------------------------------------------------------- //
    // Additional options
    // ----------------------------------------------------------------------------------------------------------- //

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@SchmidtNumber", 
        OpenSMOKEpp::SINGLE_DOUBLE, 
        "Schmidt number (default: 50)", 
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@MinimumTiO2Units", 
        OpenSMOKEpp::SINGLE_INT, 
        "Minimum TiO2 units (default: 2)", 
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@NucleatedParticleTiO2Units",
        OpenSMOKEpp::SINGLE_INT,
        "Number of TiO2 units in the nucleated particle (default: 5)",
        false));

    // ----------------------------------------------------------------------------------------------------------- //
    // Minimum values for properties calculation
    // ----------------------------------------------------------------------------------------------------------- //

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@MinimumNs",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Minimum Ns for the calculation of TiO2 properties (default: 1e3 #/m3)",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@MinimumFv",
        OpenSMOKEpp::SINGLE_DOUBLE,
        "Minimum fv for the calculation of TiO2 properties (default: 1e-16)",
        false));

    // ----------------------------------------------------------------------------------------------------------- //
    // Debug mode
    // ----------------------------------------------------------------------------------------------------------- //

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@DebugMode", 
        OpenSMOKEpp::SINGLE_BOOL, 
        "Debug mode (default: false)", 
        false));
}

} // namespace MOM
