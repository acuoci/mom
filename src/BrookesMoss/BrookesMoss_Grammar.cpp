#include "BrookesMoss/BrookesMoss_Grammar.h"

/**
 * @file BrookesMoss_Grammar.cpp
 * @brief OpenSMOKE++ dictionary keyword rules for the BrookesMoss model.
 */

namespace MOM
{
void BrookesMoss_Grammar::DefineRules()
{

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@BrookesMoss",
        OpenSMOKEpp::SINGLE_BOOL,
        "BrookesMoss model: on/off (default: true)",
        true));

    // Process models

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@NucleationModel",
        OpenSMOKEpp::SINGLE_STRING,
        "Nucleation model: 0=none, 1=BrookesMoss, 2=BrookesMossHall (default: 1)",
        true));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@SurfaceGrowthModel",
        OpenSMOKEpp::SINGLE_INT,
        "Surface growth model: 0=off, 1=on (default: 1)",
        true));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@OxidationModel",
        OpenSMOKEpp::SINGLE_STRING,
        "Oxidation model: 0=none, 1=BrookesMoss, 2=BrookesMossHall (default: 1)",
        true));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@CoagulationModel",
        OpenSMOKEpp::SINGLE_INT,
        "Coagulation model: 0=off, 1=on (default: 1)",
        true));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@ThermophoreticModel",
        OpenSMOKEpp::SINGLE_INT,
        "Thermophoretic model: 0=off, 1=on (default: 0)",
        true));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@DebugMode",
        OpenSMOKEpp::SINGLE_BOOL,
        "Enable verbose diagnostic output (default: false)",
        false));

    // Gas species

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@Precursors",
        OpenSMOKEpp::SINGLE_STRING,
        "Species to be assumed as soot precursors (example: @Precursors C2H2;)",
        true));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@SurfaceGrowthSpecies",
        OpenSMOKEpp::SINGLE_STRING,
        "Species to be assumed as participating surface growth species (example: @SurfaceGrowthSpecies C2H2;)",
        true));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@Benzene",
        OpenSMOKEpp::SINGLE_STRING,
        "Benzene species used by BrookesMossHall kinetics",
        false));  
        
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@PhenylRadical",
        OpenSMOKEpp::SINGLE_STRING,
        "Phenyl radical species used by BrookesMossHall kinetics",
        false));          

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@SootParticleDiameter", 
        OpenSMOKEpp::SINGLE_MEASURE, 
        "Soot particle diameter (default: 1 nm)", 
        true));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@SootParticleMolecularWeight",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Soot particle molecular weight (default: 144 kg/kmol)",
        true));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@GasClosureDummySpecies",
        OpenSMOKEpp::SINGLE_STRING,
        "Species to be assumed as gaseous dummy species (example: @GasClosureDummySpecies CSOOT;)",
        true));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@GasConsumption",
        OpenSMOKEpp::SINGLE_BOOL,
        "Consumption of gaseous species is accounted for (default: false)",
        true));

    // Transport and radiation

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@RadiativeHeatTransfer",
        OpenSMOKEpp::SINGLE_BOOL,
        "Radiative heat transfer (default: true)",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@PlanckCoefficient",
        OpenSMOKEpp::SINGLE_STRING,
        "Planck coefficient model: Smooke (default) | Kent | Sazhin | none",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@SchmidtNumber", 
        OpenSMOKEpp::SINGLE_DOUBLE, 
        "Schmidt number (default: 50)", 
        false));

    // Soot properties

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@SootDensity",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Density of soot particles (default: 1800 kg/m3)",
        false));

    // Brookes-Moss kinetic constants

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@Calpha",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Soot inception rate constant (default: 54 1/s)",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@Talpha",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Activation temperature of soot inception (default: 21000 K)",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@Cbeta",
        OpenSMOKEpp::SINGLE_DOUBLE,
        "Model constant for coagulation rate (default: 1)",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@Cgamma",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Surface growth rate scaling factor (default: 11700 kg*m/kmol/s)",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@Tgamma",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Activation temperature of surface growth (default: 12100 K)",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@Comega",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Oxidation model constant (default: 105.8125 kgm/kmol/sqrt(K)/s)",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@EtaColl",
        OpenSMOKEpp::SINGLE_DOUBLE,
        "Collisional efficiency parameter (default: 0.04)",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@Coxid",
        OpenSMOKEpp::SINGLE_DOUBLE,
        "Oxidation rate scaling parameter (default: 0.015)",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@NucleationExponent",
        OpenSMOKEpp::SINGLE_DOUBLE,
        "Exponent of concentration in nucleation (default: 1)",
        false));

    AddKeyWord(
        OpenSMOKEpp::DictionaryKeyWord(
            "@SurfaceGrowthExponent1",
            OpenSMOKEpp::SINGLE_DOUBLE,
            "Exponent of concentration in surface growth (default: 1)",
            false));

    AddKeyWord(
        OpenSMOKEpp::DictionaryKeyWord(
            "@SurfaceGrowthExponent2",
            OpenSMOKEpp::SINGLE_DOUBLE,
            "Exponent of mass concentration in surface growth (default: 1)",
            false));

    // Number-density normalization

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@NsNorm", 
        OpenSMOKEpp::SINGLE_MEASURE,
        "Normalization factor (default: 1e15 #/m3)", 
        false));

    // Brookes-Moss-Hall kinetic constants
    
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@Calpha1",
        OpenSMOKEpp::SINGLE_MEASURE,
        "BrookesMossHall channel-1 inception rate constant (default: 127*10^8.88 kg*m3/kmol2/s)",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@Talpha1",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Activation temperature of soot inception (default: 4378 K)",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@Calpha2",
        OpenSMOKEpp::SINGLE_MEASURE,
        "BrookesMossHall channel-2 inception rate constant (default: 178*10^9.50 kg*m3/kmol2/s)",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@Talpha2",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Activation temperature of soot inception (default: 6390 K)",
        false));
        
    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@Comega2",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Model constant for soot oxidation (default: 8903.51 kg*m/kmol/s/sqrt(K))",
        false));

    AddKeyWord(OpenSMOKEpp::DictionaryKeyWord(
        "@Tomega2",
        OpenSMOKEpp::SINGLE_MEASURE,
        "Activation temperature of soot oxidation (default: 19778 K)",
        false));        
}
} // namespace MOM
