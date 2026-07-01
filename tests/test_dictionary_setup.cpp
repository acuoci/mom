/*-----------------------------------------------------------------------*\
|   MOM Library - Dictionary setup regression test                        |
\*-----------------------------------------------------------------------*/

// ParseConfig is a function template. This test intentionally instantiates it
// with a dictionary test double, so it must see the header-only .tpp bodies even
// when the linked MOM target exposes the precompiled-library macro.
#if defined(MOM_COMPILED_LIBRARY)
#undef MOM_COMPILED_LIBRARY
#endif

#include "MOM/MOM.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

#if !defined(MOM_USE_DICTIONARY)
#error "test_dictionary_setup.cpp must be built only with MOM_USE_DICTIONARY enabled"
#endif

namespace
{

struct FakeDictionary
{
    std::unordered_map<std::string, bool> bools;
    std::unordered_map<std::string, int> ints;
    std::unordered_map<std::string, double> doubles;
    std::unordered_map<std::string, std::string> strings;
    std::unordered_map<std::string, std::pair<double, std::string>> measures;
    bool grammar_was_set = false;

    template <typename Grammar>
    void SetGrammar(Grammar&)
    {
        grammar_was_set = true;
    }

    [[nodiscard]] bool CheckOption(const char* key) const
    {
        const std::string k{key};
        return bools.contains(k) || ints.contains(k) || doubles.contains(k) ||
               strings.contains(k) || measures.contains(k);
    }

    void ReadBool(const char* key, bool& value) const { value = bools.at(std::string{key}); }

    void ReadInt(const char* key, int& value) const { value = ints.at(std::string{key}); }

    void ReadDouble(const char* key, double& value) const { value = doubles.at(std::string{key}); }

    void ReadString(const char* key, std::string& value) const { value = strings.at(std::string{key}); }

    void ReadMeasure(const char* key, double& value, std::string& units) const
    {
        const auto& entry = measures.at(std::string{key});
        value = entry.first;
        units = entry.second;
    }
};

MOM::BasicThermoData buildSootThermo()
{
    MOM::BasicThermoData th;
    th.names = {"H", "OH", "O2", "H2", "H2O", "C2H2", "N2"};
    th.mw    = {1.008, 17.008, 31.999, 2.016, 18.015, 26.038, 28.014};
    th.nc    = {0, 0, 0, 0, 0, 2, 0};
    th.nh    = {1, 1, 0, 2, 2, 2, 0};
    th.no    = {0, 1, 2, 0, 1, 0, 0};
    th.nn    = {0, 0, 0, 0, 0, 0, 2};
    th.nti   = {0, 0, 0, 0, 0, 0, 0};
    return th;
}

MOM::BasicThermoData buildTiO2Thermo()
{
    MOM::BasicThermoData th;
    th.names = {"TiOH4", "N2"};
    th.mw    = {115.899, 28.014};
    th.nc    = {0, 0};
    th.nh    = {4, 0};
    th.no    = {4, 2};
    th.nn    = {0, 2};
    th.nti   = {1, 0};
    return th;
}

MOM::BasicThermoData buildTiO2GasThermo()
{
    MOM::BasicThermoData th;
    th.names = {"TiOH4", "H2O", "N2"};
    th.mw    = {115.899, 18.015, 28.014};
    th.nc    = {0, 0, 0};
    th.nh    = {4, 2, 0};
    th.no    = {4, 1, 0};
    th.nn    = {0, 0, 2};
    th.nti   = {1, 0, 0};
    return th;
}

FakeDictionary buildHMOMDictionary()
{
    FakeDictionary dict;

    dict.bools["@HMOM"] = true;
    dict.bools["@GasConsumption"] = false;
    dict.bools["@RadiativeHeatTransfer"] = false;
    dict.bools["@SimplifiedPAHMass"] = true;

    dict.ints["@NucleationModel"] = 0;
    dict.ints["@SurfaceGrowthModel"] = 1;
    dict.ints["@OxidationModel"] = 0;
    dict.ints["@CondensationModel"] = 1;
    dict.ints["@CoagulationModel"] = 0;
    dict.ints["@ContinuousCoagulationModel"] = 1;
    dict.ints["@ThermophoreticModel"] = 0;
    dict.ints["@FractalDiameterModel"] = 0;
    dict.ints["@CollisionDiameterModel"] = 1;

    dict.doubles["@SchmidtNumber"] = 42.5;
    dict.doubles["@StickingCoefficientConstant"] = 0.01;
    dict.doubles["@Efficiency6"] = 0.25;

    dict.strings["@PAH"] = "C2H2";
    dict.strings["@GasClosureDummySpecies"] = "none";
    dict.strings["@PlanckCoefficient"] = "none";
    dict.strings["@StickingCoefficientModel"] = "constant";

    dict.measures["@SootDensity"] = {2.0, "g/cm3"};
    dict.measures["@SurfaceDensity"] = {2.0, "#/cm2"};
    dict.measures["@A1f"] = {10.0, "cm3,mol,s"};
    dict.measures["@E1f"] = {20.0, "kJ/mol"};

    return dict;
}

FakeDictionary buildThreeEquationsDictionary()
{
    FakeDictionary dict;

    dict.bools["@ThreeEquations"] = true;
    dict.bools["@GasConsumption"] = false;
    dict.bools["@RadiativeHeatTransfer"] = false;
    dict.bools["@SimplifiedPAHMass"] = true;

    dict.ints["@NucleationModel"] = 0;
    dict.ints["@SurfaceGrowthModel"] = 1;
    dict.ints["@OxidationModel"] = 0;
    dict.ints["@CondensationModel"] = 1;
    dict.ints["@CoagulationModel"] = 0;
    dict.ints["@ThermophoreticModel"] = 0;

    dict.doubles["@SchmidtNumber"] = 41.0;
    dict.doubles["@epsNucleation"] = 2.0;
    dict.doubles["@epsCondensation"] = 3.0;
    dict.doubles["@epsCoagulation"] = 4.0;
    dict.doubles["@StickingCoefficientConstant"] = 0.02;
    dict.doubles["@CorrectionCoefficientPAHPAH"] = 1.5;

    dict.strings["@PAH"] = "C2H2";
    dict.strings["@GasClosureDummySpecies"] = "none";
    dict.strings["@PlanckCoefficient"] = "none";
    dict.strings["@SurfaceChemistryModel"] = "hmom";
    dict.strings["@DimerModel"] = "qssa-rodrigues";
    dict.strings["@StickingCoefficientModel"] = "constant";

    dict.measures["@SootDensity"] = {1.9, "g/cm3"};
    dict.measures["@MinimumNs"] = {2.0, "#/cm3"};

    return dict;
}

FakeDictionary buildBrookesMossDictionary()
{
    FakeDictionary dict;

    dict.bools["@BrookesMoss"] = true;
    dict.bools["@GasConsumption"] = true;
    dict.bools["@RadiativeHeatTransfer"] = false;

    dict.ints["@SurfaceGrowthModel"] = 1;
    dict.ints["@CoagulationModel"] = 0;
    dict.ints["@ThermophoreticModel"] = 0;

    dict.doubles["@SchmidtNumber"] = 40.0;
    dict.doubles["@Cbeta"] = 0.5;
    dict.doubles["@EtaColl"] = 0.06;
    dict.doubles["@Coxid"] = 0.02;
    dict.doubles["@NucleationExponent"] = 1.2;
    dict.doubles["@SurfaceGrowthExponent1"] = 1.1;
    dict.doubles["@SurfaceGrowthExponent2"] = 0.9;

    dict.strings["@NucleationModel"] = "BrookesMoss";
    dict.strings["@OxidationModel"] = "none";
    dict.strings["@Precursors"] = "C2H2";
    dict.strings["@SurfaceGrowthSpecies"] = "C2H2";
    dict.strings["@GasClosureDummySpecies"] = "none";
    dict.strings["@PlanckCoefficient"] = "none";

    dict.measures["@SootDensity"] = {1.8, "g/cm3"};
    dict.measures["@SootParticleDiameter"] = {2.0, "nm"};
    dict.measures["@SootParticleMolecularWeight"] = {150.0, "kg/kmol"};
    dict.measures["@NsNorm"] = {3.0, "#/cm3"};
    dict.measures["@Calpha"] = {55.0, "1/s"};
    dict.measures["@Talpha"] = {21001.0, "K"};
    dict.measures["@Cgamma"] = {11701.0, "kg*m/kmol/s"};
    dict.measures["@Tgamma"] = {12101.0, "K"};
    dict.measures["@Comega"] = {106.0, "kg*m/kmol/sqrt(K)/s"};

    return dict;
}

FakeDictionary buildTiO2Dictionary()
{
    FakeDictionary dict;

    dict.bools["@MetalOxide"] = true;
    dict.bools["@GasConsumption"] = false;
    dict.bools["@SinteringDeferred"] = true;

    dict.ints["@SinteringModel"] = 0;
    dict.ints["@CoagulationModel"] = 1;
    dict.ints["@CondensationModel"] = 0;
    dict.ints["@ThermophoreticModel"] = 0;
    dict.ints["@MinimumFormulaUnits"] = 3;
    dict.ints["@NucleatedParticleFormulaUnits"] = 6;

    dict.doubles["@SchmidtNumber"] = 39.0;
    dict.doubles["@MinimumFv"] = 1.e-20;
    dict.doubles["@ns"] = 1.25;
    dict.doubles["@SolidFormulaUnitsPerPrecursor"] = 2.0;

    dict.strings["@Precursor"] = "TiOH4";
    dict.strings["@SolidName"] = "GenericOxide";
    dict.strings["@GasClosureDummySpecies"] = "none";
    dict.strings["@NucleationModel"] = "fixed-cluster";

    dict.measures["@SolidMolecularWeight"] = {100.0, "kg/kmol"};
    dict.measures["@SolidDensity"] = {4.5, "g/cm3"};
    dict.measures["@SinteringDpMinimum"] = {3.0, "nm"};
    dict.measures["@SinteringTauMinimum"] = {1.e-8, "s"};
    dict.measures["@SinteringKMaximum"] = {1.e9, "1/s"};
    dict.measures["@As"] = {8.0e16, "s,K,m"};
    dict.measures["@Ts"] = {32000.0, "K"};
    dict.measures["@MinimumNs"] = {4.0, "#/cm3"};

    return dict;
}

FakeDictionary buildTiO2GasDictionary()
{
    FakeDictionary dict;

    dict.bools["@MetalOxide"] = true;
    dict.bools["@GasConsumption"] = true;

    dict.strings["@Precursor"] = "TiOH4";
    dict.strings["@GasClosureDummySpecies"] = "none";
    dict.strings["@NucleationModel"] = "binary";
    dict.strings["@GasStoichiometry"] = "\"TiOH4:-1,H2O:2\"";

    return dict;
}

void require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}

void requireNear(double value, double expected, const char* message)
{
    const double scale = std::max({1.0, std::abs(value), std::abs(expected)});
    if (std::abs(value - expected) > 1.e-12 * scale)
        throw std::runtime_error(message);
}

bool containsLabel(const std::vector<std::string>& labels, std::string_view wanted)
{
    return std::find(labels.begin(), labels.end(), wanted) != labels.end();
}

void checkHMOMDictionarySetup()
{
    const auto thermo = buildSootThermo();

    auto parse_dict = buildHMOMDictionary();
    auto cfg = MOM::HMOM<MOM::BasicThermoData>::ParseConfig(parse_dict);

    require(parse_dict.grammar_was_set, "ParseConfig did not install the HMOM grammar");
    require(cfg.has_value(), "HMOM ParseConfig returned an unexpected error");
    require(cfg->is_active, "@HMOM was not parsed");
    require(cfg->pah_species == "C2H2", "HMOM @PAH was not parsed");
    require(cfg->nucleation_model == 0, "HMOM @NucleationModel was not parsed");
    require(cfg->oxidation_model == 0, "HMOM @OxidationModel was not parsed");
    require(cfg->coagulation_model == 0, "HMOM @CoagulationModel was not parsed");
    require(cfg->thermophoretic_model == 0, "HMOM @ThermophoreticModel was not parsed");
    require(!cfg->gas_consumption, "HMOM @GasConsumption was not parsed");
    require(!cfg->radiative_heat_transfer, "HMOM @RadiativeHeatTransfer was not parsed");
    requireNear(cfg->soot_density_kg_m3, 2000.0, "HMOM @SootDensity unit conversion failed");
    requireNear(cfg->surface_density_per_m2, 2.0e4, "HMOM @SurfaceDensity unit conversion failed");
    requireNear(cfg->schmidt_number, 42.5, "HMOM @SchmidtNumber was not parsed");
    requireNear(cfg->A1f, 10.0, "HMOM @A1f was not parsed");
    requireNear(cfg->E1f, 20.0, "HMOM @E1f was not parsed");

    auto setup_dict = buildHMOMDictionary();
    auto model = MOM::MakeAnyMomentMethod<MOM::BasicThermoData>(thermo, "HMOM");
    MOM::SetupFromDictionary(model, setup_dict);

    require(setup_dict.grammar_was_set, "HMOM SetupFromDictionary did not invoke ParseConfig");

    const auto& hmom = std::get<MOM::HMOM<MOM::BasicThermoData>>(model);
    require(hmom.is_active(), "HMOM model was not activated");
    require(hmom.precursor_species() == "C2H2", "HMOM PAH species was not applied");
    require(hmom.nucleation_model() == 0, "HMOM nucleation flag was not applied");
    require(hmom.oxidation_model() == 0, "HMOM oxidation flag was not applied");
    require(hmom.coagulation_model() == 0, "HMOM coagulation flag was not applied");
    require(hmom.thermophoretic_model() == 0, "HMOM thermophoretic flag was not applied");
    require(!hmom.gas_consumption(), "HMOM gas-consumption flag was not applied");
    require(!hmom.radiative_heat_transfer(), "HMOM radiation flag was not applied");
    requireNear(hmom.particle_density(), 2000.0, "HMOM soot density was not applied");
    requireNear(hmom.schmidt_number(), 42.5, "HMOM Schmidt number was not applied");
}

void checkThreeEquationsDictionarySetup()
{
    const auto thermo = buildSootThermo();

    auto parse_dict = buildThreeEquationsDictionary();
    auto cfg = MOM::ThreeEquations<MOM::BasicThermoData>::ParseConfig(parse_dict);

    require(parse_dict.grammar_was_set, "ParseConfig did not install the ThreeEquations grammar");
    require(cfg.has_value(), "ThreeEquations ParseConfig returned an unexpected error");
    require(cfg->is_active, "ThreeEquations activation flag was not parsed");
    require(cfg->pah_species == "C2H2", "ThreeEquations @PAH was not parsed");
    require(cfg->nucleation_model == 0, "ThreeEquations @NucleationModel was not parsed");
    require(cfg->oxidation_model == 0, "ThreeEquations @OxidationModel was not parsed");
    require(cfg->coagulation_model == 0, "ThreeEquations @CoagulationModel was not parsed");
    require(cfg->thermophoretic_model == 0, "ThreeEquations @ThermophoreticModel was not parsed");
    require(!cfg->gas_consumption, "ThreeEquations @GasConsumption was not parsed");
    require(!cfg->radiative_heat_transfer, "ThreeEquations @RadiativeHeatTransfer was not parsed");
    requireNear(cfg->soot_density_kg_m3, 1900.0, "ThreeEquations @SootDensity conversion failed");
    requireNear(cfg->ns_minimum_per_m3, 2.0e6, "ThreeEquations @MinimumNs conversion failed");
    requireNear(cfg->schmidt_number, 41.0, "ThreeEquations @SchmidtNumber was not parsed");

    auto setup_dict = buildThreeEquationsDictionary();
    auto model = MOM::MakeAnyMomentMethod<MOM::BasicThermoData>(thermo, "ThreeEquations");
    MOM::SetupFromDictionary(model, setup_dict);

    require(setup_dict.grammar_was_set, "ThreeEquations SetupFromDictionary did not invoke ParseConfig");

    const auto& te = std::get<MOM::ThreeEquations<MOM::BasicThermoData>>(model);
    require(te.is_active(), "ThreeEquations model was not activated");
    require(te.precursor_species() == "C2H2", "ThreeEquations PAH species was not applied");
    require(te.nucleation_model() == 0, "ThreeEquations nucleation flag was not applied");
    require(te.oxidation_model() == 0, "ThreeEquations oxidation flag was not applied");
    require(te.coagulation_model() == 0, "ThreeEquations coagulation flag was not applied");
    require(te.thermophoretic_model() == 0, "ThreeEquations thermophoretic flag was not applied");
    require(!te.gas_consumption(), "ThreeEquations gas-consumption flag was not applied");
    require(!te.radiative_heat_transfer(), "ThreeEquations radiation flag was not applied");
    requireNear(te.particle_density(), 1900.0, "ThreeEquations soot density was not applied");
    requireNear(te.schmidt_number(), 41.0, "ThreeEquations Schmidt number was not applied");
}

void checkBrookesMossDictionarySetup()
{
    const auto thermo = buildSootThermo();

    auto parse_dict = buildBrookesMossDictionary();
    auto cfg = MOM::BrookesMoss<MOM::BasicThermoData>::ParseConfig(parse_dict);

    require(parse_dict.grammar_was_set, "ParseConfig did not install the BrookesMoss grammar");
    require(cfg.has_value(), "BrookesMoss ParseConfig returned an unexpected error");
    require(cfg->is_active, "BrookesMoss activation flag was not parsed");
    require(cfg->precursors_species == "C2H2", "BrookesMoss @Precursors was not parsed");
    require(cfg->surface_growth_species == "C2H2", "BrookesMoss @SurfaceGrowthSpecies was not parsed");
    require(cfg->nucleation_model == 1, "BrookesMoss @NucleationModel was not parsed");
    require(cfg->oxidation_model == 0, "BrookesMoss @OxidationModel was not parsed");
    require(cfg->coagulation_model == 0, "BrookesMoss @CoagulationModel was not parsed");
    require(cfg->thermophoretic_model == 0, "BrookesMoss @ThermophoreticModel was not parsed");
    require(cfg->gas_consumption, "BrookesMoss @GasConsumption was not parsed");
    require(!cfg->radiative_heat_transfer, "BrookesMoss @RadiativeHeatTransfer was not parsed");
    requireNear(cfg->soot_density_kg_m3, 1800.0, "BrookesMoss @SootDensity conversion failed");
    requireNear(cfg->soot_particle_diameter_m, 2.e-9, "BrookesMoss @SootParticleDiameter conversion failed");
    requireNear(cfg->ns_norm, 3.0e6, "BrookesMoss @NsNorm conversion failed");
    requireNear(cfg->schmidt_number, 40.0, "BrookesMoss @SchmidtNumber was not parsed");

    auto setup_dict = buildBrookesMossDictionary();
    auto model = MOM::MakeAnyMomentMethod<MOM::BasicThermoData>(thermo, "BrookesMoss");
    MOM::SetupFromDictionary(model, setup_dict);

    require(setup_dict.grammar_was_set, "BrookesMoss SetupFromDictionary did not invoke ParseConfig");

    const auto& bm = std::get<MOM::BrookesMoss<MOM::BasicThermoData>>(model);
    require(bm.is_active(), "BrookesMoss model was not activated");
    require(bm.precursor_species() == "C2H2", "BrookesMoss precursor species was not applied");
    require(bm.sg_species() == "C2H2", "BrookesMoss surface-growth species was not applied");
    require(bm.nucleation_model() == 1, "BrookesMoss nucleation flag was not applied");
    require(bm.oxidation_model() == 0, "BrookesMoss oxidation flag was not applied");
    require(bm.coagulation_model() == 0, "BrookesMoss coagulation flag was not applied");
    require(bm.thermophoretic_model() == 0, "BrookesMoss thermophoretic flag was not applied");
    require(bm.gas_consumption(), "BrookesMoss gas-consumption flag was not applied");
    require(!bm.radiative_heat_transfer(), "BrookesMoss radiation flag was not applied");
    requireNear(bm.particle_density(), 1800.0, "BrookesMoss soot density was not applied");
    requireNear(bm.schmidt_number(), 40.0, "BrookesMoss Schmidt number was not applied");
}

void checkTiO2DictionarySetup()
{
    const auto thermo = buildTiO2Thermo();

    auto parse_dict = buildTiO2Dictionary();
    auto cfg = MOM::TiO2<MOM::BasicThermoData>::ParseConfig(parse_dict);

    require(parse_dict.grammar_was_set, "ParseConfig did not install the TiO2 grammar");
    require(cfg.has_value(), "TiO2 ParseConfig returned an unexpected error");
    require(cfg->is_active, "TiO2 activation flag was not parsed");
    require(cfg->precursor_species == "TiOH4", "TiO2 @Precursor was not parsed");
    require(cfg->solid_name == "GenericOxide", "TiO2 @SolidName was not parsed");
    requireNear(cfg->solid_molecular_weight_kg_kmol, 100.0,
                "TiO2 @SolidMolecularWeight was not parsed");
    requireNear(cfg->solid_density_kg_m3, 4500.0, "TiO2 @SolidDensity conversion failed");
    requireNear(cfg->solid_formula_units_per_precursor, 2.0,
                "TiO2 @SolidFormulaUnitsPerPrecursor was not parsed");
    require(cfg->nucleation_model == "fixed-cluster", "TiO2 @NucleationModel was not parsed");
    require(cfg->sintering_model == 0, "TiO2 @SinteringModel was not parsed");
    require(cfg->coagulation_model == 1, "TiO2 @CoagulationModel was not parsed");
    require(cfg->condensation_model == 0, "TiO2 @CondensationModel was not parsed");
    require(cfg->thermophoretic_model == 0, "TiO2 @ThermophoreticModel was not parsed");
    require(!cfg->gas_consumption, "TiO2 @GasConsumption was not parsed");
    require(cfg->sintering_deferred, "TiO2 @SinteringDeferred was not parsed");
    requireNear(cfg->sintering_dp_min_m, 3.e-9, "TiO2 @SinteringDpMinimum conversion failed");
    requireNear(cfg->ns_minimum_per_m3, 4.e6, "TiO2 @MinimumNs conversion failed");
    requireNear(cfg->schmidt_number, 39.0, "TiO2 @SchmidtNumber was not parsed");

    auto setup_dict = buildTiO2Dictionary();
    auto model = MOM::MakeAnyMomentMethod<MOM::BasicThermoData>(thermo, "TiO2");
    MOM::SetupFromDictionary(model, setup_dict);

    require(setup_dict.grammar_was_set, "TiO2 SetupFromDictionary did not invoke ParseConfig");

    const auto& tio2 = std::get<MOM::TiO2<MOM::BasicThermoData>>(model);
    require(tio2.is_active(), "TiO2 model was not activated");
    require(tio2.precursor_species() == "TiOH4", "TiO2 precursor species was not applied");
    require(tio2.solid_name() == "GenericOxide", "TiO2 solid name was not applied");
    requireNear(tio2.solid_molecular_weight(), 100.0, "TiO2 solid molecular weight was not applied");
    requireNear(tio2.solid_density(), 4500.0, "TiO2 solid density was not applied");
    requireNear(tio2.solid_formula_units_per_precursor(), 2.0,
                "TiO2 formula units per precursor was not applied");
    require(tio2.nucleation_model() == 2, "TiO2 nucleation flag was not applied");
    require(tio2.sintering_model() == 0, "TiO2 sintering flag was not applied");
    require(tio2.coagulation_model() == 1, "TiO2 coagulation flag was not applied");
    require(tio2.condensation_model() == 0, "TiO2 condensation flag was not applied");
    require(tio2.thermophoretic_model() == 0, "TiO2 thermophoretic flag was not applied");
    require(!tio2.gas_consumption(), "TiO2 gas-consumption flag was not applied");
    requireNear(tio2.schmidt_number(), 39.0, "TiO2 Schmidt number was not applied");
}

void checkTiO2GasStoichiometrySetup()
{
    const auto thermo = buildTiO2GasThermo();

    auto parse_dict = buildTiO2GasDictionary();
    auto cfg = MOM::TiO2<MOM::BasicThermoData>::ParseConfig(parse_dict);

    require(parse_dict.grammar_was_set, "ParseConfig did not install the TiO2 gas grammar");
    require(cfg.has_value(), "TiO2 gas ParseConfig returned an unexpected error");
    require(cfg->gas_consumption, "TiO2 gas @GasConsumption was not parsed");
    require(cfg->gas_stoichiometry.size() == 2u, "TiO2 @GasStoichiometry was not parsed");
    require(cfg->gas_stoichiometry[0].species == "TiOH4",
            "TiO2 first gas-stoichiometry species was not parsed");
    requireNear(cfg->gas_stoichiometry[0].coefficient, -1.0,
                "TiO2 first gas-stoichiometry coefficient was not parsed");
    require(cfg->gas_stoichiometry[1].species == "H2O",
            "TiO2 second gas-stoichiometry species was not parsed");
    requireNear(cfg->gas_stoichiometry[1].coefficient, 2.0,
                "TiO2 second gas-stoichiometry coefficient was not parsed");

    auto setup_dict = buildTiO2GasDictionary();
    auto model = MOM::MakeAnyMomentMethod<MOM::BasicThermoData>(thermo, "TiO2");
    MOM::SetupFromDictionary(model, setup_dict);

    auto& tio2 = std::get<MOM::TiO2<MOM::BasicThermoData>>(model);
    require(tio2.gas_consumption(), "TiO2 explicit gas-consumption flag was not applied");

    const double Y[] = {0.10, 0.0, 0.90};
    tio2.SetStatus(1500.0, 101325.0, Y);
    tio2.CalculateSourceMoments();

    const auto omega = tio2.omega_gas();
    require(omega.size() == 3u, "TiO2 omega_gas size is wrong");
    require(omega[0] < 0.0, "TiO2 explicit stoichiometry did not consume precursor");
    require(omega[1] > 0.0, "TiO2 explicit stoichiometry did not produce H2O");
    requireNear(omega[1] / (-omega[0]), 2.0 * 18.015 / 115.899,
                "TiO2 explicit gas stoichiometry produced the wrong H2O/precursor mass ratio");

    bool rejected_unbalanced = false;
    try
    {
        MOM::TiO2<MOM::BasicThermoData>::Config bad;
        bad.precursor_species = "TiOH4";
        bad.gas_consumption = true;
        bad.gas_stoichiometry = {{"TiOH4", -1.0}};

        MOM::TiO2<MOM::BasicThermoData> bad_model(thermo);
        bad_model.SetupFromConfig(bad);
    }
    catch (const std::runtime_error&)
    {
        rejected_unbalanced = true;
    }
    require(rejected_unbalanced, "TiO2 accepted an unbalanced explicit gas stoichiometry");
}

void checkTiO2ReporterLabels()
{
    const auto thermo = buildTiO2GasThermo();

    MOM::TiO2<MOM::BasicThermoData> model(thermo);
    model.SetPrecursor("TiOH4");
    const std::vector<MOM::TiO2<MOM::BasicThermoData>::GasStoichiometryTerm> stoich{
        {"TiOH4", -1.0},
        {"H2O", 2.0}
    };
    model.SetGasStoichiometry(stoich);
    model.SetGasConsumption(true);

    std::vector<std::string> labels;
    model.variant_prefix_output(
        [&labels](std::string_view label, double)
        {
            labels.emplace_back(label);
        });

    require(containsLabel(labels, "omegaTot[kg/m3/s]"),
            "TiO2 reporter is missing total gas-source label");
    require(containsLabel(labels, "omegaPrecursor[kg/m3/s]"),
            "TiO2 reporter is missing generic precursor gas-source label");
    require(containsLabel(labels, "omegaGas(H2O)[kg/m3/s]"),
            "TiO2 reporter is missing generic gas-stoichiometry species label");
    require(!containsLabel(labels, "omegaPrec[kg/m3/s]"),
            "TiO2 reporter still exposes deprecated omegaPrec label");
    require(!containsLabel(labels, "omegaH2O[kg/m3/s]"),
            "TiO2 reporter still exposes fixed TiO2 H2O label");
}

void checkTiO2RequiresExplicitGasStoichiometry()
{
    const auto thermo = buildTiO2GasThermo();

    bool rejected_missing_stoichiometry = false;
    try
    {
        MOM::TiO2<MOM::BasicThermoData>::Config cfg;
        cfg.precursor_species = "TiOH4";
        cfg.gas_consumption = true;

        MOM::TiO2<MOM::BasicThermoData> model(thermo);
        model.SetupFromConfig(cfg);
    }
    catch (const std::runtime_error&)
    {
        rejected_missing_stoichiometry = true;
    }

    require(rejected_missing_stoichiometry,
            "TiO2 accepted gas consumption without explicit gas stoichiometry");
}

} // namespace

int main()
{
    try
    {
        checkHMOMDictionarySetup();
        checkThreeEquationsDictionarySetup();
        checkBrookesMossDictionarySetup();
        checkTiO2DictionarySetup();
        checkTiO2GasStoichiometrySetup();
        checkTiO2ReporterLabels();
        checkTiO2RequiresExplicitGasStoichiometry();

        std::cout << "[PASS] Dictionary parsing and setup paths for all MOM variants\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "[FAIL] " << e.what() << '\n';
        return 1;
    }
}
