/**
 * @file MOMConfig.hpp
 * @brief Shared plain-data configuration blocks used by MOM variant Config types.
 *
 * These structs collect configuration fields that are common across multiple
 * particle models. Concrete variants inherit only the blocks they need while
 * preserving direct member access such as `cfg.gas_consumption`.
 */

#pragma once

#include <string>
#include "ProcessFlags.hpp"

namespace MOM
{

/**
 * @brief Common activation, transport, and diagnostic controls.
 *
 * @tparam ThermophoreticDefault Default thermophoretic model for the variant
 *         (`ThermophoreticModel::Off` or `ThermophoreticModel::Standard`).
 */
template <ThermophoreticModel ThermophoreticDefault> struct CommonConfig
{
    bool is_active = true; //!< Enable this variant.

    ThermophoreticModel thermophoretic_model = ThermophoreticDefault; //!< Thermophoretic model.
    double schmidt_number                    = 50.;                   //!< Schmidt number for moment transport.

    bool debug_mode = false; //!< Verbose diagnostic output.
};

/**
 * @brief Common gas-consumption and closure controls.
 *
 * @tparam GasConsumptionDefault Variant-specific default for gas consumption.
 */
template <bool GasConsumptionDefault> struct GasConsumptionConfig
{
    bool gas_consumption = GasConsumptionDefault; //!< Consume gas-phase species.
    std::string gas_closure_dummy_species = "none"; //!< Dummy mass-closure species.
};

/** @brief Common optically-thin soot radiation controls. */
struct SootRadiationConfig
{
    bool radiative_heat_transfer = true;     //!< Optically-thin radiation.
    std::string planck_coefficient = "Smooke"; //!< Planck mean absorption coefficient.
};

/** @brief Common soot material density control. */
struct SootDensityConfig
{
    double soot_density_kg_m3 = 1800.; //!< Soot density [kg/m3].
};

/** @brief Common PAH setup used by soot variants that resolve a PAH species. */
struct PAHConfig : SootDensityConfig
{
    std::string pah_species = "C2H2"; //!< PAH growth species name.
    bool simplified_pah_mass = false; //!< Use Nc x WC instead of full PAH MW.
};

/** @brief Common binary soot-process switches. */
struct BinarySootProcessConfig
{
    NucleationModel    nucleation_model     = NucleationModel::Standard;    //!< Nucleation model.
    CondensationModel  condensation_model   = CondensationModel::Standard;  //!< Condensation model.
    SurfaceGrowthModel surface_growth_model = SurfaceGrowthModel::Standard; //!< Surface growth model.
    OxidationModel     oxidation_model      = OxidationModel::Standard;     //!< Oxidation model.
    CoagulationModel   coagulation_model    = CoagulationModel::Standard;   //!< Coagulation model.
};

/** @brief Common sticking-coefficient controls for PAH collision models. */
struct StickingConfig
{
    std::string sticking_model = "constant"; //!< Sticking model label.
    double sticking_coeff_constant = 2.e-3;  //!< Constant sticking coefficient [-].
};

/** @brief Collision enhancement factors shared by soot MOM variants. */
struct CollisionEnhancementConfig
{
    double eps_nucleation = 2.5;   //!< Nucleation enhancement factor [-].
    double eps_condensation = 1.3; //!< Condensation enhancement factor [-].
    double eps_coagulation = 2.2;  //!< Coagulation enhancement factor [-].
};

/** @brief Process switches for BrookesMoss/BrookesMoss-Hall. */
struct BrookesMossProcessConfig
{
    NucleationModel    nucleation_model     = NucleationModel::Standard;    //!< Off=0, Standard=BrookesMoss, Extended=BrookesMossHall.
    SurfaceGrowthModel surface_growth_model = SurfaceGrowthModel::Standard; //!< Surface growth model.
    OxidationModel     oxidation_model      = OxidationModel::Standard;     //!< Off=0, Standard=BrookesMoss, Extended=BrookesMossHall.
    CoagulationModel   coagulation_model    = CoagulationModel::Standard;   //!< Coagulation model.
};

} // namespace MOM
