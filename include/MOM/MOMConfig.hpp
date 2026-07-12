/*-----------------------------------------------------------------------*\
|   Shared configuration building blocks for MOM variants                  |
\*-----------------------------------------------------------------------*/

#pragma once

#include <string>

namespace MOM
{

/**
 * @brief Common activation, transport, and diagnostic controls.
 *
 * @tparam ThermophoreticDefault Integer default for the variant's
 *         thermophoretic model. BrookesMoss historically defaults to 0,
 *         while HMOM, ThreeEquations, and MetalOxide default to 1.
 */
template <int ThermophoreticDefault> struct CommonConfig
{
    bool is_active = true; //!< Enable this variant.

    int thermophoretic_model = ThermophoreticDefault; //!< Thermophoretic model index.
    double schmidt_number    = 50.;                   //!< Schmidt number for moment transport.

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
    int nucleation_model     = 1; //!< Nucleation model.
    int condensation_model   = 1; //!< Condensation model.
    int surface_growth_model = 1; //!< Surface growth model.
    int oxidation_model      = 1; //!< Oxidation model.
    int coagulation_model    = 1; //!< Coagulation model.
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
    int nucleation_model     = 1; //!< 0=off, 1=BrookesMoss, 2=BrookesMossHall.
    int surface_growth_model = 1; //!< Surface growth model index.
    int oxidation_model      = 1; //!< 0=off, 1=BrookesMoss, 2=BrookesMossHall.
    int coagulation_model    = 1; //!< Coagulation model index.
};

} // namespace MOM
