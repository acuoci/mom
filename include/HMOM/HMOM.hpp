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

#pragma once

#include "MOM/MomentMethodBase.hpp"
#include "MOM/ThermoProxy.hpp"
#include "Utilities/OutputFileColumns.h"
#include "Eigen/Dense"
#include <array>
#include <span>
#include <string>
#include <string_view>
#if defined(MOM_USE_DICTIONARY)
#  include <expected>
#endif

namespace MOM {

// ============================================================================
// HMOM<Thermo> — Hybrid Method of Moments, 4 transported equations
// ============================================================================
//
// Reference:
//   Mueller, Pitsch & Raman, Proc. Combust. Inst. 32 (2009) 785–792.
//   Attili, Bisetti, Mueller, Pitsch, Combust. Flame (2014).
//
// Transported moments (normalized, [mol/m3]):
//   moments[0] = M00  — zeroth-order moment (total number density proxy)
//   moments[1] = M10  — first-order volume moment
//   moments[2] = M01  — first-order surface moment
//   moments[3] = N0   — small-particle (nucleation mode) number density
//
// Physical processes modelled:
//   Nucleation    (PAH dimerization, free-molecular regime)
//   Surface growth (HACA mechanism, 5-step + efficiency)
//   Condensation  (PAH adsorption on existing soot)
//   Oxidation     (O2 + OH)
//   Coagulation   (discrete + continuous contributions)
//   Thermophoresis (via effective diffusion coefficient)
//
// @tparam Thermo  Must satisfy MOM::ThermoMap.
// ============================================================================

template <ThermoMap Thermo>
class HMOM : public MomentMethodBase<HMOM<Thermo>, 4>
{
    using Base = MomentMethodBase<HMOM<Thermo>, 4>;

public:
    using typename Base::MomentVector;

    /// Labels accepted by MOM::MakeAnyMomentMethod for runtime variant selection.
    static constexpr std::array<std::string_view, 2> variant_labels { "HMOM", "hmom" };

    // ── Method-specific sub-model enums ─────────────────────────────────────

    enum class StickingModel : int
    {
        Constant = 0,   //!< Fixed sticking coefficient (default: 2e-3)
        PAH4     = 1    //!< Sticking model based on PAH 4-ring collision cross-section
    };

    enum class FractalDiameterModel : int { Model0 = 0, Model1 = 1 };
    enum class CollisionDiameterModel : int { Model1 = 1, Model2 = 2 };

    // ── NDF two-node reconstruction ─────────────────────────────────────────
    //
    // HMOM reconstructs the NDF as two delta nodes:
    //   node[0] = small (nucleation) mode at (V0, S0)
    //   node[1] = large (coagulation/surface-growth) mode at (VL, SL)
    //
    // Mueller et al. (2009), Sec. 3.2.

    struct NDFNode {
        double number_density;     //!< N  [#/m3]
        double volume;             //!< V  [m3/#]  mean particle volume
        double surface;            //!< S  [m2/#]  mean particle surface
        double primary_diameter;   //!< dp [m]
        double primary_particles;  //!< np [-]
        double collision_diameter; //!< dc [m]
    };

    // ── Construction ────────────────────────────────────────────────────────

    /// Constructs HMOM bound to the given thermodynamics map.
    /// Does not allocate computational memory; call SetupFromDictionary()
    /// or the individual Set* methods before CalculateSourceMoments().
    explicit HMOM(const Thermo& thermo);

    /// Non-copyable (holds reference to external thermo and non-trivial state).
    HMOM(const HMOM&)            = delete;
    HMOM& operator=(const HMOM&) = delete;
    HMOM(HMOM&&)                 = default;
    HMOM& operator=(HMOM&&)      = default;

    /// Configure from a key-value dictionary.
    /// Returns an error string on failure; never throws in computational code.

#if defined(MOM_USE_DICTIONARY)
    template <typename Dictionary>
    [[nodiscard]] std::expected<void, std::string>
        SetupFromDictionary(Dictionary& dict);
#endif

    // ── MomentMethod concept — state injection ───────────────────────────────

    /// Inject thermodynamic state for the current cell.
    /// Extracts species concentrations used by HMOM (H, OH, O2, H2, H2O, C2H2, PAH).
    /// @param T    Temperature [K]
    /// @param P_Pa Pressure [Pa]
    /// @param Y    Species mass fractions (pointer to array, size = n_species)
    void SetStatus(double T, double P_Pa, const double* Y) noexcept;

    /// Generic moment setter (satisfies MomentMethod concept).
    /// @param m  Span of size 4: [M00_norm, M10_norm, M01_norm, N0_norm] in [mol/m3]
    void SetMoments(std::span<const double> m) noexcept;

    /// HMOM-specific named setter (preferred in HMOM-aware code).
    void SetNormalizedMoments(double M00_norm, double M10_norm,
                               double M01_norm, double N0_norm) noexcept;

    // ── MomentMethod concept — core computation ──────────────────────────────

    /// Computes all moment source terms and omega_gas for the current cell state.
    /// Updates source_all_, source_nucleation_, source_growth_, source_oxidation_,
    /// source_condensation_, source_coagulation_*, and omega_gas_.
    void CalculateSourceMoments() noexcept;

    /// Computes only gas-phase consumption terms (omega_gas_).
    /// Called internally by CalculateSourceMoments(); exposed for cases where
    /// source terms are already known and only gas coupling is needed.
    void CalculateOmegaGas() noexcept;

    // ── MomentMethod concept — particle properties ───────────────────────────

    [[nodiscard]] double VolumeFraction()        const noexcept;
    [[nodiscard]] double ParticleDiameter()      const noexcept;  //!< mean primary diameter [m]
    [[nodiscard]] double CollisionDiameter()     const noexcept;  //!< aggregate collision diameter [m]
    [[nodiscard]] double ParticleNumberDensity() const noexcept;  //!< total N = N0 + NL [#/m3]
    [[nodiscard]] double MassFraction()          const noexcept;
    [[nodiscard]] double SpecificSurface()       const noexcept;  //!< [m2/m3]
    [[nodiscard]] double NumberOfPrimaryParticles()const noexcept;
    [[nodiscard]] double DiffusionCoefficient()  const noexcept;  //!< [kg/m/s]

    // ── MomentMethod concept — initial conditions ────────────────────────────

    /// Returns the initial (near-zero) moment values for solver initialisation.
    /// Computed once during setup and cached; no allocation on repeated calls.
    [[nodiscard]] std::span<const double> initial_moments() const noexcept {
        return { initial_moments_cache_.data(), 4u };
    }

    // ── MomentMethod concept — precursor ─────────────────────────────────────

    [[nodiscard]] int         precursor_index()         const noexcept { return pah_index_; }
    [[nodiscard]] double      precursor_concentration() const noexcept { return conc_PAH_; }
    [[nodiscard]] const std::string& precursor_species()const noexcept { return pah_species_; }

    // ── MomentMethod concept — diagnostics ───────────────────────────────────

    void PrintSummary() const;

    // ── CRTP extension points — process source storage owned by HMOM ─────────
    //
    // Each `sources_X_impl()` method signals to MomentMethodBase that HMOM
    // models process X. The base class `sources_X()` getter detects these at
    // compile time via `if constexpr (requires ...)` and forwards here with
    // zero overhead. No `sources_sintering_impl()` is declared: HMOM does not
    // model sintering, so `sources_sintering()` returns a zero span automatically.

    [[nodiscard]] std::span<const double> sources_nucleation_impl()   const noexcept { return { source_nucleation_.data(),   this->n_equations }; }
    [[nodiscard]] std::span<const double> sources_coagulation_impl()  const noexcept { return { source_coagulation_.data(),  this->n_equations }; }
    [[nodiscard]] std::span<const double> sources_condensation_impl() const noexcept { return { source_condensation_.data(), this->n_equations }; }
    [[nodiscard]] std::span<const double> sources_growth_impl()       const noexcept { return { source_growth_.data(),       this->n_equations }; }
    [[nodiscard]] std::span<const double> sources_oxidation_impl()    const noexcept { return { source_oxidation_.data(),    this->n_equations }; }

    /// Discrete coagulation breakdown (small+small, small+large, large+large)
    [[nodiscard]] std::span<const double> sources_coagulation_discrete()    const noexcept;
    [[nodiscard]] std::span<const double> sources_coagulation_discrete_ss() const noexcept;
    [[nodiscard]] std::span<const double> sources_coagulation_discrete_sl() const noexcept;
    [[nodiscard]] std::span<const double> sources_coagulation_discrete_ll() const noexcept;

    /// Continuous coagulation breakdown
    [[nodiscard]] std::span<const double> sources_coagulation_continuous()    const noexcept;
    [[nodiscard]] std::span<const double> sources_coagulation_continuous_ss() const noexcept;
    [[nodiscard]] std::span<const double> sources_coagulation_continuous_sl() const noexcept;
    [[nodiscard]] std::span<const double> sources_coagulation_continuous_ll() const noexcept;

    // ── NDF reconstruction ────────────────────────────────────────────────────

    /// Two-node NDF reconstruction: node[0]=small mode, node[1]=large mode.
    [[nodiscard]] std::array<NDFNode, 2> NumberDensityFunctionNodes() const;

    [[nodiscard]] double SootSmallParticleNumberDensity()   const noexcept;  //!< N0 [#/m3]
    [[nodiscard]] double SootLargeParticleNumberDensity()   const noexcept;  //!< NL [#/m3]
    [[nodiscard]] double SootLargeParticleFraction()        const noexcept;  //!< NL/(N0+NL) [-]
    [[nodiscard]] double SootSmallParticleFraction()        const noexcept;  //!< N0/(N0+NL) [-]
    [[nodiscard]] double SootLargeMeanParticleVolume()      const noexcept;  //!< VL [m3/#]
    [[nodiscard]] double SootLargeMeanParticleSurface()     const noexcept;  //!< SL [m2/#]
    [[nodiscard]] double SootLargePrimaryParticleDiameter() const noexcept;  //!< [m]
    [[nodiscard]] double SootLargeNumberOfPrimaryParticles()const noexcept;  //!< [-]
	[[nodiscard]] double SootMeanParticleVolume() 			const noexcept;  //!< [m3/#]
	[[nodiscard]] double SootMeanParticleSurface() 			const noexcept;  //!< [m2/#]

    /// Log geometric std dev of primary particle diameter (Mueller 2009, Eq. 44).
    [[nodiscard]] double SootLogGeometricStdDevPrimaryParticleDiameter() const noexcept;
    /// Log geometric std dev of primary particle number (Mueller 2009, Eq. 45).
    [[nodiscard]] double SootLogGeometricStdDevPrimaryParticleNumber()   const noexcept;
    /// Scattering effective diameter d63 = 6*(M_{4,-3}/M_{1,0})^{1/3} (Mueller 2009, Eq. 46).
    [[nodiscard]] double SootD63() const noexcept;
	[[nodiscard]] double SootLargeLogGeometricStdDevPrimaryParticleDiameter() const noexcept;
	[[nodiscard]] double SootLargeLogGeometricStdDevPrimaryParticleNumber() const noexcept;

    /// Aggregated properties struct (convenience wrapper over individual accessors).
    void Properties(double& fv, double& dp, double& dc,
                    double& np, double& ss, double& vs) const noexcept;

    // ── Model switches ────────────────────────────────────────────────────────

    void SetNucleation(int flag) noexcept                { nucleation_model_              = flag; }
    void SetSurfaceGrowth(int flag) noexcept             { surface_growth_model_          = flag; }
    void SetCondensation(int flag) noexcept              { condensation_model_            = flag; }
    void SetOxidation(int flag) noexcept                 { oxidation_model_               = flag; }
    void SetCoagulation(int flag) noexcept               { coagulation_model_             = flag; }
    void SetCoagulationContinuous(int flag) noexcept     { coagulation_continuous_model_  = flag; }
    void SetFractalDiameterModel(int m) noexcept         { fractal_diameter_model_        = static_cast<FractalDiameterModel>(m); }
    void SetCollisionDiameterModel(int m) noexcept       { collision_diameter_model_      = static_cast<CollisionDiameterModel>(m); }

    void SetStickingCoefficientModel(std::string_view label);
    void SetStickingCoefficientConstant(double value) noexcept { sticking_coeff_constant_ = value; }

    void SetSurfaceDensity(double value) noexcept        { surface_density_ = value; }
    void SetSurfaceDensityCorrectionCoefficient(bool on) noexcept { surface_density_correction_ = on; }
    void SetSurfaceDensityCorrectionCoefficientA1(double v) noexcept { surf_dens_a1_ = v; }
    void SetSurfaceDensityCorrectionCoefficientA2(double v) noexcept { surf_dens_a2_ = v; }
    void SetSurfaceDensityCorrectionCoefficientB1(double v) noexcept { surf_dens_b1_ = v; }
    void SetSurfaceDensityCorrectionCoefficientB2(double v) noexcept { surf_dens_b2_ = v; }

    /// Sets the PAH precursor species by name (must be present in the thermo map).
    void SetPAH(std::string_view name);

    /// Sets the dummy gas-phase species used for mass-fraction closure.
    void SetGasClosureDummySpecies(std::string_view name);

    // ── HACA surface kinetics parameters ─────────────────────────────────────
    //
    // Arrhenius parameters for the 5-step HACA mechanism:
    //   R1f/b: H + Csoot-H <=> Csoot* + H2
    //   R2f/b: Csoot* + OH <=> Csoot-OH
    //   R3f/b: Csoot* + H  <=> Csoot-H
    //   R4:    Csoot* + C2H2 => Csoot-H + H
    //   R5:    Csoot* + O2  => 2CO + Csoot*
    //
    // Units: A [cm3, mol, s]; E [kJ/mol] (converted internally to [K]); n [-].

    void SetA1f(double v) noexcept; void SetA1b(double v) noexcept;
    void SetA2f(double v) noexcept; void SetA2b(double v) noexcept;
    void SetA3f(double v) noexcept; void SetA3b(double v) noexcept;
    void SetA4 (double v) noexcept;
    void SetA5 (double v) noexcept;

    void SetE1f(double kJ) noexcept; void SetE1b(double kJ) noexcept;
    void SetE2f(double kJ) noexcept; void SetE2b(double kJ) noexcept;
    void SetE3f(double kJ) noexcept; void SetE3b(double kJ) noexcept;
    void SetE4 (double kJ) noexcept;
    void SetE5 (double kJ) noexcept;

    void Setn1f(double v) noexcept; void Setn1b(double v) noexcept;
    void Setn2f(double v) noexcept; void Setn2b(double v) noexcept;
    void Setn3f(double v) noexcept; void Setn3b(double v) noexcept;
    void Setn4 (double v) noexcept;
    void Setn5 (double v) noexcept;
    void SetEfficiency6(double v) noexcept { eff6_ = v; }

    // ── Model state queries ───────────────────────────────────────────────────

    [[nodiscard]] int    nucleation_model()              const noexcept { return nucleation_model_; }
    [[nodiscard]] int    surface_growth_model()          const noexcept { return surface_growth_model_; }
    [[nodiscard]] int    condensation_model()            const noexcept { return condensation_model_; }
    [[nodiscard]] int    oxidation_model()               const noexcept { return oxidation_model_; }
    [[nodiscard]] int    coagulation_model()             const noexcept { return coagulation_model_; }
    [[nodiscard]] int    continuous_coagulation_model()  const noexcept { return coagulation_continuous_model_; }
    [[nodiscard]] double dimerization_rate()             const noexcept { return dimerization_rate_; }
    [[nodiscard]] double V0()                            const noexcept { return V0_; }
    [[nodiscard]] double S0()                            const noexcept { return S0_; }

	void WriteHeaderLine(MOM::OutputFileColumns& fOutput, const unsigned int precision);

	void WriteOutputLine( MOM::OutputFileColumns& fOutput,
							const double T, const double P_Pa, const double* Y, const double mu,
							const double* M);

private:
    // ── Private computational methods ─────────────────────────────────────────

    void MemoryAllocation();
    void Precalculations();
    void GetMoments();
    void DimerConcentration();
    void SootKineticConstants();
    void SootNucleationM4();
    void SootSurfaceGrowthM4();
    void SootOxidationM4();
    void SootCondensationM4();
    void SootCoagulationM4();
    void SootCoagulationSmallSmallM4();
    void SootCoagulationSmallLargeM4();
    void SootCoagulationLargeLargeM4();
    void SootCoagulationContinuousM4();
    void SootCoagulationContinuousSmallSmallM4(double lambda);
    void SootCoagulationContinuousSmallLargeM4(double lambda);
    void SootCoagulationContinuousLargeLargeM4(double lambda);
    void CalculateAlphaCoefficient();

    [[nodiscard]] double GetMoment(double i, double j)        const noexcept;
    [[nodiscard]] double GetMissingMoment(double i, double j) const noexcept;
    [[nodiscard]] double GetBetaC()                           const noexcept;
    [[nodiscard]] double PAHDimerizationRate()                const noexcept;
    [[nodiscard]] bool   HasSoot()                            const noexcept;
    [[nodiscard]] double SafePowPositive(double x, double a)  const noexcept;
    [[nodiscard]] double LogGeomStdDevFromMoments(
                             double M0, double M1, double M2) const noexcept;

private:
    // ── Thermodynamics reference ──────────────────────────────────────────────
    const Thermo& thermo_;

    // ── Transported (normalised) moments ─────────────────────────────────────
    double M00_normalized_ = 0.;   //!< [mol/m3]
    double M10_normalized_ = 0.;   //!< [mol/m3]
    double M01_normalized_ = 0.;   //!< [mol/m3]
    double N0_normalized_  = 0.;   //!< [mol/m3]

    // ── Reconstructed physical moments ───────────────────────────────────────
    double M00_ = 0.;   //!< [#/m3]
    double M10_ = 0.;   //!< [#]
    double M01_ = 0.;   //!< [#/m]
    double N0_  = 0.;   //!< [#/m3]   small-particle number density

    // Large-particle mode quantities (reconstructed from M00, M10, M01, N0)
    double NL_   = 0.;  //!< large-particle number density [#/m3]
    double NLVL_ = 0.;  //!< NL * mean volume of large particles [#]
    double NLSL_ = 0.;  //!< NL * mean surface of large particles [#/m]

    // ── Species concentrations [kmol/m3] ──────────────────────────────────────
    double conc_OH_   = 0., conc_H_    = 0., conc_H2O_  = 0.;
    double conc_H2_   = 0., conc_C2H2_ = 0., conc_O2_   = 0.;
    double conc_PAH_  = 0., conc_DIMER_= 0.;

    // 0-based species indices (-1 if absent in mechanism)
    int index_H_    = -1, index_OH_   = -1, index_O2_   = -1;
    int index_H2_   = -1, index_H2O_  = -1, index_C2H2_ = -1;

    // Mass fractions (needed for some surface rate expressions)
    double mass_fraction_H_  = 0.;
    double mass_fraction_OH_ = 0.;

    // ── PAH (precursor) properties ─────────────────────────────────────────────
    std::string pah_species_;
    int    pah_index_ = -1;
    double vpah_  = 0.;   //!< PAH molecule volume [m3]
    double spah_  = 0.;   //!< PAH molecule surface [m2]
    double dpah_  = 0.;   //!< PAH molecule diameter [m]
    double mpah_  = 0.;   //!< PAH molecule mass [kg]
    double mwpah_ = 0.;   //!< PAH molecular weight [kg/kmol]
    double ncpah_ = 0.;   //!< number of C atoms per PAH molecule
    double nhpah_ = 0.;   //!< number of H atoms per PAH molecule

    // ── Nucleated particle geometry ────────────────────────────────────────────
    double V0_ = 0.;    //!< nucleated particle volume [m3]
    double S0_ = 0.;    //!< nucleated particle surface [m2]
    double VC2_= 0.;    //!< volume of 2 C atoms [m3]

    // ── Dimer properties ───────────────────────────────────────────────────────
    double dimer_volume_       = 0.;
    double dimer_surface_      = 0.;
    double dimerization_rate_  = 0.;   //!< [mol/m3/s]

    // ── Kinetic intermediate quantities ───────────────────────────────────────
    double kox_     = 0.;    //!< total oxidation rate constant [1/s]
    double kox_O2_  = 0.;    //!< O2 contribution [1/s]
    double kox_OH_  = 0.;    //!< OH contribution [1/s]
    double ksg_     = 0.;    //!< surface growth rate constant
    double betaN_   = 0.;
    double Cfm_     = 0.;
    double betaN_TV_= 0.;

    // ── Fractal/collision geometry pre-factors ─────────────────────────────────
    double Av_fractal_    = 0., As_fractal_    = 0., K_fractal_    = 0.;
    double D_collisional_ = 0., Av_collisional_= 0.,
           As_collisional_= 0., K_collisional_ = 0.;

    // ── Surface density correction ─────────────────────────────────────────────
    bool   surface_density_correction_ = false;
    double surface_density_ = 1.7e19;  //!< [#/m2]
    double surf_dens_a1_ = 0., surf_dens_a2_ = 0.;
    double surf_dens_b1_ = 0., surf_dens_b2_ = 0.;
    double alpha_        = 1.;           //!< correction factor α

    // ── Model flags ────────────────────────────────────────────────────────────
    int nucleation_model_              = 0;
    int condensation_model_            = 0;
    int surface_growth_model_          = 0;
    int oxidation_model_               = 0;
    int coagulation_model_             = 0;
    int coagulation_continuous_model_  = 0;

    FractalDiameterModel   fractal_diameter_model_   = FractalDiameterModel::Model1;
    CollisionDiameterModel collision_diameter_model_ = CollisionDiameterModel::Model2;
    StickingModel          sticking_model_           = StickingModel::Constant;
    double                 sticking_coeff_constant_  = 2.e-3;

	bool is_debug_mode_              = false;  //!< enable verbose diagnostic output
    bool is_simplified_pah_mass_     = false;  //!< use Nc*WC instead of full PAH MW

    // ── Per-process source storage (owned by HMOM, not inherited from base) ───
    //
    // Only processes HMOM actually models are declared here.  The compile-time
    // CRTP dispatch in MomentMethodBase::sources_X() detects which _impl()
    // methods exist via `if constexpr (requires ...)` and selects the right
    // implementation (or a zero-span fallback) with no runtime overhead.
    //
    // HMOM models: nucleation, coagulation, condensation, growth, oxidation.
    // HMOM does NOT model: sintering → base class returns zero span automatically.

    MomentVector source_nucleation_   = MomentVector::Zero();
    MomentVector source_coagulation_  = MomentVector::Zero();
    MomentVector source_condensation_ = MomentVector::Zero();
    MomentVector source_growth_       = MomentVector::Zero();
    MomentVector source_oxidation_    = MomentVector::Zero();

    // ── HMOM-specific coagulation source breakdown ─────────────────────────────
    //
    // Coagulation in HMOM has discrete and continuous parts.
    // These are HMOM-specific vectors not present in other variants.

    MomentVector source_coagulation_discrete_    = MomentVector::Zero();
    MomentVector source_coagulation_ss_          = MomentVector::Zero();
    MomentVector source_coagulation_sl_          = MomentVector::Zero();
    MomentVector source_coagulation_ll_          = MomentVector::Zero();
    MomentVector source_coagulation_continuous_  = MomentVector::Zero();
    MomentVector source_coagulation_cont_ss_     = MomentVector::Zero();
    MomentVector source_coagulation_cont_sl_     = MomentVector::Zero();
    MomentVector source_coagulation_cont_ll_     = MomentVector::Zero();
    MomentVector source_coagulation_all_         = MomentVector::Zero();

    // ── Initial moments cache ──────────────────────────────────────────────────
    MomentVector initial_moments_cache_ = MomentVector::Zero();

    // ── HACA kinetics parameters ───────────────────────────────────────────────
    // Stored in [1/s], [cm3/mol/s] as appropriate; conversions in Set* methods.
    double A1f_=0., n1f_=0., E1f_=0.,  A1b_=0., n1b_=0., E1b_=0.;
    double A2f_=0., n2f_=0., E2f_=0.,  A2b_=0., n2b_=0., E2b_=0.;
    double A3f_=0., n3f_=0., E3f_=0.,  A3b_=0., n3b_=0., E3b_=0.;
    double A4_ =0., n4_ =0., E4_ =0.;
    double A5_ =0., n5_ =0., E5_ =0.;
    double eff6_= 0.;

    // ── Numerical floors (constexpr — zero cost) ───────────────────────────────
    static constexpr double kTinyNumberDensity = 1.e-30;   //!< [#/m3]
    static constexpr double kSootNumberFloor   = 1.e3;     //!< [#/m3]
    static constexpr double kSootVolumeFloor   = 1.e-40;   //!< [-]
    static constexpr double kSootSurfaceFloor  = 1.e-30;   //!< [m2/m3]
    static constexpr double kMomentEps         = 1.e-300;
};

} // namespace MOM

#if !defined(MOM_COMPILED_LIBRARY)
#  include "HMOM.tpp"
#else
namespace MOM 
{
    extern template class HMOM<BasicThermoData>;
}
#endif
