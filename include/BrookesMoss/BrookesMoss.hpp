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
|   Copyright (C) 2026 Alberto Cuoci                                      |
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
#include <span>
#include <string>
#include <string_view>
#if defined(MOM_USE_DICTIONARY)
#  include <expected>
#endif

namespace MOM {

// ============================================================================
// BrookesMoss<Thermo> — 2-equation soot model
// ============================================================================
//
// References:
//   Brookes & Moss, Combust. Flame 116 (1999) 486–503.
//   Hall et al., Combust. Flame 169 (2016) 191–206.  (BM-Hall variant)
//
// Transported variables:
//   moments[0] = Ys   — soot mass fraction [-]
//   moments[1] = bs   — normalised radical nuclei concentration [m3/kg]
//                       (bs = Ns / (rho * Ns_norm))
//
// Physical processes modelled:
//   Nucleation    (Brookes-Moss or Brookes-Moss-Hall variant)
//   Surface growth (C2H2 deposition)
//   Oxidation     (O2 or BM-Hall variant)
//   Coagulation   (free-molecular regime)
//   Thermophoresis (via effective diffusion coefficient)
//
// Note: condensation is not modelled in the Brookes-Moss family.
//       sources_condensation() from base returns a zero span.
//
// @tparam Thermo  Must satisfy MOM::ThermoMap.
// ============================================================================

template <ThermoMap Thermo>
class BrookesMoss : public MomentMethodBase<BrookesMoss<Thermo>, 2>
{
    using Base = MomentMethodBase<BrookesMoss<Thermo>, 2>;

public:
    using typename Base::MomentVector;

    /// Labels accepted by MOM::MakeAnyMomentMethod for runtime variant selection.
    static constexpr std::array<std::string_view, 3> variant_labels { "BrookesMoss", "brookesmoss", "BM" };

    // ── Method-specific sub-model enums ─────────────────────────────────────

    enum class NucleationVariant : int
    {
        Off              = 0,
        BrookesMoss      = 1,   //!< Original Brookes & Moss (1999) nucleation
        BrookesMossHall  = 2    //!< Hall et al. (2016) extended nucleation
    };

    enum class OxidationVariant : int
    {
        Off              = 0,
        BrookesMoss      = 1,   //!< Original Brookes & Moss (1999) oxidation
        BrookesMossHall  = 2    //!< Hall et al. (2016) extended oxidation
    };

    // ── Construction ─────────────────────────────────────────────────────────

    explicit BrookesMoss(const Thermo& thermo);

    BrookesMoss(const BrookesMoss&)            = delete;
    BrookesMoss& operator=(const BrookesMoss&) = delete;
    BrookesMoss(BrookesMoss&&)                 = default;
    BrookesMoss& operator=(BrookesMoss&&)      = default;

#if defined(MOM_USE_DICTIONARY)
    template <typename Dictionary>
    [[nodiscard]] std::expected<void, std::string>
        SetupFromDictionary(Dictionary& dict);
#endif

    // ── MomentMethod concept — state injection ────────────────────────────────

    /// @param T    Temperature [K]
    /// @param P_Pa Pressure [Pa]
    /// @param Y    Mass fractions, size = n_species
    void SetStatus(double T, double P_Pa, const double* Y) noexcept;

    /// Generic span setter. Order: [Ys, bs].
    /// Ys [-], bs [m3/kg].
    void SetMoments(std::span<const double> m) noexcept;

    /// Named setter (preferred for BrookesMoss-aware code).
    void SetMoments(double Ys, double bs) noexcept;

    // ── MomentMethod concept — core computation ───────────────────────────────

    void CalculateSourceMoments() noexcept;
    void CalculateOmegaGas() noexcept;

    // ── MomentMethod concept — particle properties ────────────────────────────

    [[nodiscard]] double VolumeFraction()        const noexcept;
    [[nodiscard]] double ParticleDiameter()      const noexcept;  //!< mean particle diameter [m]
    [[nodiscard]] double CollisionDiameter()     const noexcept;  //!< same as ParticleDiameter for BM
    [[nodiscard]] double ParticleNumberDensity() const noexcept;  //!< [#/m3]
    [[nodiscard]] double MassFraction()          const noexcept;  //!< = Ys_
    [[nodiscard]] double SpecificSurface()       const noexcept;  //!< [m2/m3]
    [[nodiscard]] double DiffusionCoefficient()  const noexcept;  //!< [kg/m/s]

    // ── MomentMethod concept — initial conditions ─────────────────────────────

    [[nodiscard]] std::span<const double> initial_moments() const noexcept {
        return { initial_moments_cache_.data(), 2u };
    }

    // ── MomentMethod concept — precursor ──────────────────────────────────────

    [[nodiscard]] int         precursor_index()         const noexcept { return prec_index_; }
    [[nodiscard]] double      precursor_concentration() const noexcept { return conc_prec_;  }
    [[nodiscard]] const std::string& precursor_species()const noexcept { return prec_species_; }

    // ── MomentMethod concept — diagnostics ────────────────────────────────────

    void PrintSummary() const;

    // ── Model switches ────────────────────────────────────────────────────────

    void SetNucleation(int flag) noexcept          { nucleation_variant_ = static_cast<NucleationVariant>(flag); }
    void SetNucleation(std::string_view label);
    void SetSurfaceGrowth(int flag) noexcept       { surface_growth_model_ = flag; }
    void SetOxidation(int flag) noexcept           { oxidation_variant_ = static_cast<OxidationVariant>(flag); }
    void SetOxidation(std::string_view label);
    void SetCoagulation(int flag) noexcept         { coagulation_model_ = flag; }

    void SetPrecursors(std::string_view name);
    void SetSurfaceGrowthSpecies(std::string_view name);
    void SetGasClosureDummySpecies(std::string_view name);

    /// Sets normalisation factor for nuclei concentration [#/m3] (default: 1e15).
    void SetNsNorm(double value) noexcept { Ns_norm_ = value; }

    /// Sets fixed soot particle diameter used in the BM model [m].
    void SetDp(double dp) noexcept        { dp_ = dp; }

    /// Sets soot particle molecular weight [kg/kmol] (used for diffusion).
    void SetMWp(double mwp) noexcept      { mwp_ = mwp; }

    // ── BrookesMoss-Hall model constants ──────────────────────────────────────

    void SetCalpha1_MBH(double v) noexcept { Calpha1_MBH_ = v; }
    void SetCalpha2_MBH(double v) noexcept { Calpha2_MBH_ = v; }
    void SetTalpha1_MBH(double v) noexcept { Talpha1_MBH_ = v; }
    void SetTalpha2_MBH(double v) noexcept { Talpha2_MBH_ = v; }
    void SetComega2_MBH(double v) noexcept { Comega2_MBH_ = v; }
    void SetTomega2_MBH(double v) noexcept { Tomega2_MBH_ = v; }

    // ── Model state queries ────────────────────────────────────────────────────

    [[nodiscard]] int nucleation_model()    const noexcept { return static_cast<int>(nucleation_variant_); }
    [[nodiscard]] int surface_growth_model()const noexcept { return surface_growth_model_; }
    [[nodiscard]] int oxidation_model()     const noexcept { return static_cast<int>(oxidation_variant_); }
    [[nodiscard]] int coagulation_model()   const noexcept { return coagulation_model_; }

    [[nodiscard]] const std::string& sg_species() const noexcept { return sg_species_; }

    // ── CRTP extension points — process source storage owned by BrookesMoss ───
    //
    // BrookesMoss models: nucleation, coagulation, growth, oxidation.
    // NOT modelled: condensation, sintering → base class returns zero span.

    [[nodiscard]] std::span<const double> sources_nucleation_impl()  const noexcept { return { source_nucleation_.data(),  this->n_equations }; }
    [[nodiscard]] std::span<const double> sources_coagulation_impl() const noexcept { return { source_coagulation_.data(), this->n_equations }; }
    [[nodiscard]] std::span<const double> sources_growth_impl()      const noexcept { return { source_growth_.data(),      this->n_equations }; }
    [[nodiscard]] std::span<const double> sources_oxidation_impl()   const noexcept { return { source_oxidation_.data(),   this->n_equations }; }

	void WriteHeaderLine(MOM::OutputFileColumns& fOutput, const unsigned int precision);

	void WriteOutputLine( MOM::OutputFileColumns& fOutput,
							const double T, const double P_Pa, const double* Y, const double mu,
							const double* M);

private:
    // ── Private computational methods ──────────────────────────────────────────

    void MemoryAllocation();
    void NucleationSourceTerms();
    void NucleationSourceTerms_BM();
    void NucleationSourceTerms_BMH();
    void SurfaceGrowthSourceTerms();
    void OxidationSourceTerms();
    void OxidationSourceTerms_BM();
    void OxidationSourceTerms_BMH();
    void CoagulationSourceTerms();
    void CheckBrookesMossHallSpecies();

private:
    // ── Thermodynamics reference ───────────────────────────────────────────────
    const Thermo& thermo_;

    // ── Transported variables ──────────────────────────────────────────────────
    double Ys_ = 0.;   //!< soot mass fraction [-]
    double bs_ = 0.;   //!< normalised nuclei concentration [m3/kg]

    // ── Precursor species ──────────────────────────────────────────────────────
    std::string prec_species_;
    int         prec_index_ = -1;
    double      prec_nc_    = 0.;   //!< C atoms per precursor molecule
    double      prec_nh_    = 0.;   //!< H atoms per precursor molecule
    double      conc_prec_  = 0.;   //!< precursor concentration [kmol/m3]

    // ── Surface growth species ─────────────────────────────────────────────────
    std::string sg_species_;
    int         sg_index_ = -1;
    double      sg_nc_    = 0.;
    double      sg_nh_    = 0.;
    double      conc_sg_  = 0.;

    // ── Key species concentrations [kmol/m3] ───────────────────────────────────
    double conc_H_    = 0., conc_OH_   = 0., conc_O2_   = 0.;
    double conc_H2_   = 0., conc_H2O_  = 0., conc_C2H2_ = 0.;

    // 0-based species indices
    int index_H_     = -1, index_OH_   = -1, index_O2_   = -1;
    int index_H2_    = -1, index_H2O_  = -1, index_C2H2_ = -1;
    int index_C6H5_  = -1, index_C6H6_ = -1;

    // Mass fractions (needed for BM-Hall expressions)
    double Y_C2H2_ = 0., Y_C6H5_ = 0., Y_C6H6_ = 0.;
    double Y_H2_   = 0., Y_OH_   = 0., Y_O2_   = 0.;

    // ── Particle properties ────────────────────────────────────────────────────
    double dp_   = 25.e-9;    //!< soot particle diameter [m] (default 25 nm)
    double mwp_  = 144.;      //!< soot particle molecular weight [kg/kmol]

    // ── Model constants ────────────────────────────────────────────────────────
    double Calpha_  = 0.;   //!< inception rate pre-exponential
    double Talpha_  = 0.;   //!< inception activation temperature [K]
    double Cbeta_   = 0.;   //!< coagulation rate constant
    double Cgamma_  = 0.;   //!< surface growth pre-exponential
    double Tgamma_  = 0.;   //!< surface growth activation temperature [K]
    double Comega_  = 0.;   //!< oxidation pre-exponential
    double etaColl_ = 0.;   //!< collision efficiency
    double Coxid_   = 0.;   //!< oxidation rate constant
    double exp_l_   = 0.;   //!< exponent for Ys in surface growth
    double exp_m_   = 0.;   //!< exponent for Ys in coagulation
    double exp_n_   = 0.;   //!< exponent for Ys in oxidation

    // ── BM-Hall extended constants ─────────────────────────────────────────────
    double Calpha1_MBH_  = 0., Calpha2_MBH_  = 0.;
    double Talpha1_MBH_  = 0., Talpha2_MBH_  = 0.;
    double Comega2_MBH_  = 0., Tomega2_MBH_  = 0.;

    // ── Model flags ────────────────────────────────────────────────────────────
    NucleationVariant nucleation_variant_ = NucleationVariant::Off;
    OxidationVariant  oxidation_variant_  = OxidationVariant::Off;
    int               surface_growth_model_ = 0;
    int               coagulation_model_    = 0;

    // ── Numerical parameters ───────────────────────────────────────────────────
    double Ys_min_  = 1.e-15;   //!< minimum Ys used in property calculations
    double bs_min_  = 0.;       //!< minimum bs used in property calculations
    double Ns_norm_ = 1.e15;    //!< normalisation factor for Ns [#/m3]

    // ── Gas consumption intermediate quantities ────────────────────────────────
    double dMdt_nucleation_      = 0.;   //!< [kg/m3/s]
    double dMdt_nucleation_BMH_1_= 0.;
    double dMdt_nucleation_BMH_2_= 0.;
    double dMdt_surface_growth_  = 0.;
    double dMdt_oxidation_       = 0.;

    // ── Per-process source storage (owned by BrookesMoss, not by base) ────────
    //
    // BrookesMoss models: nucleation, coagulation, growth, oxidation.
    // condensation and sintering are absent; base class returns zero span for both.

    MomentVector source_nucleation_  = MomentVector::Zero();
    MomentVector source_coagulation_ = MomentVector::Zero();
    MomentVector source_growth_      = MomentVector::Zero();
    MomentVector source_oxidation_   = MomentVector::Zero();

    // ── Initial moments cache ──────────────────────────────────────────────────
    MomentVector initial_moments_cache_ = MomentVector::Zero();

    // ── Numerical floors ───────────────────────────────────────────────────────
    static constexpr double kYsMin = 1.e-15;   //!< [-]
    static constexpr double kBsMin = 0.;       //!< [m3/kg]
};

} // namespace MOM

#if !defined(MOM_COMPILED_LIBRARY)
#  include "BrookesMoss.tpp"
#else
namespace MOM 
{
    extern template class BrookesMoss<BasicThermoData>;
}
#endif
