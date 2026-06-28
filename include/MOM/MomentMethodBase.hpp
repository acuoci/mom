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

#pragma once

#include "ProcessFlags.hpp"
#include "Eigen/Dense"
#include <array>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <cmath>

namespace MOM {

// ============================================================================
// MomentMethodBase<Derived, NEq>
// ============================================================================
//
// CRTP base class providing shared state and common implementations for all
// moment method classes. Zero virtual functions — all dispatch is resolved
// statically at compile time via the Curiously Recurring Template Pattern.
//
// Design rationale
// ----------------
// * NEq is a compile-time constant, so all source vectors are fixed-size
//   Eigen::Matrix<double, NEq, 1> objects. These live on the stack (for
//   NEq <= ~16), are fully unrolled by the compiler, and are SIMD-vectorised
//   without any heap allocation in the computational hot loop.
//
// * All physical sub-process source vectors are stored here (initialised to
//   zero). Each concrete derived class fills only the processes it models;
//   the rest remain zero, which is physically correct.
//
// * The omega_gas_ vector (gas-phase source terms) is an Eigen::VectorXd
//   sized once during setup (n_species elements). It is NOT re-allocated
//   in the hot loop. Eigen provides cache-line alignment and the same
//   setZero() / .data() API as the fixed-size source vectors.
//
// * The Planck absorption coefficient implementation is shared here; TiO2
//   uses PlanckCoeffModel::None and gets 0.0 from planck_coefficient().
//
// Template parameters
// -------------------
//   Derived  — The concrete subclass (HMOM, BrookesMoss, ThreeEquations, TiO2)
//   NEq      — Number of transported moment equations (compile-time constant)
// ============================================================================

template <class Derived, unsigned NEq>
class MomentMethodBase
{
public:
    // ── Compile-time interface ─────────────────────────────────────────────

    /// Number of transported equations. Satisfies the MomentMethod concept.
    static constexpr unsigned n_equations = NEq;

    /// Eigen type for a moment vector; used by derived classes.
    using MomentVector = Eigen::Matrix<double, static_cast<int>(NEq), 1>;

    // ── Common setters ─────────────────────────────────────────────────────

    /// Sets the soot/particle Schmidt number (default: 0.7).
    void SetSchmidtNumber(double value) noexcept         { schmidt_number_ = value; }

    /// Sets the particle material density [kg/m3] (default: 1800 for soot).
    void SetParticleDensity(double value) noexcept       { rho_particle_ = value; }

    /// Alias retained for backward compatibility with existing codes.
    void SetSootDensity(double value) noexcept           { rho_particle_ = value; }

    /// Sets the mixture dynamic viscosity [kg/m/s] (call before CalculateSourceMoments).
    void SetViscosity(double mu) noexcept                { mu_ = mu; }

    /// Enables/disables consumption of gas-phase precursor (default: true).
    void SetGasConsumption(bool flag) noexcept           { gas_consumption_ = flag; }

    /// Enables/disables particle contribution to radiative heat transfer.
    void SetRadiativeHeatTransfer(bool flag) noexcept    { radiative_heat_transfer_ = flag; }

    /// Sets the thermophoretic drift model (0 = off, 1 = standard).
    void SetThermophoreticModel(int flag) noexcept       { thermophoretic_model_ = static_cast<ThermophoreticModel>(flag); }
    void SetThermophoreticModel(ThermophoreticModel m) noexcept { thermophoretic_model_ = m; }

    /// Selects the Planck mean absorption coefficient correlation.
    void SetPlanckAbsorptionCoefficient(PlanckCoeffModel model) noexcept { planck_model_ = model; }
    void SetPlanckAbsorptionCoefficient(std::string_view label) noexcept {
        planck_model_ = PlanckCoeffModelFromString(label);
    }

    // ── Common getters ─────────────────────────────────────────────────────

    [[nodiscard]] constexpr unsigned n_moments()             const noexcept { return NEq; }
    [[nodiscard]] bool   is_active()                         const noexcept { return is_active_; }
    [[nodiscard]] bool   GasConsumption()                    const noexcept { return gas_consumption_; }
    [[nodiscard]] double schmidt_number()                    const noexcept { return schmidt_number_; }
    [[nodiscard]] double ParticleDensity()                   const noexcept { return rho_particle_; }
    [[nodiscard]] bool   radiative_heat_transfer()           const noexcept { return radiative_heat_transfer_; }
    [[nodiscard]] int    thermophoretic_model()              const noexcept { return static_cast<int>(thermophoretic_model_); }
    [[nodiscard]] bool   ClosureDummySpeciesIsActive()       const noexcept { return dummy_species_closure_; }
    [[nodiscard]] const std::string& closure_dummy_species() const noexcept { return dummy_species_; }
    [[nodiscard]] int    closure_dummy_index()               const noexcept { return dummy_index_; }

    // ── Source term access — span-based for zero-copy CFD interop ──────────

    /// Total source vector summed over all active processes [method-dependent units].
    /// Returns a non-owning view into the internal fixed-size storage.
    [[nodiscard]] std::span<const double> sources() const noexcept {
        return { source_all_.data(), NEq };
    }

    // Per-process source vectors.
    // Processes not modelled by a concrete class return a zero-filled span.
    [[nodiscard]] std::span<const double> sources_nucleation()   const noexcept { return { source_nucleation_.data(),   NEq }; }
    [[nodiscard]] std::span<const double> sources_coagulation()  const noexcept { return { source_coagulation_.data(),  NEq }; }
    [[nodiscard]] std::span<const double> sources_condensation() const noexcept { return { source_condensation_.data(), NEq }; }
    [[nodiscard]] std::span<const double> sources_growth()       const noexcept { return { source_growth_.data(),       NEq }; }
    [[nodiscard]] std::span<const double> sources_oxidation()    const noexcept { return { source_oxidation_.data(),    NEq }; }
    [[nodiscard]] std::span<const double> sources_sintering()    const noexcept { return { source_sintering_.data(),    NEq }; }

    /// Gas-phase consumption source terms [kg/m3/s].
    /// Size = n_species; allocated once during setup.
    [[nodiscard]] std::span<const double> omega_gas() const noexcept {
        return { omega_gas_.data(), static_cast<std::size_t>(omega_gas_.size()) };
    }

    // ── Radiative properties ───────────────────────────────────────────────

    /// Planck mean absorption coefficient [1/m].
    /// Returns 0.0 when planck_model_ == PlanckCoeffModel::None (e.g. TiO2).
    /// @param T  Gas temperature [K]
    /// @param fv Particle volume fraction [-]
    [[nodiscard]] double planck_coefficient(double T, double fv) const noexcept;

protected:
    // ── Protected constructor — only accessible through derived classes ─────

    MomentMethodBase() = default;
    ~MomentMethodBase() = default;

    // Non-copyable: moment method objects hold external thermo references
    // and large internal state. Copy semantics would be expensive and fragile.
    MomentMethodBase(const MomentMethodBase&)            = delete;
    MomentMethodBase& operator=(const MomentMethodBase&) = delete;
    MomentMethodBase(MomentMethodBase&&)                 = default;
    MomentMethodBase& operator=(MomentMethodBase&&)      = default;

    // ── Shared thermodynamic state ─────────────────────────────────────────
    // Updated by each concrete class's SetStatus() implementation.

    double T_    = 0.;   //!< temperature [K]
    double P_Pa_ = 0.;   //!< pressure [Pa]
    double rho_  = 0.;   //!< mixture density [kg/m3]
    double MW_   = 0.;   //!< mixture molecular weight [kg/kmol]
    double mu_   = 0.;   //!< dynamic viscosity [kg/m/s]

    // ── Common control flags ───────────────────────────────────────────────

    bool is_active_               = false;
    bool gas_consumption_         = true;
    bool radiative_heat_transfer_ = false;
    bool dummy_species_closure_   = false;

    double schmidt_number_        = 0.7;
    double rho_particle_          = 1800.;   //!< [kg/m3] — soot default; TiO2 overrides

    ThermophoreticModel thermophoretic_model_ = ThermophoreticModel::Off;
    PlanckCoeffModel    planck_model_         = PlanckCoeffModel::Smooke;

    std::string dummy_species_;
    int         dummy_index_ = -1;

    // ── Fixed-size source storage (stack-allocated) ────────────────────────
    //
    // All vectors initialised to zero. Concrete classes fill only the
    // processes they implement; unused entries remain zero (physically correct).

    MomentVector source_all_          = MomentVector::Zero();
    MomentVector source_nucleation_   = MomentVector::Zero();
    MomentVector source_coagulation_  = MomentVector::Zero();
    MomentVector source_condensation_ = MomentVector::Zero();
    MomentVector source_growth_       = MomentVector::Zero();
    MomentVector source_oxidation_    = MomentVector::Zero();
    MomentVector source_sintering_    = MomentVector::Zero();

    // ── Gas-phase source terms (sized at setup, not in hot loop) ──────────
    //
    // Size = n_species. Allocated once in each variant's MemoryAllocation().
    // Eigen::VectorXd gives aligned storage and a uniform API (setZero, .data,
    // .sum) consistent with the fixed-size MomentVector members above.
    Eigen::VectorXd omega_gas_;

    // ── Helpers for zero-initialising source vectors between time steps ─────

    void ZeroSources() noexcept {
        source_all_.setZero();
        source_nucleation_.setZero();
        source_coagulation_.setZero();
        source_condensation_.setZero();
        source_growth_.setZero();
        source_oxidation_.setZero();
        source_sintering_.setZero();
        omega_gas_.setZero();
    }

    // ── CRTP down-cast helpers ─────────────────────────────────────────────

    [[nodiscard]] Derived&       derived()       noexcept { return static_cast<Derived&>(*this); }
    [[nodiscard]] const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }

    // ── Mathematical constants (from <numbers>, full double precision) ────────
    //
    // std::numbers::pi_v<double> and sqrt2_v<double> carry the maximum
    // representable precision for double — no manual digit counting required.

    static constexpr double pi_    = std::numbers::pi_v<double>;    //!< π
    static constexpr double sqrt2_ = std::numbers::sqrt2_v<double>; //!< √2

    // ── Physical constants (CODATA 2018 exact values, SI units) ───────────
    //
    // kB and Nav are exact by SI 2019 redefinition.
    // Rgas = kB * Nav (exact). WC is the standard atomic weight (IUPAC 2021).

    static constexpr double kB_       = 1.380649e-23;            //!< Boltzmann [J/K]
    static constexpr double Nav_mol_  = 6.02214076e23;           //!< Avogadro [#/mol]
    static constexpr double Nav_kmol_ = 6.02214076e26;           //!< Avogadro [#/kmol]
    static constexpr double Rgas_     = 8314.46261815324;        //!< Gas constant [J/kmol/K]
    static constexpr double WC_       = 12.011;                  //!< C atomic weight [kg/kmol]

private:
    // ── Planck coefficient model implementations ───────────────────────────

    [[nodiscard]] double PlanckSmooke(double T, double fv) const noexcept;
    [[nodiscard]] double PlanckKent  (double T, double fv) const noexcept;
    [[nodiscard]] double PlanckSazhin(double T, double fv) const noexcept;
};

// ============================================================================
// planck_coefficient implementation
// ============================================================================
// Kept in the header to allow inlining. Implementations of the three models
// are in MomentMethodBase.tpp to keep this header readable.
// ============================================================================

template <class Derived, unsigned NEq>
inline double MomentMethodBase<Derived, NEq>::planck_coefficient(
    double T, double fv) const noexcept
{
    switch (planck_model_) {
        case PlanckCoeffModel::Smooke: return PlanckSmooke(T, fv);
        case PlanckCoeffModel::Kent:   return PlanckKent  (T, fv);
        case PlanckCoeffModel::Sazhin: return PlanckSazhin(T, fv);
        case PlanckCoeffModel::None:   return 0.;
        default:                       return 0.;
    }
}

} // namespace MOM

// MomentMethodBase is only instantiated via its Derived classes.
// The explicit template instantiations in src/*.cpp implicitly cover the base.
// No extern template declarations needed here.
#if !defined(MOM_COMPILED_LIBRARY)
#  include "MomentMethodBase.tpp"
#endif
