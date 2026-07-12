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

/**
 * @file NewVariant.hpp
 * @brief [SKELETON] Compilable CRTP template for adding a new MOM particle method.
 *
 * @par How to use this skeleton
 *
 * 1. **Copy** `include/NewVariant/` to `include/YourModel/`.
 * 2. **Rename** every occurrence of `NewVariant` → `YourModel` (class, file, labels).
 * 3. **Set `NEq`** (line marked with [NEq]) to your number of transported equations.
 * 4. **Fill in** every `// TODO:` block.
 * 5. **Uncomment** the `sources_X_impl()` methods for each physical process you model.
 * 6. **Create** `src/YourModel/YourModel.cpp` with one line:
 *    @code
 *      #include "YourModel/YourModel.hpp"
 *      namespace MOM { template class YourModel<BasicThermoData>; }
 *    @endcode
 * 7. **Register** the variant in `include/MOM/MomVariantList.hpp`:
 *    - Add `#include "YourModel/YourModel.hpp"` under "Registered variant headers".
 *    - Add `YourModel` to the `AllVariants` type alias.
 *
 * The `static_assert` at the bottom of this file fires immediately if any
 * concept-required member is missing — consult `MomentMethodConcept.hpp` for
 * the full requirement list.
 *
 * @par Concept requirement quick-reference
 *
 * **Compile-time statics** (must be public):
 *   - `n_equations` — INHERITED from `MomentMethodBase`
 *   - `variant_labels` — DECLARE here (see below)
 *
 * **Injectors** (noexcept, run every cell):
 *   - `SetState(T, P, Y)` — IMPLEMENT in .tpp
 *   - `SetMoments(span)` — IMPLEMENT in .tpp
 *   - `SetViscosity(mu)` — INHERITED from `MomentMethodBase`
 *
 * **Core computation** (noexcept):
 *   - `ComputeSources()` — IMPLEMENT in .tpp
 *
 * **Source spans** (zero-copy, CRTP-dispatched):
 *   - `sources()` — INHERITED from base
 *   - `sources_{nucleation,coagulation,condensation,growth,oxidation,sintering}()` —
 *     INHERITED; each returns your `sources_X_impl()` if declared, else a zero span.
 *   - `omega_gas()` — INHERITED from base
 *
 * **Particle properties** (IMPLEMENT here or in .tpp):
 *   - `volume_fraction()`, `particle_diameter()`, `collision_diameter()`
 *   - `particle_number_density()`, `number_primary_particles()`, `mass_fraction()`
 *   - `particle_density()` — INHERITED (returns `rho_particle_`)
 *   - `specific_surface_area()`
 *
 * **Transport** (IMPLEMENT):
 *   - `diffusion_coefficient()` — use `CunninghamDiffusionCoeff()` from base
 *   - `schmidt_number()` — INHERITED
 *   - `thermophoretic_model()` — INHERITED
 *
 * **Status / initial conditions** (IMPLEMENT or INHERIT):
 *   - `is_active()`, `gas_consumption()` — INHERITED
 *   - `initial_moments()` — IMPLEMENT (return span of size NEq)
 *
 * **Gas coupling** (IMPLEMENT):
 *   - `precursor_index()`, `precursor_concentration()`
 *   - `is_closure_dummy_species()`, `closure_dummy_index()` — INHERITED
 *
 * **Radiation** (IMPLEMENT or INHERIT):
 *   - `radiative_heat_transfer()` — INHERITED
 *   - `planck_coefficient(T, fv)` — INHERITED; set `planck_model_` to select correlation
 *
 * **Diagnostics** (IMPLEMENT):
 *   - `PrintSummary()` — print your model's configuration to stdout
 *
 * @par Optional output hooks (for MomentMethodReporter)
 *   - `variant_prefix_output(CB&&)` — extra columns before the source block
 *   - `variant_suffix_output(CB&&)` — extra columns after the source block (rarely needed)
 *
 * @tparam Thermo  Must satisfy `MOM::ThermoMap` (see ThermoProxy.hpp).
 *
 * @warning Do NOT add this file's include to MomVariantList.hpp until you have
 *          filled in all TODO blocks.  The `static_assert` at the bottom of this
 *          file is disabled while the skeleton is incomplete (see note there).
 */

#include <array>
#include <cmath>
#include <span>
#include <string>
#include <string_view>

#include "MOM/MomentMethodBase.hpp"
#include "MOM/MOMConfig.hpp"
#include "MOM/ThermoProxy.hpp"

namespace MOM
{

/**
 * @class NewVariant
 * @brief [SKELETON] Replace this Doxygen block with your model's full description.
 *
 * TODO: describe transported variables, physical processes modelled,
 *       physical processes NOT modelled, key references, and thread safety.
 *
 * @tparam Thermo  Must satisfy `MOM::ThermoMap`.
 */
template <ThermoMap Thermo>
class NewVariant : public MomentMethodBase<NewVariant<Thermo>, 2u>  // [NEq] change 2u to your NEq
{
    using Base = MomentMethodBase<NewVariant<Thermo>, 2u>;           // [NEq] keep in sync

    static constexpr unsigned NEq = 2u;   // [NEq] convenience alias — change to your value

public:

    using typename Base::MomentVector;

    // =========================================================================
    // [REQUIRED] Factory labels
    // =========================================================================
    //
    // At least one label is needed.  The first is the canonical human-readable
    // name; subsequent entries are accepted aliases (case-sensitive).
    // MakeAnyMomentMethod uses these to map a runtime string → this type.

    static constexpr std::array<std::string_view, 2> variant_labels
        { "NewVariant", "newvariant" };   // TODO: replace with your model's names

    // =========================================================================
    // Constructor
    // =========================================================================

    /**
     * @brief Constructs a dormant model; call `ApplyConfig()` before use.
     * @param thermo  Reference to the thermodynamics backend.  Must outlive this object.
     */
    explicit NewVariant(const Thermo& thermo) noexcept : thermo_(thermo) {}

    NewVariant(const NewVariant&)            = delete;
    NewVariant& operator=(const NewVariant&) = delete;
    NewVariant(NewVariant&&)                 = default;
    NewVariant& operator=(NewVariant&&)      = default;

    // =========================================================================
    // Config struct (optional but strongly recommended)
    // =========================================================================
    //
    // Collect every user-tunable parameter in one POD-like struct.  ApplyConfig()
    // sets the defaults; the user overrides individual fields before calling it.
    // This gives a named-parameter-idiom API without requiring a builder.

    /** @brief All tunable parameters for NewVariant.  All fields carry safe defaults. */
    struct Config
    {
        bool   is_active        = false;
        double particle_density = 1800.;   // [kg/m3]  TODO: adjust for your material
        // TODO: add your model-specific parameters
    };

    /** @brief Applies @p cfg, re-computes derived quantities, allocates gas arrays. */
    void ApplyConfig(const Config& cfg)
    {
        this->is_active_ = cfg.is_active;
        this->SetParticleDensity(cfg.particle_density);
        // TODO: copy remaining config fields, call Precalculations() if needed,
        //       call MemoryAllocation(thermo_) or equivalent to size omega_gas_.
    }

    // =========================================================================
    // [REQUIRED] State injection — runs every cell / time-step
    // =========================================================================

    /**
     * @brief Updates the gas thermodynamic state.
     * @param T     Temperature [K].
     * @param P_Pa  Pressure [Pa].
     * @param Y     Species mass fractions (pointer, length = thermo_.NumberOfSpecies()).
     */
    void SetState(double T, double P_Pa, const double* Y) noexcept;

    /**
     * @brief Ingests the current transported moment values.
     * @param m  Span of size `n_equations`.  Unpack into your physical variables.
     */
    void SetMoments(std::span<const double> m) noexcept;

    // =========================================================================
    // [REQUIRED] Core computation — the only method a CFD hot loop needs
    // =========================================================================

    /**
     * @brief Evaluates all source terms and gas-phase consumption for the
     *        current state (set by SetState + SetMoments + SetViscosity).
     *
     * Implementation pattern (in .tpp):
     * @code
     *   void NewVariant<Thermo>::ComputeSources() noexcept
     *   {
     *       if (!this->is_active_) return;
     *       this->ZeroSources();
     *       source_nucleation_.setZero();
     *       // ... zero other per-process vectors ...
     *
     *       SootNucleation();      // fills source_nucleation_
     *       SootCoagulation();     // fills source_coagulation_
     *       // ... other processes ...
     *
     *       this->source_all_ = source_nucleation_ + source_coagulation_ + ...;
     *   }
     * @endcode
     */
    void ComputeSources() noexcept;

    // =========================================================================
    // [REQUIRED] Particle properties
    // =========================================================================

    /** @brief Particle volume fraction [-]. */
    [[nodiscard]] double volume_fraction() const noexcept;

    /** @brief Mean primary particle diameter [m]. */
    [[nodiscard]] double particle_diameter() const noexcept;

    /**
     * @brief Collision (aggregate) diameter [m].
     *
     * For simple models without fractal aggregates this equals `particle_diameter()`.
     * For HMOM-style fractal models it is the aggregate collision diameter.
     */
    [[nodiscard]] double collision_diameter() const noexcept;

    /** @brief Total particle number density [#/m3]. */
    [[nodiscard]] double particle_number_density() const noexcept;

    /**
     * @brief Mean number of primary spheres per aggregate [-].
     *
     * Return 1.0 for models without aggregate structure.
     * For models that carry aggregate state, derive this from the sphere
     * geometry helpers: `this->NumberPrimaryParticles(ss, vs)`.
     */
    [[nodiscard]] double number_primary_particles() const noexcept;

    /** @brief Particle mass fraction [-]. */
    [[nodiscard]] double mass_fraction() const noexcept;

    /** @brief Specific surface area of the particle phase [m2/m3]. */
    [[nodiscard]] double specific_surface_area() const noexcept;

    // =========================================================================
    // [REQUIRED] Transport
    // =========================================================================

    /**
     * @brief Effective diffusion coefficient [kg/m/s].
     *
     * Use `this->CunninghamDiffusionCoeff(dc_safe)` for Stokes-Einstein-Cunningham
     * and fall back to the Schmidt-number diffusivity:
     * @code
     *   const double dc_safe = std::max(collision_diameter(), 1.e-12);
     *   const double Gamma_B = this->CunninghamDiffusionCoeff(dc_safe);
     *   return std::max(Gamma_B, this->mu_ / this->schmidt_number_);
     * @endcode
     */
    [[nodiscard]] double diffusion_coefficient() const noexcept;

    // =========================================================================
    // [REQUIRED] Gas coupling
    // =========================================================================

    /**
     * @brief 0-based index of the precursor species in the thermo map.
     *
     * Set this in `SetState()` (or once in `ApplyConfig()` if the species name
     * is fixed): `prec_index_ = thermo_.IndexOfSpecies("A4");`
     * Return -1 if no precursor species.
     */
    [[nodiscard]] int precursor_index() const noexcept { return prec_index_; }

    /**
     * @brief Molar concentration of the precursor species [kmol/m3].
     *
     * Typically computed in `SetState()`:
     * `conc_prec_ = SpeciesConcentrationKmolM3(prec_index_, Y, cTot, thermo_);`
     */
    [[nodiscard]] double precursor_concentration() const noexcept { return conc_prec_; }

    // =========================================================================
    // [REQUIRED] Initial moments & activation
    // =========================================================================

    /**
     * @brief Initial values for the transported moment equations [variant units].
     *
     * Called once at solver startup to seed the transport fields.
     * Fill `initial_moments_` in `ApplyConfig()` and return a span here.
     * @return Span of size `n_equations` into stack-allocated storage.
     */
    [[nodiscard]] std::span<const double> initial_moments() const noexcept
    {
        return {initial_moments_.data(), NEq};
    }

    // =========================================================================
    // [REQUIRED] Diagnostics
    // =========================================================================

    /** @brief Prints model configuration and key parameters to stdout. */
    void PrintSummary() const;

    // =========================================================================
    // [OPTIONAL] Per-process source extension points
    // =========================================================================
    //
    // Declare `sources_X_impl()` for each physical process your model implements.
    // MomentMethodBase detects the _impl() at compile time (CRTP + requires-expression)
    // and routes `sources_X()` to it; variants WITHOUT the _impl get a zero span at
    // zero cost — no virtual dispatch, no runtime branch.
    //
    // Rule: only uncomment blocks for processes you actually compute.
    //
    // PROCESS CAPABILITY MATRIX — tick what this model implements:
    //
    //   [ ] Nucleation          →  uncomment sources_nucleation_impl()
    //   [ ] Coagulation         →  uncomment sources_coagulation_impl()
    //   [ ] Surface growth      →  uncomment sources_growth_impl()
    //   [ ] Oxidation           →  uncomment sources_oxidation_impl()
    //   [ ] Condensation        →  uncomment sources_condensation_impl()
    //   [ ] Sintering           →  uncomment sources_sintering_impl()
    //                              (currently only MetalOxide uses sintering)

    // -- Nucleation -----------------------------------------------------------
    // [[nodiscard]] [[gnu::always_inline]]
    // std::span<const double> sources_nucleation_impl() const noexcept
    // { return {source_nucleation_.data(), NEq}; }

    // -- Coagulation ----------------------------------------------------------
    // [[nodiscard]] [[gnu::always_inline]]
    // std::span<const double> sources_coagulation_impl() const noexcept
    // { return {source_coagulation_.data(), NEq}; }

    // -- Surface growth -------------------------------------------------------
    // [[nodiscard]] [[gnu::always_inline]]
    // std::span<const double> sources_growth_impl() const noexcept
    // { return {source_growth_.data(), NEq}; }

    // -- Oxidation ------------------------------------------------------------
    // [[nodiscard]] [[gnu::always_inline]]
    // std::span<const double> sources_oxidation_impl() const noexcept
    // { return {source_oxidation_.data(), NEq}; }

    // -- Condensation ---------------------------------------------------------
    // [[nodiscard]] [[gnu::always_inline]]
    // std::span<const double> sources_condensation_impl() const noexcept
    // { return {source_condensation_.data(), NEq}; }

    // -- Sintering ------------------------------------------------------------
    // [[nodiscard]] [[gnu::always_inline]]
    // std::span<const double> sources_sintering_impl() const noexcept
    // { return {source_sintering_.data(), NEq}; }

    // =========================================================================
    // [OPTIONAL] Reporter output hooks (for MomentMethodReporter)
    // =========================================================================
    //
    // Implement `variant_prefix_output(CB&&)` to append extra diagnostic columns
    // BEFORE the per-process source block in the output file.  Use `EmitOmegaGas`
    // (from MomentMethodBase) for safe gas-source column output.
    //
    // Implement `variant_suffix_output(CB&&)` for columns AFTER the source block
    // (rarely needed — HMOM uses it for coagulation sub-process breakdowns).
    //
    // The callback CB satisfies: cb(std::string_view label, double value).
    // The reporter calls this with a column-registration lambda during WriteHeader
    // and a value-writing lambda during WriteRow; the _impl always calls cb identically.

    // template <typename CB>
    // void variant_prefix_output(CB&& cb) const
    // {
    //     cb("fv[-]",   volume_fraction());
    //     cb("dp[nm]",  particle_diameter() * 1.e9);
    //     cb("N[#/m3]", particle_number_density());
    //     // TODO: add your diagnostic columns
    //     this->EmitOmegaGas(cb, "omegaPrec[kg/m3/s]", prec_index_);
    // }

    // template <typename CB>
    // void variant_suffix_output(CB&& cb) const { /* TODO (optional) */ }

private:

    // =========================================================================
    // Thermo reference
    // =========================================================================

    const Thermo& thermo_;   //!< Non-owning ref to the thermo backend. Must outlive *this.

    // =========================================================================
    // Precursor tracking
    // =========================================================================

    int    prec_index_ = -1;  //!< 0-based index; -1 = species absent in mechanism.
    double conc_prec_  = 0.;  //!< Precursor molar concentration [kmol/m3].

    // =========================================================================
    // Transported moment variables
    // =========================================================================
    //
    // Declare one member per transported equation.  Unpack from the span in SetMoments().
    //
    // TODO: replace these placeholders with your actual variable names and units.

    // double Ys_ = 0.;   //!< Example: soot mass fraction [-].
    // double bs_ = 0.;   //!< Example: normalised nuclei concentration [m3/kg].

    // =========================================================================
    // Derived particle state (updated in SetMoments or ComputeSources)
    // =========================================================================

    // TODO: declare particle state variables (fv, dp, N, etc.) if you cache them.

    // =========================================================================
    // Per-process source vectors
    // =========================================================================
    //
    // Declare one MomentVector per process you model (must match the _impl() above).
    // MomentVector lives on the stack for NEq ≤ ~16.  No heap allocation in the hot loop.

    // MomentVector source_nucleation_  = MomentVector::Zero();
    // MomentVector source_coagulation_ = MomentVector::Zero();
    // MomentVector source_growth_      = MomentVector::Zero();
    // MomentVector source_oxidation_   = MomentVector::Zero();
    // MomentVector source_condensation_= MomentVector::Zero();
    // MomentVector source_sintering_   = MomentVector::Zero();

    // =========================================================================
    // Initial conditions cache (filled once in ApplyConfig)
    // =========================================================================

    MomentVector initial_moments_ = MomentVector::Zero();

    // =========================================================================
    // Private physics helpers (declare, implement in .tpp)
    // =========================================================================

    // TODO: declare your per-process subroutines, e.g.:
    // void SootNucleation()  noexcept;
    // void SootCoagulation() noexcept;
    // void SootGrowth()      noexcept;
    // void SootOxidation()   noexcept;
};

// =============================================================================
// Concept gate — fires at include time if any required member is missing
// =============================================================================
//
// NOTE: This static_assert is INTENTIONALLY COMMENTED OUT while you are filling
//       in the TODO blocks.  The stub implementations intentionally do not
//       implement all physics.  Uncomment it once your model is complete and
//       you want the compiler to verify the full contract.
//
// static_assert(MomentMethod<NewVariant<BasicThermoData>>,
//     "[NewVariant] Does not satisfy MomentMethod. "
//     "See MomentMethodConcept.hpp for the full required-expression list.");

} // namespace MOM

// Include the template implementation body.
// All member function definitions live in the .tpp so the header stays readable.
#include "NewVariant/NewVariant.tpp"
