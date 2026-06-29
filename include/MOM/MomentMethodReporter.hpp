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

// ============================================================================
// MomentMethodReporter
// ============================================================================
//
// Single-Responsibility: formats and writes output for any MOM variant.
//
// Design contract
// ---------------
// • Accepts any model satisfying MomentMethod<M> via a read-only const&.
// • Reads state exclusively through the public interface — no private access,
//   no friend declarations, no casts.
// • Variant-specific columns are provided BY THE VARIANT, not detected here.
//   The reporter is fully closed to modification when new variants are added.
// • Owns no model and performs no numerical computation.
// • The caller (CFD code) is responsible for calling SetStatus,
//   SetMoments, and CalculateSourceMoments before every WriteRow.
//   The reporter only observes — it never mutates the model.
//
// Extensibility protocol — variant_output hooks
// ---------------------------------------------
// Each variant may optionally implement one or both of these template methods:
//
//   template <typename CB>
//   void variant_prefix_output(CB&& cb) const;   // extra cols BEFORE omega_gas
//
//   template <typename CB>
//   void variant_suffix_output(CB&& cb) const;   // extra cols AFTER source terms
//
// The callback signature is: cb(std::string_view label, double value).
// It is called once per extra column.  The reporter supplies the lambda:
//   • in WriteHeader mode — uses the label to register the column
//   • in WriteRow mode    — uses the value to write the data
// The variant calls cb identically in both modes; the reporter controls
// which dimension of the (label, value) pair is used.
//
// To ADD a new variant with custom columns:
//   1. Implement variant_prefix_output / variant_suffix_output in the new class.
//   2. No changes to MomentMethodReporter are required.
//
// Usage sketch
// ------------
//   MOM::OutputFileColumns file("soot.out");
//   MOM::MomentMethodReporter reporter(file, thermo.names);
//   reporter.WriteHeader(model);    // invokes variant's hooks automatically
//   file.Complete();
//
//   for (auto& cell : grid) {
//       model.SetStatus(cell.T, cell.P, cell.Y.data());
//       model.SetMoments(cell.M);
//       model.CalculateSourceMoments();
//       // ... apply sources to CFD residuals ...
//       if (output_step)
//           reporter.WriteRow(model);  // pure observer — never mutates model
//   }
//
// AnyMomentMethod shims
// ---------------------
// std::visit is called ONCE per WriteHeader/WriteRow, not per cell.
// For high-frequency use, prefer the direct template overloads.
//
// ============================================================================

#include "MomentMethodConcept.hpp"
#include "AnyMomentMethod.hpp"
#include "Utilities/OutputFileColumns.h"

#include <cmath>
#include <numeric>
#include <span>
#include <string>
#include <vector>

namespace MOM
{

class MomentMethodReporter
{
public:

    // -- Construction ----------------------------------------------------------

    /// @param out            Output file managed externally. Must outlive the reporter.
    /// @param species_names  (optional) Species names from the thermo map, used to
    ///                       label per-species gas-consumption columns.
    ///                       If empty, only the total omega_gas column is written.
    explicit MomentMethodReporter(OutputFileColumns& out, std::vector<std::string> species_names = {})
        : out_(out), species_names_(std::move(species_names))
    {
    }

    MomentMethodReporter(const MomentMethodReporter&)            = delete;
    MomentMethodReporter& operator=(const MomentMethodReporter&) = delete;
    MomentMethodReporter(MomentMethodReporter&&)                 = default;

    // -- Static-dispatch API (preferred — zero overhead) -----------------------

    /// Register all output columns for variant Model.
    /// Call exactly once, before OutputFileColumns::Complete() and WriteRow.
    template <MomentMethod Model> void WriteHeader(const Model& model, unsigned precision = 8);

    /// Write one output row from the current model state.
    /// Precondition: CalculateSourceMoments() has been called by the CFD code.
    /// This method is a pure observer — it never calls SetStatus/SetMoments
    /// or CalculateSourceMoments on the model.
    template <MomentMethod Model> void WriteRow(const Model& model);

    // -- Runtime-dispatch API (AnyMomentMethod) --------------------------------

    template <ThermoMap Thermo>
    void WriteHeader(const AnyMomentMethod<Thermo>& any, unsigned precision = 8)
    {
        std::visit([&](const auto& m) { this->WriteHeader(m, precision); }, any);
    }

    template <ThermoMap Thermo> void WriteRow(const AnyMomentMethod<Thermo>& any)
    {
        std::visit([&](const auto& m) { this->WriteRow(m); }, any);
    }

private:

    // -- Internals -------------------------------------------------------------

    OutputFileColumns& out_;
    std::vector<std::string> species_names_;

    // Column-label helpers
    static std::string col(std::string_view prefix,
                           unsigned j,
                           bool zf               = false,
                           std::string_view unit = "mol/m3/s")
    {
        return std::string(prefix) + "(" + std::to_string(j) + ")" + (zf ? "[ZF]" : "") + "[" +
               std::string(unit) + "]";
    }

    // Write all values of a span to the current row
    void writeSpan(std::span<const double> s)
    {
        for (auto v : s)
            out_ << v;
    }
};

// ============================================================================
// WriteHeader — template implementation
// ============================================================================

template <MomentMethod Model>
void MomentMethodReporter::WriteHeader(const Model& model, unsigned precision)
{
    constexpr unsigned N = Model::n_equations;

    // Compile-time per-process ownership flags (for [ZF] tagging).
    // These are the ONLY compile-time checks in the reporter — they query the
    // concept-mandated process interface, not variant identity.
    constexpr bool has_nuc = requires(const Model& m) { m.sources_nucleation_impl(); };
    constexpr bool has_coa = requires(const Model& m) { m.sources_coagulation_impl(); };
    constexpr bool has_con = requires(const Model& m) { m.sources_condensation_impl(); };
    constexpr bool has_gro = requires(const Model& m) { m.sources_growth_impl(); };
    constexpr bool has_oxi = requires(const Model& m) { m.sources_oxidation_impl(); };
    constexpr bool has_sin = requires(const Model& m) { m.sources_sintering_impl(); };

    // Lambda passed to variant_prefix_output / variant_suffix_output in header mode.
    // The variant calls cb(label, value); here we use only the label.
    auto add_col = [&](std::string_view label, double /*unused_in_header*/)
    {
        out_.AddColumn(std::string(label), precision);
    };

    // -- Block 1: Core particle state (concept-mandated, truly common) ---------
    out_.AddColumn("Ys[-]", precision);
    out_.AddColumn("Ns[#/m3]", precision);
    out_.AddColumn("Ss[m2/m3]", precision);
    out_.AddColumn("fv[-]", precision);
    out_.AddColumn("dp[nm]", precision);
    out_.AddColumn("dc[nm]", precision);

    // -- Variant prefix columns (np, ss, vs, aggregate props, statistics…) -----
    // The variant self-describes its extra columns by implementing
    // variant_prefix_output(cb).  If the method is absent (e.g. BrookesMoss),
    // this block is a no-op at compile time.
    if constexpr (requires(const Model& m) {
                      m.variant_prefix_output([](std::string_view, double) {});
                  })
        model.variant_prefix_output(add_col);


    // -- Block 3: Transport (concept-mandated) ---------------------------------
    out_.AddColumn("D[kg/m/s]", precision);

    // -- Block 4: Total source terms (concept-mandated) ------------------------
    for (unsigned j = 0; j < N; ++j)
        out_.AddColumn(col("Sall", j, false, "mol/m3/s"), precision);

    // -- Block 5: Per-process source terms ([ZF]-tagged for zero-fallback) ------
    for (unsigned j = 0; j < N; ++j)
        out_.AddColumn(col("Snuc", j, !has_nuc), precision);
    for (unsigned j = 0; j < N; ++j)
        out_.AddColumn(col("Scoa", j, !has_coa), precision);
    for (unsigned j = 0; j < N; ++j)
        out_.AddColumn(col("Scon", j, !has_con), precision);
    for (unsigned j = 0; j < N; ++j)
        out_.AddColumn(col("Sgro", j, !has_gro), precision);
    for (unsigned j = 0; j < N; ++j)
        out_.AddColumn(col("Soxi", j, !has_oxi), precision);
    for (unsigned j = 0; j < N; ++j)
        out_.AddColumn(col("Ssin", j, !has_sin), precision);

    // -- Variant suffix columns (detailed breakdowns, sub-process vectors…) -----
    // Same protocol as prefix.  HMOM uses this for the coagulation sub-breakdown.
    if constexpr (requires(const Model& m) {
                      m.variant_suffix_output([](std::string_view, double) {});
                  })
        model.variant_suffix_output(add_col);
}

// ============================================================================
// WriteRow — template implementation
// ============================================================================

template <MomentMethod Model> void MomentMethodReporter::WriteRow(const Model& model)
{
    // Lambda passed to variant hooks in row mode.
    // The variant calls cb(label, value); here we use only the value.
    auto add_val = [&](std::string_view /*unused_in_row*/, double value)
    {
        out_ << value;
    };


    // -- Block 1: Core particle state (concept-mandated) -----------------------
    out_ << model.mass_fraction();
    out_ << model.particle_number_density();
    out_ << model.specific_surface();
    out_ << model.volume_fraction();
    out_ << model.particle_diameter() * 1.e9;  // m → nm
    out_ << model.collision_diameter() * 1.e9; // m → nm

    // -- Variant prefix values -------------------------------------------------
    if constexpr (requires(const Model& m) {
                      m.variant_prefix_output([](std::string_view, double) {});
                  })
        model.variant_prefix_output(add_val);


    // -- Block 3: Transport (concept-mandated) ---------------------------------
    out_ << model.diffusion_coefficient();

    // -- Block 4: Total source terms (concept-mandated) ------------------------
    writeSpan(model.sources());

    // -- Block 5: Per-process source terms (concept-mandated, zero-fallback) ----
    writeSpan(model.sources_nucleation());
    writeSpan(model.sources_coagulation());
    writeSpan(model.sources_condensation());
    writeSpan(model.sources_growth());
    writeSpan(model.sources_oxidation());
    writeSpan(model.sources_sintering());

    // -- Variant suffix values -------------------------------------------------
    if constexpr (requires(const Model& m) {
                      m.variant_suffix_output([](std::string_view, double) {});
                  })
        model.variant_suffix_output(add_val);
}

} // namespace MOM
