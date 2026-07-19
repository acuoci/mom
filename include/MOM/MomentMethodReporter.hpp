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
 * @file MomentMethodReporter.hpp
 * @brief Output-column reporter for MOM particle state, sources, and NDF data.
 */

#include "MomentMethodConcept.hpp"
#include "AnyMomentMethod.hpp"
#include "Utilities/OutputFileColumns.h"

#include <cassert>
#include <cmath>
#include <numbers>
#include <numeric>
#include <span>
#include <string>
#include <vector>

namespace MOM
{

/**
 * @class MomentMethodReporter
 * @brief Writes diagnostic columns for any type satisfying `MomentMethod`.
 *
 * The reporter is a read-only observer. The caller must update the model state
 * and call `ComputeSources()` before each `WriteRow()`. Variant-specific columns
 * are supplied through optional callback hooks detected by concepts.
 *
 * @note `OutputFileColumns` and the optional species-name list are not owned by
 *       the model. The output object passed to the constructor must outlive the
 *       reporter.
 */
class MomentMethodReporter
{
public:

    // -- Construction ----------------------------------------------------------

    /**
     * @brief Constructs a disconnected reporter.
     *
     * The reporter must be assigned from a connected reporter before output is
     * written.
     */
    MomentMethodReporter() = default;

    /**
     * @brief Constructs a reporter bound to an output file.
     * @param out Output file object. Must outlive the reporter.
     * @param species_names Species names used to label gas-source columns.
     */
    explicit MomentMethodReporter(OutputFileColumns& out, std::vector<std::string> species_names = {})
        : out_(&out), species_names_(std::move(species_names))
    {
    }

    MomentMethodReporter(const MomentMethodReporter&)            = delete;
    MomentMethodReporter& operator=(const MomentMethodReporter&) = delete;
    MomentMethodReporter(MomentMethodReporter&&)                 = default;
    MomentMethodReporter& operator=(MomentMethodReporter&&)      = default;

    // -- Static-dispatch API ---------------------------------------------------

    /**
     * @brief Registers all columns required by @p model.
     * @param model Configured moment method instance.
     * @param precision Numeric precision (significant digits) used by `OutputFileColumns`.
     *                  Defaults to 8, which is sufficient for source-term diagnostics.
     *
     * Call once before `OutputFileColumns::Complete()` and before any row output.
     *
     * @note The default precision here (8) differs from `WriteHeaderLineReconstructedNDF`
     *       (16).  The asymmetry is intentional: NDF reconstruction integrates moment
     *       ratios that can be numerically sensitive, so higher output precision aids
     *       post-processing and round-trip verification.  Source-term output does not
     *       require that extra resolution.
     */
    template <MomentMethod Model> void WriteHeader(const Model& model, unsigned precision = 8);

    /**
     * @brief Writes one row from the current model state.
     * @param model Moment method instance whose sources have already been computed.
     *
     * @pre `ComputeSources()` has been called for the current cell/state.
     */
    template <MomentMethod Model> void WriteRow(const Model& model);

    // -- Runtime-dispatch API (AnyMomentMethod) --------------------------------

    /**
     * @brief Runtime-dispatch overload of `WriteHeader()`.
     * @param any Runtime-selected model.
     * @param precision Numeric precision used by `OutputFileColumns`.
     */
    template <ThermoMap Thermo>
    void WriteHeader(const AnyMomentMethod<Thermo>& any, unsigned precision = 8)
    {
        std::visit([&](const auto& m) { this->WriteHeader(m, precision); }, any);
    }

    /**
     * @brief Runtime-dispatch overload of `WriteRow()`.
     * @param any Runtime-selected model whose sources have already been computed.
     */
    template <ThermoMap Thermo> void WriteRow(const AnyMomentMethod<Thermo>& any)
    {
        std::visit([&](const auto& m) { this->WriteRow(m); }, any);
    }

    // -- NDF snapshot API ------------------------------------------------------

    /**
     * @brief Registers reconstructed NDF output columns.
     *
     * Core columns use nm and nm³ units: particle volume, dimensional NDF,
     * normalized NDF, sphere-equivalent diameter, diameter-space NDF, and
     * normalized diameter-space NDF.
     *
     * @param model NDF-capable moment method instance.
     * @param ndf_out Output file object for the NDF table.
     * @param precision Numeric precision (significant digits) used by `OutputFileColumns`.
     *                  Defaults to 16 — see the note below.
     *
     * @note The default precision here (16) differs from `WriteHeader` (8).  The
     *       asymmetry is intentional: NDF reconstruction integrates moment ratios
     *       that can be numerically sensitive, so higher output precision aids
     *       post-processing and round-trip verification.  Source-term output does
     *       not require that extra resolution.
     */
    template <MomentMethod Model>
    requires HasReconstructedNDF<Model>
    void WriteHeaderLineReconstructedNDF(   const Model& model,
                                            OutputFileColumns& ndf_out,
                                            unsigned precision);

    /**
     * @brief Writes a complete reconstructed NDF table.
     *
     * @param model NDF-capable moment method instance.
     * @param ndf_out Output file object for the NDF table.
     * @param nv Number of logarithmically spaced volume points.
     * @param vmin_nm3 Minimum particle volume [nm3].
     * @param vmax_nm3 Maximum particle volume [nm3].
     * @param use_regularized_moments If true, use the model's regularized NDF moments.
     */
    template <MomentMethod Model>
    requires HasReconstructedNDF<Model>
    void WriteReconstructedNDF(const Model& model,
                               OutputFileColumns& ndf_out,
                               int    nv                       = 100,
                               double vmin_nm3                 = 1.0,
                               double vmax_nm3                 = 1.0e6,
                               bool   use_regularized_moments  = false);

    /**
     * @brief Runtime-dispatch overload of `WriteHeaderLineReconstructedNDF()`.
     *
     * Does nothing when the active variant has no reconstructed NDF.
     */
    template <ThermoMap Thermo>
    void WriteHeaderLineReconstructedNDF(   const AnyMomentMethod<Thermo>& any,
                                            OutputFileColumns& ndf_out,
                                            unsigned precision = 16)
    {
        std::visit(
            [&](const auto& m) {
                using M = std::decay_t<decltype(m)>;
                if constexpr (HasReconstructedNDF<M>)
                    this->WriteHeaderLineReconstructedNDF(m, ndf_out, precision);
            },
            any);
    }

    /**
     * @brief Runtime-dispatch overload of `WriteReconstructedNDF()`.
     *
     * Does nothing when the active variant has no reconstructed NDF.
     */
    template <ThermoMap Thermo>
    void WriteReconstructedNDF(const AnyMomentMethod<Thermo>& any,
                               OutputFileColumns& ndf_out,
                               int    nv                       = 100,
                               double vmin_nm3                 = 1.0,
                               double vmax_nm3                 = 1.0e6,
                               bool   use_regularized_moments  = false)
    {
        std::visit(
            [&](const auto& m) {
                using M = std::decay_t<decltype(m)>;
                if constexpr (HasReconstructedNDF<M>)
                    this->WriteReconstructedNDF(m, ndf_out, nv, vmin_nm3, vmax_nm3,
                                                use_regularized_moments);
            },
            any);
    }

    // -- Convenience overloads using this reporter's own output file ------------

    template <MomentMethod Model>
    requires HasReconstructedNDF<Model>
    void WriteHeaderLineReconstructedNDF(const Model& model, unsigned precision = 16)
    {
        assert(out_ != nullptr && "MomentMethodReporter: WriteHeaderLineReconstructedNDF called on a disconnected reporter (out_ is null).");
        WriteHeaderLineReconstructedNDF(model, *out_, precision);
    }

    template <ThermoMap Thermo>
    void WriteHeaderLineReconstructedNDF(const AnyMomentMethod<Thermo>& any,
                                         unsigned precision = 16)
    {
        assert(out_ != nullptr && "MomentMethodReporter: WriteHeaderLineReconstructedNDF called on a disconnected reporter (out_ is null).");
        WriteHeaderLineReconstructedNDF(any, *out_, precision);
    }

    template <MomentMethod Model>
    requires HasReconstructedNDF<Model>
    void WriteReconstructedNDF(const Model& model,
                               int    nv                      = 100,
                               double vmin_nm3                = 1.0,
                               double vmax_nm3                = 1.0e6,
                               bool   use_regularized_moments = false)
    {
        assert(out_ != nullptr && "MomentMethodReporter: WriteReconstructedNDF called on a disconnected reporter (out_ is null).");
        WriteReconstructedNDF(model, *out_, nv, vmin_nm3, vmax_nm3, use_regularized_moments);
    }

    template <ThermoMap Thermo>
    void WriteReconstructedNDF(const AnyMomentMethod<Thermo>& any,
                               int    nv                      = 100,
                               double vmin_nm3                = 1.0,
                               double vmax_nm3                = 1.0e6,
                               bool   use_regularized_moments = false)
    {
        assert(out_ != nullptr && "MomentMethodReporter: WriteReconstructedNDF called on a disconnected reporter (out_ is null).");
        WriteReconstructedNDF(any, *out_, nv, vmin_nm3, vmax_nm3, use_regularized_moments);
    }

private:

    // -- Internals -------------------------------------------------------------

    OutputFileColumns* out_ = nullptr;
    std::vector<std::string> species_names_;

    // Column-label helper for moment-source vectors.
    static std::string col(std::string_view prefix,
                           unsigned j,
                           bool zf               = false,
                           std::string_view unit = "mol/m3/s")
    {
        return std::string(prefix) + "(" + std::to_string(j) + ")" + (zf ? "[ZF]" : "") + "[" +
               std::string(unit) + "]";
    }

    // Write all values of a span to the current row.
    void writeSpan(std::span<const double> s)
    {
        for (auto v : s)
            *out_ << v;
    }
};

template <MomentMethod Model>
void MomentMethodReporter::WriteHeader(const Model& model, unsigned precision)
{
    assert(out_ != nullptr && "MomentMethodReporter: WriteHeader called on a disconnected reporter (out_ is null).");
    constexpr unsigned N = Model::n_equations;

    // [ZF] marks columns backed by the base-class zero fallback for this variant.
    constexpr bool has_nuc = ModelsNucleation<Model>;
    constexpr bool has_coa = ModelsCoagulation<Model>;
    constexpr bool has_con = ModelsCondensation<Model>;
    constexpr bool has_gro = ModelsSurfaceGrowth<Model>;
    constexpr bool has_oxi = ModelsOxidation<Model>;
    constexpr bool has_sin = ModelsSintering<Model>;

    // Header-mode callback for optional variant columns.
    auto add_col = [&](std::string_view label, double /*unused_in_header*/)
    {
        out_->AddColumn(std::string(label), precision);
    };

    // -- Block 1: Core particle state (concept-mandated, truly common) ---------
    out_->AddColumn("Ys[-]", precision);
    out_->AddColumn("Ns[#/m3]", precision);
    out_->AddColumn("Ss[m2/m3]", precision);
    out_->AddColumn("fv[-]", precision);
    out_->AddColumn("dp[nm]", precision);
    out_->AddColumn("dc[nm]", precision);

    // -- Variant prefix columns ------------------------------------------------
    if constexpr (HasVariantPrefixOutput<Model>)
        model.variant_prefix_output(add_col);


    // -- Block 3: Transport (concept-mandated) ---------------------------------
    out_->AddColumn("D[kg/m/s]", precision);

    // -- Block 4: Total source terms (concept-mandated) ------------------------
    for (unsigned j = 0; j < N; ++j)
        out_->AddColumn(col("Sall", j, false, "mol/m3/s"), precision);

    // -- Block 5: Per-process source terms ([ZF]-tagged for zero-fallback) ------
    for (unsigned j = 0; j < N; ++j)
        out_->AddColumn(col("Snuc", j, !has_nuc), precision);
    for (unsigned j = 0; j < N; ++j)
        out_->AddColumn(col("Scoa", j, !has_coa), precision);
    for (unsigned j = 0; j < N; ++j)
        out_->AddColumn(col("Scon", j, !has_con), precision);
    for (unsigned j = 0; j < N; ++j)
        out_->AddColumn(col("Sgro", j, !has_gro), precision);
    for (unsigned j = 0; j < N; ++j)
        out_->AddColumn(col("Soxi", j, !has_oxi), precision);
    for (unsigned j = 0; j < N; ++j)
        out_->AddColumn(col("Ssin", j, !has_sin), precision);

    // -- Variant suffix columns ------------------------------------------------
    if constexpr (HasVariantSuffixOutput<Model>)
        model.variant_suffix_output(add_col);
}

template <MomentMethod Model> void MomentMethodReporter::WriteRow(const Model& model)
{
    assert(out_ != nullptr && "MomentMethodReporter: WriteRow called on a disconnected reporter (out_ is null).");
    // Row-mode callback for optional variant columns.
    auto add_val = [&](std::string_view /*unused_in_row*/, double value)
    {
        *out_ << value;
    };


    // -- Block 1: Core particle state (concept-mandated) -----------------------
    *out_ << model.mass_fraction();
    *out_ << model.particle_number_density();
    *out_ << model.specific_surface_area();
    *out_ << model.volume_fraction();
    *out_ << model.particle_diameter() * 1.e9;  // m → nm
    *out_ << model.collision_diameter() * 1.e9; // m → nm

    // -- Variant prefix values -------------------------------------------------
    if constexpr (HasVariantPrefixOutput<Model>)
        model.variant_prefix_output(add_val);


    // -- Block 3: Transport (concept-mandated) ---------------------------------
    *out_ << model.diffusion_coefficient();

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
    if constexpr (HasVariantSuffixOutput<Model>)
        model.variant_suffix_output(add_val);
}

template <MomentMethod Model>
requires HasReconstructedNDF<Model>
void MomentMethodReporter::WriteHeaderLineReconstructedNDF( const Model& model,
                                                            OutputFileColumns& ndf_out,
                                                            unsigned precision)
{
    // -- Header ---------------------------------------------------------------
    // Core columns (same for all NDF-capable variants)
    ndf_out.AddColumn("nu[nm3]",     precision);  // particle volume [nm³]
    ndf_out.AddColumn("n[#/m3/nm3]", precision);  // dimensional NDF n(v) [#/m³_gas/nm³]
    ndf_out.AddColumn("nbar[1/nm3]", precision);  // normalised NDF nbar(v) [1/nm³]
    ndf_out.AddColumn("dsph[nm]",    precision);  // sphere-equivalent diameter [nm]
    ndf_out.AddColumn("nd[#/m3/nm]", precision);  // diameter-space NDF n_d(d) [#/m³_gas/nm]
    ndf_out.AddColumn("ndbar[1/nm]", precision);  // normalised diameter-space NDF [1/nm]

    // Variant-specific extra NDF columns.
    auto add_col = [&](std::string_view label, double /*value*/)
    {
        ndf_out.AddColumn(std::string(label), precision);
    };
    if constexpr (HasNDFExtraOutput<Model>)
        model.ndf_extra_output(add_col);

    ndf_out.Complete();
}

template <MomentMethod Model>
requires HasReconstructedNDF<Model>
void MomentMethodReporter::WriteReconstructedNDF(   const Model& model,
                                                    OutputFileColumns& ndf_out,
                                                    int    nv,
                                                    double vmin_nm3,
                                                    double vmax_nm3,
                                                    bool   use_regularized_moments)
{
    // -- Volume grid: nv points log-spaced in [vmin_nm3, vmax_nm3] nm3 -------
    constexpr double pi = std::numbers::pi_v<double>;

    if (nv < 2)
        nv = 2;
    const double log_ratio = std::log(vmax_nm3 / vmin_nm3) / static_cast<double>(nv - 1);

    // Row-mode callback for optional NDF columns.
    auto add_val = [&](std::string_view /*label*/, double value)
    {
        ndf_out << value;
    };

    for (int i = 0; i < nv; ++i)
    {
        const double nu_nm3 = vmin_nm3 * std::exp(static_cast<double>(i) * log_ratio);
        const double nu_m3  = nu_nm3 * 1.e-27;   // nm3 -> m3

        // Query the model in SI units
        const double n_SI    = model.ReconstructedNDF(nu_m3, use_regularized_moments);
        const double nbar_SI = model.ReconstructedNormalizedNDF(nu_m3, use_regularized_moments);

        // Convert to nm-based units (1 m3 = 1e27 nm3)
        const double n_nm    = n_SI    * 1.e-27;   // [#/m³_gas / nm³_particle]
        const double nbar_nm = nbar_SI * 1.e-27;   // [1/nm³]

        // Sphere-equivalent diameter [nm] from volume [nm3]: d_sph = (6v/pi)^(1/3)
        const double dsph_nm = std::pow(6. * nu_nm3 / pi, 1. / 3.);

        // Diameter-space NDF via Jacobian |dv/dd_sph| = (pi/2) * d_sph^2.
        const double jacobian = (pi / 2.) * dsph_nm * dsph_nm;
        const double nd_nm    = n_nm    * jacobian;   // [#/m³_gas / nm]
        const double ndbar_nm = nbar_nm * jacobian;   // [1/nm]

        ndf_out.NewRow();
        ndf_out << nu_nm3;
        ndf_out << n_nm;
        ndf_out << nbar_nm;
        ndf_out << dsph_nm;
        ndf_out << nd_nm;
        ndf_out << ndbar_nm;

        if constexpr (HasNDFExtraOutput<Model>)
            model.ndf_extra_output(add_val);
    }
}

} // namespace MOM
