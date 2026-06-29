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

/**
 * @file MomentMethodBase.tpp
 * @brief Template method implementations for MomentMethodBase<Derived, NEq>.
 *
 * This file is `#include`d at the **bottom** of MomentMethodBase.hpp and must
 * never be compiled as a standalone translation unit.  It exists solely to keep
 * the header readable: the class definition stays in the `.hpp` while the
 * out-of-line template bodies live here.
 *
 * @par Contents
 * Planck mean absorption coefficient correlations — three peer-reviewed empirical
 * fits relating the local soot volume fraction @p fv and gas temperature @p T to
 * the spectrally-integrated Planck mean absorption coefficient kP [1/m].
 *
 * kP enters the optically-thin radiative source term as
 *
 *   @f[
 *     \dot{q}_\mathrm{rad} = -4\,\sigma\,k_P\!\left(T^4 - T_\infty^4\right)
 *   @f]
 *
 * where @f$\sigma = 5.670\times10^{-8}\ \mathrm{W\,m^{-2}\,K^{-4}}@f$ is the
 * Stefan–Boltzmann constant and @f$T_\infty@f$ is an ambient reference
 * temperature.  The sign convention makes @f$\dot{q}_\mathrm{rad}@f$ negative
 * (a heat *loss* from the gas phase) whenever @f$T > T_\infty@f$.
 *
 * The active correlation is selected at construction time via
 * ProcessFlags::PlanckCoeffModel and dispatched in
 * MomentMethodBase::planck_coefficient().  The three implementations below are
 * otherwise functionally independent — each is a pure arithmetic function with
 * no side effects and no access to member state.
 *
 * @note All three correlations are empirical curve-fits valid over a limited
 *       temperature range (typically 1000–2500 K) and for soot volume fractions
 *       characteristic of laminar/mildly turbulent hydrocarbon diffusion flames.
 *       They should not be extrapolated outside these conditions without
 *       verification against independent data.
 */

#pragma once

#include <cmath>

namespace MOM
{

// ============================================================================
// PlanckSmooke
// ============================================================================

/**
 * @brief Planck mean absorption coefficient — Smooke et al. (2005) correlation.
 *
 * Linear fit in both @p fv and @p T:
 *
 *   @f[ k_P = C_1 \, f_v \, T \quad \left[\mathrm{m}^{-1}\right] @f]
 *
 * with @f$C_1 = 1232.4\ \mathrm{K^{-1}\,m^{-1}}@f$.
 *
 * @par Reference
 * M.D. Smooke, R.J. Hall, M.B. Colket, P. Tomboulides, M.B. Long,
 * C.S. McEnally, L.D. Pfefferle, *Combust. Theory Modelling* **8** (2004)
 * 593–606; M.D. Smooke et al., *Combust. Flame* **143** (2005) 613–628.
 *
 * @par Validity
 * Calibrated against laminar diffusion flame data; intended temperature range
 * roughly 1000–2500 K for hydrocarbon soot.
 *
 * @param T  Local gas temperature [K].  Must be positive; no guard is applied.
 * @param fv Soot volume fraction [-].  Typically 1×10⁻⁸ – 1×10⁻⁵ in flames.
 * @return   Planck mean absorption coefficient kP [1/m].
 *
 * @note The function is @c const and @c noexcept; it reads no member state and
 *       performs only two multiplications — safe to call in inner loops.
 *       Dispatched by planck_coefficient() when PlanckCoeffModel::Smooke is set.
 */
template <class Derived, unsigned NEq>
double MomentMethodBase<Derived, NEq>::PlanckSmooke(double T, double fv) const noexcept
{
    constexpr double C1 = 1232.4; // [K^{-1} m^{-1}]  — Smooke et al. (2005)
    return C1 * fv * T;
}

// ============================================================================
// PlanckKent
// ============================================================================

/**
 * @brief Planck mean absorption coefficient — Kent & Honnery (1990) correlation.
 *
 * Linear in @p fv, affine in @p T (temperature-offset form):
 *
 *   @f[ k_P = C_1 \, f_v \, (C_2 + T) \quad \left[\mathrm{m}^{-1}\right] @f]
 *
 * with @f$C_1 = 1.3\times10^5\ \mathrm{m^{-1}}@f$ and @f$C_2 = 0\ \mathrm{K}@f$.
 *
 * @par Why C2 = 0
 * The original Kent & Honnery fit contains a polynomial in T whose constant
 * term is negligible over the flame-temperature range.  @p C2 is retained as a
 * named constant (rather than being silently absorbed into C1) so the
 * affine structure of the formula remains legible and the offset can be
 * restored to its original non-zero value without restructuring the expression.
 *
 * @par Reference
 * J.H. Kent, D. Honnery, "Soot and mixture fraction in turbulent diffusion
 * flames," *Combust. Sci. Tech.* **75** (1990) 167–177.
 *
 * @par Validity
 * Derived from turbulent flame measurements; applicable over ~1200–2400 K.
 *
 * @param T  Local gas temperature [K].  Must be positive; no guard is applied.
 * @param fv Soot volume fraction [-].
 * @return   Planck mean absorption coefficient kP [1/m].
 *
 * @note The function is @c const and @c noexcept.
 *       Dispatched by planck_coefficient() when PlanckCoeffModel::Kent is set.
 */
template <class Derived, unsigned NEq>
double MomentMethodBase<Derived, NEq>::PlanckKent(double T, double fv) const noexcept
{
    constexpr double C1 = 1.3e5; // [m^{-1}]   — Kent & Honnery (1990), leading linear coefficient
    constexpr double C2 = 0.;    // [K]         — temperature offset; zero in the adopted linearisation
    return C1 * fv * (C2 + T);
}

// ============================================================================
// PlanckSazhin
// ============================================================================

/**
 * @brief Planck mean absorption coefficient — Sazhin (1994) polynomial correlation.
 *
 * Third-degree polynomial in temperature, linear in @p fv:
 *
 *   @f[
 *     k_P = f_v \left( C_0 + C_1 T + C_2 T^2 + C_3 T^3 \right)
 *           \quad \left[\mathrm{m}^{-1}\right]
 *   @f]
 *
 * Adopted coefficient values:
 *
 * | Symbol | Value    | Unit              | Physical role              |
 * |--------|----------|-------------------|----------------------------|
 * | C0     | 6.30e+2  | m⁻¹               | constant absorption offset |
 * | C1     | 6.30e-1  | m⁻¹ K⁻¹           | dominant linear growth     |
 * | C2     | -1.00e-4 | m⁻¹ K⁻²           | negative curvature (mild saturation at high T) |
 * | C3     | 0        | m⁻¹ K⁻³           | cubic term — zero in adopted quadratic fit |
 *
 * @par Why C3 = 0
 * The quadratic truncation already reproduces the mild negative curvature of
 * kP at temperatures above ~2000 K.  C3 is kept explicit (rather than removed)
 * so the polynomial degree remains obvious and upgrading to a full cubic fit
 * requires only changing the one constant, not restructuring the expression.
 *
 * @par Implementation
 * @c T2 = T*T is computed once and shared between the quadratic and (potential)
 * cubic terms, eliminating a redundant multiplication.
 *
 * @par Reference
 * S.S. Sazhin, "An approximation for the absorption of thermal radiation by
 * soot particles," *Prog. Energy Combust. Sci.* **20** (1994) 297–318.
 *
 * @par Validity
 * Fitted to soot absorption data over approximately 1000–3000 K.  This is the
 * widest validated range among the three correlations and is therefore the
 * recommended default when high-temperature accuracy is required.
 *
 * @param T  Local gas temperature [K].  Must be positive; no guard is applied.
 * @param fv Soot volume fraction [-].
 * @return   Planck mean absorption coefficient kP [1/m].
 *
 * @note The function is @c const and @c noexcept.
 *       Dispatched by planck_coefficient() when PlanckCoeffModel::Sazhin is set.
 */
template <class Derived, unsigned NEq>
double MomentMethodBase<Derived, NEq>::PlanckSazhin(double T, double fv) const noexcept
{
    constexpr double C0 = 6.3e2;   // [m^{-1}]           — constant term (absorption at T→0, extrapolated)
    constexpr double C1 = 6.3e-1;  // [m^{-1} K^{-1}]   — linear coefficient (dominant contribution)
    constexpr double C2 = -1.0e-4; // [m^{-1} K^{-2}]   — quadratic coefficient (negative → saturation at high T)
    constexpr double C3 = 0.;      // [m^{-1} K^{-3}]   — cubic coefficient (zero: quadratic fit is sufficient)
    const double T2     = T * T;   // T^2 precomputed — shared by C2 and C3 terms
    return fv * (C0 + C1 * T + C2 * T2 + C3 * T2 * T);
}

} // namespace MOM
