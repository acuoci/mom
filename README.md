# OpenSMOKE++ MOM - Method of Moments Library

> Production C++20/23 source-term kernels for particle population models in CFD solvers.

Developed at the **CRECK Modeling Lab**, Department of Chemistry, Materials and Chemical
Engineering, **Politecnico di Milano**.

---

## Overview

The MOM library provides high-performance Method of Moments source terms for particle-laden
reactive flows. It currently ships five variants:

- `HMOM`: four-equation Hybrid Method of Moments soot model.
- `HMOM6`: seven-equation Hybrid Method of Moments soot model with bivariate MOMIC closure.
- `ThreeEquations`: three-equation soot model with explicit surface-area transport.
- `BrookesMoss`: compact two-equation soot model.
- `MetalOxide`: generic metal-oxide nanoparticle model, including the TiO2 case.

The library is intended to be embedded directly in CFD cell loops. The hot path uses static
polymorphism, fixed-size Eigen vectors, `std::span` views, and compile-time process
fallbacks. Runtime model selection is provided through `AnyMomentMethod<Thermo>`, a thin
`std::variant` wrapper whose dispatch can be hoisted outside the spatial loop.

The public entry point is:

```cpp
#include "MOM/MOM.hpp"
```

This umbrella header exposes concrete variants, concepts, runtime selection, source access,
particle-property access, operator-splitting utilities, and reporting.

---

## Key Features

### Zero-Overhead Variant Implementations

Each model derives from `MomentMethodBase<Derived, NEq>` using CRTP. There are no virtual
calls in source-term evaluation. Process accessors such as `sources_nucleation()` resolve
at compile time:

```cpp
if constexpr (requires(const Derived& d) { d.sources_sintering_impl(); })
    return derived().sources_sintering_impl();
else
    return { kZeroData, NEq };
```

If a process is not modeled, the base class returns a span over a static zero array of the
correct size. No heap allocation, no dynamic branch, and no per-cell polymorphic lookup are
introduced.

### Unified Concept API

Every variant satisfies `MOM::MomentMethod<M>`. A CFD solver can be written against this
concept and remain independent of the concrete particle model:

```cpp
template <MOM::MomentMethod Model>
void AdvanceParticleSources(Model& model, /* solver fields */);
```

### Plain Configuration Structs

Each variant owns a nested `Config` struct. This keeps setup explicit, dependency-free, and
easy to populate from solver input files:

```cpp
using Thermo = MOM::BasicThermoData;
using Model  = MOM::HMOM<Thermo>;

Model model(thermo);

Model::Config cfg;
cfg.pah_species = "C16H10";
cfg.gas_closure_dummy_species = "N2";
cfg.nucleation_model = 1;
cfg.coagulation_model = 1;
cfg.condensation_model = 1;
cfg.surface_growth_model = 1;
cfg.oxidation_model = 1;

model.SetupFromConfig(cfg);
```

When `MOM_USE_DICTIONARY` is enabled, the same configuration path is used internally:
dictionary parsing produces a `Config`, then calls `SetupFromConfig()`.

### Runtime Selection Without Inner-Loop Dispatch

`AnyMomentMethod<Thermo>` is a `std::variant` over the registered concrete variants. Use
`MOM::ForEachCell()` to perform the `std::visit` once before the CFD loop:

```cpp
auto model = MOM::MakeAnyMomentMethod(thermo, variant_name);

MOM::ForEachCell(model, [&](auto& concrete)
{
    using Model = std::decay_t<decltype(concrete)>;
    constexpr unsigned neq = Model::n_equations;

    for (int c = 0; c < n_cells; ++c)
    {
        MOM::ComputeCell(concrete,
                         T[c],
                         P[c],
                         Y + c * n_species,
                         mu[c],
                         { moments + c * neq, neq });

        const auto src = concrete.sources();
        std::copy(src.begin(), src.end(), source_terms + c * neq);
    }
});
```

Inside the loop, `concrete` has a statically known type. The generated code is equivalent
to a compile-time model selection path.

---

## Supported Variants

| Variant | Equations | Transported variables | Main use |
|---|---:|---|---|
| `HMOM` | 4 | `M00`, `M10`, `M01`, `N0` | Detailed soot with two-node HMOM reconstruction |
| `HMOM6` | 7 | `M00`, `M10`, `M01`, `M20`, `M11`, `M02`, `N0` | Detailed soot with bivariate MOMIC closure |
| `ThreeEquations` | 3 | `Ys`, `NsNorm`, `Ss` | Soot with explicit surface-area evolution |
| `BrookesMoss` | 2 | `Ys`, `bs` | Economical soot model for large CFD calculations |
| `MetalOxide` | 3 | `Ysolid`, `NsolidN`, `Ssolid` | Generic metal-oxide nanoparticle synthesis |

### Process Capability Matrix

| Process | HMOM | HMOM6 | ThreeEquations | BrookesMoss | MetalOxide |
|---|---|---|---|---|---|
| Nucleation | yes | yes | yes | yes | yes |
| Coagulation | yes | yes | yes | yes | yes |
| Condensation | yes | yes | yes | zero span | yes |
| Surface growth | yes | yes | yes | yes | zero span |
| Oxidation | yes | yes | yes | yes | zero span |
| Sintering | zero span | zero span | zero span | zero span | yes |

The zero-span entries are compile-time fallbacks supplied by `MomentMethodBase`.

### MetalOxide Configuration Example

`MetalOxide` is configured through explicit material and stoichiometry data. The default
configuration reproduces TiO2 behavior, but the model is not tied to titanium:

```cpp
using Thermo = MOM::BasicThermoData;
using Model  = MOM::MetalOxide<Thermo>;

Model model(thermo);

Model::Config cfg;
cfg.solid_name = "TiO2";
cfg.precursor_species = "H4O4TI";
cfg.gas_closure_dummy_species = "TIO2RU";
cfg.gas_consumption = false;

cfg.solid_molecular_weight_kg_kmol = 79.866;
cfg.solid_density_kg_m3 = 4230.0;
cfg.solid_formula_units_per_precursor = 1.0;

cfg.gas_stoichiometry = {
    { "H2O", 2.0 }
};

cfg.nucleation_model = "binary";
cfg.closure_model    = "monodisperse";   // or "lognormal"
cfg.condensation_model = 1;
cfg.coagulation_model = 1;
cfg.sintering_model = 1;

model.SetupFromConfig(cfg);
```

---

## Architecture

### Header Domains

The public API is partitioned into small, focused header domains and re-exported by
`MOM/MOM.hpp`:

| Header | Responsibility |
|---|---|
| `MomentMethodConcept.hpp` | C++20 concepts and compile-time `ComputeCell()` |
| `MomentMethodBase.hpp` | CRTP base, common source buffers, process fallbacks |
| `MomVariantList.hpp` | Authoritative variant registry |
| `AnyMomentMethod.hpp` | Runtime `std::variant`, factory, `ForEachCell()` |
| `Dispatch.hpp` | State injection and computation for `AnyMomentMethod` |
| `Sources.hpp` | Total, gas, per-process, capability, and activation queries |
| `Properties.hpp` | Particle properties, transport, radiation, status queries |
| `Splitting.hpp` | Operator-splitting source combinations |
| `MomentMethodReporter.hpp` | Read-only diagnostic output observer |
| `MOMConfig.hpp` | Shared plain-data configuration blocks |

This layout keeps the user interface clean while allowing implementation-heavy dispatch,
source, property, and splitting helpers to remain small and highly inlineable. Translation
units can include `MOM/MOM.hpp` for the complete API or include a domain header directly
when only a subset is required.

### Runtime Wrapper

`AnyMomentMethod<Thermo>` is generated from the type list in `MomVariantList.hpp`:

```cpp
using AllVariants =
    detail::TypeList<HMOM, HMOM6, BrookesMoss, ThreeEquations, MetalOxide>;
```

The same registry drives:

- `AnyMomentMethod<Thermo>`
- `MakeAnyMomentMethod(thermo, label)`
- compile-time concept checks for every registered variant
- dispatch helpers in `Dispatch`, `Sources`, `Properties`, and `Splitting`

### Thermodynamics Boundary

Variants are templated on a thermodynamics backend satisfying `MOM::ThermoMap`. The
backend must provide species counts, species lookup, molecular weights, and atom counts.
`MOM::BasicThermoData` is provided for standalone use and tests. External solvers can
pass their own thermo adapter without coupling MOM to OpenFOAM, Cantera, or OpenSMOKE++.

---

## Quick Start

```cpp
#include "MOM/MOM.hpp"

#include <algorithm>
#include <span>
#include <vector>

using Thermo = MOM::BasicThermoData;
using Model  = MOM::ThreeEquations<Thermo>;

Thermo thermo;
// Fill thermo species names, molecular weights, and atom counts here.

Model model(thermo);

Model::Config cfg;
cfg.pah_species = "C16H10";
cfg.gas_closure_dummy_species = "N2";
cfg.nucleation_model = 1;
cfg.condensation_model = 1;
cfg.coagulation_model = 1;
cfg.surface_growth_model = 1;
cfg.oxidation_model = 1;
cfg.surface_chemistry_model = "rcpah";

model.SetupFromConfig(cfg);

static_assert(MOM::MomentMethod<Model>);

for (int c = 0; c < n_cells; ++c)
{
    MOM::ComputeCell(model,
                     T[c],
                     P[c],
                     Y + c * n_species,
                     mu[c],
                     { moments + c * Model::n_equations, Model::n_equations });

    const auto src = model.sources();              // zero-copy span
    const auto omega = model.omega_gas();          // gas sources [kg/m3/s]

    std::copy(src.begin(),
              src.end(),
              source_terms + c * Model::n_equations);
}
```

---

## Integration Patterns

### Compile-Time Variant

Use this path when the particle model is a solver template parameter or compile-time
choice:

```cpp
template <MOM::MomentMethod Model>
void CellLoop(Model& model,
              int n_cells,
              int n_species,
              const double* T,
              const double* P,
              const double* Y,
              const double* mu,
              const double* moments,
              double* source_terms)
{
    constexpr unsigned neq = Model::n_equations;

    for (int c = 0; c < n_cells; ++c)
    {
        MOM::ComputeCell(model,
                         T[c],
                         P[c],
                         Y + c * n_species,
                         mu[c],
                         { moments + c * neq, neq });

        const auto src = model.sources();
        std::copy(src.begin(), src.end(), source_terms + c * neq);
    }
}
```

### Runtime Variant

Use this path when the model name comes from an input file:

```cpp
MOM::AnyMomentMethod<Thermo> model =
    MOM::MakeAnyMomentMethod(thermo, variant_name);

MOM::ForEachCell(model, [&](auto& concrete)
{
    using Model = std::decay_t<decltype(concrete)>;
    constexpr unsigned neq = Model::n_equations;

    for (int c = 0; c < n_cells; ++c)
    {
        MOM::ComputeCell(concrete,
                         T[c],
                         P[c],
                         Y + c * n_species,
                         mu[c],
                         { moments + c * neq, neq });

        const auto src = concrete.sources();
        std::copy(src.begin(), src.end(), source_terms + c * neq);
    }
});
```

`ForEachCell()` performs exactly one `std::visit`. The loop body sees the concrete type and
is compiled as a normal template instantiation.

### Runtime Setup with Config

When the variant is selected at runtime and configured programmatically, configure inside a
single visit:

```cpp
MOM::ForEachCell(model, [&](auto& concrete)
{
    using Model = std::decay_t<decltype(concrete)>;
    typename Model::Config cfg;

    cfg.gas_closure_dummy_species = "N2";

    if constexpr (requires { cfg.pah_species; })
        cfg.pah_species = "C16H10";

    concrete.SetupFromConfig(cfg);
});
```

For dictionary-based setup, enable `MOM_USE_DICTIONARY` and use:

```cpp
MOM::SetupFromDictionary(model, dict);
```

---

## BrookesMoss Dictionary Configuration

The `BrookesMoss` dictionary configures the two-equation soot model based on soot mass fraction and normalized soot number density. The model supports the original Brookes-Moss kinetics and the extended Brookes-Moss-Hall nucleation/oxidation paths.

```cpp
Dictionary BrookesMoss
{
    @BrookesMoss                    true;

    @Precursors                     A2;
    @SurfaceGrowthSpecies           C2H2;
    @GasClosureDummySpecies         CSOOT;

    @NucleationModel                BrookesMoss;
    @SurfaceGrowthModel             1;
    @OxidationModel                 0;
    @CoagulationModel               1;
    @ThermophoreticModel            1;

    @GasConsumption                 false;

    @SootParticleDiameter           1 nm;
    @SootParticleMolecularWeight    144 kg/kmol;

    @Calpha                         54 1/s;
    @Talpha                         21000 K;
    @Cbeta                          1.;
    @Cgamma                         11700 kg*m/kmol/s;
    @Tgamma                         12100 K;
    @Comega                         105.8125 kg*m/kmol/sqrt(K)/s;
    @EtaColl                        0.04;
    @Coxid                          0.015;
    @NucleationExponent             1;
    @SurfaceGrowthExponent1         1;
    @SurfaceGrowthExponent2         1;
}
```

### Required Keywords

| Keyword | Type / units | Accepted values | Default in C++ config | Effect |
|---|---:|---|---:|---|
| `@BrookesMoss` | `bool` | `true`, `false` | `true` | Enables or disables the BrookesMoss variant. |
| `@Precursors` | `string` | Mechanism species name | `C2H2` | Species used by original Brookes-Moss nucleation. The species must exist and contain carbon. |
| `@SurfaceGrowthSpecies` | `string` | Mechanism species name | `C2H2` | Gas species consumed by surface growth. The species must exist and contain carbon. |
| `@GasClosureDummySpecies` | `string` | Mechanism species name or `none` | `none` | Optional dummy gas species used to close the gas-phase mass balance. |
| `@NucleationModel` | `string` | `0`, `none`, `1`, `BrookesMoss`, `2`, `BrookesMossHall` | `1` | Selects nucleation kinetics. BM-Hall requires `@Benzene` and `@PhenylRadical`. |
| `@SurfaceGrowthModel` | `int` | `0`, `1` | `1` | Enables surface growth when `1`. |
| `@OxidationModel` | `string` | `0`, `none`, `1`, `BrookesMoss`, `2`, `BrookesMossHall` | `1` | Selects oxidation kinetics. BM-Hall requires `@Benzene` and `@PhenylRadical`. |
| `@CoagulationModel` | `int` | `0`, `1` | `1` | Enables free-molecular coagulation when `1`. |
| `@ThermophoreticModel` | `int` | `0`, `1` | `0` | Enables thermophoretic drift contribution when `1`. |
| `@GasConsumption` | `bool` | `true`, `false` | `false` | If enabled, computes gas-phase source terms from soot nucleation, growth, and oxidation. |
| `@SootParticleDiameter` | measure | `m`, `mm`, `nm` | `1e-9 m` | Stored soot particle diameter. Current kernels use the dynamic diameter reconstructed from soot mass and number density. |
| `@SootParticleMolecularWeight` | measure | `kg/kmol`, `g/mol` | `144 kg/kmol` | Molecular weight assigned to the soot inception particle. BM-Hall defaults to `1200 kg/kmol` if this key keeps the default value. |

### Optional Physical, Transport, and Diagnostic Keywords

| Keyword | Type / units | Accepted values | Default | Effect |
|---|---:|---|---:|---|
| `@SootDensity` | measure | `kg/m3`, `g/cm3` | `1800 kg/m3` | Soot material density used in volume, diameter, surface-area, and coagulation calculations. |
| `@NsNorm` | measure | `#/m3`, `#/cm3` | `1e15 #/m3` | Normalization factor for the transported soot number-density variable. |
| `@RadiativeHeatTransfer` | `bool` | `true`, `false` | `true` | Enables particle contribution to optically thin radiation coupling. |
| `@PlanckCoefficient` | `string` | `Smooke`, `Kent`, `Sazhin`, `none` | `Smooke` | Selects the soot Planck mean absorption coefficient correlation. |
| `@SchmidtNumber` | `double` | Positive scalar expected | `50` | Particle Schmidt number used by `diffusion_coefficient() = mu / Sc`. |
| `@DebugMode` | `bool` | `true`, `false` | `false` | Enables verbose diagnostics. |

### Original Brookes-Moss Kinetic Constants

| Keyword | Type / units | Default | Kernel role |
|---|---:|---:|---|
| `@Calpha` | measure, `1/s` | `54 1/s` | Nucleation pre-exponential factor. |
| `@Talpha` | measure, `K` | `21000 K` | Nucleation activation temperature. |
| `@Cbeta` | `double` | `1` | Coagulation scaling coefficient. |
| `@Cgamma` | measure, `kg*m/kmol/s` | `11700 kg*m/kmol/s` | Surface-growth pre-exponential factor. |
| `@Tgamma` | measure, `K` | `12100 K` | Surface-growth activation temperature. |
| `@Comega` | measure, `kg*m/kmol/sqrt(K)/s` | `105.8125 kg*m/kmol/sqrt(K)/s` | OH oxidation pre-exponential factor. |
| `@EtaColl` | `double` | `0.04` | Collision efficiency in the OH oxidation rate. |
| `@Coxid` | `double` | `0.015` | Oxidation rate multiplier. |
| `@NucleationExponent` | `double` | `1` | Reaction-order exponent applied to precursor concentration in original BM nucleation. |
| `@SurfaceGrowthExponent1` | `double` | `1` | Reaction-order exponent applied to surface-growth species concentration. |
| `@SurfaceGrowthExponent2` | `double` | `1` | Exponent applied to particle surface area in the surface-growth rate. |

### Brookes-Moss-Hall Extension Keywords

These keys are only needed when `@NucleationModel BrookesMossHall;`, `@NucleationModel 2;`, `@OxidationModel BrookesMossHall;`, or `@OxidationModel 2;` is used.

| Keyword | Type / units | Default | Effect |
|---|---:|---:|---|
| `@Benzene` | `string` | `C6H6` | Benzene species for BM-Hall channel 2. Required by the parser when BM-Hall is active. Must have composition C6H6. |
| `@PhenylRadical` | `string` | `C6H5` | Phenyl radical species for BM-Hall inception. Required by the parser when BM-Hall is active. Must have composition C6H5. |
| `@Calpha1` | measure, `kg*m3/kmol2/s` | `127*10^8.88` | BM-Hall channel-1 nucleation pre-exponential factor. |
| `@Talpha1` | measure, `K` | `4378 K` | BM-Hall channel-1 activation temperature. |
| `@Calpha2` | measure, `kg*m3/kmol2/s` | `178*10^9.50` | BM-Hall channel-2 nucleation pre-exponential factor. |
| `@Talpha2` | measure, `K` | `6390 K` | BM-Hall channel-2 activation temperature. |
| `@Comega2` | measure, `kg*m/kmol/s/sqrt(K)` | `8903.51 kg*m/kmol/s/sqrt(K)` | BM-Hall O2 oxidation pre-exponential factor. |
| `@Tomega2` | measure, `K` | `19778 K` | BM-Hall O2 oxidation activation temperature. |

When BM-Hall is selected, the C++ configuration also applies BM-Hall-specific defaults unless the user explicitly overrides them: the soot particle molecular weight becomes `1200 kg/kmol`, the surface-growth coefficient becomes `9000.6 kg*m/kmol/s`, and the OH collision efficiency becomes `0.13`.

---

## HMOM Dictionary Configuration

The `HMOM` dictionary configures the four-equation Hybrid Method of Moments soot model. The transported variables are the normalized moments `[M00, M10, M01, N0]`, and the active source terms may include PAH dimerization nucleation, HACA surface growth, PAH condensation, oxidation, discrete coagulation, continuous coagulation, gas-phase coupling, thermophoresis, and soot radiation.

```cpp
Dictionary HMOM
{
    @HMOM                          true;

    @PAH                           A2;
    @SimplifiedPAHMass             true;
    @GasClosureDummySpecies        CSOOT;

    @CollisionDiameterModel        2;
    @FractalDiameterModel          1;
    @ThermophoreticModel           1;

    @NucleationModel               1;
    @SurfaceGrowthModel            1;
    @OxidationModel                0;
    @CondensationModel             1;
    @CoagulationModel              1;
    @ContinuousCoagulationModel    1;

    @GasConsumption                true;

    @DebugMode                     false;
}
```

### Required Keywords

| Keyword | Type / units | Accepted values | Default in C++ config | Effect |
|---|---:|---|---:|---|
| `@HMOM` | `bool` | `true`, `false` | `true` | Enables or disables the HMOM variant. |
| `@GasClosureDummySpecies` | `string` | Mechanism species name or `none` | `none` | Optional gas species used to close the gas-phase mass balance. It must exist when not `none`, cannot be the PAH precursor, and cannot be `H`, `H2`, `O2`, `OH`, `H2O`, or `C2H2`. |

### Core Species and Gas-Coupling Controls

| Keyword | Type / units | Accepted values | Default | Effect |
|---|---:|---|---:|---|
| `@PAH` | `string` | Mechanism species name | `C2H2` | PAH precursor used for dimerization nucleation and condensation. The species must exist in the thermodynamic map. |
| `@SimplifiedPAHMass` | `bool` | `true`, `false` | `false` | If enabled, the PAH molecular weight used by HMOM is computed as `nC * WC`, ignoring hydrogen. |
| `@GasConsumption` | `bool` | `true`, `false` | `true` | If enabled, computes gas-phase source terms from active soot processes. |
| `@DebugMode` | `bool` | `true`, `false` | `false` | Enables verbose diagnostics. |

### Process and Geometry Model Switches

| Keyword | Type | Accepted values | Default | Effect |
|---|---:|---|---:|---|
| `@NucleationModel` | `int` | `0`, `1` | `1` | Enables PAH dimerization nucleation when `1`. |
| `@SurfaceGrowthModel` | `int` | `0`, `1` | `1` | Enables HACA surface growth when `1`. |
| `@OxidationModel` | `int` | `0`, `1` | `1` | Enables O2/OH soot oxidation when `1`. |
| `@CondensationModel` | `int` | `0`, `1` | `1` | Enables PAH condensation on soot particles when `1`. |
| `@CoagulationModel` | `int` | `0`, `1` | `1` | Enables discrete HMOM coagulation when `1`. |
| `@ContinuousCoagulationModel` | `int` | `0`, `1` | `1` | Enables the continuum coagulation correction when `1`. |
| `@ThermophoreticModel` | `int` | `0`, `1` | `1` | Enables thermophoretic drift contribution when `1`. |
| `@FractalDiameterModel` | `int` | `0`, `1` | `1` | Selects the primary-particle/fractal-diameter closure. |
| `@CollisionDiameterModel` | `int` | `1`, `2` | `2` | Selects the aggregate collision-diameter closure. |

### Transport, Radiation, and Material Properties

| Keyword | Type / units | Accepted values | Default | Effect |
|---|---:|---|---:|---|
| `@SootDensity` | measure | `kg/m3`, `g/cm3` | `1800 kg/m3` | Soot material density used for PAH volume, soot volume, particle diameter, and coagulation geometry. |
| `@SchmidtNumber` | `double` | Positive scalar expected | `50` | Schmidt number used in the effective particle diffusion coefficient. |
| `@RadiativeHeatTransfer` | `bool` | `true`, `false` | `true` | Enables soot contribution to radiative heat transfer. |
| `@PlanckCoefficient` | `string` | `Smooke`, `Kent`, `Sazhin`, `none` | `Smooke` | Selects the soot Planck mean absorption coefficient model. Unrecognized labels map to `none` in the current C++ helper. |

### Surface Density and Active-Site Correction

| Keyword | Type / units | Accepted values | Default | Effect |
|---|---:|---|---:|---|
| `@SurfaceDensity` | measure | `#/m2`, `#/cm2`, `#/mm2` | `1.7e19 #/m2` | Active surface-site density used by HACA kinetics. |
| `@SurfaceDensityCorrectionCoefficient` | `bool` | `true`, `false` | `false` | Enables the temperature-dependent surface-density correction. |
| `@SurfaceDensityCorrectionCoefficientA1` | `double` | Any scalar | `12.65` | Coefficient `A1` for the surface-density correction. |
| `@SurfaceDensityCorrectionCoefficientA2` | `double` | Any scalar | `-0.00563` | Coefficient `A2` for the surface-density correction `[1/K]`. |
| `@SurfaceDensityCorrectionCoefficientB1` | `double` | Any scalar | `-1.38` | Coefficient `B1` for the surface-density correction. |
| `@SurfaceDensityCorrectionCoefficientB2` | `double` | Any scalar | `0.00069` | Coefficient `B2` for the surface-density correction `[1/K]`. |

### PAH Sticking Coefficient

| Keyword | Type | Accepted values | Default | Effect |
|---|---:|---|---:|---|
| `@StickingCoefficientModel` | `string` | `constant`, `pah-dependent` | `constant` | Selects the PAH-PAH sticking model used in dimerization nucleation. |
| `@StickingCoefficientConstant` | `double` | Non-negative scalar expected | `2e-3` | Base sticking coefficient. For `pah-dependent`, the implementation multiplies this value by a PAH-mass-dependent factor. |

### HACA Kinetic Parameters

Frequency factors must use the literal dictionary unit `cm3,mol,s`. Activation energies must use `kJ/mol`. The C++ implementation converts activation energies internally to activation temperatures.

| Keyword | Type / units | Default | Reaction role |
|---|---:|---:|---|
| `@A1f` | measure, `cm3,mol,s` | `6.72e1` | Forward rate for reaction 1. |
| `@A1b` | measure, `cm3,mol,s` | `6.44e-1` | Backward rate for reaction 1. |
| `@n1f` | `double` | `3.33` | Temperature exponent for reaction 1 forward. |
| `@n1b` | `double` | `3.79` | Temperature exponent for reaction 1 backward. |
| `@E1f` | measure, `kJ/mol` | `6.09` | Activation energy for reaction 1 forward. |
| `@E1b` | measure, `kJ/mol` | `27.96` | Activation energy for reaction 1 backward. |
| `@A2f` | measure, `cm3,mol,s` | `1.00e8` | Forward rate for reaction 2. |
| `@A2b` | measure, `cm3,mol,s` | `8.68e4` | Backward rate for reaction 2. |
| `@n2f` | `double` | `1.80` | Temperature exponent for reaction 2 forward. |
| `@n2b` | `double` | `2.36` | Temperature exponent for reaction 2 backward. |
| `@E2f` | measure, `kJ/mol` | `68.42` | Activation energy for reaction 2 forward. |
| `@E2b` | measure, `kJ/mol` | `25.46` | Activation energy for reaction 2 backward. |
| `@A3f` | measure, `cm3,mol,s` | `1.13e16` | Forward rate for reaction 3. |
| `@A3b` | measure, `cm3,mol,s` | `4.17e13` | Backward rate for reaction 3. |
| `@n3f` | `double` | `-0.06` | Temperature exponent for reaction 3 forward. |
| `@n3b` | `double` | `0.15` | Temperature exponent for reaction 3 backward. |
| `@E3f` | measure, `kJ/mol` | `476.05` | Activation energy for reaction 3 forward. |
| `@E3b` | measure, `kJ/mol` | `0.00` | Activation energy for reaction 3 backward. |
| `@A4` | measure, `cm3,mol,s` | `2.52e9` | C2H2-addition rate. |
| `@n4` | `double` | `1.10` | Temperature exponent for C2H2 addition. |
| `@E4` | measure, `kJ/mol` | `17.13` | Activation energy for C2H2 addition. |
| `@A5` | measure, `cm3,mol,s` | `2.20e12` | O2 oxidation rate. |
| `@n5` | `double` | `0.00` | Temperature exponent for O2 oxidation. |
| `@E5` | measure, `kJ/mol` | `31.38` | Activation energy for O2 oxidation. |
| `@Efficiency6` | `double` | `0.13` | Efficiency factor for the OH oxidation contribution. |

---

## HMOM6 Dictionary Configuration

The `HMOM6` dictionary configures the seven-equation Hybrid Method of Moments soot model with bivariate MOMIC closure. The transported variables are six normalized bivariate moments `[M00, M10, M01, M20, M11, M02]` plus the small-mode number density `N0`. The large mode is represented by a second-order polynomial in log-space (MOMIC, R=2), fitted analytically to the six large-mode moment residuals at each cell. The active source terms are identical to HMOM: PAH dimerization nucleation, HACA surface growth, PAH condensation, O2/OH oxidation, discrete coagulation, and continuum coagulation.

```cpp
Dictionary HMOM6
{
    @HMOM6                         true;

    @PAH                           A2;
    @SimplifiedPAHMass             true;
    @GasClosureDummySpecies        CSOOT;

    @CollisionDiameterModel        2;
    @FractalDiameterModel          1;
    @ThermophoreticModel           1;

    @NucleationModel               1;
    @SurfaceGrowthModel            1;
    @OxidationModel                0;
    @CondensationModel             1;
    @CoagulationModel              1;
    @ContinuousCoagulationModel    1;

    @GasConsumption                true;

    @DebugMode                     false;
}
```

### Transported Variables

| Index | Symbol | Physical meaning |
|---:|---|---|
| 0 | M00 | Zeroth-order moment ∝ total number density |
| 1 | M10 | First-order volume moment ∝ volume fraction |
| 2 | M01 | First-order surface moment ∝ specific surface area |
| 3 | M20 | Second-order volume moment |
| 4 | M11 | Cross volume-surface moment |
| 5 | M02 | Second-order surface moment |
| 6 | N0  | Small-mode (nucleation-mode) number density |

All variables are normalized as `M_{x,y,norm} = M_{x,y} / (V0^x · S0^y · Nav)` in units of mol/m³, where V0 and S0 are the volume and surface area of the dimer-derived inception particle.

### Required Keywords

| Keyword | Type / units | Accepted values | Default in C++ config | Effect |
|---|---:|---|---:|---|
| `@HMOM6` | `bool` | `true`, `false` | `true` | Enables or disables the HMOM6 variant. |
| `@GasClosureDummySpecies` | `string` | Mechanism species name or `none` | `none` | Optional gas species used to close the gas-phase mass balance. It must exist when not `none`, cannot be the PAH precursor, and cannot be `H`, `H2`, `O2`, `OH`, `H2O`, or `C2H2`. |

### Core Species and Gas-Coupling Controls

| Keyword | Type / units | Accepted values | Default | Effect |
|---|---:|---|---:|---|
| `@PAH` | `string` | Mechanism species name | `C2H2` | PAH precursor used for dimerization nucleation and condensation. The species must exist in the thermodynamic map. |
| `@SimplifiedPAHMass` | `bool` | `true`, `false` | `false` | If enabled, the PAH molecular weight is computed as `nC * WC`, ignoring hydrogen. |
| `@GasConsumption` | `bool` | `true`, `false` | `true` | If enabled, computes gas-phase source terms from active soot processes. |
| `@DebugMode` | `bool` | `true`, `false` | `false` | Enables verbose diagnostics. |

### Process and Geometry Model Switches

| Keyword | Type | Accepted values | Default | Effect |
|---|---:|---|---:|---|
| `@NucleationModel` | `int` | `0`, `1` | `1` | Enables PAH dimerization nucleation when `1`. |
| `@SurfaceGrowthModel` | `int` | `0`, `1` | `1` | Enables HACA surface growth when `1`. |
| `@OxidationModel` | `int` | `0`, `1` | `1` | Enables O2/OH soot oxidation when `1`. |
| `@CondensationModel` | `int` | `0`, `1` | `1` | Enables PAH condensation on soot particles when `1`. |
| `@CoagulationModel` | `int` | `0`, `1` | `1` | Enables discrete coagulation when `1`. |
| `@ContinuousCoagulationModel` | `int` | `0`, `1` | `1` | Enables the continuum coagulation correction when `1`. |
| `@ThermophoreticModel` | `int` | `0`, `1` | `1` | Enables thermophoretic drift contribution when `1`. |
| `@FractalDiameterModel` | `int` | `0`, `1` | `1` | Selects the primary-particle/fractal-diameter closure. Model 0 uses a spherical approximation; model 1 uses the Attili et al. fractal-aggregate law. |
| `@CollisionDiameterModel` | `int` | `1`, `2` | `2` | Selects the aggregate collision-diameter closure. |

### Transport, Radiation, and Material Properties

These keywords are identical to HMOM:

| Keyword | Type / units | Accepted values | Default | Effect |
|---|---:|---|---:|---|
| `@SootDensity` | measure | `kg/m3`, `g/cm3` | `1800 kg/m3` | Soot material density. |
| `@SchmidtNumber` | `double` | Positive scalar expected | `50` | Schmidt number for particle diffusion. |
| `@RadiativeHeatTransfer` | `bool` | `true`, `false` | `true` | Enables soot contribution to radiative heat transfer. |
| `@PlanckCoefficient` | `string` | `Smooke`, `Kent`, `Sazhin`, `none` | `Smooke` | Planck mean absorption coefficient model. |

### Surface Density and HACA Parameters

All surface density and HACA kinetic keywords are identical to HMOM. See the HMOM section
for `@SurfaceDensity`, `@SurfaceDensityCorrectionCoefficient`, `@StickingCoefficientModel`,
`@StickingCoefficientConstant`, and the full set of `@A1f`–`@Efficiency6` parameters.

### Operator-Splitting Notes

When oxidation is treated via operator splitting (`GetOxidationRateCoefficients`), HMOM6
applies a Cauchy-Schwarz realizability guard to the second-order moment decay rates. The
MOMIC polynomial must be extrapolated beyond its calibrated domain y ∈ {0,1,2} to evaluate
the M02 oxidation source (requires the moment at y = As_f+3, where As_f = 0 for the
spherical geometry model). Unconstrained, this extrapolation can over-estimate the M02
decay rate by several orders of magnitude for broad distributions, collapsing M02 to zero
in a single splitting step and invalidating the MOMIC closure. The guard bounds each
second-order kappa using the moment inequalities:

```
κ₃ ≤ max(0, 2·κ₁ − κ₀)   // M20·M00 ≥ M10²
κ₅ ≤ max(0, 2·κ₂ − κ₀)   // M02·M00 ≥ M01²
κ₄ ≤ max(0, (κ₃+κ₅)/2)   // M11² ≤ M20·M02
```

The source terms themselves are unaffected; only the splitting decay rates are bounded.

### Reporter Output

HMOM6 emits 17 model-specific prefix columns in addition to the standard gas-phase and source-term columns:

| Column | Unit | Description |
|---|---:|---|
| `np[-]` | `[-]` | Mean primary particles per aggregate |
| `ss[m2/#]` | `[m²/#]` | Mean surface per particle |
| `vs[m3/#]` | `[m³/#]` | Mean volume per particle |
| `N0[#/m3]` | `[#/m³]` | Small-mode number density |
| `NL[#/m3]` | `[#/m³]` | Large-mode number density |
| `alphaL[-]` | `[-]` | Large-mode number fraction |
| `dpL[nm]` | `[nm]` | Large-mode mean primary particle diameter |
| `npL[-]` | `[-]` | Large-mode mean primary particle count |
| `d63[nm]` | `[nm]` | Scattering effective diameter d63 |
| `sigma_dp[-]` | `[-]` | Log geometric std dev of dp (total NDF) |
| `sigma_np[-]` | `[-]` | Log geometric std dev of np (total NDF) |
| `sigma_dp_L[-]` | `[-]` | Log geometric std dev of dp (large mode) |
| `sigma_np_L[-]` | `[-]` | Log geometric std dev of np (large mode) |
| `gsd_dp[-]` | `[-]` | Geometric std dev of dp = exp(sigma_dp) |
| `gsd_np[-]` | `[-]` | Geometric std dev of np = exp(sigma_np) |
| `gsd_dp_L[-]` | `[-]` | Geometric std dev of dp for large mode |
| `gsd_np_L[-]` | `[-]` | Geometric std dev of np for large mode |

The `coagulation_detail()` accessor provides nine additional sub-vectors (discrete small-small, small-large, large-large, and their continuum counterparts) available through the `variant_suffix_output` hook.

---

## ThreeEquations Dictionary Configuration

The `ThreeEquations` dictionary configures the three-equation soot model. The transported variables are soot mass fraction `Ys`, scaled soot number density `NsNorm = Ns / 1e15`, and soot specific surface area `Ss`. The model supports PAH dimerization nucleation, PAH condensation, surface growth, oxidation, coagulation, thermophoretic transport, gas-phase coupling, and soot radiation.

```cpp
Dictionary ThreeEquations
{
    @ThreeEquations                 true;

    @NucleationModel                1;
    @SurfaceGrowthModel             1;
    @OxidationModel                 0;
    @CondensationModel              1;
    @CoagulationModel               1;

    @DimerModel                     qssa-rodrigues;

    @SurfaceChemistryModel          HMOM;
    @SootDensity                    1800 kg/m3;
    @ThermophoreticModel            1;

    @StickingCoefficientModel       constant;
    @StickingCoefficientConstant    2e-3;

    @SimplifiedPAHMass              true;
    @PAH                            A2;
    @GasConsumption                 false;
    @GasClosureDummySpecies         CSOOT;

    // @RadiativeHeatTransfer       true;
    // @PlanckCoefficient           Smooke;
    // @SchmidtNumber               50;

    @epsNucleation                  2.2;
    @epsCondensation                1.3;
    @epsCoagulation                 2.2;

    @DebugMode                      false;
}
```

### Required Dictionary Keywords

These keys are marked as compulsory by the OpenSMOKE++ grammar for the dictionary path.

| Keyword | Type / units | Accepted values | C++ default | Effect |
|---|---:|---|---:|---|
| `@ThreeEquations` | `bool` | `true`, `false` | `true` | Enables or disables the ThreeEquations variant. |
| `@PAH` | `string` | Mechanism species name | `C2H2` | PAH precursor used for dimerization nucleation and condensation. The species must exist in the thermodynamic map. |
| `@GasConsumption` | `bool` | `true`, `false` | `false` | Enables gas-phase source terms associated with soot processes. |
| `@GasClosureDummySpecies` | `string` | Mechanism species name or `none` | `none` | Optional dummy species used to close gas-phase mass balance. It cannot be the PAH species and cannot be `H`, `H2`, `O2`, `OH`, `H2O`, or `C2H2`. |

### Process Switches

| Keyword | Type | Accepted values | Default | Effect |
|---|---:|---|---:|---|
| `@NucleationModel` | `int` | `0`, `1` | `1` | Enables PAH dimerization nucleation when `1`. |
| `@SurfaceGrowthModel` | `int` | `0`, `1` | `1` | Enables soot surface growth when `1`. |
| `@OxidationModel` | `int` | `0`, `1` | `1` | Enables soot oxidation when `1`. |
| `@CondensationModel` | `int` | `0`, `1` | `1` | Enables PAH condensation on existing soot particles when `1`. |
| `@CoagulationModel` | `int` | `0`, `1` | `1` | Enables soot coagulation when `1`. |
| `@ThermophoreticModel` | `int` | `0`, `1` | `1` | Enables thermophoretic drift contribution when `1`. |

### PAH, Dimer, and Surface-Chemistry Models

| Keyword | Type | Accepted values | Default | Effect |
|---|---:|---|---:|---|
| `@DimerModel` | `string` | `qssa-rodrigues` | `qssa-rodrigues` | Selects the PAH dimer concentration closure. This is currently the only accepted model. |
| `@SurfaceChemistryModel` | `string` | `RC-PAH`, `rc-pah`, `RCPAH`, `rcpah`, `HMOM`, `hmom` | `rcpah` | Selects the surface-growth and oxidation chemistry formulation. |
| `@SimplifiedPAHMass` | `bool` | `true`, `false` | `false` | If enabled, the PAH molecular weight is computed as `nC * WC`, ignoring hydrogen atoms. |
| `@CorrectionCoefficientPAHPAH` | `double` | Scalar | `4.4` | Multiplicative correction applied to the PAH-PAH nucleation collision kernel. |
| `@StickingCoefficientModel` | `string` | `constant`, `pah-dependent` | `constant` | Selects the PAH sticking-coefficient model used in dimerization. |
| `@StickingCoefficientConstant` | `double` | Scalar | `2e-3` | Base sticking coefficient. For `pah-dependent`, the implementation multiplies this value by `mwpah^4`. |

### Collision Enhancement Factors

| Keyword | Type | Default | Effect |
|---|---:|---:|---|
| `@epsNucleation` | `double` | `2.5` | Collision enhancement factor for PAH dimerization nucleation. |
| `@epsCondensation` | `double` | `1.3` | Collision enhancement factor for PAH condensation. |
| `@epsCoagulation` | `double` | `2.2` | Collision enhancement factor for soot coagulation. |

### Material, Transport, Radiation, and Numerical Floors

| Keyword | Type / units | Accepted values | Default | Effect |
|---|---:|---|---:|---|
| `@SootDensity` | measure | `kg/m3`, `g/cm3` | `1800 kg/m3` | Soot material density used for PAH-derived geometry, soot volume fraction, particle diameter, and surface-area closure. |
| `@MinimumNs` | measure | `#/m3`, `#/cm3` | `1e6 #/m3` | Minimum soot number density used by regularized property calculations and initial transported values. |
| `@RadiativeHeatTransfer` | `bool` | `true`, `false` | `true` | Enables soot contribution to radiative heat transfer. |
| `@PlanckCoefficient` | `string` | `Smooke`, `Kent`, `Sazhin`, `none` | `Smooke` | Selects the soot Planck mean absorption coefficient model. Unrecognized labels map to `none` in the current helper. |
| `@SchmidtNumber` | `double` | Positive scalar expected | `50` | Schmidt number used by the particle diffusion coefficient. |
| `@DebugMode` | `bool` | `true`, `false` | `false` | Enables verbose diagnostic output. |

### Additional notes

- When using `pah-dependent`, provide an explicit calibrated `@StickingCoefficientConstant`. In case of `pah-dependent` sticking coefficient must be set to `1.5e-11`. The current default is always `2e-3`. 

---

## MetalOxide Dictionary Configuration

`MetalOxide` is a three-equation solid-oxide nanoparticle model. The transported variables are solid mass fraction `Ysolid`, scaled particle number density `NsolidN = N / 1e15`, and total particle surface area `Ssolid`. The model supports precursor-driven nucleation, condensation, coagulation, sintering, thermophoresis, explicit gas stoichiometry, and optional gas-phase mass closure.

The source-term closure for coagulation, condensation, and sintering is selected by `@ClosureModel`. The default `monodisperse` closure uses the mean-particle approximation. The `lognormal` closure reconstructs a dimensionless log-normal size distribution from the three transported fields, providing more accurate rates for polydisperse populations.

```cpp
Dictionary MetalOxide
{
    @MetalOxide                     true;

    @NucleationModel                binary;
    @ClosureModel                   monodisperse;
    @SinteringModel                 1;
    @CondensationModel              1;
    @CoagulationModel               1;
    @ThermophoreticModel            0;

    @Precursor                      H4O4TI;
    @GasClosureDummySpecies         TIO2RU;
    @GasConsumption                 false;

    // Required only when @GasConsumption true.
    // Coefficients are per precursor molecule: negative = reactant, positive = product.
    // @GasStoichiometry              H4O4TI:-1,H2O:2;
    // @GasStoichiometryMassTolerance 1e-3;

    // TiO2 defaults shown explicitly for clarity.
    @SolidName                      TiO2;
    @SolidMolecularWeight           79.866 kg/kmol;
    @SolidDensity                   4230 kg/m3;
    @SolidFormulaUnitsPerPrecursor  1;

    @MinimumFormulaUnits            2;
    @NucleatedParticleFormulaUnits  5;

    @SinteringDeferred              false;
    @SinteringDpMinimum             2e-9 m;
    @SinteringTauMinimum            1e-8 s;
    @SinteringKMaximum              1e5 1/s;

    // Sintering law: tau_s = As * T^ns * dp^4 * exp(Ts/T)
    // @As                           7.44e16 s,K,m;
    // @ns                           1;
    // @Ts                           31000 K;

    @SchmidtNumber                  50;

    // Optional numerical floors for property reconstruction.
    // @MinimumNs                    1e3 #/m3;
    // @MinimumFv                    1e-16;

    @DebugMode                      false;
}
```

### Required Dictionary Keywords

| Keyword | Type / units | Accepted values | C++ default | Effect |
|---|---:|---|---:|---|
| `@Precursor` | `string` | Mechanism species name or `none` | `none` | Gas-phase precursor used by nucleation and condensation. If not `none`, the species must exist in the thermodynamic map. |
| `@GasClosureDummySpecies` | `string` | Mechanism species name or `none` | `none` | Optional dummy species used to close the gas-phase mass balance. It must exist when not `none` and cannot be the precursor species. |
| `@GasConsumption` | `bool` | `true`, `false` | `false` | Enables gas-phase source terms from solid formation. If `true`, explicit gas stoichiometry must be provided and mass-balanced. |

### Activation and Process Switches

| Keyword | Type | Accepted values | Default | Effect |
|---|---:|---|---:|---|
| `@MetalOxide` | `bool` | `true`, `false` | `true` | Enables or disables the MetalOxide variant. |
| `@NucleationModel` | `string` | `0`, `none`, `1`, `binary`, `2`, `fixed-cluster` | `binary` | Selects the nucleation model. `binary` uses precursor-precursor collisions; `fixed-cluster` nucleates a cluster with `@NucleatedParticleFormulaUnits` formula units. |
| `@ClosureModel` | `string` | `monodisperse`, `lognormal` | `monodisperse` | Selects the closure used by coagulation, condensation, and sintering source terms. `lognormal` reconstructs a dimensionless log-normal population from `Ysolid`, `NsolidN`, and `Ssolid`. |
| `@SinteringModel` | `int` | `0`, `1` | `1` | Enables sintering source terms when `1`. |
| `@CondensationModel` | `int` | `0`, `1` | `1` | Enables precursor condensation on existing particles when `1`. |
| `@CoagulationModel` | `int` | `0`, `1` | `1` | Enables particle coagulation when `1`. |
| `@ThermophoreticModel` | `int` | `0`, `1` | `1` | Enables thermophoretic drift contribution when `1`. |

### Material and Stoichiometry

| Keyword | Type / units | Accepted values | Default | Effect |
|---|---:|---|---:|---|
| `@SolidName` | `string` | Label | `TiO2` | Solid product label used in summaries and diagnostics. |
| `@SolidMolecularWeight` | measure | `kg/kmol` | `79.866 kg/kmol` | Molecular weight of one solid formula unit. |
| `@SolidDensity` | measure | `kg/m3`, `g/cm3` | `4230 kg/m3` | Solid material density used for particle volume, diameter, collision kernels, and sintering. |
| `@SolidFormulaUnitsPerPrecursor` | `double` | Positive scalar | `1` | Number of solid formula units produced per precursor molecule. |
| `@GasStoichiometry` | `string` | `Species:coeff` or `Species=coeff` entries | empty | Explicit gas stoichiometry per precursor molecule. Negative coefficients consume gas species; positive coefficients produce gas species. Entries may be comma-separated, e.g. `H4O4TI:-1,H2O:2`. |
| `@GasStoichiometryMassTolerance` | `double` | Non-negative scalar | `1e-3` | Relative tolerance used to validate gas/solid mass balance when gas stoichiometry is supplied. |

When `@GasConsumption true;`, `@GasStoichiometry` must include the precursor with coefficient `-1`. The parser validates that gas stoichiometry plus `@SolidFormulaUnitsPerPrecursor * @SolidMolecularWeight` is mass-balanced within `@GasStoichiometryMassTolerance`.

`@ClosureModel lognormal;` currently requires `@SinteringDeferred false;` because deferred sintering is implemented only for the monodisperse closure.

### Cluster Size and Numerical Floors

| Keyword | Type / units | Accepted values | Default | Effect |
|---|---:|---|---:|---|
| `@MinimumFormulaUnits` | `int` | Positive integer | `2` | Minimum number of solid formula units used by regularized particle geometry. |
| `@NucleatedParticleFormulaUnits` | `int` | Positive integer | `5` | Number of solid formula units in a newly nucleated particle for the fixed-cluster nucleation path. |
| `@MinimumNs` | measure | `#/m3`, `#/cm3` | `1e3 #/m3` | Minimum particle number density used by regularized property reconstruction. |
| `@MinimumFv` | `double` | Non-negative scalar expected | `1e-16` | Minimum solid volume fraction used by regularized property reconstruction. |

### Sintering Controls

| Keyword | Type / units | Accepted values | Default | Effect |
|---|---:|---|---:|---|
| `@SinteringDeferred` | `bool` | `true`, `false` | `false` | If enabled, sintering is removed from the direct source vector and handled through `SinteringDeferredUpdate(dt)`. |
| `@SinteringDpMinimum` | measure | `m`, `mm`, `nm` | `2e-9 m` | Minimum primary particle diameter for active sintering. |
| `@SinteringTauMinimum` | measure | `s` | `1e-10 s` | Lower bound for the sintering time scale. |
| `@SinteringKMaximum` | measure | `1/s` | `1e6 1/s` | Upper bound applied to the effective sintering relaxation rate. |
| `@As` | measure | `s,K,m` | `7.44e16 s,K,m` | Sintering pre-exponential factor in `tau_s = As * T^ns * dp^4 * exp(Ts/T)`. |
| `@ns` | `double` | Scalar | `1` | Temperature exponent in the sintering time-scale law. |
| `@Ts` | measure | `K` | `31000 K` | Sintering activation temperature used as `exp(Ts/T)`. |

### Transport and Diagnostics

| Keyword | Type | Accepted values | Default | Effect |
|---|---:|---|---:|---|
| `@SchmidtNumber` | `double` | Positive scalar expected | `50` | Schmidt number used by the particle diffusion coefficient. |
| `@DebugMode` | `bool` | `true`, `false` | `false` | Enables verbose diagnostic output. |

Reporter output includes lognormal-closure diagnostics in dedicated columns:
`closureValid[-]`, `closureSigmaGM[-]`, `closureSigma[-]`, `closureM[-]`,
`closureKmean[-]`, `closureDppMean[nm]`, and `closureDcMean[nm]`. These are
separate from the legacy reconstructed-NDF fields such as `sigma[-]`.

### Code-Alignment Notes

- `@epsNucleation`, `@epsCondensation`, and `@epsCoagulation` are not currently registered by `MetalOxide_Grammar.cpp` and are not parsed by `MetalOxide::ParseConfig()`. The model has internal enhancement-factor defaults `2.5`, `1.3`, and `2.2`, but they are not dictionary-configurable in the current implementation.
- MetalOxide radiation is disabled by construction: the constructor sets the Planck model to `none`, and the dictionary does not expose `@RadiativeHeatTransfer` or `@PlanckCoefficient` for this variant.

---

## API Reference

### Variant Setup

| API | Description |
|---|---|
| `Model model(thermo)` | Construct a concrete model bound to a thermo backend |
| `typename Model::Config cfg` | Plain setup data for that variant |
| `model.SetupFromConfig(cfg)` | Apply all setup data and validate species/model options |
| `MOM::MakeAnyMomentMethod(thermo, label)` | Runtime construction from registered labels |
| `MOM::SetupFromDictionary(model, dict)` | Dictionary setup when `MOM_USE_DICTIONARY` is enabled |

### Per-Cell Computation

| API | Description |
|---|---|
| `MOM::ComputeCell(model, T, P, Y, mu, moments)` | Preferred hot-path entry point |
| `model.SetState(T, P, Y)` | Inject gas state |
| `model.SetMoments(span)` | Inject transported moment values |
| `model.SetViscosity(mu)` | Inject mixture viscosity |
| `model.ComputeSources()` | Compute all moment and gas source terms |

### Source Access

All accessors return `std::span<const double>` and do not allocate.

| API | Description |
|---|---|
| `model.sources()` | Total moment source vector |
| `model.sources_nucleation()` | Nucleation contribution |
| `model.sources_coagulation()` | Coagulation contribution |
| `model.sources_condensation()` | Condensation contribution or zero span |
| `model.sources_growth()` | Surface-growth contribution or zero span |
| `model.sources_oxidation()` | Oxidation contribution or zero span |
| `model.sources_sintering()` | Sintering contribution or zero span |
| `model.omega_gas()` | Gas-phase species source terms [kg/m3/s] |

For `AnyMomentMethod<Thermo>`, use the free functions in `Sources.hpp`:
`GetSources`, `GetOmegaGas`, `GetNucleationSources`, `GetSinteringSources`, and related
activation/capability queries.

### Particle Properties

| API | Unit |
|---|---|
| `volume_fraction()` | `[-]` |
| `particle_diameter()` | `[m]` |
| `collision_diameter()` | `[m]` |
| `particle_number_density()` | `[#/m3]` |
| `mass_fraction()` | `[-]` |
| `particle_density()` | `[kg/m3]` |
| `specific_surface_area()` | `[m2/m3]` |
| `number_primary_particles()` | `[-]` |
| `diffusion_coefficient()` | `[kg/m/s]` |
| `planck_coefficient(T, fv)` | `[1/m]` |

---

## Single-File Extensibility

Adding a new MOM variant is localized and mechanical.

1. Create `include/MyVariant/MyVariant.hpp`.
   - Define `template <ThermoMap Thermo> class MyVariant`.
   - Derive from `MomentMethodBase<MyVariant<Thermo>, NEq>`.
   - Define `static constexpr variant_labels`.
   - Define `struct Config` and `void SetupFromConfig(const Config&)`.
   - Satisfy `MOM::MomentMethod<MyVariant<Thermo>>`.

2. Create `include/MyVariant/MyVariant.tpp`.
   - Implement `SetState`, `SetMoments`, `ComputeSources`, property accessors, and optional
     process `sources_X_impl()` functions.
   - Omit a `sources_X_impl()` when the process is not modeled; the base class supplies the
     zero-span fallback.

3. Optionally create dictionary files.
   - `include/MyVariant/MyVariant_Grammar.h`
   - `src/MyVariant/MyVariant_Grammar.cpp`
   - Implement `ParseConfig(dict)` only when dictionary setup is needed.

4. Register the variant in `include/MOM/MomVariantList.hpp`.

```cpp
#include "MyVariant/MyVariant.hpp"

using AllVariants =
    detail::TypeList<HMOM, HMOM6, BrookesMoss, ThreeEquations, MetalOxide, MyVariant>;
```

No global factory switch, central enum, reporter change, dispatch-header edit, or global
configuration file is required. `AnyMomentMethod`, `MakeAnyMomentMethod`, concept checks,
and the domain free functions update from the registry.

---

## Diagnostics

`MomentMethodReporter` is a read-only observer. It uses the common concept interface plus
optional variant hooks:

```cpp
MOM::OutputFileColumns file("particle_sources.out");
MOM::MomentMethodReporter reporter(file, thermo.names);

reporter.WriteHeader(model);
file.Complete();

file.NewRow();
reporter.WriteRow(model);
```

Variant-specific columns are emitted through optional
`variant_prefix_output`, `variant_suffix_output`, and `ndf_extra_output` hooks when present.

---

## Building

### Requirements

| Component | Minimum |
|---|---|
| C++ compiler | C++20-capable compiler |
| CMake | 3.20 |
| Eigen | 3.4 |

### CMake Integration

```cmake
add_subdirectory(MOM)
target_link_libraries(MySolver PRIVATE MOM::MOM)
```

For solvers instantiating templates with custom thermo backends, use the header target:

```cmake
target_link_libraries(MySolver PRIVATE MOM::MOM_headers)
```

### Build Options

| Option | Default | Description |
|---|---:|---|
| `MOM_BUILD_SHARED` | `OFF` | Build shared library |
| `MOM_BUILD_TESTS` | `ON` | Build tests |
| `MOM_INSTALL` | `ON` | Generate install rules |
| `MOM_USE_DICTIONARY` | `OFF` | Enable OpenSMOKE++ dictionary parsing |

### Typical Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Recommended production flags:

```bash
-std=c++20 -O3 -march=native
```

---

## Testing

The test suite covers:

- construction and source evaluation for all registered variants
- runtime dictionary setup when enabled
- zero-span fallback behavior for unmodeled processes
- compile-time and runtime loop patterns
- reporter output on representative model states

Run:

```bash
ctest --test-dir build --output-on-failure
```

---

## References

Please cite the relevant physical model when using the library in scientific work.

**HMOM**

- M.E. Mueller, H. Pitsch, "Hybrid Method of Moments for modelling soot formation and
  growth," *Combustion and Flame* **156** (2009) 1143-1155.
- A. Attili, F. Bisetti, M.E. Mueller, H. Pitsch, soot moment-method extensions for
  turbulent flames.

**HMOM6 (bivariate MOMIC closure)**

- M.E. Mueller, G. Blanquart, H. Pitsch, "Hybrid Method of Moments for modeling soot
  formation and growth," *Combustion and Flame* **156** (2009) 1143-1155.
- M.E. Mueller, G. Blanquart, H. Pitsch, "A joint volume-surface model of soot aggregation
  with the method of moments," *Proceedings of the Combustion Institute* **32** (2009)
  785-792.

**Three-Equation Soot Model**

- B. Franzelli, A. Vie, N. Darabiha, "A three-equation model for soot formation in
  combustion," *Proceedings of the Combustion Institute* **37** (2019) 5411-5419.

**Brookes-Moss Soot Model**

- S.J. Brookes, J.B. Moss, "Predictions of soot and thermal radiation properties in
  confined turbulent jet diffusion flames," *Combustion and Flame* **116** (1999) 486-503.

**Metal-Oxide Nanoparticles**

- S.E. Pratsinis, "Simultaneous nucleation, condensation, and coagulation in aerosol
  reactors," *Journal of Colloid and Interface Science* **124** (1988) 416-427.
- R. Jossen, S.E. Pratsinis, W.J. Stark, L. Madler, flame-spray synthesis criteria for
  oxide nanoparticles.

---

## License

Copyright (C) 2026 Alberto Cuoci.

This library is distributed under the **GNU General Public License v3.0 or later**. See
the [`LICENSE`](LICENSE) file for the full text.

---

*CRECK Modeling Lab - Politecnico di Milano*<br>
*Department of Chemistry, Materials, and Chemical Engineering*<br>
*P.zza Leonardo da Vinci 32, 20133 Milano, Italy*<br>
*<https://www.creckmodeling.polimi.it>*
