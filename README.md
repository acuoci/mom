# OpenSMOKE++ MOM - Method of Moments Library

> Production C++20/23 source-term kernels for particle population models in CFD solvers.

Developed at the **CRECK Modeling Lab**, Department of Chemistry, Materials and Chemical
Engineering, **Politecnico di Milano**.

---

## Overview

The MOM library provides high-performance Method of Moments source terms for particle-laden
reactive flows. It currently ships four variants:

- `HMOM`: four-equation Hybrid Method of Moments soot model.
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
| `ThreeEquations` | 3 | `Ys`, `NsNorm`, `Ss` | Soot with explicit surface-area evolution |
| `BrookesMoss` | 2 | `Ys`, `bs` | Economical soot model for large CFD calculations |
| `MetalOxide` | 3 | `Ysolid`, `NsolidN`, `Ssolid` | Generic metal-oxide nanoparticle synthesis |

### Process Capability Matrix

| Process | HMOM | ThreeEquations | BrookesMoss | MetalOxide |
|---|---|---|---|---|
| Nucleation | yes | yes | yes | yes |
| Coagulation | yes | yes | yes | yes |
| Condensation | yes | yes | zero span | yes |
| Surface growth | yes | yes | yes | zero span |
| Oxidation | yes | yes | yes | zero span |
| Sintering | zero span | zero span | zero span | yes |

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
    detail::TypeList<HMOM, BrookesMoss, ThreeEquations, MetalOxide>;
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

Adding a fifth MOM variant is localized and mechanical.

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
    detail::TypeList<HMOM, BrookesMoss, ThreeEquations, MetalOxide, MyVariant>;
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
