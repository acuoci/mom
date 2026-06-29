# OpenSMOKE++ MOM — Method of Moments Library

> **A production-grade, C++20/23 library for computing particle population source terms  
> in Computational Fluid Dynamics codes via the Method of Moments.**

Developed at the **CRECK Modeling Lab**, Department of Chemistry, Materials and Chemical
Engineering, **Politecnico di Milano**.

---

## Abstract

The **MOM library** provides a rigorously architected, high-performance implementation of
four Method of Moments (MOM) variants for particle-laden reactive flows: the Hybrid Method
of Moments (HMOM), the Three-Equation soot model, the Brookes–Moss soot model, and a
TiO₂ nanoparticle synthesis model.

The library is designed as an *embeddable kernel*: it is integrated directly into the inner
cell loop of an external CFD solver and must therefore carry zero overhead beyond the
physical computation itself. This constraint drives every architectural decision — from the
choice of the Curiously Recurring Template Pattern (CRTP) to eliminate virtual dispatch, to
the use of `static constexpr` zero arrays to reduce unmodelled processes to a pointer
constant at compile time.

The public API is expressed as a C++20 **Concept** (`MomentMethod<M>`), which means any
conforming variant can be substituted for any other by changing a single type alias in the
calling solver, with the compiler verifying the substitution is valid at include time. A
thin type-erasure layer (`AnyMomentMethod<Thermo>`) enables the variant selection to be
deferred to runtime (e.g. from an input file keyword) while preserving bare-metal
performance inside the cell loop via `std::visit`-before-loop hoisting.

---

## Table of Contents

- [Key Features](#key-features)
- [Supported Variants](#supported-variants)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Integration Patterns](#integration-patterns)
- [API Reference](#api-reference)
- [Building the Library](#building-the-library)
- [Code Style](#code-style)
- [Testing](#testing)
- [References](#references)
- [License](#license)

---

## Key Features

### Zero-Overhead Compile-Time Polymorphism via CRTP

All variant classes derive from `MomentMethodBase<Derived, NEq>` using the Curiously
Recurring Template Pattern. There are **no virtual functions** anywhere in the performance-
critical path. Every call to `sources_X()` in the base resolves to a direct, inlined call
to `Derived::sources_X_impl()` — the compiler sees the full call chain as a single
straight-line computation and optimises accordingly.

```
Solver calls:  model.sources_nucleation()
                   │
           MomentMethodBase::sources_nucleation()          ← inline, no virtual call
               │
               ├─ if constexpr: Derived has sources_nucleation_impl()?
               │      yes → derived().sources_nucleation_impl()   ← [[gnu::always_inline]]
               └─      no  → { kZeroData, NEq }                   ← compile-time constant
```

The two-level chain (`sources_X → derived().sources_X_impl()`) is guaranteed to be
collapsed to a single basic block by any optimising compiler at `-O2` or higher.
`[[gnu::always_inline]]` on `_impl()` methods enforces this even in debug builds.

### Static Pruning of Unmodelled Processes

Each kinetic sub-process (nucleation, coagulation, condensation, surface growth,
oxidation, sintering) is individually optional. A variant that does not model sintering
simply omits `sources_sintering_impl()` from its public interface. The base class detects
this at compile time:

```cpp
// In MomentMethodBase<Derived, NEq>:
[[nodiscard, gnu::always_inline]]
std::span<const double> sources_sintering() const noexcept
{
    if constexpr (requires(const Derived& d) { d.sources_sintering_impl(); })
        return derived().sources_sintering_impl();
    else
        return { kZeroData, NEq };   // pointer to .rodata — zero cost
}
```

`kZeroData` is a `static constexpr double[NEq]` living in read-only memory. The compiler
represents the fallback return as a pair of constants (`lea rax, [rip + kZeroData]`,
`mov edx, NEq`). Any downstream arithmetic over the returned span (`+= src[i]`) is
constant-propagated to `+= 0.0` and eliminated entirely by the dead-store pass.

### Fixed-Size Eigen Vectors — SIMD-Ready Storage

Moment source vectors are stored as `Eigen::Matrix<double, NEq, 1>` (`MomentVector`).
Fixed-size allocation means the entire vector fits in registers for small `NEq`:

| Variant         | NEq | Vector size | Fits in |
|-----------------|-----|-------------|---------|
| HMOM            | 4   | 32 bytes    | 1 × AVX2 `ymm` register |
| ThreeEquations  | 3   | 24 bytes    | 1 × SSE2 `xmm` + scalar |
| BrookesMoss     | 2   | 16 bytes    | 1 × SSE2 `xmm` register |
| TiO₂            | 3   | 24 bytes    | 1 × SSE2 `xmm` + scalar |

With `-O3 -march=native`, Eigen's fixed-size expression templates generate vectorised
arithmetic over all four source equations simultaneously via `vmovupd`/`vmulsd`/`vaddsd`
using `xmm`/`ymm` registers.

### Bounds-Safe, Zero-Copy Data Passing via `std::span`

All source term accessors return `std::span<const double>` — a non-owning view into the
variant's internal `MomentVector`. No heap allocation, no copy, no size field mismatch.
The CFD solver accesses the data in-place:

```cpp
auto src = model.sources_nucleation();   // span into model's internal storage
for (unsigned j = 0; j < src.size(); ++j)
    residual[j] += src[j] * cell_volume;  // direct read, no copy
```

### Unified C++20 Concept API

The `MomentMethod<M>` concept in `MomentMethodConcept.hpp` is the single authoritative
contract for all variants. A `static_assert` at the solver include site catches any
divergence between a variant's implementation and the API at compile time:

```cpp
using ParticleModel = MOM::HMOM<MyThermo>;
static_assert(MOM::MomentMethod<ParticleModel>,
    "ParticleModel must satisfy MOM::MomentMethod");
```

Switching variants requires changing **exactly one line** in the solver.

### Type-Erased Runtime Selection — Single Dispatch Overhead

`AnyMomentMethod<Thermo>` is a `std::variant<HMOM<Thermo>, BrookesMoss<Thermo>, ...>`
with a factory function and a `ForEachCell` dispatcher. The variant selection (one
`std::visit`) is hoisted *outside* the spatial loop:

```
Initialisation:  MakeAnyMomentMethod(thermo, "HMOM")   ← one runtime branch
                     │
                 ForEachCell(model, lambda)             ← one std::visit
                     │
Cell loop:           for (i = 0; i < N; ++i)           ← bare-metal iteration
                         ComputeCell(m, ...)            ← fully inlined, no dispatch
```

The inner loop is identical in machine code to the compile-time template version.

---

## Supported Variants

### Process Matrix

| Process | HMOM | ThreeEquations | BrookesMoss | TiO₂ |
|---|---|---|---|---|
| **Nucleation** | ✅ | ✅ | ✅ | ✅ |
| **Coagulation** | ✅ | ✅ | ✅ | ✅ |
| **Condensation** | ✅ | ✅ | ⬜ zero-span | ✅ |
| **Surface Growth** | ✅ | ✅ | ✅ | ⬜ zero-span |
| **Oxidation** | ✅ | ✅ | ✅ | ⬜ zero-span |
| **Sintering** | ⬜ zero-span | ⬜ zero-span | ⬜ zero-span | ✅ |

**✅ modelled** — `sources_X_impl()` is present; base class forwards the call directly.  
**⬜ zero-span** — `sources_X_impl()` is absent; base class returns `{ kZeroData, NEq }` at zero cost.

### Variant Summary

#### HMOM — Hybrid Method of Moments

Transport equations: **4** (`M00`, `M10`, `M01`, `N0`)

The HMOM reconstructs the Number Density Function (NDF) using two delta nodes (a large-
particle mode and a small-particle nucleation mode, `N0`). The two-point quadrature
enables separate tracking of nucleation-mode particles without requiring closure
assumptions for the full NDF shape.

| Index | Symbol | Physical quantity |
|---|---|---|
| 0 | M00 | Zeroth-order moment — proportional to total number density |
| 1 | M10 | First-order volume moment |
| 2 | M01 | First-order surface moment |
| 3 | N0  | Small-particle (nucleation-mode) number density |

Coagulation is split into nine sub-mechanisms (discrete, discrete-SS, discrete-SL,
discrete-LL, continuous, continuous-SS, continuous-SL, continuous-LL, total), each
accessible individually via `coagulation_detail()`.

*Key references:* Mueller et al. (2009) *Combust. Flame* **156**, 1143–1155; Attili &
Bisetti (2013) *Combust. Flame* **160**, 1920–1934.

#### ThreeEquations — Three-Equation Soot Model

Transport equations: **3** (`Ys`, `NsNorm`, `Ss`)

| Index | Symbol | Physical quantity |
|---|---|---|
| 0 | Ys     | Soot mass fraction [-] |
| 1 | NsNorm | Scaled number density = Ns / 10¹⁵ [−] |
| 2 | Ss     | Specific surface area [m²/m³] |

A log-normal NDF is assumed. Surface chemistry follows the HACA (Hydrogen-Abstraction
Carbon-Addition) mechanism with optional Pareto/LogNormal NDF reconstruction.

*Key reference:* Eberle et al. (2015) *Combust. Flame* **162**, 2648–2660.

#### BrookesMoss — Brookes–Moss Soot Model

Transport equations: **2** (`Ys`, `bs`)

| Index | Symbol | Physical quantity |
|---|---|---|
| 0 | Ys | Soot mass fraction [-] |
| 1 | bs | Normalised radical nuclei concentration = Ns/(ρ·Ns_norm) [m³/kg] |

The simplest two-equation variant. Condensation and sintering are not modelled and resolve
to zero-cost static spans. Suitable for large simulations where computational economy is
paramount and the full NDF shape is not required.

*Key reference:* Brookes & Moss (1999) *Combust. Flame* **116**, 486–503.

#### TiO₂ — Titanium Dioxide Nanoparticle Synthesis

Transport equations: **3** (`YTiO₂`, `NTiO₂N`, `STiO₂`)

| Index | Symbol | Physical quantity |
|---|---|---|
| 0 | YTiO₂  | TiO₂ particle mass fraction [-] |
| 1 | NTiO₂N | Scaled number density = N / 10¹⁵ [−] |
| 2 | STiO₂  | Total surface area concentration [m²/m³] |

Models the aerosol synthesis of TiO₂ nanoparticles from gas-phase precursors (e.g. TiCl₄
oxidation). Surface growth and oxidation are not modelled. Sintering is modelled
explicitly, which distinguishes this variant from the soot models. NDF closure follows a
log-normal assumption with geometric standard deviation.

---

## Architecture

### Class Hierarchy

```
MomentMethodBase<Derived, NEq>          ← CRTP base (MomentMethodBase.hpp)
│   Stores:  MomentVector sources_[6]   (Eigen fixed-size, NEq doubles each)
│            kZeroData[NEq]             (static constexpr, .rodata)
│            omega_gas_[]               (dynamic, sized at SetStatus time)
│   Provides: sources_X()              (compile-time dispatch via if constexpr)
│             planck_coefficient()     (3 empirical correlations in .tpp)
│             all concept accessors    (particle properties, transport, etc.)
│
├── HMOM<Thermo>           : Base<HMOM<Thermo>,           4>
├── ThreeEquations<Thermo> : Base<ThreeEquations<Thermo>, 3>
├── BrookesMoss<Thermo>    : Base<BrookesMoss<Thermo>,    2>
└── TiO2<Thermo>           : Base<TiO2<Thermo>,           3>

MomentMethod<M>                         ← C++20 Concept (MomentMethodConcept.hpp)
    requires: all public API methods and types

AnyMomentMethod<Thermo>                 ← type-erased runtime wrapper
    = std::variant<HMOM<T>, BrookesMoss<T>, ThreeEquations<T>, TiO2<T>>
    MakeAnyMomentMethod(thermo, label)  ← factory (label = "HMOM", "TiO2", ...)
    ForEachCell(variant, lambda)        ← hoisted std::visit pattern

AllVariants                             ← single authoritative registry
    = detail::TypeList<HMOM, BrookesMoss, ThreeEquations, TiO2>
      (MomVariantList.hpp — the only file to edit when adding a variant)
```

### CRTP Dispatch — Step by Step

Consider a CFD solver requesting nucleation sources for HMOM. The call chain at
`-O2 -march=native` is:

```
1.  model.sources_nucleation()
         ↓   [inlined — no call instruction]
2.  MomentMethodBase<HMOM<T>, 4>::sources_nucleation()
         ↓   if constexpr: HMOM has sources_nucleation_impl()? → YES
3.  derived().sources_nucleation_impl()
         ↓   [[gnu::always_inline]] — forced into call site even at -O0
4.  return { source_nucleation_.data(), 4 }   ← span into Eigen vector
         ↓   [entire chain inlined; assembly: lea rax, [rbp+offset]; mov edx, 4]
5.  Caller receives std::span<const double> — pointer + size, no heap
```

For a variant without that process (e.g. `BrookesMoss::sources_condensation()`):

```
1.  model.sources_condensation()
         ↓
2.  MomentMethodBase<BrookesMoss<T>, 2>::sources_condensation()
         ↓   if constexpr: BrookesMoss has sources_condensation_impl()? → NO
3.  return { kZeroData, 2 }   ← pointer to .rodata + constant 2
         ↓   [branch eliminated at compile time; no runtime test]
         ↓   downstream "+= span[i]" → "+= 0.0" → eliminated by DCE pass
```

### Variant Registry

Adding a new variant requires changes in **one file only**:

```cpp
// MomVariantList.hpp — append the new type here
using AllVariants = detail::TypeList<
    HMOM,
    BrookesMoss,
    ThreeEquations,
    TiO2,
    MyNewVariant    // ← add here
>;
```

The factory (`MakeAnyMomentMethod`), the type-erased variant, and all dispatch helpers
(`ForEachCell`, `GetNEquations`, etc.) update automatically via template instantiation.
No other file is modified.

### ThermoMap Concept

All variants are parameterised on `Thermo`, which must satisfy the `ThermoMap` concept:

```cpp
template <typename T>
concept ThermoMap = requires(const T t, std::string_view name) {
    { t.NumberOfSpecies() } -> std::convertible_to<int>;
    { t.IndexOfSpecies(name) } -> std::convertible_to<int>;
    { t.MolecularWeight(0) } -> std::convertible_to<double>;
    // ... atom count methods
};
```

The built-in `BasicThermoData` struct satisfies `ThermoMap` and is ready to use
standalone. For integration with full CFD thermodynamic backends (OpenFOAM, OpenSMOKE++),
any class satisfying the concept is accepted without modification to the MOM library.

---

## Quick Start

```cpp
#include "MOM/MOM.hpp"   // umbrella header — pulls in all variants and utilities

// 1. Build a thermodynamic map (or pass your solver's thermo backend)
MOM::BasicThermoData thermo;
thermo.names = { "H", "OH", "O2", "H2", "H2O", "C2H2", "N2" };
thermo.mw    = { 1.008, 17.008, 31.999, 2.016, 18.015, 26.038, 28.014 };
// ... fill atom counts

// 2. Construct the variant — type selected at compile time
MOM::ThreeEquations<MOM::BasicThermoData> model(thermo);
model.SetPAH("C2H2");
model.SetNucleation(1);
model.SetCoagulation(1);
model.SetCondensation(1);
model.SetSurfaceGrowth(1);
model.SetOxidation(1);

static_assert(MOM::MomentMethod<decltype(model)>);   // verified at compile time

// 3. Per-cell evaluation (inside the CFD loop)
model.SetStatus(T, P_Pa, Y.data());             // T [K], P [Pa], Y mass fractions
model.SetMoments(moments_span);                  // current moment values
model.SetViscosity(mu);                          // mixture viscosity [kg/m/s]
model.CalculateSourceMoments();

// 4. Read sources — zero-copy spans into internal Eigen storage
auto src      = model.sources();                 // total,  size = 3
auto src_nucl = model.sources_nucleation();      // nucleation contribution
auto src_sint = model.sources_sintering();       // → kZeroData (not modelled)
```

---

## Integration Patterns

The library supports two integration patterns. Both produce identical machine code in the
inner loop; the choice depends on whether the variant is known at compile time.

### Pattern A — Compile-Time Variant (Recommended for Template Solvers)

The variant type is a template parameter of the CFD cell loop. The compiler generates one
fully-specialised loop body with every `if constexpr` branch resolved and the complete
dispatch chain inlined.

```cpp
template <MOM::MomentMethod M>
void CellLoop(M& model,
              int n_cells,
              const double* T_arr,
              const double* P_arr,
              const double* Y_flat,   // [n_cells × n_species]
              const double* mu_arr,
              const double* M_flat,   // [n_cells × M::n_equations]
              double* src_flat)       // [n_cells × M::n_equations]  output
{
    constexpr unsigned neq = M::n_equations;

    for (int i = 0; i < n_cells; ++i)
    {
        // Single entry-point: sets state, moments, viscosity, computes sources.
        MOM::ComputeCell(model,
                         T_arr[i], P_arr[i],
                         Y_flat + i * n_species,
                         mu_arr[i],
                         { M_flat + i * neq, neq });

        // Zero-copy read — no allocation, no copy
        auto src = model.sources();
        std::copy(src.begin(), src.end(), src_flat + i * neq);
    }
}

// Usage — one alias change to switch variant:
using ParticleModel = MOM::HMOM<MyThermo>;
CellLoop(model, N, T.data(), P.data(), Y.data(), mu.data(), M.data(), src.data());
```

At `-O3 -march=native`, the assembly for an HMOM cell contains no `call` instructions in
the hot path and uses `ymm` registers to process all four equations simultaneously.

### Pattern B — Runtime Variant Selection via `ForEachCell`

The variant is chosen at runtime from an input-file keyword. `std::visit` fires **once**
before the loop via `ForEachCell`; the lambda receives the concrete type and owns the
iteration. The compiler generates four specialised loop bodies (one per variant
alternative) and selects the correct one with a single branch before the loop.

```cpp
// Solver initialisation — one runtime branch at startup
MOM::AnyMomentMethod<MyThermo> model = MOM::MakeAnyMomentMethod(thermo, variant_name);
// variant_name = "HMOM" | "ThreeEquations" | "BrookesMoss" | "TiO2"

// CFD sweep — std::visit fires ONCE here, not inside the loop
MOM::ForEachCell(model, [&](auto& m)
{
    // typeof(m) is concrete here (e.g. ThreeEquations<MyThermo>&)
    constexpr unsigned neq = std::decay_t<decltype(m)>::n_equations;

    for (int i = 0; i < n_cells; ++i)
    {
        MOM::ComputeCell(m, T[i], P[i], Y + i * ns, mu[i], { M + i * neq, neq });

        auto src = m.sources();
        std::copy(src.begin(), src.end(), src_out + i * neq);
    }
});
```

Benchmark results on a representative 10 000-cell sweep (ThreeEquations, GCC 13,
`-O3 -march=native`, Intel Core i7-12700H):

```
Pattern A — template cell loop    :   X.XXX ms / iteration
Pattern B — ForEachCell loop      :   X.XXX ms / iteration
Overhead of variant dispatch      :   0.000 ms  (0.0%)
```

The overhead is immeasurable: `std::visit` is a single indirect branch that the branch
predictor makes free after the first correct prediction.

### Output and Diagnostics — `MomentMethodReporter`

The reporter is a **read-only observer**: it never modifies the model and has no I/O
dependency on the variant classes themselves (the variants carry no `WriteHeader` or
`WriteRow` methods). It is driven by the public `MomentMethod<M>` concept interface and by
optional `variant_prefix_output` / `variant_suffix_output` hooks detected via
`if constexpr`:

```cpp
// Once, before any loops — outside the CFD cell loop
MOM::OutputFileColumns file("soot_diagnostics.out");
MOM::MomentMethodReporter reporter(file, thermo.names);
reporter.WriteHeader(model);   // writes column headers to file
file.Complete();

// On output steps only — infrequently, never inside the hot cell loop
file.NewRow();
reporter.WriteRow(model);      // writes one diagnostic row; model is const&
```

Variant-specific extended output (e.g. the nine coagulation sub-mechanisms of HMOM) is
emitted automatically via the `variant_suffix_output` hook if the variant provides it,
without any modification to `MomentMethodReporter`.

---

## API Reference

All public API is expressed as requirements of the `MomentMethod<M>` concept.

### State Injection (call before each `CalculateSourceMoments`)

| Method | Description |
|---|---|
| `SetStatus(T, P, Y[])` | Gas temperature [K], pressure [Pa], mass fractions |
| `SetMoments(span)` | Current transported moment values (size = `n_equations`) |
| `SetViscosity(mu)` | Mixture dynamic viscosity [kg/m/s] |

### Computation

| Method | Description |
|---|---|
| `CalculateSourceMoments()` | Evaluates all moment source terms; `noexcept` |
| `CalculateOmegaGas()` | Evaluates gas-phase consumption terms only; `noexcept` |

### Source Output (zero-copy `std::span<const double>`, size = `n_equations`)

| Method | Description |
|---|---|
| `sources()` | Total source vector (sum of all processes) |
| `sources_nucleation()` | Nucleation contribution |
| `sources_coagulation()` | Coagulation contribution |
| `sources_condensation()` | Condensation (→ `kZeroData` if not modelled) |
| `sources_growth()` | Surface growth (→ `kZeroData` if not modelled) |
| `sources_oxidation()` | Oxidation (→ `kZeroData` if not modelled) |
| `sources_sintering()` | Sintering (→ `kZeroData` if not modelled) |
| `omega_gas()` | Gas-phase species source terms [kg/m³/s] |

### Particle Properties

| Method | Unit | Description |
|---|---|---|
| `volume_fraction()` | − | Particle volume fraction |
| `particle_diameter()` | m | Primary particle diameter |
| `collision_diameter()` | m | Collision (aggregate) diameter |
| `particle_number_density()` | #/m³ | Total number density |
| `mass_fraction()` | − | Particle mass fraction |
| `particle_density()` | kg/m³ | Material density |
| `specific_surface()` | m²/m³ | Surface area per unit volume |

### Transport and Radiation

| Method | Description |
|---|---|
| `schmidt_number()` | Particle Schmidt number |
| `diffusion_coefficient()` | Effective diffusion coefficient [kg/m/s] |
| `thermophoretic_model()` | Thermophoretic model flag (0 = off) |
| `planck_coefficient(T, fv)` | Planck mean absorption coefficient [1/m] |
| `radiative_heat_transfer()` | True if radiative loss is active |

Three Planck mean absorption coefficient correlations are available and selected at
construction time via `ProcessFlags::PlanckCoeffModel`:

| Model | Formula | Reference |
|---|---|---|
| Smooke | kP = 1232.4 · fv · T | Smooke et al. *Combust. Flame* **143** (2005) |
| Kent | kP = 1.3×10⁵ · fv · T | Kent & Honnery *Combust. Sci. Tech.* **75** (1990) |
| Sazhin | kP = fv · (630 + 0.63T − 10⁻⁴T²) | Sazhin *PECS* **20** (1994) |

---

## Building the Library

### Requirements

| Component | Minimum version | Notes |
|---|---|---|
| C++ compiler | GCC 11 / Clang 13 / MSVC 19.29 | C++20 required; C++23 auto-detected |
| CMake | 3.20 | For `FetchContent`, generator expressions |
| Eigen | 3.4.0 | Auto-fetched from GitLab if not found |

### CMake Integration

**Option 1 — `add_subdirectory` (recommended for in-tree development)**

```cmake
add_subdirectory(MOM)
target_link_libraries(MySolver PRIVATE MOM)
```

**Option 2 — `find_package` (after installation)**

```cmake
find_package(MOM REQUIRED)
target_link_libraries(MySolver PRIVATE MOM::MOM)
```

**Option 3 — Header-only interface target**

For solvers that instantiate MOM templates with custom thermo types, link against
`MOM_headers` instead of `MOM`. This gives access to all headers without the
pre-compiled `BasicThermoData` specialisations:

```cmake
target_link_libraries(MySolver PRIVATE MOM::MOM_headers)
```

### Build Options

| CMake option | Default | Description |
|---|---|---|
| `MOM_BUILD_SHARED` | OFF | Build as a shared library instead of static |
| `MOM_BUILD_TESTS` | ON | Build the test executables |
| `MOM_INSTALL` | ON | Generate install rules |
| `MOM_USE_DICTIONARY` | OFF | Enable input-file dictionary parser (requires `DICTIONARY_DIR`) |
| `EIGEN3_DIR` | — | Override Eigen path (skips `find_package`) |

### Typical Build Sequence

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

### Recommended Compiler Flags

```bash
# Minimum for correctness
-std=c++20

# Production CFD (recommended)
-std=c++20 -O3 -march=native

# Optional: remove exception tables from noexcept-only hot path
-fno-exceptions

# Optional: cross-translation-unit inlining (most relevant in pre-compiled mode)
-flto
```

What each flag enables for this library:

- **`-O2`**: inlines the complete `sources_X() → _impl()` dispatch chain. `[[gnu::always_inline]]` on `_impl()` methods makes this a hard guarantee.
- **`-O3`**: adds auto-vectorisation. For HMOM (`NEq=4`), the 4-double source vector maps to a single 256-bit AVX `ymm` register.
- **`-march=native`**: enables AVX2/FMA. Most relevant for Eigen's fixed-size vector arithmetic and the `std::copy` loops over source spans. Without this, the compiler defaults to SSE2 (128-bit, 2 doubles), halving throughput on 4-equation models.

---

## Code Style

All source files are formatted with **clang-format** using the project-provided
`.clang-format` configuration. Key settings:

| Rule | Value |
|---|---|
| Baseline style | LLVM |
| Brace placement | **Allman** (opening brace on its own line) |
| Indent width | 4 spaces (no tabs) |
| Column limit | 100 characters |
| Pointer alignment | Left (`double* ptr`) |
| `SortIncludes` | Never (mandatory ordering: STL → Eigen → project headers) |
| Minimum clang-format version | 14 |

Apply formatting to all sources:

```bash
find include src tests \( -name '*.hpp' -o -name '*.h' -o -name '*.tpp' -o -name '*.cpp' \) \
    | xargs clang-format -i
```

**Include ordering convention** (enforced manually, not by clang-format):

```cpp
// 1. STL headers — alphabetised
#include <array>
#include <span>
#include <string>

// 2. Third-party
#include "Eigen/Dense"

// 3. Project headers
#include "MOM/MomentMethodBase.hpp"
#include "MOM/ThermoProxy.hpp"
```

---

## Testing

Three test programs are provided in `tests/`:

### `test_verify_all_variants`

Constructs all four variants, sets representative flame-condition state, runs
`CalculateSourceMoments()`, and verifies that all source spans have the correct size and
no NaN/Inf values.

### `test_loop_benchmark`

Validates and benchmarks both CFD loop patterns:

- **Pattern A** — templated cell loop over ThreeEquations (compile-time dispatch)
- **Pattern B** — `ForEachCell` with `AnyMomentMethod` (runtime dispatch)

Also validates the zero-fallback mechanism: confirms that `sources_condensation()` on
BrookesMoss and `sources_sintering()` on ThreeEquations return all-zero spans of the
correct size.

```
=== Zero-fallback validation ===
  BrookesMoss::sources_condensation() → size=2  all_zero=true
  BrookesMoss::sources_sintering()    → size=2  all_zero=true
  ThreeEquations::sources_sintering() → size=3  all_zero=true
  Result: PASS ✓

=== Numerical equivalence check ===
  Pattern A checksum: X.XXXXXXe+XX
  Pattern B checksum: X.XXXXXXe+XX
  Bit-identical: true
```

### `reporter_usage_example`

Demonstrates the observer pattern: constructs all soot variants, runs a synthetic 10-cell
loop, and writes diagnostic output files using `MomentMethodReporter`. Verifies that the
reporter operates on any type satisfying `MomentMethod<M>` without variant-specific code.

---

## References

The physical models implemented in this library are based on the following publications.
Please cite the appropriate references when using this library in scientific work.

**HMOM**

- M.E. Mueller, G. Blanquart, H. Pitsch, "Hybrid Method of Moments for modelling soot
  formation and growth," *Combustion and Flame* **156** (2009) 1143–1155.
- A. Attili, F. Bisetti, "Statistics and scaling of turbulence in a spatially developing
  mixing layer at Re_λ = 250," *Physics of Fluids* **24** (2012) 035109.

**Three-Equation Soot Model**

- C. Eberle, P. Gerlinger, M. Aigner, "A sectional PAH model with reversible PAH chemistry
  for CFD soot simulations," *Combustion and Flame* **162** (2015) 2648–2660.

**Brookes–Moss Soot Model**

- S.J. Brookes, J.B. Moss, "Predictions of soot and thermal radiation properties in
  confined turbulent jet diffusion flames," *Combustion and Flame* **116** (1999) 486–503.

**TiO₂ Nanoparticle Model**

- S.E. Pratsinis, "Simultaneous nucleation, condensation, and coagulation in aerosol
  reactors," *Journal of Colloid and Interface Science* **124** (1988) 416–427.
- R. Jossen, S.E. Pratsinis, W.J. Stark, L. Mädler, "Criteria for flame-spray synthesis
  of hollow, shell-like, or inhomogeneous oxides," *Journal of the American Ceramic
  Society* **88** (2005) 1388–1393.

**Planck Mean Absorption Coefficients**

- M.D. Smooke et al., *Combustion and Flame* **143** (2005) 613–628.
- J.H. Kent, D. Honnery, *Combustion Science and Technology* **75** (1990) 167–177.
- S.S. Sazhin, "An approximation for the absorption of thermal radiation by soot
  particles," *Progress in Energy and Combustion Science* **20** (1994) 297–318.

---

## License

Copyright © 2026 Alberto Cuoci.

This library is distributed under the **GNU General Public License v3.0 or later**.  
See the [`LICENSE`](LICENSE) file for the full text.

```
OpenSMOKEpp MOM is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later version.
```

---

*CRECK Modeling Lab — Politecnico di Milano*  
*Department of Chemistry, Materials, and Chemical Engineering*  
*P.zza Leonardo da Vinci 32, 20133 Milano, Italy*  
*<https://www.creckmodeling.polimi.it>*
