/**
 * @file MetalOxide_BasicThermoData.cpp
 * @brief Explicit instantiation of `MetalOxide<BasicThermoData>`.
 */

#if defined(MOM_COMPILED_LIBRARY)
#error "Do not define MOM_COMPILED_LIBRARY when compiling library sources"
#endif

#include "MetalOxide/MetalOxide.hpp"

namespace MOM
{
template class MetalOxide<BasicThermoData>;
}
