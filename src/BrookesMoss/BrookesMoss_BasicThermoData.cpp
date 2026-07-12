/**
 * @file BrookesMoss_BasicThermoData.cpp
 * @brief Explicit instantiation of `BrookesMoss<BasicThermoData>`.
 */

#if defined(MOM_COMPILED_LIBRARY)
#error "Do not define MOM_COMPILED_LIBRARY when compiling library sources"
#endif

#include "BrookesMoss/BrookesMoss.hpp"

namespace MOM
{
template class BrookesMoss<BasicThermoData>;
}
