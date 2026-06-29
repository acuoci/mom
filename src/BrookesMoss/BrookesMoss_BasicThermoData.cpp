/*-----------------------------------------------------------------------*\
|   MOM Library — BrookesMoss<BasicThermoData> explicit instantiation     |
|   Compiled unconditionally. Requires no external dependencies.          |
\*-----------------------------------------------------------------------*/

#if defined(MOM_COMPILED_LIBRARY)
#error "Do not define MOM_COMPILED_LIBRARY when compiling library sources"
#endif

#include "BrookesMoss/BrookesMoss.hpp"

namespace MOM
{
template class BrookesMoss<BasicThermoData>;
}

