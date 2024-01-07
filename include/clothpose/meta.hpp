#include <type_traits>

namespace cp
{
template <class T>
struct wrapt
{
    using type = T;
};
}  // namespace cp