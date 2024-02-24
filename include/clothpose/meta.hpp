#pragma once
#include <functional>
#include <tuple>
#include <type_traits>
namespace cp
{
template <class T>
struct wrapt
{
    using type = T;
};

// reference:
// https://codereview.stackexchange.com/questions/136770/hashing-a-tuple-in-c17
template <class T>
std::size_t hashBuiltin(const T& value)
{
    return std::hash<T>{}(value);
}

template <class... Args>
struct TupleHash
{
    using value_type = std::tuple<Args...>;

    template <std::size_t... Indices>
    std::size_t get_tuple_hash_from_indices(
        const value_type& tuple, const std::index_sequence<Indices...>&) const
    {
        std::size_t result{0};
        for (const auto& hashValue : {hashBuiltin(std::get<Indices>(tuple))...})
            result ^= hashValue + 0x9e3779b9 + (result << 6) + (result >> 2);
        return result;
    }

    std::size_t operator()(const value_type& value) const
    {
        return get_tuple_hash_from_indices(
            value, std::make_index_sequence<sizeof...(Args)>{});
    }
};
}  // namespace cp