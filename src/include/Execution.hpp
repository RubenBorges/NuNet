#pragma once

#include <algorithm>
#include <cstddef>
#include <future>
#include <functional>
#include <type_traits>
#include <utility>

#if __has_include(<experimental/simd>)
#include <experimental/simd>
#define NUNET_HAS_SIMD 1
#endif

namespace bpy {

enum class ExecutionPolicy {
    CPU,
    ParallelCPU,
    SYCLGPU
};

namespace detail {

inline void simd_relu(
    const double* input,
    double* output,
    std::size_t size)
{
#if defined(NUNET_HAS_SIMD)
    namespace sx = std::experimental;
    using Simd = sx::native_simd<double>;
    const Simd zero{0.0};
    std::size_t index = 0;

    for (; index + Simd::size() <= size; index += Simd::size())
    {
        Simd values{input + index, sx::element_aligned};
        sx::where(!(zero < values), values) = zero;
        values.copy_to(output + index, sx::element_aligned);
    }

    for (; index < size; ++index)
        output[index] = std::max(0.0, input[index]);
#else
    for (std::size_t index = 0; index < size; ++index)
        output[index] = std::max(0.0, input[index]);
#endif
}

inline void simd_multiply_add(
    double* output,
    const double* right,
    double left,
    std::size_t size)
{
#if defined(NUNET_HAS_SIMD)
    namespace sx = std::experimental;
    using Simd = sx::native_simd<double>;
    const Simd left_values{left};
    std::size_t index = 0;

    for (; index + Simd::size() <= size; index += Simd::size())
    {
        Simd output_values{output + index, sx::element_aligned};
        const Simd right_values{right + index, sx::element_aligned};
        output_values += left_values * right_values;
        output_values.copy_to(output + index, sx::element_aligned);
    }

    for (; index < size; ++index)
        output[index] += left * right[index];
#else
    for (std::size_t index = 0; index < size; ++index)
        output[index] += left * right[index];
#endif
}

} // namespace detail

template <class T>
class Sender {
public:
    using value_type = T;

    explicit Sender(std::future<T> future)
        : future_{std::move(future)}
    {
    }

    static Sender just(T value)
    {
        std::promise<T> promise;
        auto future = promise.get_future();
        promise.set_value(std::move(value));
        return Sender{std::move(future)};
    }

    template <class Function>
    static auto submit(Function&& function)
    {
        using Result = std::invoke_result_t<std::decay_t<Function>>;
        return Sender<Result>{
            std::async(
                std::launch::async,
                std::forward<Function>(function))
        };
    }

    template <class Function>
    auto then(Function&& function) &&
    {
        using StoredFunction = std::decay_t<Function>;
        using Result = std::invoke_result_t<StoredFunction, T>;
        auto source = std::move(future_);

        return Sender<Result>::submit(
            [source = std::move(source),
             function = StoredFunction{std::forward<Function>(function)}]() mutable {
                return std::invoke(function, source.get());
            });
    }

    template <class Function>
    auto let_value(Function&& function) &&
    {
        using StoredFunction = std::decay_t<Function>;
        using NextSender = std::invoke_result_t<StoredFunction, T>;
        using Result = typename NextSender::value_type;
        auto source = std::move(future_);

        return Sender<Result>::submit(
            [source = std::move(source),
             function = StoredFunction{std::forward<Function>(function)}]() mutable {
                auto next = std::invoke(function, source.get());
                return next.get();
            });
    }

    T get()
    {
        return future_.get();
    }

private:
    std::future<T> future_;
};

} // namespace bpy
