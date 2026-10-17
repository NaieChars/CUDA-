#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include "cuda_buffer.cuh"
#include "cuda_check.cuh"
#include "cuda_timer.cuh"
#include "kernels.cuh"

namespace
{
constexpr int kInvalidArguments = 2;
constexpr int kNotImplemented = 3;
constexpr int kValidationFailed = 4;

struct Options
{
    int mode = 0;
    size_t n = 1u << 24;
    int block = 256;
    int items = 1;
    int warmup = 10;
    int iterations = 100;
    double atol = 1e-3;
    double rtol = 1e-6;
    bool list = false;
    bool help = false;
    bool plan_only = false;
    bool check_only = false;
    bool profile = false;
};

struct Pass
{
    size_t n;
    unsigned int blocks;
};

size_t parse_unsigned(const std::string& text, const char* name)
{
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument(std::string(name) + " must be an unsigned integer");
    const auto value = std::stoull(text);
    if (value > std::numeric_limits<size_t>::max())
        throw std::invalid_argument(std::string(name) + " is too large");
    return static_cast<size_t>(value);
}

int parse_int(const std::string& text, const char* name)
{
    const size_t value = parse_unsigned(text, name);
    if (value > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(std::string(name) + " is too large");
    return static_cast<int>(value);
}

double parse_tolerance(const std::string& text)
{
    size_t used = 0;
    const double value = std::stod(text, &used);
    if (used != text.size() || !std::isfinite(value) || value < 0)
        throw std::invalid_argument("tolerances must be finite and nonnegative");
    return value;
}

Options parse_options(int argc, char** argv)
{
    Options options;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") options.help = true;
        else if (arg == "--list") options.list = true;
        else if (arg == "--plan") options.plan_only = true;
        else if (arg == "--check-only") options.check_only = true;
        else if (arg == "--profile") options.profile = true;
        else if (arg == "--warmup" || arg == "--iterations" ||
                 arg == "--atol" || arg == "--rtol")
        {
            if (++i >= argc) throw std::invalid_argument(arg + " needs a value");
            if (arg == "--warmup") options.warmup = parse_int(argv[i], "warmup");
            else if (arg == "--iterations") options.iterations = parse_int(argv[i], "iterations");
            else if (arg == "--atol") options.atol = parse_tolerance(argv[i]);
            else options.rtol = parse_tolerance(argv[i]);
        }
        else if (arg.rfind("--", 0) == 0)
            throw std::invalid_argument("unknown option: " + arg);
        else positional.push_back(arg);
    }
    if (positional.size() > 4)
        throw std::invalid_argument("expected: mode n block items_per_thread");
    if (!positional.empty()) options.mode = parse_int(positional[0], "mode");
    const auto& variant = reduction_variant(options.mode);
    options.items = variant.multiple_items ? 4 : 1;
    if (positional.size() >= 2) options.n = parse_unsigned(positional[1], "n");
    if (positional.size() >= 3) options.block = parse_int(positional[2], "block");
    if (positional.size() >= 4) options.items = parse_int(positional[3], "items_per_thread");
    if (options.n == 0 || options.n > std::numeric_limits<size_t>::max() / sizeof(float))
        throw std::invalid_argument("n must be positive and its byte size must fit size_t");
    if (options.block < 32 || options.block > 1024 ||
        (options.block & (options.block - 1)) != 0)
        throw std::invalid_argument("block must be a power of two in [32, 1024]");
    if (options.items < 1 || options.items > 64)
        throw std::invalid_argument("items_per_thread must be in [1, 64]");
    if (!variant.multiple_items && options.items != 1)
        throw std::invalid_argument("mode 0 requires items_per_thread=1");
    if (options.iterations < 1)
        throw std::invalid_argument("iterations must be positive");
    if (options.profile && options.check_only)
        throw std::invalid_argument("--profile and --check-only are mutually exclusive");
    return options;
}

void print_help()
{
    std::puts(
        "Usage: reduction [mode=0] [n=16777216] [block=256] [items_per_thread]\n"
        "  mode 0: shared-memory baseline; items must be 1\n"
        "  mode 1: multi-item reduction; default items=4\n"
        "  mode 2: warp-shuffle reduction; default items=4\n"
        "Options:\n"
        "  --list          List algorithms and TODO/implemented status (no GPU needed)\n"
        "  --plan          Print host-side pass sizes without launching kernels\n"
        "  --check-only    Validate one complete reduction without benchmarking\n"
        "  --profile       Validate, warm up, then expose only the first pass to ncu\n"
        "  --warmup N      Complete warmup reductions; default 10\n"
        "  --iterations N  Complete timed reductions; default 100\n"
        "  --atol X        Absolute validation tolerance; default 1e-3\n"
        "  --rtol X        Relative validation tolerance; default 1e-6\n"
        "  --help          Show this help\n"
        "Exit codes: 0=success, 1=CUDA error, 2=invalid/setup error,\n"
        "            3=kernel TODO, 4=validation failed.");
}

std::vector<Pass> make_plan(const Options& options)
{
    std::vector<Pass> plan;
    const size_t chunk = static_cast<size_t>(options.block) * options.items;
    size_t current = options.n;
    // Even n=1 gets one pass, so the same output contract is exercised.
    do
    {
        const size_t blocks = current / chunk + (current % chunk != 0);
        if (blocks > std::numeric_limits<unsigned int>::max())
            throw std::invalid_argument("grid is too large; increase items or reduce n");
        plan.push_back({current, static_cast<unsigned int>(blocks)});
        current = blocks;
    } while (current > 1);
    return plan;
}

bool validate(float actual, double reference, const Options& options, const char* phase)
{
    const double error = std::abs(static_cast<double>(actual) - reference);
    const double allowed = options.atol + options.rtol * std::abs(reference);
    const bool passed = std::isfinite(actual) && error <= allowed;
    std::printf("Check | phase=%s | status=%s | expected=%.12g | actual=%.12g | "
                "abs_error=%.9g | allowed_error=%.9g\n",
                phase, passed ? "PASS" : "FAIL", reference,
                static_cast<double>(actual), error, allowed);
    return passed;
}

int run(const Options& options)
{
    if (options.help) { print_help(); return 0; }
    if (options.list)
    {
        for (const auto& variant : reduction_variants())
            std::printf("Variant | mode=%d | name=%s | kernel=%s | implemented=%d | multiple_items=%d\n",
                        variant.mode, variant.name, variant.kernel_name,
                        variant.implemented ? 1 : 0, variant.multiple_items ? 1 : 0);
        return 0;
    }
    const auto& variant = reduction_variant(options.mode);
    const auto plan = make_plan(options);
    if (options.plan_only)
    {
        std::printf("Plan | mode=%d | name=%s | n=%zu | block=%d | items=%d | passes=%zu\n",
                    options.mode, variant.name, options.n, options.block, options.items, plan.size());
        for (size_t i = 0; i < plan.size(); ++i)
            std::printf("Pass | index=%zu | input_n=%zu | blocks=%u | output_n=%u\n",
                        i, plan[i].n, plan[i].blocks, plan[i].blocks);
        return 0;
    }
    if (!variant.implemented)
    {
        std::fprintf(stderr, "TODO | mode=%d | kernel=%s | implement kernels.cu and enable its flag first.\n",
                     options.mode, variant.kernel_name);
        return kNotImplemented;
    }

    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (options.block > properties.maxThreadsPerBlock)
        throw std::invalid_argument("block exceeds this device's thread limit");
    if (plan.front().blocks > static_cast<unsigned int>(properties.maxGridSize[0]))
        throw std::invalid_argument("first-pass grid exceeds this device's grid limit");

    std::vector<float> input(options.n);
    double reference = 0.0;
    for (size_t i = 0; i < options.n; ++i)
    {
        // Deterministic, nontrivial mixed-sign data. Binary fractions avoid input
        // conversion noise; double reference and tolerances handle sum order.
        input[i] = 0.25f + static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
        reference += static_cast<double>(input[i]);
    }
    CudaBuffer<float> device_input(options.n);
    CudaBuffer<float> scratch_a(plan.front().blocks);
    CudaBuffer<float> scratch_b(plan.front().blocks);
    CUDA_CHECK(cudaMemcpy(device_input.get(), input.data(), options.n * sizeof(float),
                          cudaMemcpyHostToDevice));
    // NaN sentinels make an unwritten output fail validation, even if the
    // expected sum happens to be zero. Poisoning is outside the timed region.
    CUDA_CHECK(cudaMemset(scratch_a.get(), 0xff, scratch_a.size() * sizeof(float)));
    CUDA_CHECK(cudaMemset(scratch_b.get(), 0xff, scratch_b.size() * sizeof(float)));

    auto complete_reduction = [&]() {
        const float* source = device_input.get();
        for (size_t i = 0; i < plan.size(); ++i)
        {
            float* destination = (i % 2 == 0) ? scratch_a.get() : scratch_b.get();
            launch_reduction_pass(options.mode, source, destination, plan[i].n,
                                  plan[i].blocks, options.block, options.items);
            source = destination;
        }
    };
    auto read_result = [&]() {
        float result = 0;
        const float* final = (plan.size() % 2 == 1) ? scratch_a.get() : scratch_b.get();
        CUDA_CHECK(cudaMemcpy(&result, final, sizeof(float), cudaMemcpyDeviceToHost));
        return result;
    };

    std::printf("Config | mode=%d | name=%s | kernel=%s | n=%zu | block=%d | items=%d | "
                "passes=%zu | warmup=%d | iterations=%d | gpu=%s\n",
                options.mode, variant.name, variant.kernel_name, options.n, options.block,
                options.items, plan.size(), options.warmup, options.iterations, properties.name);
    complete_reduction();
    CUDA_CHECK(cudaDeviceSynchronize());
    if (!validate(read_result(), reference, options, "before")) return kValidationFailed;
    if (options.check_only) return 0;

    if (options.profile)
    {
        for (int i = 0; i < options.warmup; ++i) complete_reduction();
        CUDA_CHECK(cudaDeviceSynchronize());
        std::printf("Profile | stage=first_pass | input_n=%zu | blocks=%u | full_passes=%zu\n",
                    options.n, plan.front().blocks, plan.size());
        // ncu --profile-from-start off: validation/warmup launches stay outside
        // the capture range. Only the first large-array pass is profiled.
        CUDA_CHECK(cudaProfilerStart());
        launch_reduction_pass(options.mode, device_input.get(), scratch_a.get(),
                              options.n, plan.front().blocks, options.block, options.items);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaProfilerStop());
        std::puts("ProfileComplete | stage=first_pass");
        return 0;
    }

    CudaTimer timer;
    const float ms = timer.benchmark(complete_reduction, options.warmup, options.iterations);
    if (!validate(read_result(), reference, options, "after")) return kValidationFailed;
    if (!std::isfinite(ms) || ms <= 0)
        throw std::runtime_error("invalid event duration; increase iterations");
    const double input_bytes = static_cast<double>(options.n) * sizeof(float);
    const double bandwidth = input_bytes / (static_cast<double>(ms) * 1e6);
    std::printf("Reduction | mode=%d | name=%s | n=%zu | block=%d | items=%d | passes=%zu | "
                "time_ms=%.8f | input_bandwidth_GBs=%.6f | check=PASS\n",
                options.mode, variant.name, options.n, options.block, options.items,
                plan.size(), static_cast<double>(ms), bandwidth);
    return 0;
}
}

int main(int argc, char** argv)
{
    try { return run(parse_options(argc, argv)); }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "Error: %s\nUse --help for usage.\n", error.what());
        return kInvalidArguments;
    }
}
