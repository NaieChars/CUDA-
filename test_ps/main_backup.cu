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
    size_t m = 512, n = 512, k = 512;
    int warmup = 10;
    int iterations = 100;
    double atol = 1e-4, rtol = 1e-4;
    bool help = false, list = false, plan_only = false;
    bool check_only = false, profile = false;
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

size_t matrix_elements(size_t rows, size_t columns, size_t element_bytes)
{
    const size_t limit = std::numeric_limits<size_t>::max();
    if (rows == 0 || columns == 0 || rows > limit / columns)
        throw std::invalid_argument("matrix dimensions must be positive and fit size_t");
    const size_t count = rows * columns;
    if (count > limit / element_bytes)
        throw std::invalid_argument("matrix byte size does not fit size_t");
    return count;
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
        throw std::invalid_argument("expected: mode M N K");
    if (!positional.empty()) options.mode = parse_int(positional[0], "mode");
    gemm_variant(options.mode);
    if (positional.size() > 1) options.m = parse_unsigned(positional[1], "M");
    if (positional.size() > 2) options.n = parse_unsigned(positional[2], "N");
    if (positional.size() > 3) options.k = parse_unsigned(positional[3], "K");
    matrix_elements(options.m, options.k, sizeof(float));
    matrix_elements(options.k, options.n, sizeof(float));
    matrix_elements(options.m, options.n, sizeof(double));
    if (options.iterations < 1) throw std::invalid_argument("iterations must be positive");
    if (options.profile && options.check_only)
        throw std::invalid_argument("--profile and --check-only are mutually exclusive");
    return options;
}

void print_help()
{
    std::puts(
        "Usage: sgemm [mode=0] [M=512] [N=512] [K=512]\n"
        "  Contiguous row-major float matrices: C[M,N] = A[M,K] * B[K,N]\n"
        "  mode 0: naive, one output per thread\n"
        "  mode 1: shared-memory tiling, one output per thread\n"
        "  mode 2: register tiling, multiple outputs per thread\n"
        "Options:\n"
        "  --list          List implementation status and tile contracts (no GPU)\n"
        "  --plan          Print grid/block/scratch and FLOP count (no GPU)\n"
        "  --check-only    One GPU launch, full CPU-double validation, no timing\n"
        "  --profile       Validate/warm up; expose one GEMM kernel to ncu\n"
        "  --warmup N      Warmup launches; default 10\n"
        "  --iterations N  Timed launches; default 100\n"
        "  --atol X        Absolute tolerance; default 1e-4\n"
        "  --rtol X        Relative tolerance; default 1e-4\n"
        "  --help          Show help\n"
        "  Full CPU validation costs O(M*N*K); start with small shapes.\n"
        "Exit codes: 0=success, 1=CUDA error, 2=invalid/setup error,\n"
        "            3=kernel TODO, 4=validation failed.");
}

bool validate(const std::vector<float>& actual, const std::vector<double>& reference,
              const Options& options, const char* phase)
{
    double max_absolute = 0, max_scaled = 0;
    size_t failed = 0, first_bad = 0;
    for (size_t i = 0; i < reference.size(); ++i)
    {
        const double error = std::isfinite(actual[i])
            ? std::abs(static_cast<double>(actual[i]) - reference[i])
            : std::numeric_limits<double>::infinity();
        const double allowed = options.atol + options.rtol * std::abs(reference[i]);
        const double scaled = !std::isfinite(error) ? std::numeric_limits<double>::infinity() :
            (allowed > 0 ? error / allowed :
             (error == 0 ? 0 : std::numeric_limits<double>::infinity()));
        max_absolute = std::max(max_absolute, error);
        max_scaled = std::max(max_scaled, scaled);
        if (!std::isfinite(actual[i]) || error > allowed)
        {
            if (failed == 0) first_bad = i;
            ++failed;
        }
    }
    std::printf("Check | phase=%s | status=%s | checked=full | elements=%zu | "
                "failed=%zu | max_abs_error=%.9g | max_scaled_error=%.9g\n",
                phase, failed == 0 ? "PASS" : "FAIL", reference.size(),
                failed, max_absolute, max_scaled);
    if (failed != 0)
        std::printf("Mismatch | row=%zu | col=%zu | expected=%.12g | actual=%.12g\n",
                    first_bad / options.n, first_bad % options.n,
                    reference[first_bad], static_cast<double>(actual[first_bad]));
    return failed == 0;
}

int run(const Options& options)
{
    if (options.help) { print_help(); return 0; }
    if (options.list)
    {
        for (const auto& v : gemm_variants())
            std::printf("Variant | mode=%d | name=%s | kernel=%s | implemented=%d | "
                        "block_x=%d | block_y=%d | tile_m=%d | tile_n=%d | tile_k=%d | "
                        "thread_m=%d | thread_n=%d | shared_bytes=%zu\n",
                        v.mode, v.name, v.kernel_name, v.implemented ? 1 : 0,
                        v.block_x, v.block_y, v.tile_m, v.tile_n, v.tile_k,
                        v.thread_m, v.thread_n, v.shared_bytes);
        return 0;
    }
    const auto& variant = gemm_variant(options.mode);
    const auto launch = make_gemm_launch(options.mode, options.m, options.n);
    const double flops = 2.0 * static_cast<double>(options.m)
                       * static_cast<double>(options.n) * static_cast<double>(options.k);
    if (options.plan_only)
    {
        std::printf("Plan | mode=%d | name=%s | M=%zu | N=%zu | K=%zu | "
                    "grid_x=%u | grid_y=%u | block_x=%u | block_y=%u | "
                    "tile_m=%d | tile_n=%d | tile_k=%d | thread_m=%d | thread_n=%d | "
                    "shared_bytes=%zu | flops=%.0f\n",
                    options.mode, variant.name, options.m, options.n, options.k,
                    launch.grid.x, launch.grid.y, launch.block.x, launch.block.y,
                    variant.tile_m, variant.tile_n, variant.tile_k,
                    variant.thread_m, variant.thread_n, launch.shared_bytes, flops);
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
    if (launch.block.x > static_cast<unsigned int>(properties.maxThreadsDim[0]) ||
        launch.block.y > static_cast<unsigned int>(properties.maxThreadsDim[1]) ||
        static_cast<size_t>(launch.block.x) * launch.block.y > static_cast<size_t>(properties.maxThreadsPerBlock) ||
        launch.grid.x > static_cast<unsigned int>(properties.maxGridSize[0]) ||
        launch.grid.y > static_cast<unsigned int>(properties.maxGridSize[1]) ||
        launch.shared_bytes > properties.sharedMemPerBlock)
        throw std::invalid_argument("launch dimensions or shared memory exceed device limits");

    const size_t count_a = matrix_elements(options.m, options.k, sizeof(float));
    const size_t count_b = matrix_elements(options.k, options.n, sizeof(float));
    const size_t count_c = matrix_elements(options.m, options.n, sizeof(float));
    std::vector<float> a(count_a), b(count_b), actual(count_c);
    std::vector<double> reference(count_c, 0.0);
    // Distinct, deterministic, mixed-sign, bounded binary fractions.
    for (size_t i = 0; i < count_a; ++i)
        a[i] = static_cast<float>(static_cast<int>((i * 17 + 3) % 29) - 14) * 0.03125f;
    for (size_t i = 0; i < count_b; ++i)
        b[i] = static_cast<float>(static_cast<int>((i * 11 + 5) % 31) - 15) * 0.03125f;
    std::printf("Reference | status=computing | M=%zu | N=%zu | K=%zu | checked=full\n",
                options.m, options.n, options.k);
    std::fflush(stdout);
    // CPU oracle, outside timing; contiguous inner loop avoids a strided B read.
    for (size_t row = 0; row < options.m; ++row)
        for (size_t inner = 0; inner < options.k; ++inner)
        {
            const double value = static_cast<double>(a[row * options.k + inner]);
            for (size_t col = 0; col < options.n; ++col)
                reference[row * options.n + col] +=
                    value * static_cast<double>(b[inner * options.n + col]);
        }

    CudaBuffer<float> device_a(count_a), device_b(count_b), device_c(count_c);
    CUDA_CHECK(cudaMemcpy(device_a.get(), a.data(), count_a * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_b.get(), b.data(), count_b * sizeof(float), cudaMemcpyHostToDevice));
    // An unwritten C element must fail even when its reference happens to be zero.
    CUDA_CHECK(cudaMemset(device_c.get(), 0xff, count_c * sizeof(float)));
    auto gemm = [&]() {
        launch_gemm(options.mode, device_a.get(), device_b.get(), device_c.get(),
                    options.m, options.n, options.k, launch);
    };
    auto check_result = [&](const char* phase) {
        CUDA_CHECK(cudaMemcpy(actual.data(), device_c.get(), count_c * sizeof(float), cudaMemcpyDeviceToHost));
        return validate(actual, reference, options, phase);
    };
    std::printf("Config | mode=%d | name=%s | kernel=%s | M=%zu | N=%zu | K=%zu | "
                "grid_x=%u | grid_y=%u | block_x=%u | block_y=%u | shared_bytes=%zu | "
                "warmup=%d | iterations=%d | gpu=%s\n",
                options.mode, variant.name, variant.kernel_name, options.m, options.n, options.k,
                launch.grid.x, launch.grid.y, launch.block.x, launch.block.y, launch.shared_bytes,
                options.warmup, options.iterations, properties.name);
    gemm();
    CUDA_CHECK(cudaDeviceSynchronize());
    if (!check_result("before")) return kValidationFailed;
    if (options.check_only) return 0;

    if (options.profile)
    {
        for (int i = 0; i < options.warmup; ++i) gemm();
        CUDA_CHECK(cudaDeviceSynchronize());
        std::puts("Profile | stage=kernel");
        CUDA_CHECK(cudaProfilerStart());
        gemm();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaProfilerStop());
        if (!check_result("after")) return kValidationFailed;
        std::puts("ProfileComplete | stage=kernel");
        return 0;
    }

    CudaTimer timer;
    const float ms = timer.benchmark(gemm, options.warmup, options.iterations);
    if (!check_result("after")) return kValidationFailed;
    if (!std::isfinite(ms) || ms <= 0)
        throw std::runtime_error("invalid event duration; increase iterations");
    const double gflops = flops / (static_cast<double>(ms) * 1e6);
    std::printf("GEMM | mode=%d | name=%s | M=%zu | N=%zu | K=%zu | "
                "time_ms=%.8f | gflops=%.6f | check=PASS\n",
                options.mode, variant.name, options.m, options.n, options.k,
                static_cast<double>(ms), gflops);
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
