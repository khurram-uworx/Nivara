// SYCL/DPC++ probe runner for the Nivara GPU kernel probe (commit 5 sandbox).
//
// Compile (oneAPI Base Toolkit + VS Build Tools, see build.cmd):
//   icpx -fsycl -O2 sycl_runner.cpp -o sycl_runner.exe
//
// Usage:
//   sycl_runner.exe <mode> <in.bin> <out.bin> [dim0] [dim1]
//     mode  dot16 : in = 32 x u16 (a[0..15] then b[0..15])    out = 1 x f32   (dims implied 16/16)
//     mode  gemv  : in = (dim0*dim1 + dim1) x u16: W row-major [dim0 x dim1], then x[dim1]
//                   out = dim0 x f32
//     mode  silu  : in = dim0 x u16                          out = dim0 x f32
//
// BF16 data travels as raw u16 (System.Numerics.BFloat16 bit pattern, same layout as the
// .NET probe's fixtures); the device widens in-register via shift-left-16 => f32 (the same
// emul-widen the L0 leg uses — no SPV_INTEL_bfloat16 dependency).
//
// Exit codes: 0 = success, 1 = device/queue/kernel failure, 2 = bad args / I/O.
#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

inline float WidenBf16(uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, sizeof(u));
    return f;
}

int Fail(const char* msg, int code) {
    std::fprintf(stderr, "sycl_runner: %s\n", msg);
    return code;
}

std::vector<uint16_t> ReadU16s(const char* path) {
    std::ifstream in(path, std::ios::binary);
    std::vector<uint16_t> data;
    if (!in)
        return data;
    in.seekg(0, std::ios::end);
    std::streamoff len = in.tellg();
    in.seekg(0, std::ios::beg);
    data.resize(static_cast<size_t>(len) / sizeof(uint16_t));
    if (!data.empty())
        in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(uint16_t)));
    return data;
}

int RunDot16(sycl::queue& q, const std::vector<uint16_t>& input, std::vector<float>& out) {
    if (input.size() < 32)
        return Fail("dot16: need 32 u16 (a[16], b[16])", 2);

    auto a = sycl::malloc_shared<uint16_t>(16, q);
    auto b = sycl::malloc_shared<uint16_t>(16, q);
    auto c = sycl::malloc_shared<float>(1, q);
    if (!a || !b || !c)
        return Fail("dot16: device allocation failed", 1);

    std::memcpy(a, input.data(), 16 * sizeof(uint16_t));
    std::memcpy(b, input.data() + 16, 16 * sizeof(uint16_t));

    int rc = 0;
    try {
        q.submit([&](sycl::handler& h) {
            h.single_task([=] {
                float acc = 0.0f;
                for (int i = 0; i < 16; ++i)
                    acc += WidenBf16(a[i]) * WidenBf16(b[i]);
                c[0] = acc;
            });
        });
        q.wait();
        out.assign(1, c[0]);
        std::fprintf(stdout, "  dot16 = %.9g\n", c[0]);
    } catch (const sycl::exception& e) {
        rc = Fail(("dot16 kernel failed: " + std::string(e.what())).c_str(), 1);
    }

    sycl::free(a, q); sycl::free(b, q); sycl::free(c, q);
    return rc;
}

int RunGemv(sycl::queue& q, const std::vector<uint16_t>& input, int rows, int cols, std::vector<float>& out) {
    if (cols <= 0 || rows <= 0)
        return Fail("gemv: bad dims", 2);
    const size_t need = static_cast<size_t>(rows) * cols + cols;
    if (input.size() < need)
        return Fail("gemv: input too small (need W[rows x cols] + x[cols])", 2);

    auto w = sycl::malloc_shared<uint16_t>(static_cast<size_t>(rows) * cols, q);
    auto x = sycl::malloc_shared<uint16_t>(static_cast<size_t>(cols), q);
    auto y = sycl::malloc_shared<float>(static_cast<size_t>(rows), q);
    if (!w || !x || !y)
        return Fail("gemv: device allocation failed", 1);

    std::memcpy(w, input.data(), static_cast<size_t>(rows) * cols * sizeof(uint16_t));
    std::memcpy(x, input.data() + static_cast<size_t>(rows) * cols, static_cast<size_t>(cols) * sizeof(uint16_t));

    int rc = 0;
    try {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(static_cast<size_t>(rows)), [=](sycl::id<1> r) {
                float acc = 0.0f;
                for (int k = 0; k < cols; ++k)
                    acc += WidenBf16(w[r * cols + k]) * WidenBf16(x[k]);
                y[r] = acc;
            });
        });
        q.wait();
        out.assign(y, y + rows);
    } catch (const sycl::exception& e) {
        rc = Fail(("gemv kernel failed: " + std::string(e.what())).c_str(), 1);
    }

    sycl::free(w, q); sycl::free(x, q); sycl::free(y, q);
    return rc;
}

int RunSilu(sycl::queue& q, const std::vector<uint16_t>& input, int n, std::vector<float>& out) {
    if (n <= 0)
        return Fail("silu: bad dims", 2);
    const size_t need = static_cast<size_t>(n);
    if (input.size() < need)
        return Fail("silu: input too small", 2);

    auto in = sycl::malloc_shared<uint16_t>(need, q);
    auto result = sycl::malloc_shared<float>(need, q);
    if (!in || !result)
        return Fail("silu: device allocation failed", 1);

    std::memcpy(in, input.data(), need * sizeof(uint16_t));

    int rc = 0;
    try {
        q.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(need), [=](sycl::id<1> i) {
                float xv = WidenBf16(in[i]);
                result[i] = xv / (1.0f + sycl::exp(-xv));
            });
        });
        q.wait();
        out.assign(result, result + need);
    } catch (const sycl::exception& e) {
        rc = Fail(("silu kernel failed: " + std::string(e.what())).c_str(), 1);
    }

    sycl::free(in, q); sycl::free(result, q);
    return rc;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: sycl_runner <dot16|gemv|silu> <in.bin> <out.bin> [dim0] [dim1]\n");
        return 2;
    }

    std::string mode = argv[1];
    const char* inPath = argv[2];
    const char* outPath = argv[3];
    int dim0 = argc > 4 ? std::atoi(argv[4]) : 0;
    int dim1 = argc > 5 ? std::atoi(argv[5]) : 0;

    auto input = ReadU16s(inPath);
    if (input.empty())
        return Fail("cannot read input file", 2);

    sycl::queue q;
    try {
        q = sycl::queue(sycl::gpu_selector_v);
        std::fprintf(stdout, "device: %s (%s)\n",
                     q.get_device().get_info<sycl::info::device::name>().c_str(),
                     q.get_device().get_info<sycl::info::device::driver_version>().c_str());
    } catch (const sycl::exception& e) {
        return Fail(("no usable GPU device: " + std::string(e.what())).c_str(), 1);
    }

    std::vector<float> out;
    int rc = 0;
    if (mode == "dot16") {
        rc = RunDot16(q, input, out);
    } else if (mode == "gemv") {
        rc = RunGemv(q, input, dim0, dim1, out);
    } else if (mode == "silu") {
        rc = RunSilu(q, input, dim0, out);
    } else {
        return Fail("unknown mode (dot16|gemv|silu)", 2);
    }

    if (rc != 0)
        return rc;

    std::ofstream ofile(outPath, std::ios::binary);
    if (!ofile)
        return Fail("cannot write output file", 2);
    ofile.write(reinterpret_cast<const char*>(out.data()),
                static_cast<std::streamsize>(out.size() * sizeof(float)));
    if (!ofile.good())
        return Fail("output write failed", 2);
    return 0;
}