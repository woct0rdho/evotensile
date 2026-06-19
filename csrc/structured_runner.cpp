// EvoTensile structured runner for the current gfx1151 FP16 NT HHS target.
//
// The Python scheduler writes exact runnable pairs as JSONL. This binary loads
// the TensileLite-generated library/code objects, runs exactly the requested
// solution index for each pair, validates numerics, and writes JSONL samples
// carrying shape_id/candidate_hash identity on every row.

#include <Tensile/Activation.hpp>
#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/KernelLanguageTypes.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/PerformanceMetricTypes.hpp>
#include <Tensile/Task.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/Utils.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/hip/HipUtils.hpp>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;

namespace
{
    using TensileLite::ContractionInputs;
    using TensileLite::ContractionProblemGemm;
    using TensileLite::ContractionSolution;
    using TensileLite::Half;

    constexpr size_t DEFAULT_WORKSPACE_BYTES = 128ull * 1024ull * 1024ull;
    constexpr size_t SYNCHRONIZER_ELEMENTS   = 409600;

    struct Args
    {
        std::string pairs;
        std::string output;
        std::string libraryDir;
        std::string libraryFile;
        std::string codeObject;
        int         device        = 0;
        bool        useUserArgs    = false;
        bool        validateOnly   = false;
        int         primeEnqueues  = 0;
        size_t      workspaceSize  = DEFAULT_WORKSPACE_BYTES;
    };

    struct Pair
    {
        std::string shapeId;
        std::string candidateHash;
        int64_t     m                      = 0;
        int64_t     n                      = 0;
        int64_t     batch                  = 0;
        int64_t     k                      = 0;
        int         problemIndex           = 0;
        int         requestedSolutionIndex = 0;
        int         librarySolutionIndex   = 0;
        int         numWarmups             = 10;
        int         numBenchmarks          = 10;
        int         enqueuesPerSync        = 10;
        int         syncsPerBenchmark      = 1;
        int         numElementsToValidate  = 128;
    };

    template <typename T>
    class DeviceBuffer
    {
    public:
        DeviceBuffer() = default;
        explicit DeviceBuffer(size_t count)
        {
            reset(count);
        }
        DeviceBuffer(DeviceBuffer const&)            = delete;
        DeviceBuffer& operator=(DeviceBuffer const&) = delete;
        DeviceBuffer(DeviceBuffer&& other) noexcept
            : m_ptr(other.m_ptr)
            , m_count(other.m_count)
        {
            other.m_ptr   = nullptr;
            other.m_count = 0;
        }
        DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
        {
            if(this != &other)
            {
                release();
                m_ptr         = other.m_ptr;
                m_count       = other.m_count;
                other.m_ptr   = nullptr;
                other.m_count = 0;
            }
            return *this;
        }
        ~DeviceBuffer()
        {
            release();
        }

        void reset(size_t count)
        {
            release();
            m_count = count;
            if(count > 0)
                HIP_CHECK_EXC(hipMalloc(reinterpret_cast<void**>(&m_ptr), count * sizeof(T)));
        }

        T* get() const
        {
            return m_ptr;
        }
        size_t count() const
        {
            return m_count;
        }

    private:
        void release()
        {
            if(m_ptr != nullptr)
                HIP_CHECK_PRINT(hipFree(m_ptr), [](hipError_t e) {
                    std::cerr << "hipFree failed: " << hipGetErrorString(e) << "\n";
                });
            m_ptr   = nullptr;
            m_count = 0;
        }

        T*     m_ptr   = nullptr;
        size_t m_count = 0;
    };

    class HipEvent
    {
    public:
        HipEvent()
        {
            HIP_CHECK_EXC(hipEventCreate(&m_event));
        }
        HipEvent(HipEvent const&)            = delete;
        HipEvent& operator=(HipEvent const&) = delete;
        ~HipEvent()
        {
            if(m_event != nullptr)
                HIP_CHECK_PRINT(hipEventDestroy(m_event), [](hipError_t e) {
                    std::cerr << "hipEventDestroy failed: " << hipGetErrorString(e) << "\n";
                });
        }
        operator hipEvent_t() const
        {
            return m_event;
        }

    private:
        hipEvent_t m_event = nullptr;
    };

    class HipStream
    {
    public:
        HipStream()
        {
            HIP_CHECK_EXC(hipStreamCreate(&m_stream));
        }
        HipStream(HipStream const&)            = delete;
        HipStream& operator=(HipStream const&) = delete;
        ~HipStream()
        {
            if(m_stream != nullptr)
                HIP_CHECK_PRINT(hipStreamDestroy(m_stream), [](hipError_t e) {
                    std::cerr << "hipStreamDestroy failed: " << hipGetErrorString(e) << "\n";
                });
        }
        operator hipStream_t() const
        {
            return m_stream;
        }

    private:
        hipStream_t m_stream = nullptr;
    };

    struct Buffers
    {
        std::vector<Half>  hostA;
        std::vector<Half>  hostB;
        std::vector<Half>  hostC;
        std::vector<Half>  hostD;
        std::vector<Half>  hostBias;
        std::vector<float> hostScaleAlphaVec;
        std::vector<Half>  resultD;

        DeviceBuffer<Half>  devA;
        DeviceBuffer<Half>  devB;
        DeviceBuffer<Half>  devC;
        DeviceBuffer<Half>  devD;
        DeviceBuffer<Half>  devBias;
        DeviceBuffer<float> devScaleAlphaVec;
        DeviceBuffer<char>  devWorkspace;
        DeviceBuffer<float> devSynchronizer;
    };

    [[noreturn]] void usage(std::ostream& os, int code)
    {
        os << "usage: evotensile-structured-runner --pairs pairs.jsonl --output results.jsonl "
              "--library-dir DIR [--library-file FILE] [--code-object FILE] [--device IDX] "
              "[--prime-enqueues N]\n";
        std::exit(code);
    }

    Args parseArgs(int argc, char** argv)
    {
        Args args;
        for(int i = 1; i < argc; ++i)
        {
            std::string key(argv[i]);
            auto needValue = [&](std::string const& name) -> std::string {
                if(i + 1 >= argc)
                    throw std::runtime_error("missing value for " + name);
                return std::string(argv[++i]);
            };

            if(key == "--pairs")
                args.pairs = needValue(key);
            else if(key == "--output")
                args.output = needValue(key);
            else if(key == "--library-dir")
                args.libraryDir = needValue(key);
            else if(key == "--library-file")
                args.libraryFile = needValue(key);
            else if(key == "--code-object")
                args.codeObject = needValue(key);
            else if(key == "--device")
                args.device = std::stoi(needValue(key));
            else if(key == "--workspace-bytes")
                args.workspaceSize = static_cast<size_t>(std::stoull(needValue(key)));
            else if(key == "--prime-enqueues")
                args.primeEnqueues = std::stoi(needValue(key));
            else if(key == "--use-user-args")
                args.useUserArgs = true;
            else if(key == "--validate-only")
                args.validateOnly = true;
            else if(key == "--help" || key == "-h")
                usage(std::cout, 0);
            else
                throw std::runtime_error("unknown argument: " + key);
        }
        if(args.pairs.empty() || args.output.empty())
            throw std::runtime_error("--pairs and --output are required");
        if(args.libraryDir.empty() && args.libraryFile.empty())
            throw std::runtime_error("--library-dir or --library-file is required");
        if(args.primeEnqueues < 0)
            throw std::runtime_error("--prime-enqueues must be non-negative");
        return args;
    }

    std::string stripArch(std::string arch)
    {
        auto pos = arch.find(':');
        if(pos != std::string::npos)
            arch.resize(pos);
        return arch;
    }

    fs::path defaultLibraryFile(fs::path const& libraryDir, std::string const& arch)
    {
        std::vector<fs::path> candidates = {
            libraryDir / ("TensileLibrary_" + arch + ".yaml"),
            libraryDir / "TensileLibrary.yaml",
            libraryDir / ("TensileLibrary_lazy_" + arch + ".yaml"),
            libraryDir / ("TensileLibrary_lazy_" + arch + ".dat"),
        };
        for(auto const& candidate : candidates)
            if(fs::exists(candidate))
                return candidate;
        return candidates.front();
    }

    fs::path defaultCodeObject(fs::path const& libraryDir, std::string const& arch)
    {
        return libraryDir / ("TensileLibrary_" + arch + ".co");
    }

    std::string jsonEscape(std::string_view value)
    {
        std::ostringstream os;
        for(char ch : value)
        {
            switch(ch)
            {
            case '"':
                os << "\\\"";
                break;
            case '\\':
                os << "\\\\";
                break;
            case '\b':
                os << "\\b";
                break;
            case '\f':
                os << "\\f";
                break;
            case '\n':
                os << "\\n";
                break;
            case '\r':
                os << "\\r";
                break;
            case '\t':
                os << "\\t";
                break;
            default:
                if(static_cast<unsigned char>(ch) < 0x20)
                    os << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                       << static_cast<int>(static_cast<unsigned char>(ch)) << std::dec;
                else
                    os << ch;
            }
        }
        return os.str();
    }

    size_t findJsonValue(std::string const& line, std::string const& key)
    {
        std::string needle = "\"" + key + "\"";
        size_t      pos    = line.find(needle);
        if(pos == std::string::npos)
            throw std::runtime_error("missing JSON key: " + key);
        pos = line.find(':', pos + needle.size());
        if(pos == std::string::npos)
            throw std::runtime_error("missing ':' for JSON key: " + key);
        ++pos;
        while(pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos])))
            ++pos;
        return pos;
    }

    std::optional<size_t> findOptionalJsonValue(std::string const& line, std::string const& key)
    {
        std::string needle = "\"" + key + "\"";
        size_t      pos    = line.find(needle);
        if(pos == std::string::npos)
            return std::nullopt;
        pos = line.find(':', pos + needle.size());
        if(pos == std::string::npos)
            throw std::runtime_error("missing ':' for JSON key: " + key);
        ++pos;
        while(pos < line.size() && std::isspace(static_cast<unsigned char>(line[pos])))
            ++pos;
        return pos;
    }

    std::string parseJsonStringAt(std::string const& line, size_t pos)
    {
        if(pos >= line.size() || line[pos] != '"')
            throw std::runtime_error("expected JSON string");
        ++pos;
        std::string out;
        while(pos < line.size())
        {
            char ch = line[pos++];
            if(ch == '"')
                return out;
            if(ch == '\\')
            {
                if(pos >= line.size())
                    throw std::runtime_error("unterminated JSON escape");
                char esc = line[pos++];
                switch(esc)
                {
                case '"':
                case '\\':
                case '/':
                    out.push_back(esc);
                    break;
                case 'b':
                    out.push_back('\b');
                    break;
                case 'f':
                    out.push_back('\f');
                    break;
                case 'n':
                    out.push_back('\n');
                    break;
                case 'r':
                    out.push_back('\r');
                    break;
                case 't':
                    out.push_back('\t');
                    break;
                default:
                    throw std::runtime_error("unsupported JSON escape");
                }
            }
            else
            {
                out.push_back(ch);
            }
        }
        throw std::runtime_error("unterminated JSON string");
    }

    std::string jsonString(std::string const& line, std::string const& key)
    {
        return parseJsonStringAt(line, findJsonValue(line, key));
    }

    int64_t jsonInt(std::string const& line, std::string const& key)
    {
        size_t pos = findJsonValue(line, key);
        size_t end = pos;
        if(end < line.size() && line[end] == '-')
            ++end;
        while(end < line.size() && std::isdigit(static_cast<unsigned char>(line[end])))
            ++end;
        if(end == pos || (line[pos] == '-' && end == pos + 1))
            throw std::runtime_error("expected integer for JSON key: " + key);
        return std::stoll(line.substr(pos, end - pos));
    }

    int64_t jsonOptionalInt(std::string const& line, std::string const& key, int64_t defaultValue)
    {
        auto pos = findOptionalJsonValue(line, key);
        if(!pos)
            return defaultValue;
        size_t end = *pos;
        if(end < line.size() && line[end] == '-')
            ++end;
        while(end < line.size() && std::isdigit(static_cast<unsigned char>(line[end])))
            ++end;
        if(end == *pos || (line[*pos] == '-' && end == *pos + 1))
            throw std::runtime_error("expected integer for JSON key: " + key);
        return std::stoll(line.substr(*pos, end - *pos));
    }

    std::vector<Pair> readPairs(std::string const& path)
    {
        std::ifstream in(path);
        if(!in)
            throw std::runtime_error("could not open pairs file: " + path);
        std::vector<Pair> pairs;
        std::string       line;
        int               lineNo = 0;
        while(std::getline(in, line))
        {
            ++lineNo;
            if(line.find_first_not_of(" \t\r\n") == std::string::npos)
                continue;
            try
            {
                Pair pair;
                pair.shapeId                = jsonString(line, "shape_id");
                pair.candidateHash          = jsonString(line, "candidate_hash");
                pair.m                      = jsonInt(line, "m");
                pair.n                      = jsonInt(line, "n");
                pair.batch                  = jsonInt(line, "batch");
                pair.k                      = jsonInt(line, "k");
                pair.problemIndex           = static_cast<int>(jsonOptionalInt(line, "problem_index", 0));
                pair.requestedSolutionIndex = static_cast<int>(jsonInt(line, "requested_solution_index"));
                pair.librarySolutionIndex
                    = static_cast<int>(jsonOptionalInt(line, "library_solution_index", pair.requestedSolutionIndex));
                pair.numWarmups            = static_cast<int>(jsonOptionalInt(line, "num_warmups", 10));
                pair.numBenchmarks         = static_cast<int>(jsonOptionalInt(line, "num_benchmarks", 10));
                pair.enqueuesPerSync       = static_cast<int>(jsonOptionalInt(line, "enqueues_per_sync", 10));
                pair.syncsPerBenchmark     = static_cast<int>(jsonOptionalInt(line, "syncs_per_benchmark", 1));
                pair.numElementsToValidate = static_cast<int>(jsonOptionalInt(line, "num_elements_to_validate", 128));
                if(pair.m <= 0 || pair.n <= 0 || pair.batch <= 0 || pair.k <= 0)
                    throw std::runtime_error("shape dimensions must be positive");
                if(pair.numWarmups < 0 || pair.numBenchmarks < 0 || pair.enqueuesPerSync <= 0
                   || pair.syncsPerBenchmark <= 0)
                    throw std::runtime_error("invalid benchmark protocol");
                pairs.push_back(std::move(pair));
            }
            catch(std::exception const& exc)
            {
                std::ostringstream msg;
                msg << path << ':' << lineNo << ": " << exc.what();
                throw std::runtime_error(msg.str());
            }
        }
        return pairs;
    }

    bool isPrime(size_t value)
    {
        if(value < 2)
            return false;
        if(value % 2 == 0)
            return value == 2;
        for(size_t factor = 3; factor <= value / factor; factor += 2)
            if(value % factor == 0)
                return false;
        return true;
    }

    size_t nextPrime(size_t value)
    {
        if(value <= 2)
            return 2;
        if(value % 2 == 0)
            ++value;
        while(!isPrime(value))
            value += 2;
        return value;
    }

    uint32_t mix32(uint64_t x)
    {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33;
        return static_cast<uint32_t>(x >> 32);
    }

    float deterministicValue(uint64_t index, uint64_t salt)
    {
        uint32_t v      = mix32(index + 0x9e3779b97f4a7c15ULL * (salt + 1));
        int      bucket = static_cast<int>(v % 17u) - 8;
        return static_cast<float>(bucket) / 8.0f;
    }

    bool almostEqualHalf(float reference, float result)
    {
        float absDiff = std::fabs(reference - result);
        return reference == result
               || absDiff < 0.01f * (std::fabs(reference) + std::fabs(result) + 1.0f);
    }

    ContractionProblemGemm makeProblem(Pair const& pair, size_t workspaceSize, bool useUserArgs)
    {
        size_t m     = static_cast<size_t>(pair.m);
        size_t n     = static_cast<size_t>(pair.n);
        size_t k     = static_cast<size_t>(pair.k);
        size_t batch = static_cast<size_t>(pair.batch);

        auto problem = ContractionProblemGemm::GEMM_Strides(false,
                                                            true,
                                                            rocisa::DataType::Half,
                                                            rocisa::DataType::Half,
                                                            rocisa::DataType::Half,
                                                            rocisa::DataType::Half,
                                                            m,
                                                            n,
                                                            k,
                                                            batch,
                                                            m,
                                                            m * k,
                                                            n,
                                                            n * k,
                                                            m,
                                                            m * n,
                                                            m,
                                                            m * n,
                                                            2.0);
        problem.setComputeInputTypeA(rocisa::DataType::Half);
        problem.setComputeInputTypeB(rocisa::DataType::Half);
        problem.setF32XdlMathOp(rocisa::DataType::Float);
        problem.setActivationComputeType(rocisa::DataType::Float);
        problem.setAlphaType(rocisa::DataType::Float);
        problem.setBetaType(rocisa::DataType::Float);
        problem.setAlphaRestriction(TensileLite::toScalarValueEnum(2.0));
        problem.setBetaRestriction(TensileLite::toScalarValueEnum(2.0));
        problem.setHighPrecisionAccumulate(true);
        problem.setCEqualsD(false);
        problem.setStridedBatched(true);
        problem.setUseGradient(false);
        problem.setUseBias(1);
        problem.setUseE(false);
        problem.setOutputAmaxD(false);
        problem.setKernelLanguage(TensileLite::KernelLanguage::Assembly);
        problem.setPerformanceMetric(TensileLite::PerformanceMetric::DeviceEfficiency);
        problem.setDeterministicMode(false);
        problem.setSparse(0, 0);
        problem.setActivationType(TensileLite::ActivationType::Hipblaslt_all);
        problem.setActivationNoGuard(false);
        problem.setWorkspaceSize(workspaceSize);
        problem.setSwizzleTensorA(false);
        problem.setSwizzleTensorB(false);
        problem.setBias(rocisa::DataType::Half,
                        m,
                        0,
                        false,
                        ContractionProblemGemm::TENSOR::D,
                        0);
        problem.setUseScaleAB("");
        problem.setUseScaleCD(false);
        problem.setUseScaleAlphaVec(1);
        problem.setScaleAlphaVec(rocisa::DataType::Float, m, 0);
        problem.setSynchronizer(rocisa::DataType::Float, SYNCHRONIZER_ELEMENTS);
        problem.setGroupedGemm(false);
        problem.setUseDeviceUserArguments(useUserArgs);
        problem.setParams().setActivationEnum(TensileLite::ActivationType::None);
        return problem;
    }

    Buffers makeBuffers(Pair const& pair, size_t workspaceSize)
    {
        size_t m     = static_cast<size_t>(pair.m);
        size_t n     = static_cast<size_t>(pair.n);
        size_t k     = static_cast<size_t>(pair.k);
        size_t batch = static_cast<size_t>(pair.batch);

        Buffers buffers;
        buffers.hostA.resize(m * k * batch);
        buffers.hostB.resize(n * k * batch);
        buffers.hostC.resize(m * n * batch);
        buffers.hostD.resize(m * n * batch, static_cast<Half>(0.0f));
        buffers.hostBias.resize(m);
        buffers.hostScaleAlphaVec.resize(m);
        buffers.resultD.resize(m * n * batch);

        for(size_t i = 0; i < buffers.hostA.size(); ++i)
            buffers.hostA[i] = static_cast<Half>(deterministicValue(i, 1));
        for(size_t i = 0; i < buffers.hostB.size(); ++i)
            buffers.hostB[i] = static_cast<Half>(deterministicValue(i, 2));
        for(size_t i = 0; i < buffers.hostC.size(); ++i)
            buffers.hostC[i] = static_cast<Half>(deterministicValue(i, 3));
        for(size_t i = 0; i < buffers.hostBias.size(); ++i)
            buffers.hostBias[i] = static_cast<Half>(deterministicValue(i, 4));
        for(size_t i = 0; i < buffers.hostScaleAlphaVec.size(); ++i)
            buffers.hostScaleAlphaVec[i] = deterministicValue(i, 5);

        buffers.devA.reset(buffers.hostA.size());
        buffers.devB.reset(buffers.hostB.size());
        buffers.devC.reset(buffers.hostC.size());
        buffers.devD.reset(buffers.hostD.size());
        buffers.devBias.reset(buffers.hostBias.size());
        buffers.devScaleAlphaVec.reset(buffers.hostScaleAlphaVec.size());
        buffers.devWorkspace.reset(workspaceSize);
        buffers.devSynchronizer.reset(SYNCHRONIZER_ELEMENTS);

        HIP_CHECK_EXC(hipMemcpy(buffers.devA.get(),
                                buffers.hostA.data(),
                                buffers.hostA.size() * sizeof(Half),
                                hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemcpy(buffers.devB.get(),
                                buffers.hostB.data(),
                                buffers.hostB.size() * sizeof(Half),
                                hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemcpy(buffers.devC.get(),
                                buffers.hostC.data(),
                                buffers.hostC.size() * sizeof(Half),
                                hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemcpy(buffers.devD.get(),
                                buffers.hostD.data(),
                                buffers.hostD.size() * sizeof(Half),
                                hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemcpy(buffers.devBias.get(),
                                buffers.hostBias.data(),
                                buffers.hostBias.size() * sizeof(Half),
                                hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemcpy(buffers.devScaleAlphaVec.get(),
                                buffers.hostScaleAlphaVec.data(),
                                buffers.hostScaleAlphaVec.size() * sizeof(float),
                                hipMemcpyHostToDevice));
        HIP_CHECK_EXC(hipMemset(buffers.devWorkspace.get(), 0, workspaceSize));
        HIP_CHECK_EXC(hipMemset(buffers.devSynchronizer.get(), 0, SYNCHRONIZER_ELEMENTS * sizeof(float)));
        return buffers;
    }

    ContractionInputs makeInputs(Buffers& buffers, ContractionProblemGemm const& problem)
    {
        ContractionInputs inputs(buffers.devA.get(),
                                 buffers.devB.get(),
                                 buffers.devC.get(),
                                 buffers.devD.get(),
                                 static_cast<float>(2.0f),
                                 static_cast<float>(2.0f));
        inputs.bias          = buffers.devBias.get();
        inputs.scaleAlphaVec = buffers.devScaleAlphaVec.get();
        inputs.ws            = buffers.devWorkspace.get();
        inputs.Synchronizer  = buffers.devSynchronizer.get();
        inputs.gpu           = true;
        inputs.maxElements.resize(problem.tensors().size(), 0);
        for(size_t i = 0; i < problem.tensors().size(); ++i)
            inputs.maxElements[i] = problem.tensors()[i].totalAllocatedElements();
        return inputs;
    }

    float referenceElement(Pair const& pair, Buffers const& buffers, size_t dIndex)
    {
        size_t m     = static_cast<size_t>(pair.m);
        size_t n     = static_cast<size_t>(pair.n);
        size_t k     = static_cast<size_t>(pair.k);
        size_t batch = static_cast<size_t>(pair.batch);
        (void)n;
        size_t batchStrideD = m * static_cast<size_t>(pair.n);
        size_t b            = dIndex / batchStrideD;
        size_t rem          = dIndex % batchStrideD;
        size_t col          = rem / m;
        size_t row          = rem % m;
        if(b >= batch)
            throw std::runtime_error("reference index out of range");

        float accum = 0.0f;
        size_t aBatchOffset = b * m * k;
        size_t bBatchOffset = b * static_cast<size_t>(pair.n) * k;
        for(size_t kk = 0; kk < k; ++kk)
        {
            float av = static_cast<float>(buffers.hostA[aBatchOffset + row + kk * m]);
            float bv = static_cast<float>(
                buffers.hostB[bBatchOffset + col + kk * static_cast<size_t>(pair.n)]);
            accum += av * bv;
        }
        float result = 2.0f * accum;
        result *= buffers.hostScaleAlphaVec[row];
        result += 2.0f * static_cast<float>(buffers.hostC[dIndex]);
        result += static_cast<float>(buffers.hostBias[row]);
        return static_cast<float>(static_cast<Half>(result));
    }

    bool validateResult(Pair const& pair, Buffers& buffers, std::string& message)
    {
        if(pair.numElementsToValidate == 0)
        {
            message = "NO_CHECK";
            return true;
        }

        size_t total = static_cast<size_t>(pair.m) * static_cast<size_t>(pair.n)
                       * static_cast<size_t>(pair.batch);
        HIP_CHECK_EXC(hipMemcpy(buffers.resultD.data(),
                                buffers.devD.get(),
                                buffers.resultD.size() * sizeof(Half),
                                hipMemcpyDeviceToHost));

        size_t stride = 1;
        if(pair.numElementsToValidate > 0 && static_cast<size_t>(pair.numElementsToValidate) < total)
            stride = nextPrime(total / static_cast<size_t>(pair.numElementsToValidate));

        size_t checked = 0;
        for(size_t elem = 0; elem < total; elem += stride)
        {
            float expected = referenceElement(pair, buffers, elem);
            float actual   = static_cast<float>(buffers.resultD[elem]);
            if(!almostEqualHalf(expected, actual))
            {
                std::ostringstream os;
                os << "FAILED elem=" << elem << " expected=" << expected << " actual=" << actual
                   << " stride=" << stride;
                message = os.str();
                return false;
            }
            ++checked;
        }
        std::ostringstream os;
        os << "PASSED checked=" << checked << " stride=" << stride;
        message = os.str();
        return true;
    }

    std::string validationToken(std::string const& message)
    {
        if(message.rfind("PASSED", 0) == 0)
            return "PASSED";
        if(message.rfind("NO_CHECK", 0) == 0)
            return "NO_CHECK";
        return "FAILED";
    }

    bool checkSolution(ContractionSolution& solution,
                       ContractionProblemGemm& problem,
                       TensileLite::Hardware const& hardware,
                       std::string& reason)
    {
        if(!(*solution.hardwarePredicate)(hardware))
        {
            reason = "WRONG_HARDWARE";
            return false;
        }
        problem.checkPersistentKernelEligibility(solution, hardware);
        TensileLite::Task task(hardware, problem, solution);
        if(!(*solution.problemPredicate)(problem) || !(*solution.taskPredicate)(task))
        {
            reason = "DID_NOT_SATISFY_ASSERTS";
            return false;
        }
        if(solution.requiredHostWorkspaceSizePerProblem == static_cast<size_t>(-1))
            solution.requiredHostWorkspaceSizePerProblem
                = solution.requiredHostSizeGroupedGemmSingle(problem, hardware);
        return true;
    }

    void emitRow(std::ofstream& out,
                 Pair const&    pair,
                 std::string const& status,
                 int            sampleIndex,
                 std::optional<double> timeUs,
                 std::optional<double> gflops,
                 std::string const& validation,
                 std::string const& validationDetail,
                 int solutionIndex)
    {
        out << "{\"candidate_hash\":\"" << jsonEscape(pair.candidateHash) << "\""
            << ",\"shape_id\":\"" << jsonEscape(pair.shapeId) << "\""
            << ",\"status\":\"" << jsonEscape(status) << "\""
            << ",\"sample_index\":";
        if(sampleIndex >= 0)
            out << sampleIndex;
        else
            out << "null";
        out << ",\"time_us\":";
        if(timeUs)
            out << std::setprecision(10) << *timeUs;
        else
            out << "null";
        out << ",\"gflops\":";
        if(gflops)
            out << std::setprecision(10) << *gflops;
        else
            out << "null";
        out << ",\"validation\":\"" << jsonEscape(validation) << "\""
            << ",\"validation_detail\":\"" << jsonEscape(validationDetail) << "\""
            << ",\"solution_index\":" << solutionIndex
            << ",\"requested_solution_index\":" << pair.requestedSolutionIndex
            << ",\"library_solution_index\":" << pair.librarySolutionIndex
            << ",\"problem_index\":" << pair.problemIndex << "}\n";
    }

    void runPair(std::ofstream& out,
                 Pair const& pair,
                 TensileLite::MasterSolutionLibrary<ContractionProblemGemm, ContractionSolution> const& library,
                 TensileLite::Hardware const& hardware,
                 TensileLite::hip::SolutionAdapter& adapter,
                 hipStream_t stream,
                 size_t workspaceSize,
                 bool useUserArgs,
                 bool validateOnly,
                 int primeEnqueues)
    {
        try
        {
            auto problem = makeProblem(pair, workspaceSize, useUserArgs);
            auto solution = library.getSolutionByIndex(problem, hardware, pair.librarySolutionIndex);
            if(!solution)
            {
                emitRow(out,
                        pair,
                        "solution_not_found",
                        -1,
                        std::nullopt,
                        std::nullopt,
                        "FAILED",
                        "solution index not found",
                        pair.librarySolutionIndex);
                return;
            }

            std::string predicateReason;
            if(!checkSolution(*solution, problem, hardware, predicateReason))
            {
                emitRow(out,
                        pair,
                        "rejected",
                        -1,
                        std::nullopt,
                        std::nullopt,
                        predicateReason,
                        predicateReason,
                        pair.librarySolutionIndex);
                return;
            }

            auto buffers = makeBuffers(pair, workspaceSize);
            auto inputs  = makeInputs(buffers, problem);
            std::vector<TensileLite::KernelInvocation> kernels;
            void* dUA     = nullptr;
            void* dUAHost = nullptr;
            if(useUserArgs)
                kernels = solution->solveTensileGPU(problem, inputs, hardware, &dUA, &dUAHost, nullptr, 0, stream);
            else
                kernels = solution->solve(problem, inputs, hardware, nullptr, 0, stream);

            if(kernels.empty())
                throw std::runtime_error("solution produced no kernel invocations");

            for(int i = 0; i < primeEnqueues; ++i)
                HIP_CHECK_EXC(adapter.launchKernels(kernels, stream, nullptr, nullptr));
            if(primeEnqueues > 0)
                HIP_CHECK_EXC(hipStreamSynchronize(stream));

            std::string validationDetail = pair.numElementsToValidate == 0 ? "NO_CHECK" : "PASSED";
            bool        validationOk     = true;
            for(int i = 0; i < pair.numWarmups; ++i)
            {
                HIP_CHECK_EXC(adapter.launchKernels(kernels, stream, nullptr, nullptr));
                if(i == 0 && pair.numElementsToValidate != 0)
                {
                    HIP_CHECK_EXC(hipStreamSynchronize(stream));
                    validationOk = validateResult(pair, buffers, validationDetail);
                    if(!validationOk)
                        break;
                }
            }
            if(pair.numWarmups == 0 && pair.numElementsToValidate != 0)
            {
                HIP_CHECK_EXC(adapter.launchKernels(kernels, stream, nullptr, nullptr));
                HIP_CHECK_EXC(hipStreamSynchronize(stream));
                validationOk = validateResult(pair, buffers, validationDetail);
            }
            HIP_CHECK_EXC(hipStreamSynchronize(stream));

            if(useUserArgs)
                solution->relaseDeviceUserArgs(dUA, dUAHost);

            if(!validationOk)
            {
                emitRow(out,
                        pair,
                        "validation_fail",
                        0,
                        std::nullopt,
                        std::nullopt,
                        "FAILED",
                        validationDetail,
                        pair.librarySolutionIndex);
                return;
            }
            if(validateOnly)
            {
                emitRow(out,
                        pair,
                        "ok",
                        0,
                        std::nullopt,
                        std::nullopt,
                        validationToken(validationDetail),
                        validationDetail,
                        pair.librarySolutionIndex);
                return;
            }

            double flopCount = static_cast<double>(problem.flopCount());
            int    launches  = pair.enqueuesPerSync * pair.syncsPerBenchmark;
            for(int sample = 0; sample < pair.numBenchmarks; ++sample)
            {
                HipEvent start;
                HipEvent stop;
                HIP_CHECK_EXC(hipEventRecord(start, stream));
                for(int sync = 0; sync < pair.syncsPerBenchmark; ++sync)
                    for(int enqueue = 0; enqueue < pair.enqueuesPerSync; ++enqueue)
                        HIP_CHECK_EXC(adapter.launchKernels(kernels, stream, nullptr, nullptr));
                HIP_CHECK_EXC(hipEventRecord(stop, stream));
                HIP_CHECK_EXC(hipEventSynchronize(stop));
                float elapsedMs = 0.0f;
                HIP_CHECK_EXC(hipEventElapsedTime(&elapsedMs, start, stop));
                double timeUs = static_cast<double>(elapsedMs) * 1000.0 / static_cast<double>(launches);
                double gflops = flopCount / timeUs / 1000.0;
                emitRow(out,
                        pair,
                        "ok",
                        sample,
                        timeUs,
                        gflops,
                        validationToken(validationDetail),
                        validationDetail,
                        pair.librarySolutionIndex);
            }
        }
        catch(std::exception const& exc)
        {
            emitRow(out,
                    pair,
                    "failed",
                    -1,
                    std::nullopt,
                    std::nullopt,
                    "FAILED",
                    exc.what(),
                    pair.librarySolutionIndex);
        }
    }
}

int main(int argc, char** argv)
{
    try
    {
        Args args = parseArgs(argc, argv);
        auto pairs = readPairs(args.pairs);
        if(pairs.empty())
            throw std::runtime_error("pairs file contains no runnable pairs");

        HIP_CHECK_EXC(hipSetDevice(args.device));
        hipDeviceProp_t prop;
        HIP_CHECK_EXC(hipGetDeviceProperties(&prop, args.device));
        std::string arch = stripArch(prop.gcnArchName);

        fs::path libraryDir = args.libraryDir.empty() ? fs::path(args.libraryFile).parent_path()
                                                      : fs::path(args.libraryDir);
        fs::path libraryFile = args.libraryFile.empty() ? defaultLibraryFile(libraryDir, arch)
                                                        : fs::path(args.libraryFile);
        fs::path codeObject = args.codeObject.empty() ? defaultCodeObject(libraryDir, arch)
                                                      : fs::path(args.codeObject);

        if(!fs::exists(libraryFile))
            throw std::runtime_error("library file does not exist: " + libraryFile.string());
        if(!fs::exists(codeObject))
            throw std::runtime_error("code object does not exist: " + codeObject.string());

        auto hardware = TensileLite::hip::GetCurrentDevice();
        if(!hardware)
            throw std::runtime_error("failed to initialize TensileLite hardware descriptor");

        auto baseLibrary = TensileLite::LoadLibraryFile<ContractionProblemGemm>(libraryFile.string());
        auto library = std::dynamic_pointer_cast<
            TensileLite::MasterSolutionLibrary<ContractionProblemGemm, ContractionSolution>>(baseLibrary);
        if(!library)
            throw std::runtime_error("failed to load a MasterSolutionLibrary from: "
                                     + libraryFile.string());

        TensileLite::hip::SolutionAdapter adapter;
        HIP_CHECK_EXC(adapter.loadCodeObjectFile(codeObject.string()));
        HIP_CHECK_EXC(adapter.initializeLazyLoading(hardware->archName(), libraryDir.string()));

        HipStream stream;
        std::ofstream out(args.output);
        if(!out)
            throw std::runtime_error("could not open output file: " + args.output);

        std::cerr << "evotensile structured runner: " << pairs.size() << " pair(s), arch=" << arch
                  << ", library=" << libraryFile << ", codeObject=" << codeObject << "\n";
        for(auto const& pair : pairs)
            runPair(out,
                    pair,
                    *library,
                    *hardware,
                    adapter,
                    stream,
                    args.workspaceSize,
                    args.useUserArgs,
                    args.validateOnly,
                    args.primeEnqueues);
        out.flush();
        HIP_CHECK_EXC(hipDeviceSynchronize());
        return 0;
    }
    catch(std::exception const& exc)
    {
        std::cerr << "error: " << exc.what() << "\n";
        return 1;
    }
}
