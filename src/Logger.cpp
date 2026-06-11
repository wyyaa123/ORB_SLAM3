#include "Logger.h"

#include <mutex>
#include <vector>

#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>

namespace ORB_SLAM3
{
namespace
{
const char *kLoggerName = "ORB_SLAM3";
const std::size_t kMaxLogFileSize = 1024 * 1024 * 10;
const std::size_t kMaxLogFiles = 3;

std::mutex gLoggerMutex;
std::shared_ptr<spdlog::logger> gLogger;

void ConfigureLogger(const std::shared_ptr<spdlog::logger> &logger,
                     spdlog::level::level_enum level)
{
    logger->set_level(level);
    for (auto &sink : logger->sinks())
    {
        sink->set_level(level);
    }
    logger->flush_on(spdlog::level::warn);
    logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] [%n] [%t] %v");
}
} // namespace

void Logger::Initialize(const std::string &logFile,
                        spdlog::level::level_enum level,
                        bool logToConsole,
                        bool logToFile)
{
    std::lock_guard<std::mutex> lock(gLoggerMutex);

    std::vector<spdlog::sink_ptr> sinks;
    sinks.reserve(2);

    if (logToConsole)
    {
        auto consoleSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        sinks.push_back(consoleSink);
    }

    if (logToFile)
    {
        auto fileSink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            logFile, kMaxLogFileSize, kMaxLogFiles);
        sinks.push_back(fileSink);
    }

    if (sinks.empty())
    {
        sinks.push_back(std::make_shared<spdlog::sinks::null_sink_mt>());
    }

    auto logger = std::make_shared<spdlog::logger>(kLoggerName, sinks.begin(), sinks.end());
    ConfigureLogger(logger, level);

    spdlog::drop(kLoggerName);
    spdlog::register_logger(logger);
    spdlog::set_default_logger(logger);

    gLogger = logger;
}

std::shared_ptr<spdlog::logger> Logger::Get()
{
    std::lock_guard<std::mutex> lock(gLoggerMutex);

    if (!gLogger)
    {
        auto existingLogger = spdlog::get(kLoggerName);
        if (existingLogger)
        {
            gLogger = existingLogger;
        }
        else
        {
            std::vector<spdlog::sink_ptr> sinks;
            sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());

            gLogger = std::make_shared<spdlog::logger>(kLoggerName, sinks.begin(), sinks.end());
            ConfigureLogger(gLogger, spdlog::level::info);

            spdlog::register_logger(gLogger);
            spdlog::set_default_logger(gLogger);
        }
    }

    return gLogger;
}

void Logger::SetLevel(spdlog::level::level_enum level)
{
    auto logger = Get();

    std::lock_guard<std::mutex> lock(gLoggerMutex);

    logger->set_level(level);
    for (auto &sink : logger->sinks())
    {
        sink->set_level(level);
    }
}

void Logger::Shutdown()
{
    std::lock_guard<std::mutex> lock(gLoggerMutex);

    if (gLogger)
    {
        gLogger->flush();
        gLogger.reset();
    }

    spdlog::drop(kLoggerName);
}

} // namespace ORB_SLAM3
