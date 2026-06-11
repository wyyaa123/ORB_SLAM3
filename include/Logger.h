#ifndef ORB_SLAM3_LOGGER_H
#define ORB_SLAM3_LOGGER_H

#include <memory>
#include <string>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

namespace ORB_SLAM3
{

class Logger
{
public:
    static void Initialize(const std::string &logFile = "logs/orb_slam3.log",
                           spdlog::level::level_enum level = spdlog::level::info,
                           bool logToConsole = true,
                           bool logToFile = true);

    static std::shared_ptr<spdlog::logger> Get();
    static void SetLevel(spdlog::level::level_enum level);
    static void Shutdown();
};

} // namespace ORB_SLAM3

#define ORB_LOG_TRACE(...) SPDLOG_LOGGER_TRACE(ORB_SLAM3::Logger::Get(), __VA_ARGS__)
#define ORB_LOG_DEBUG(...) SPDLOG_LOGGER_DEBUG(ORB_SLAM3::Logger::Get(), __VA_ARGS__)
#define ORB_LOG_INFO(...) SPDLOG_LOGGER_INFO(ORB_SLAM3::Logger::Get(), __VA_ARGS__)
#define ORB_LOG_WARN(...) SPDLOG_LOGGER_WARN(ORB_SLAM3::Logger::Get(), __VA_ARGS__)
#define ORB_LOG_ERROR(...) SPDLOG_LOGGER_ERROR(ORB_SLAM3::Logger::Get(), __VA_ARGS__)
#define ORB_LOG_CRITICAL(...) SPDLOG_LOGGER_CRITICAL(ORB_SLAM3::Logger::Get(), __VA_ARGS__)

#endif // ORB_SLAM3_LOGGER_H
