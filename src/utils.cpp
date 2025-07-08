#include "utils.h"

#include <chrono>

uint64_t PRNG::operator()()
{
    uint64_t z = (x += 0x9e3779b97f4a7c15);
    z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
    z          = (z ^ (z >> 27)) * 0x94d049bb133111eb;
    return z ^ (z >> 31);
}

// -------------------------------------------------

Time Now()
{
    using namespace std::chrono;
    static_assert(sizeof(Time) == sizeof(milliseconds::rep), "Time should be 64 bits");

    auto now      = high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto millis   = duration_cast<milliseconds>(duration).count();
    return static_cast<Time>(millis);
}

// -------------------------------------------------

std::string timeText(Time time)
{
    if (time < 10000)
        return std::to_string(time) + "ms";
    else if (time < 1000000)
        return std::to_string(time / 1000) + "s";
    else if (time < 360000000)
        return std::to_string(time / 60000) + "min";
    else
        return std::to_string(time / 3600000) + "h";
}

std::string nodesText(uint64_t nodes)
{
    if (nodes < 10000)
        return std::to_string(nodes);
    else if (nodes < 10000000)
        return std::to_string(nodes / 1000) + "K";
    else if (nodes < 100000000000)
        return std::to_string(nodes / 1000000) + "M";
    else if (nodes < 100000000000000)
        return std::to_string(nodes / 1000000000) + "G";
    else
        return std::to_string(nodes / 1000000000000) + "T";
}

std::string speedText(uint64_t nodesPerSecond)
{
    if (nodesPerSecond < 100000)
        return std::to_string(nodesPerSecond);
    else if (nodesPerSecond < 100000000)
        return std::to_string(nodesPerSecond / 1000) + "K";
    else
        return std::to_string(nodesPerSecond / 1000000) + "M";
}