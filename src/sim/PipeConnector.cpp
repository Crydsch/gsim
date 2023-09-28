#include "PipeConnector.hpp"
#include "Config.hpp"
#include "spdlog/spdlog.h"
#include <arpa/inet.h>
#include <assert.h>
#include <signal.h>
#include <string>
#include <sstream>

namespace sim
{

void sigpipe_handler(int /*sig*/)
{
    SPDLOG_ERROR("Broken Pipe: Communication failed. The other end closed the pipe; probably because of an error.");
    std::exit(1);  // cannot recover
}

PipeConnector::PipeConnector()
{
    // add signal handler for SIGPIPE (when other end closes pipe on error)
    signal(SIGPIPE, sigpipe_handler);
    static_assert(sizeof(float) == 4);
    static_assert(sizeof(int) == sizeof(float));
    static_assert(std::numeric_limits<float>::is_iec559);  // Is IEEE 754 float

    SPDLOG_INFO("Opening input pipe ({})", Config::pipe_in_filepath.c_str());
    pipe_in = fopen(Config::pipe_in_filepath.c_str(), "r");
    if (pipe_in == nullptr)
    {
        SPDLOG_ERROR("Cannot open pipe for reading: " + Config::pipe_in_filepath.string());
		std::exit(1);
    }

    SPDLOG_INFO("Opening output pipe ({})", Config::pipe_out_filepath.c_str());
    pipe_out = fopen(Config::pipe_out_filepath.c_str(), "w");
    if (pipe_out == nullptr)
    {
        SPDLOG_ERROR("Cannot open pipe for writing: " + Config::pipe_out_filepath.string());
		std::exit(1);
    }
}

PipeConnector::~PipeConnector()
{
    int res = fclose(pipe_in);
    pipe_in = nullptr;
    if (res != 0) {
        SPDLOG_WARN("Error closing input pipe");
    }

    res = fclose(pipe_out);
    pipe_out = nullptr;
    if (res != 0) {
        SPDLOG_WARN("Error closing output pipe");
    }
}

void PipeConnector::flush_output()
{
    [[maybe_unused]] int res = fflush(pipe_out);
    assert(res == 0);
}

void PipeConnector::write_uint32(const uint32_t _value)
{
    uint32_t value = htonl(_value);
    [[maybe_unused]] size_t res = fwrite(&value, sizeof(uint32_t), 1, pipe_out);
    assert (res == 1);
}

uint32_t PipeConnector::read_uint32()
{
    uint32_t value;
    [[maybe_unused]] size_t res = fread(&value, sizeof(uint32_t), 1, pipe_in);
    assert (res == 1);
    return ntohl(value);
}

void PipeConnector::write_uint16(const uint16_t _value)
{
    uint16_t value = htons(_value);
    [[maybe_unused]] size_t res = fwrite(&value, sizeof(uint16_t), 1, pipe_out);
    assert(res == 1);
}

uint16_t PipeConnector::read_uint16()
{
    uint16_t value;
    [[maybe_unused]] size_t res = fread(&value, sizeof(uint16_t), 1, pipe_in);
    assert (res == 1);
    return ntohs(value);
}

void PipeConnector::write_float(const float _value)
{
    union
    {
        uint32_t i;
        float f;
    } u;
    u.f = _value;
    write_uint32(u.i);  // handles byte ordering
}

float PipeConnector::read_float()
{
    union
    {
        uint32_t i;
        float f;
    } u;
    u.i = read_uint32();  // handles byte ordering
    return u.f;
}

void PipeConnector::write_string(const std::string _string)
{
    // Write size
    write_uint16(_string.size());

    // Write string
    [[maybe_unused]] size_t res = fwrite(_string.data(), sizeof(char), _string.size(), pipe_out);
    assert(res == _string.size());
}

std::string PipeConnector::read_string()
{
    // Read size
    uint16_t size = read_uint16();

    // Read string
    std::string buf;
    buf.resize(size);
    [[maybe_unused]] size_t res = fread(buf.data(), sizeof(char), size, pipe_in);
    assert (res == size);
    return buf;
}

void PipeConnector::write_vec2(const Vec2 _vec)
{
    write_float(_vec.x);
    write_float(_vec.y);
}

Vec2 PipeConnector::read_vec2()
{
    return Vec2(read_float(), read_float());
}

Header PipeConnector::read_header()
{
    assert(sizeof(Header) == sizeof(uint16_t));
    uint16_t header = read_uint16();
    assert(header < Header::Count);
    return (Header) header;
}

void PipeConnector::write_header(const Header _header)
{
    assert(sizeof(Header) == sizeof(uint16_t));
    write_uint16(_header);
}

std::vector<std::string> PipeConnector::read_config_args()
{
    // We read one string with all options separated by spaces
    std::string options = read_string();

    // Split into args
    std::vector<std::string> args;
    std::stringstream ss(options);
    std::string arg;

    while (!getline(ss, arg, ' ').eof())
    {
        args.push_back(arg);
    }
    args.push_back(arg);

    return args;
}

/**
 * This method is intended for debugging purposes.
 * It sends data back and forth between processes and checks whether they are
 * transmitted and converted as expected.
 */
void PipeConnector::testDataExchange()
{
    SPDLOG_DEBUG("Testing pipe data exchange");

    SPDLOG_DEBUG("Testing int/uint32 exchange");
    // Recv int value, send value+1, expect value+2
    for (int i = 0; i < 4; i++)
    {
        uint32_t value = read_uint32();
        SPDLOG_DEBUG("Received {}", value);
        write_uint32(value + 1);
        flush_output();
        SPDLOG_DEBUG("Sent {}", value + 1);
        uint32_t result = read_uint32();
        SPDLOG_DEBUG("Received {}", result);
        if (result != value + 2)
        {
            SPDLOG_ERROR("Expected {}, but got {}", value + 2, result);
            SPDLOG_ERROR("Pipe DataExchange failed!");
			std::exit(1);
        }
    }

    SPDLOG_DEBUG("Testing float exchange");
    // Recv float value, send value*2, expect value*4
    for (int i = 0; i < 3; i++)
    {
        float value = read_float();
        SPDLOG_DEBUG("Received {}", value);
        write_float(value * 2.0f);
        flush_output();
        SPDLOG_DEBUG("Sent {}", value * 2.0f);
        float result = read_float();
        SPDLOG_DEBUG("Received {}", result);
        if (result != value * 4.0f)
        {
            SPDLOG_ERROR("Expected {}, but got {}", value * 4.0f, result);
            SPDLOG_ERROR("Pipe DataExchange failed!");
			std::exit(1);
        }
    }

    SPDLOG_DEBUG("Testing string exchange");
    // Recv "foo", send "foobar", expect "foobarbaz"
    std::string value = read_string();
    SPDLOG_DEBUG("Received {}", value);
    value += "bar";
    write_string(value);
    flush_output();
    SPDLOG_DEBUG("Sent {}", value);
    std::string result = read_string();
    value += "baz";
    SPDLOG_DEBUG("Received {}", result);
    if (result != value)
    {
        SPDLOG_ERROR("Expected {}, but got {}", value, result);
        SPDLOG_ERROR("Pipe DataExchange failed!");
		std::exit(1);
    }

    SPDLOG_DEBUG("Testing pipe data exchange - DONE");
}

}  // namespace sim
