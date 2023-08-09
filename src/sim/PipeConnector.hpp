#pragma once

#include "Entity.hpp"
#include <fstream>
#include <vector>
#include <stdint.h>

namespace sim {

// Note: An extra namespace is necessary to work around C++20 enums
//  We want to access the enum only via a safe namespace (ex. MyEnum::Value)
//  But also be able to cast the enum value to an integer (ex. int v = MyEnum::Value)

// Identifier for stream based communication.
// Indicates a request/response and determines subsequent data.
namespace header_ns {
enum header : uint16_t {
    Ok = 0,
    Initialize = 1,
    Move = 2,
    GetEntityPositions = 3,

    Count
};
}
typedef header_ns::header Header;

class PipeConnector {
 private:
    std::ifstream pipe_in;
    std::ofstream pipe_out;

 public:
    PipeConnector();
    ~PipeConnector();

    void flush_output();

    void write_uint32(const uint32_t _value);
    uint32_t read_uint32();
    void write_uint16(const uint16_t _value);
    uint16_t read_uint16();
    void write_float(const float _value);
    float read_float();
    void write_string(const std::string _string);
    std::string read_string();
    void write_vec2(const Vec2 _vec);
    Vec2 read_vec2();

    Header read_header();
    void write_header(Header _type);
    std::vector<std::string> read_config_args();

    void testDataExchange();
};

}  // namespace sim