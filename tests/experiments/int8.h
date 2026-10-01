#pragma once
namespace rokoko::cpu {
extern bool int8_generator;
struct Int8Scope {
    bool previous = int8_generator;
    Int8Scope() { int8_generator = true; }
    ~Int8Scope() { int8_generator = previous; }
};
}
