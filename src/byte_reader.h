#pragma once
#include <cstddef>
#include <cstring>

namespace rokoko {
// Reads untrusted binary data without alignment assumptions or out-of-range pointers.
class ByteReader {
    const unsigned char* data_;
    size_t remaining_;
public:
    ByteReader(const void* data, size_t size) : data_(static_cast<const unsigned char*>(data)), remaining_(data ? size : 0) {}
    bool read(void* out, size_t size) {
        if (size > remaining_) return false;
        if (size) { std::memcpy(out, data_, size); data_ += size; remaining_ -= size; }
        return true;
    }
    size_t remaining() const { return remaining_; }
};
}
