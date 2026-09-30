// bundle.h — Load a .bundle archive (mmap, parse TOC, return named spans)
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct Span {
    const char* data;
    size_t size;
};

struct Bundle {
    void* mmap_ptr = nullptr;
    size_t mmap_size = 0;
    std::unordered_map<std::string, Span> entries;

    static Bundle load(const std::string& path) {
        Bundle b;

        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            fprintf(stderr, "Cannot open bundle: %s\n", path.c_str());
            std::exit(1);
        }
        struct stat st;
        fstat(fd, &st);
        b.mmap_size = st.st_size;
        b.mmap_ptr = mmap(nullptr, b.mmap_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
        close(fd);
        if (b.mmap_ptr == MAP_FAILED) {
            fprintf(stderr, "mmap failed: %s\n", path.c_str());
            std::exit(1);
        }
        madvise(b.mmap_ptr, b.mmap_size, MADV_SEQUENTIAL);

        const uint8_t* base = (const uint8_t*)b.mmap_ptr;

        // Parse header
        uint32_t magic;
        memcpy(&magic, base, 4);
        if (magic != 0x4F4B4F52) { // "ROKO" little-endian
            fprintf(stderr, "Bad magic in %s: expected ROKO\n", path.c_str());
            std::exit(1);
        }

        uint32_t version, count;
        memcpy(&version, base + 4, 4);
        memcpy(&count, base + 8, 4);
        if (version != 1) {
            fprintf(stderr, "Unsupported bundle version %u\n", version);
            std::exit(1);
        }

        // Parse entries (72 bytes each, starting at offset 16)
        for (uint32_t i = 0; i < count; i++) {
            const uint8_t* e = base + 16 + i * 72;
            char name[57] = {};
            memcpy(name, e, 56);
            uint64_t offset, size;
            memcpy(&offset, e + 56, 8);
            memcpy(&size, e + 64, 8);
            b.entries[name] = {(const char*)base + offset, size};
        }

        return b;
    }

    Span get(const std::string& name) const {
        auto it = entries.find(name);
        if (it == entries.end()) return {nullptr, 0};
        return it->second;
    }

    void free() {
        if (mmap_ptr) {
            munmap(mmap_ptr, mmap_size);
            mmap_ptr = nullptr;
        }
    }
};
