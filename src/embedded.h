#pragma once
#include <cstddef>

namespace rokoko {
struct AssetView {
    const unsigned char* data;
    size_t size;
};
struct ModelAssets {
    AssetView weights, g2p, voice;
};
// Read-only process-lifetime storage linked into the executable/library.
ModelAssets embedded_assets();
AssetView embedded_build_info();
}
