#include "embedded.h"
#include <cstdint>
#include <string>

extern "C" {
extern const unsigned char rokoko_weights_start[], rokoko_weights_end[];
extern const unsigned char rokoko_g2p_start[], rokoko_g2p_end[];
extern const unsigned char rokoko_voice_start[], rokoko_voice_end[];
extern const unsigned char rokoko_info_start[], rokoko_info_end[];
}
namespace rokoko {
static AssetView view(const unsigned char* start, const unsigned char* end) {
    return {start, size_t(reinterpret_cast<uintptr_t>(end) - reinterpret_cast<uintptr_t>(start))};
}
ModelAssets embedded_assets() {
    return {view(rokoko_weights_start, rokoko_weights_end),
            view(rokoko_g2p_start, rokoko_g2p_end), view(rokoko_voice_start, rokoko_voice_end)};
}
AssetView embedded_build_info() {
    static const std::string info=[] {
        auto original=view(rokoko_info_start,rokoko_info_end);
        std::string json(reinterpret_cast<const char*>(original.data),original.size);
#ifdef ROKOKO_CPU
        json.insert(1,"\"backend\":\"cpu\",");
#else
        json.insert(1,"\"backend\":\"cuda\",");
#endif
        return json;
    }();
    return {reinterpret_cast<const unsigned char*>(info.data()),info.size()};
}
}
