// CPU-only, bounds-checked reader for the fixed Kokoro tensor schema.
#pragma once
#include "model_schema.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
namespace rokoko {
struct TensorDesc {
    std::string name;
    size_t offset, size_bytes;
    std::string dtype;
    std::vector<int> shape;
};
struct ArtifactIndex {
    int version;
    size_t start, bytes;
    std::vector<TensorDesc> tensors;
};
inline ArtifactIndex read_artifact_index(const void* data, size_t size) {
    auto fail=[]() { throw std::runtime_error("invalid or truncated Kokoro weights"); };
    if (size<16) fail();
    auto p=static_cast<const uint8_t*>(data);
    uint32_t magic,version; uint64_t length;
    memcpy(&magic,p,4);memcpy(&version,p+4,4);memcpy(&length,p+8,8);
    if (magic!=0x4f4b4f4b || (version!=1 && version!=2) || length>size-16 || length>1024*1024) fail();
    size_t start=(16+length+4095)&~size_t(4095);
    if (start>size) fail();
    const TensorSpec* schema=version==1?MODEL_V1:MODEL_V2;
    size_t count=version==1?std::size(MODEL_V1):std::size(MODEL_V2);
    std::unordered_map<std::string,const TensorSpec*> expected;
    for (size_t i=0;i<count;++i) expected.emplace(schema[i].name,schema+i);
    ArtifactIndex out{int(version),start,0,{}};
    std::istringstream lines(std::string(reinterpret_cast<const char*>(p+16),length));
    std::string line;
    std::vector<std::pair<size_t,size_t>> spans;
    while (std::getline(lines,line)) {
        if (line.empty()) continue;
        std::istringstream in(line); TensorDesc t{};
        if (!(in>>t.name>>t.offset>>t.size_bytes>>t.dtype)) fail();
        if (t.dtype=="float16") t.dtype="fp16";
        if (t.dtype=="float32") t.dtype="fp32";
        auto it=expected.find(t.name);
        if (it==expected.end() || t.dtype!=it->second->dtype) fail();
        size_t bytes=t.dtype=="fp16"?2:4;
        for (int dim:it->second->shape) {
            int actual; if (!(in>>actual) || actual!=dim) fail();
            t.shape.push_back(dim);bytes*=size_t(dim);
        }
        std::string extra; if (in>>extra) fail();
        if (t.offset%256 || t.size_bytes!=bytes || t.offset>size-start || bytes>size-start-t.offset) fail();
        spans.emplace_back(t.offset,t.offset+bytes);
        out.bytes=std::max(out.bytes,t.offset+bytes);
        expected.erase(it);out.tensors.push_back(std::move(t));
    }
    if (!expected.empty()) fail();
    std::sort(spans.begin(),spans.end());
    for (size_t i=1;i<spans.size();++i) if (spans[i].first<spans[i-1].second) fail();
    return out;
}
}
