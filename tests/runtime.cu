// Library integration and diagnostic runner; no Python/reference implementation here.
#include "rokoko.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
bool g_verbose=false;
using namespace rokoko;
namespace fs=std::filesystem;
static void require(bool ok,const std::string& msg) {if(!ok) throw std::runtime_error(msg);}
template<class T> void dump(const fs::path& p,const std::vector<T>& v) {
    std::ofstream f(p,std::ios::binary);f.write((const char*)v.data(),v.size()*sizeof(T));require(f.good(),"dump failed");
}
static void record(TtsPipeline& pipe,const Chunk& c,const fs::path& out,int frames=0) {
    fs::create_directories(out);
    auto style=pipe.style_for(c);
    dump(out/"tokens.i32",c.tokens);dump(out/"style.f32",std::vector<float>(style,style+256));
    std::ofstream(out/"phonemes.txt")<<c.phonemes;
    InferenceTrace trace;trace.force_frames=frames;
    auto audio=rokoko_infer(pipe.weights,c.tokens.data(),c.tokens.size(),style,pipe.stream,pipe.encode_arena,
                            pipe.decode_arena,pipe.d_workspace,pipe.ws_bytes,&trace);
    for(float x:audio) require(std::isfinite(x),"nonfinite audio");
    auto st=inference_stats();
    require(st.decode_frames==st.true_frames && st.upsample_frames==st.true_frames*2 &&
            st.stft_samples==st.true_frames*600 && audio.size()==size_t(st.true_frames)*600,"true frame length assertion");
    dump(out/"audio.f32",audio);dump(out/"duration.f32",trace.durations);
    dump(out/"rounded.i32",trace.rounded);dump(out/"f0.f32",trace.f0);dump(out/"noise.f32",trace.noise);
    std::ofstream(out/"stats.json")<<"{\"true_frames\":"<<st.true_frames<<",\"decode_frames\":"<<st.decode_frames
      <<",\"upsample_frames\":"<<st.upsample_frames<<",\"stft_samples\":"<<st.stft_samples
      <<",\"encode_hits\":"<<st.encode_hits<<",\"encode_misses\":"<<st.encode_misses
      <<",\"decode_hits\":"<<st.decode_hits<<",\"decode_misses\":"<<st.decode_misses
      <<",\"invalidations\":"<<st.invalidations<<",\"arena_bytes\":"<<st.arena_bytes<<"}";
}
int main(int argc,char** argv) {
    try {
        if (argc==5 && std::string(argv[1])=="--check-assets") {
            std::vector<unsigned char> buffers[3];
            for (int i=0;i<3;++i) {
                std::ifstream f(argv[i+2],std::ios::binary);
                require(bool(f),"missing asset");
                buffers[i]=std::vector<unsigned char>(std::istreambuf_iterator<char>(f),{});
            }
            ModelAssets assets{{buffers[0].data(),buffers[0].size()},
                {buffers[1].data(),buffers[1].size()},{buffers[2].data(),buffers[2].size()}};
            TtsContext ctx;require(ctx.init(assets),ctx.last_error);
            return 0;
        }
        if(argc<2 || argc>3) throw std::runtime_error("runtime OUT [TEXT_FILE] or --check-assets WEIGHTS G2P VOICE_FILE");
        fs::path root=argv[1];fs::create_directories(root);
        for(int context=0;context<(argc>2?1:2);++context) {
            TtsContext ctx;require(ctx.init(),ctx.last_error);
            auto pipe=ctx.pipeline();
            if(argc>2) {
                std::ifstream f(argv[2]);require(bool(f),"missing input file");
                std::string text((std::istreambuf_iterator<char>(f)),{});
                TtsPipeline::Prepared prepared;auto error=pipe.prepare(text,prepared);require(error.empty(),error);
                std::ofstream(root/"normalized.txt")<<prepared.normalized;
                std::ofstream spans(root/"spans.tsv");for(auto span:prepared.source_spans) spans<<span.begin<<'\t'<<span.end<<'\n';
                for(size_t i=0;i<prepared.chunks.size();++i) record(pipe,prepared.chunks[i],root/std::to_string(i));
                break;
            }
            fs::path out=root/((context==0)?"first":"recreated");
            auto a=chunk_ipa("hɛlˈoʊ wˈɜːld.").at(0);
            auto b=chunk_ipa("ɡʊdbˈaɪ wˈɜld.").at(0);
            // A repeated five times, changed input at identical T, changed style, A again.
            require(a.tokens.size()==b.tokens.size(),"same-key fixture token lengths");
            auto styled=a;styled.phonemes+="☃"; // Same tokens; next official style row before filtering.
            for(int i=0;i<5;++i) record(pipe,a,out/("a"+std::to_string(i)));
            record(pipe,b,out/"b");record(pipe,styled,out/"style_change");
            record(pipe,a,out/"aba");
            clear_inference_graphs();record(pipe,a,out/"isolated");
            record(pipe,a,out/"key_a",64);
            auto hits=inference_stats().decode_hits;
            record(pipe,b,out/"key_b",64);
            record(pipe,styled,out/"key_style",64);
            record(pipe,a,out/"key_aba",64);
#ifndef ROKOKO_CPU
            require(inference_stats().decode_hits==hits+3,"changed-input decode replay not exercised");
#else
            require(inference_stats().encode_graphs==0 && inference_stats().decode_graphs==0,"CPU must not create CUDA graphs");
#endif
            for(int frames:{31,32,33,63,64,65,127,128,129}) record(pipe,a,out/("frames"+std::to_string(frames)),frames);
            auto large=chunk_ipa(std::string(300,'a')).at(0);record(pipe,large,out/"grow");
            record(pipe,a,out/"after_growth");
            auto unknown=chunk_ipa("hɛlˈoʊ☃").at(0);record(pipe,unknown,out/"style_af_heart");
            std::vector<float> audio;require(!pipe.synthesize("  ",audio).empty() && audio.empty(),"empty input must fail");
            require(!pipe.synthesize("\xff",audio).empty(),"invalid UTF-8 must fail");
            require(!pipe.synthesize("☃",audio,true).empty(),"all-unknown IPA must fail");
            auto err=pipe.synthesize_streaming("Hello world.",[](const float*,size_t){return false;});
            require(err=="synthesis cancelled","cancellation must propagate");
            require(pipe.synthesize("Hello again.",audio).empty() && !audio.empty(),"request after cancellation");
            TtsContext other;require(!other.init(),"simultaneous contexts must be rejected");
        }
        std::cout<<"PASS library contracts and true frame boundaries\n";return 0;
    } catch(const std::exception& e) {std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
