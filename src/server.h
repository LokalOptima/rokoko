// server.h — HTTP server for Rokoko TTS (header-only, templated)
//
// PipelineT must expose:
//   std::string synthesize(const std::string& text, const std::string& voice,
//                          std::vector<float>& audio_out)
//   double last_preprocess_ms, last_g2p_ms, last_tts_ms
#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "cpp-httplib/httplib.h"
#include "weights.h"
#include "request_json.h"

using namespace rokoko;

static inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c)<32) {
                    char escaped[7];snprintf(escaped,sizeof(escaped),"\\u%04x",static_cast<unsigned char>(c));out+=escaped;
                } else out+=c;
        }
    }
    return out;
}

static thread_local std::string t_log_detail;

static inline void log_request(const httplib::Request& req, const httplib::Response& res) {
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    struct tm tm;
    localtime_r(&tt, &tm);
    char ts[20];
    strftime(ts, sizeof(ts), "%H:%M:%S", &tm);

    fprintf(stderr, "%s  %s %s  %d\n", ts, req.method.c_str(), req.path.c_str(), res.status);

    if (!t_log_detail.empty()) {
        fprintf(stderr, "         %s\n", t_log_detail.c_str());
        t_log_detail.clear();
    }
}

template<typename PipelineT>
static void run_server(PipelineT& pipeline, const std::string& host, int port) {
    httplib::Server svr;
    std::mutex mtx;

    svr.set_logger(log_request);

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    svr.Post("/shutdown", [&svr](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"shutting down\"}", "application/json");
        svr.stop();
    });

    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(R"html(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rokoko TTS</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, sans-serif; background: #0a0a0a; color: #e0e0e0;
         display: flex; justify-content: center; padding: 2rem; min-height: 100vh; }
  .container { width: 100%; max-width: 600px; }
  h1 { font-size: 1.3rem; font-weight: 600; margin-bottom: 1.5rem; color: #fff; }
  textarea { width: 100%; height: 120px; background: #1a1a1a; color: #e0e0e0;
             border: 1px solid #333; border-radius: 8px; padding: 12px; font-size: 15px;
             font-family: inherit; resize: vertical; outline: none; }
  textarea:focus { border-color: #555; }
  .controls { display: flex; gap: 10px; margin-top: 12px; align-items: center; }
  button { background: #2563eb; color: #fff; border: none; border-radius: 6px;
           padding: 8px 20px; font-size: 14px; font-weight: 500; cursor: pointer; }
  button:hover { background: #1d4ed8; }
  button:disabled { background: #333; color: #666; cursor: default; }
  .status { font-size: 13px; color: #888; margin-left: auto; white-space: nowrap; }
  audio { width: 100%; margin-top: 16px; outline: none; }
  .timing { font-size: 12px; color: #666; margin-top: 8px; font-variant-numeric: tabular-nums; }
  kbd { display: inline-block; font-size: 11px; color: #666; margin-top: 6px; }
</style>
</head>
<body>
<div class="container">
  <h1>Rokoko TTS</h1>
  <textarea id="text" placeholder="Type something..." autofocus>The quick brown fox jumps over the lazy dog.</textarea>
  <div class="controls">
    <button id="btn" onclick="speak()">Speak</button>
    <span class="status" id="status"></span>
  </div>
  <audio id="audio" controls style="display:none"></audio>
  <div class="timing" id="timing"></div>
  <kbd>Ctrl+Enter to speak</kbd>
</div>
<script>
const $ = id => document.getElementById(id);
let curCtx = null;
async function speak() {
  const text = $('text').value.trim();
  if (!text) return;
  $('btn').disabled = true;
  $('status').textContent = 'generating...';
  $('timing').textContent = '';
  $('audio').style.display = 'none';
  if (curCtx) { curCtx.close(); curCtx = null; }
  const t0 = performance.now();
  let tfirst = 0;
  try {
    const r = await fetch('/synthesize/stream', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text})
    });
    if (!r.ok) {
      $('status').textContent = (await r.json()).error || 'error';
      return;
    }
    const ctx = new AudioContext({sampleRate: 24000});
    curCtx = ctx;
    let nt = 0, ns = 0;
    const pcm = [];
    let rem = new Uint8Array(0);
    const rd = r.body.getReader();
    for (;;) {
      const {done, value} = await rd.read();
      if (done) break;
      if (!tfirst) {
        tfirst = performance.now();
        $('status').textContent = 'streaming...';
        nt = ctx.currentTime + 0.05;
      }
      const c = new Uint8Array(rem.length + value.length);
      c.set(rem); c.set(value, rem.length);
      const u = c.length - c.length % 4;
      rem = u < c.length ? c.slice(u) : new Uint8Array(0);
      if (!u) continue;
      const ab = new ArrayBuffer(u);
      new Uint8Array(ab).set(c.subarray(0, u));
      const f = new Float32Array(ab);
      pcm.push(f); ns += f.length;
      const b = ctx.createBuffer(1, f.length, 24000);
      b.getChannelData(0).set(f);
      const s = ctx.createBufferSource();
      s.buffer = b; s.connect(ctx.destination);
      if (nt < ctx.currentTime) nt = ctx.currentTime;
      s.start(nt); nt += b.duration;
    }
    if (!ns) { $('status').textContent = 'no audio'; return; }
    $('audio').src = URL.createObjectURL(makeWav(pcm, ns));
    $('audio').style.display = 'block';
    $('status').textContent = '';
    const dur = (ns / 24000).toFixed(1);
    const first = tfirst ? (tfirst - t0).toFixed(0) : '?';
    const total = ((performance.now() - t0) / 1000).toFixed(2);
    $('timing').textContent = dur + 's audio  \u00b7  first chunk ' + first + 'ms  \u00b7  total ' + total + 's';
  } catch(e) {
    $('status').textContent = 'error';
  } finally {
    $('btn').disabled = false;
  }
}
function makeWav(chunks, n) {
  const buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
  const w = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  w(0,'RIFF'); v.setUint32(4, 36 + n * 2, true); w(8,'WAVE');
  w(12,'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, 24000, true); v.setUint32(28, 48000, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36,'data'); v.setUint32(40, n * 2, true);
  let o = 44;
  for (const c of chunks) for (let i = 0; i < c.length; i++) {
    const x = Math.max(-1, Math.min(1, c[i]));
    v.setInt16(o, x < 0 ? x * 0x8000 : x * 0x7FFF, true); o += 2;
  }
  return new Blob([buf], {type: 'audio/wav'});
}
$('text').addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter') { e.preventDefault(); speak(); }
});
</script>
</body>
</html>)html", "text/html");
    });

    auto error=[](httplib::Response& res,int status,const std::string& msg) {
        res.status=status;res.set_content("{\"error\":\""+json_escape(msg)+"\"}","application/json");
    };
    auto prepare=[&](const httplib::Request& req,httplib::Response& res,typename PipelineT::Prepared& speech) {
        try {
            auto fields=parse_request(req.body);
            auto it=fields.find("text");
            if (it==fields.end()) {error(res,400,"missing text field");return false;}
            auto voice=fields.count("voice")?fields.at("voice"):"af_heart";
            auto err=pipeline.prepare(it->second,voice,speech,fields.count("input") && fields.at("input")=="phonemes");
            if (!err.empty()) {error(res,400,err);return false;}
            return true;
        } catch (const std::exception& e) {error(res,400,e.what());return false;}
    };
    svr.Get("/stats",[&](const httplib::Request&,httplib::Response& res) {
        std::lock_guard<std::mutex> lock(mtx);auto s=inference_stats();size_t available=0,total=0;cudaMemGetInfo(&available,&total);
        std::ostringstream out;
        out<<"{\"encode_graphs\":"<<s.encode_graphs<<",\"decode_graphs\":"<<s.decode_graphs
           <<",\"g2p_graphs\":"<<pipeline.g2p.graph_count()<<",\"arena_bytes\":"<<s.arena_bytes
           <<",\"gpu_used_bytes\":"<<total-available<<",\"invalidations\":"<<s.invalidations<<"}";
        res.set_content(out.str(),"application/json");
    });
    svr.Post("/synthesize",[&](const httplib::Request& req,httplib::Response& res) {
        std::lock_guard<std::mutex> lock(mtx);
        typename PipelineT::Prepared speech;
        if (!prepare(req,res,speech)) return;
        auto before=inference_stats();std::vector<float> audio;
        auto err=pipeline.render(speech,[&](const float* p,size_t n) {audio.insert(audio.end(),p,p+n);return true;});
        if (!err.empty()) {error(res,500,err);return;}
        auto after=inference_stats();
        res.set_header("X-Preprocess-Ms",std::to_string(pipeline.last_preprocess_ms));
        res.set_header("X-G2P-Ms",std::to_string(pipeline.last_g2p_ms));
        res.set_header("X-TTS-Ms",std::to_string(pipeline.last_tts_ms));
        res.set_header("X-Encode-Hits",std::to_string(after.encode_hits-before.encode_hits));
        res.set_header("X-Encode-Misses",std::to_string(after.encode_misses-before.encode_misses));
        res.set_header("X-Decode-Hits",std::to_string(after.decode_hits-before.decode_hits));
        res.set_header("X-Decode-Misses",std::to_string(after.decode_misses-before.decode_misses));
        res.set_header("X-Graph-Invalidations",std::to_string(after.invalidations-before.invalidations));
        res.set_header("X-Chunks",std::to_string(speech.chunks.size()));
        std::ostringstream wav(std::ios::binary);write_wav_to_(wav,audio.data(),int(audio.size()),SAMPLE_RATE);
        res.set_content(wav.str(),"audio/wav");
    });
    svr.Post("/synthesize/stream",[&](const httplib::Request& req,httplib::Response& res) {
        auto speech=std::make_shared<typename PipelineT::Prepared>();
        { std::lock_guard<std::mutex> lock(mtx); if (!prepare(req,res,*speech)) return; }
        res.set_header("X-Sample-Rate",std::to_string(SAMPLE_RATE));
        res.set_chunked_content_provider("audio/pcm",[&pipeline,&mtx,speech](size_t,httplib::DataSink& sink) {
            std::lock_guard<std::mutex> lock(mtx);
            auto err=pipeline.render(*speech,[&](const float* p,size_t n) {
                return sink.is_writable() && sink.write(reinterpret_cast<const char*>(p),n*sizeof(float));
            });
            // A failure after headers terminates the transfer without a successful terminator.
            if (!err.empty()) {fprintf(stderr,"stream: %s\n",err.c_str());return false;}
            sink.done();return true;
        });
    });

    const char* display_host = (host == "0.0.0.0") ? "localhost" : host.c_str();
    fprintf(stderr, "listening on http://%s:%d\n", display_host, port);
    fprintf(stderr, "\n");
    if (!svr.listen(host, port)) {
        fprintf(stderr, "failed to bind %s:%d\n", host.c_str(), port);
        std::exit(1);
    }
}
