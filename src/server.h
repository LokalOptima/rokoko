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
            default:   out += c;
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

// Minimal JSON string value extractor (no dependency on a JSON library).
// Finds "key":"value" and returns value. Returns fallback if not found.
static inline std::string json_get_string(const std::string& body,
                                           const std::string& key,
                                           const std::string& fallback = "") {
    std::string needle = "\"" + key + "\"";
    auto pos = body.find(needle);
    if (pos == std::string::npos) return fallback;
    pos = body.find(':', pos + needle.size());
    if (pos == std::string::npos) return fallback;
    pos = body.find('"', pos + 1);
    if (pos == std::string::npos) return fallback;
    pos++; // skip opening quote
    std::string result;
    while (pos < body.size() && body[pos] != '"') {
        if (body[pos] == '\\' && pos + 1 < body.size()) {
            pos++;
            switch (body[pos]) {
                case '"':  result += '"'; break;
                case '\\': result += '\\'; break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                default:   result += body[pos]; break;
            }
        } else {
            result += body[pos];
        }
        pos++;
    }
    return result;
}

template<typename PipelineT>
static void run_server(PipelineT& pipeline, const std::string& host, int port) {
    httplib::Server svr;
    std::mutex mtx;

    svr.set_logger(log_request);

    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
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
  select { background: #1a1a1a; color: #e0e0e0; border: 1px solid #333;
           border-radius: 6px; padding: 8px 12px; font-size: 14px; outline: none; }
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
    <select id="voice">
      <option value="af_heart">af_heart</option>
      <option value="af_bella">af_bella</option>
      <option value="af_sky">af_sky</option>
      <option value="af_nicole">af_nicole</option>
    </select>
    <button id="btn" onclick="speak()">Speak</button>
    <span class="status" id="status"></span>
  </div>
  <audio id="audio" controls style="display:none"></audio>
  <div class="timing" id="timing"></div>
  <kbd>Ctrl+Enter to speak</kbd>
</div>
<script>
const $ = id => document.getElementById(id);
async function speak() {
  const text = $('text').value.trim();
  if (!text) return;
  $('btn').disabled = true;
  $('status').textContent = 'generating...';
  $('timing').textContent = '';
  const t0 = performance.now();
  try {
    const res = await fetch('/synthesize', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({text, voice: $('voice').value})
    });
    if (!res.ok) {
      const err = await res.json();
      $('status').textContent = err.error || 'error';
      return;
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const audio = $('audio');
    audio.src = url;
    audio.style.display = 'block';
    audio.play();
    const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
    const dur = res.headers.get('X-Audio-Duration');
    const g2p = res.headers.get('X-G2P-Ms');
    const tts = res.headers.get('X-TTS-Ms');
    $('status').textContent = '';
    const parts = [];
    if (dur) parts.push(parseFloat(dur).toFixed(1) + 's audio');
    if (g2p) parts.push('g2p ' + parseFloat(g2p).toFixed(0) + 'ms');
    if (tts) parts.push('tts ' + parseFloat(tts).toFixed(0) + 'ms');
    parts.push('total ' + elapsed + 's');
    $('timing').textContent = parts.join('  ·  ');
  } catch(e) {
    $('status').textContent = 'network error';
  } finally {
    $('btn').disabled = false;
  }
}
$('text').addEventListener('keydown', e => {
  if (e.ctrlKey && e.key === 'Enter') { e.preventDefault(); speak(); }
});
</script>
</body>
</html>)html", "text/html");
    });

    svr.Post("/synthesize", [&](const httplib::Request& req, httplib::Response& res) {
        // Parse JSON body
        std::string text = json_get_string(req.body, "text");
        std::string voice = json_get_string(req.body, "voice", "af_heart");

        if (text.empty()) {
            res.status = 400;
            res.set_content("{\"error\":\"missing 'text' field\"}", "application/json");
            return;
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<float> audio;
        std::string err;
        {
            std::lock_guard<std::mutex> lock(mtx);
            err = pipeline.synthesize(text, voice, audio);
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        if (!err.empty()) {
            res.status = 500;
            std::string body = "{\"error\":\"" + json_escape(err) + "\"}";
            res.set_content(body, "application/json");
            return;
        }

        double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double audio_dur = (double)audio.size() / SAMPLE_RATE;
        double rtfx = audio_dur / (elapsed_ms / 1000.0);

        // Timing headers
        res.set_header("X-Preprocess-Ms", std::to_string(pipeline.last_preprocess_ms));
        res.set_header("X-G2P-Ms", std::to_string(pipeline.last_g2p_ms));
        res.set_header("X-TTS-Ms", std::to_string(pipeline.last_tts_ms));
        res.set_header("X-Audio-Duration", std::to_string(audio_dur));

        // Write WAV to string via ostringstream
        std::ostringstream wav_stream(std::ios::binary);
        write_wav_to_(wav_stream, audio.data(), (int)audio.size(), SAMPLE_RATE);
        res.set_content(wav_stream.str(), "audio/wav");

        // Log detail
        std::string preview = text.substr(0, 80);
        if (text.size() > 80) preview += "...";
        char detail[256];
        snprintf(detail, sizeof(detail), "audio=%.1fs  inference=%.0fms  RTFx=%.0fx  \"%s\"",
                 audio_dur, elapsed_ms, rtfx, preview.c_str());
        t_log_detail = detail;
    });

    const char* display_host = (host == "0.0.0.0") ? "localhost" : host.c_str();
    fprintf(stderr, "listening on http://%s:%d\n", display_host, port);
    fprintf(stderr, "\n");
    if (!svr.listen(host, port)) {
        fprintf(stderr, "failed to bind %s:%d\n", host.c_str(), port);
        std::exit(1);
    }
}
