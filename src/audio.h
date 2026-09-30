#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
namespace rokoko {
// ---------------------------------------------------------------------------
// WAV I/O
// ---------------------------------------------------------------------------

inline void write_wav_to_(std::ostream& f, const float* audio, int n_samples,
                          int sample_rate) {
    int16_t bits_per_sample = 16;
    int16_t num_channels = 1;
    int32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    int16_t block_align = num_channels * bits_per_sample / 8;
    int32_t data_size = n_samples * block_align;
    int32_t chunk_size = 36 + data_size;

    f.write("RIFF", 4);
    f.write(reinterpret_cast<char*>(&chunk_size), 4);
    f.write("WAVE", 4);

    f.write("fmt ", 4);
    int32_t fmt_size = 16;
    int16_t audio_format = 1;
    f.write(reinterpret_cast<char*>(&fmt_size), 4);
    f.write(reinterpret_cast<char*>(&audio_format), 2);
    f.write(reinterpret_cast<char*>(&num_channels), 2);
    f.write(reinterpret_cast<char*>(&sample_rate), 4);
    f.write(reinterpret_cast<char*>(&byte_rate), 4);
    f.write(reinterpret_cast<char*>(&block_align), 2);
    f.write(reinterpret_cast<char*>(&bits_per_sample), 2);

    f.write("data", 4);
    f.write(reinterpret_cast<char*>(&data_size), 4);

    for (int i = 0; i < n_samples; i++) {
        float s = std::max(-1.0f, std::min(1.0f, audio[i]));
        int16_t sample = (int16_t)(s * 32767.0f);
        f.write(reinterpret_cast<char*>(&sample), 2);
    }
}

inline bool write_wav(const std::string& path, const float* audio, int n_samples,
                      int sample_rate) {
    if (path == "-") {
        write_wav_to_(std::cout, audio, n_samples, sample_rate);
        std::cout.flush();
        return true;
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    write_wav_to_(f, audio, n_samples, sample_rate);
    return f.good();
}

// streambuf adapter for FILE* so we can reuse write_wav_to_() with popen pipes
class stdio_streambuf : public std::streambuf {
    FILE* f_;
protected:
    std::streamsize xsputn(const char* s, std::streamsize n) override {
        return fwrite(s, 1, n, f_);
    }
    int overflow(int c) override {
        return (c != EOF && fputc(c, f_) != EOF) ? c : EOF;
    }
public:
    stdio_streambuf(FILE* f) : f_(f) {}
};

inline bool play_wav(const float* audio, int n_samples, int sample_rate) {
    char paplay_cmd[128], pwplay_cmd[128];
    snprintf(paplay_cmd, sizeof(paplay_cmd),
             "paplay --raw --format=s16le --rate=%d --channels=1", sample_rate);
    snprintf(pwplay_cmd, sizeof(pwplay_cmd),
             "pw-play --format=s16 --rate=%d --channels=1 -", sample_rate);

    const char* players[] = {
        "aplay -q -",
        paplay_cmd,
        pwplay_cmd,
        "ffplay -nodisp -autoexit -loglevel quiet -",
        nullptr
    };
    for (int i = 0; players[i]; i++) {
        std::string cmd = players[i];
        std::string bin = cmd.substr(0, cmd.find(' '));
        if (system(("command -v " + bin + " >/dev/null 2>&1").c_str()) != 0) continue;

        FILE* pipe = popen(cmd.c_str(), "w");
        if (!pipe) continue;

        stdio_streambuf buf(pipe);
        std::ostream os(&buf);
        write_wav_to_(os, audio, n_samples, sample_rate);
        os.flush();

        if (pclose(pipe) == 0) return true;
    }
    fprintf(stderr, "Error: no audio player found. Install alsa-utils, pulseaudio, pipewire, or ffmpeg.\n");
    return false;
}

}
