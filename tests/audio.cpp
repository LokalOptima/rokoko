#include "audio.h"
int main() {
    float values[]={-2,-1,-0.5f,-1.0f/65536,0,1.0f/65536,0.5f,1,2};
    rokoko::write_wav_to_(std::cout,values,9,24000);
}
