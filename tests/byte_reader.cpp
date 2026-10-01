#include "byte_reader.h"
#include <cassert>
#include <cstdint>
#include <limits>
int main() {
    unsigned char data[]={0,1,2,3,4};
    rokoko::ByteReader reader(data+1,4); // Deliberately unaligned input.
    unsigned char out[5]={9,9,9,9,9};
    assert(!reader.read(out,std::numeric_limits<size_t>::max()));
    assert(reader.remaining()==4 && out[0]==9);
    assert(reader.read(out,3) && out[0]==1 && out[2]==3 && reader.remaining()==1);
    assert(!reader.read(out,2) && reader.remaining()==1);
    assert(reader.read(out,1) && out[0]==4 && reader.remaining()==0);
    assert(!reader.read(out,1));
    assert(reader.read(nullptr,0));
    rokoko::ByteReader empty(nullptr,100);
    assert(!empty.read(out,1) && empty.remaining()==0);
}
