#pragma once
#include "phonemes.h"
#include <map>
namespace rokoko {
// The synthesis API accepts an object of string fields. Parse the complete
// document, including JSON Unicode escapes; reject malformed/ambiguous input.
inline std::map<std::string,std::string> parse_request(const std::string& body) {
    size_t i=0;
    auto fail=[]() { throw std::invalid_argument("expected a JSON object with string fields"); };
    auto ws=[&]() { while(i<body.size() && (body[i]==' ' || body[i]=='\t' || body[i]=='\r' || body[i]=='\n')) ++i; };
    auto hex=[&]() { uint32_t n=0; for(int j=0;j<4;++j) { if(i==body.size()) fail(); char c=body[i++];
        int v=c>='0'&&c<='9'?c-'0':c>='a'&&c<='f'?c-'a'+10:c>='A'&&c<='F'?c-'A'+10:-1;
        if(v<0) fail();n=n*16+v; } return n; };
    auto str=[&]() {
        if(i==body.size() || body[i++]!='"') fail();std::string out;
        while(i<body.size()) {
            unsigned char c=body[i++];
            if(c=='"') { detail::utf8_len(out);return out; }
            if(c<32) fail();
            if(c!='\\') { out+=char(c);continue; }
            if(i==body.size()) fail();c=body[i++];
            switch(c) {
            case '"': case '\\': case '/': out+=char(c);break;
            case 'b':out+='\b';break;case 'f':out+='\f';break;
            case 'n':out+='\n';break;case 'r':out+='\r';break;case 't':out+='\t';break;
            case 'u': {
                uint32_t cp=hex();
                if(cp>=0xd800 && cp<=0xdbff) {
                    if(i+2>body.size() || body.substr(i,2)!="\\u") fail();i+=2;uint32_t low=hex();
                    if(low<0xdc00 || low>0xdfff) fail();cp=0x10000+((cp-0xd800)<<10)+(low-0xdc00);
                } else if(cp>=0xdc00 && cp<=0xdfff) fail();
                if(cp<128) out+=char(cp);
                else if(cp<2048) {out+=char(0xc0|(cp>>6));out+=char(0x80|(cp&63));}
                else if(cp<65536) {out+=char(0xe0|(cp>>12));out+=char(0x80|((cp>>6)&63));out+=char(0x80|(cp&63));}
                else {out+=char(0xf0|(cp>>18));out+=char(0x80|((cp>>12)&63));out+=char(0x80|((cp>>6)&63));out+=char(0x80|(cp&63));}
                break;
            }
            default:fail();
            }
        }
        fail();return out;
    };
    std::map<std::string,std::string> out;
    ws();if(i==body.size() || body[i++]!='{') fail();ws();
    if(i<body.size() && body[i]=='}') ++i;
    else for(;;) {
        auto key=str();ws();if(i==body.size() || body[i++]!=':') fail();ws();auto value=str();
        if(!out.emplace(key,value).second) fail();ws();if(i==body.size()) fail();
        char c=body[i++];if(c=='}') break;if(c!=',') fail();ws();
    }
    ws();if(i!=body.size()) fail();return out;
}
}
