#include "cpu/convolution.h"
#include "cpu/math.h"
#include "int8.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace rokoko;
static void require(bool value, const char *message) { if (!value) throw std::runtime_error(message); }
static float value(int i) { return ((i * 37) % 101 - 50) / 53.f; }
int main() {
    try {
        cpu::configure();
        for (int ci : {3, 16}) {
            const int co=13, kernel=3;
            std::vector<Half> w(co*ci*kernel);
            std::vector<float> scales(co), bias(co);
            std::vector<int> qw(w.size());
            for (int generation=0;generation<2;++generation) {
                for (int c=0;c<co;++c) {
                    float peak=0;bias[c]=value(c+9);
                    for (int k=0;k<ci*kernel;++k) {
                        int i=c*ci*kernel+k;
                        w[i]=Half(c==0 ? 0.f : value(i+17)*(generation+1));
                        peak=std::max(peak,std::abs(float(w[i])));
                    }
                    scales[c]=peak>0 ? peak/127.f : 1.f;
                    for (int k=0;k<ci*kernel;++k) {
                        int i=c*ci*kernel+k;
                        qw[i]=int(std::nearbyint(float(w[i])/scales[c]));
                    }
                }
                for (int T : {1,17,129,4097}) for (int stride : {1,2})
                    for (int dilation : {1,3}) for (int mode=0;mode<3;++mode) {
                        int pad=dilation, to=(T-1)/stride+1;
                        std::vector<float> x(T*ci),y(to*co+2,-999),residual(to*co);
                        float peak=0;
                        for (int i=0;i<T*ci;++i) {
                            x[i]=T==1 ? 0.f : value(i+23)*.37f;
                            peak=std::max(peak,std::abs(x[i]));
                        }
                        float scale=peak>0 ? peak/127.f : 1.f;
                        for (int i=0;i<to*co;++i) {
                            residual[i]=value(i+11);
                            y[i+1]=mode ? residual[i] : std::numeric_limits<float>::quiet_NaN();
                        }
                        {
                            cpu::Int8Scope scope;
                            cpu::convolution(x.data(),w.data(),bias.data(),y.data()+1,
                                mode==0 ? nullptr : mode==1 ? residual.data() : y.data()+1,
                                ci,co,T,kernel,stride,pad,dilation,ci);
                        }
                        require(!cpu::int8_generator,"scope leaked");
                        for (int t=0;t<to;++t) for (int c=0;c<co;++c) {
                            int sum=0;
                            for (int k=0;k<kernel;++k) {
                                int p=t*stride-pad+k*dilation;
                                if (p>=0 && p<T) for (int i=0;i<ci;++i)
                                    sum+=int(std::nearbyint(x[p*ci+i]/scale))*qw[(c*kernel+k)*ci+i];
                            }
                            double expected=double(sum)*scale*scales[c]+bias[c]+(mode ? residual[t*co+c] : 0);
                            float actual=y[t*co+c+1];
                            require(std::isfinite(actual) && std::abs(actual-expected)<2e-5*(1+std::abs(expected)),"INT8 integer oracle mismatch");
                        }
                        require(y.front()==-999 && y.back()==-999,"output guard overwritten");
                    }
                for (float bad : {std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}) {
                    std::vector<float> x(17*ci,0),y(17*co);x[0]=bad;
                    bool rejected=false;
                    try {cpu::Int8Scope scope;cpu::convolution(x.data(),w.data(),nullptr,y.data(),nullptr,ci,co,17,3,1,1,1,ci);}
                    catch (const std::runtime_error &) {rejected=true;}
                    require(rejected && !cpu::int8_generator,"nonfinite input or scope recovery");
                }
                cpu::clear_weights(); // Reuse the same weight address with changed values.
            }
        }
        std::cout<<"PASS INT8 integer oracle, zero tensors, channels/tails, padding, dilation, stride, residual aliasing, cleanup and exception recovery\n";
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
