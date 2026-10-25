// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Reference generator for tests/vectors/cpp_contract_{on,off}.txt (not built by cargo).
// The pinned header is scipy/_lib/pocketfft @ 9367142748fcc9696a1c9e5a99b76ed9897c9daa:
//   clang++ -std=c++17 -O3 -ffp-contract=on  -I<pocketfft> contraction_ref.cpp -o on  \&\& ./on  > ../tests/vectors/cpp_contract_on.txt
//   clang++ -std=c++17 -O3 -ffp-contract=off -I<pocketfft> contraction_ref.cpp -o off \&\& ./off > ../tests/vectors/cpp_contract_off.txt
// (macOS arm64: clang merges each cos/sin pair into __sincos_stret, so the twiddles are Twiddles::AppleSinCos.)
#define POCKETFFT_CACHE_SIZE 16
#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft_hdronly.h"
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <complex>
using namespace pocketfft;
static std::vector<double> sig(size_t n, uint64_t seed){uint64_t s=(seed*0x9E3779B97F4A7C15ull)|1;std::vector<double> v(n);for(auto&x:v){s^=s<<13;s^=s>>7;s^=s<<17;x=double(s>>11)/double(1ull<<53)*2.0-1.0;}return v;}
static uint64_t fnv(const double*p,size_t m){uint64_t h=0xcbf29ce484222325ull;const unsigned char*b=(const unsigned char*)p;for(size_t i=0;i<m*8;i++){h=(h^b[i])*0x100000001b3ull;}return h;}
int main(){
  for(size_t n=1;n<=600;n++){
    auto x=sig(n,n), y=sig(n,7+n);
    std::vector<std::complex<double>> X(n/2+1), Z(n), O(n); std::vector<double> xr(n);
    r2c(shape_t{n},stride_t{8},stride_t{16},0,true,x.data(),X.data(),1.0,1);
    c2r(shape_t{n},stride_t{16},stride_t{8},0,false,X.data(),xr.data(),1.0/n,1);
    for(size_t i=0;i<n;i++)Z[i]={x[i],y[i]};
    c2c(shape_t{n},stride_t{16},stride_t{16},shape_t{0},true,Z.data(),O.data(),1.0,1);
    uint64_t a=fnv((double*)X.data(),2*(n/2+1)),b=fnv(xr.data(),n),c=fnv((double*)O.data(),2*n);
    printf("%zu %016llx %016llx %016llx\n",n,(unsigned long long)a,(unsigned long long)b,(unsigned long long)c);
  }
}
