// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// SciPy xsf (cephes) oracle for rlx-special's tests. Build as SciPy builds xsf:
//   clang++ -std=c++17 -O2 -ffp-contract=on -I<xsf>/include xsf_oracle.cpp -o xsf_oracle
// "<fn> <args hex...>" -> "<fn> <args hex...> <result hex>".
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include "xsf/erf.h"
#include "xsf/cephes/ndtr.h"
#include "xsf/cephes/ndtri.h"
#include "xsf/cephes/igam.h"
#include "xsf/cephes/chdtr.h"
#include "xsf/cephes/gamma.h"
#include "xsf/cephes/zeta.h"
#include "xsf/cephes/unity.h"
static uint64_t bits(double v){uint64_t u;memcpy(&u,&v,8);return u;}
static double parse(const std::string&s){if(s.rfind("0x",0)==0){uint64_t u=std::stoull(s,nullptr,16);double v;memcpy(&v,&u,8);return v;}return std::stod(s);}
int main(){std::string line;while(std::getline(std::cin,line)){std::istringstream ss(line);std::string fn,t;std::vector<double>a;ss>>fn;while(ss>>t)a.push_back(parse(t));double r;
 if(fn=="erf")r=xsf::erf(a[0]);else if(fn=="erfc")r=xsf::erfc(a[0]);
 else if(fn=="ndtr")r=xsf::cephes::ndtr(a[0]);else if(fn=="ndtri")r=xsf::cephes::ndtri(a[0]);
 else if(fn=="igam")r=xsf::cephes::igam(a[0],a[1]);else if(fn=="igamc")r=xsf::cephes::igamc(a[0],a[1]);
 else if(fn=="chdtrc")r=xsf::cephes::chdtrc(a[0],a[1]);else if(fn=="lgam")r=xsf::cephes::lgam(a[0]);
 else if(fn=="Gamma")r=xsf::cephes::Gamma(a[0]);
 else if(fn=="zeta")r=xsf::cephes::zeta(a[0],a[1]);else if(fn=="c_log1p")r=xsf::cephes::log1p(a[0]);
 else if(fn=="c_expm1")r=xsf::cephes::expm1(a[0]);else if(fn=="c_log1pmx")r=xsf::cephes::log1pmx(a[0]);
 else if(fn=="lgam1p")r=xsf::cephes::lgam1p(a[0]);
 else{fprintf(stderr,"unknown %s\n",fn.c_str());return 1;}
 printf("%s",fn.c_str());for(double v:a)printf(" %016llx",(unsigned long long)bits(v));printf(" %016llx\n",(unsigned long long)bits(r));}}
