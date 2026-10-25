// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// Boost.Math oracle for rlx-special's tests. Build exactly as SciPy builds Boost:
//   clang++ -std=c++17 -O2 -ffp-contract=on -I<boost> boost_oracle.cpp -o boost_oracle
// Reads "<fn> <args...>" lines (args as decimal or 0x-hex bit patterns), writes
// "<fn> <arg hex...> <result hex>" — every value as its IEEE-754 bit pattern.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <boost/math/special_functions/log1p.hpp>
#include <boost/math/special_functions/powm1.hpp>
#include <boost/math/distributions/students_t.hpp>
#include <boost/math/distributions/chi_squared.hpp>
using namespace boost::math::policies;
// SciPy's error handling: never throw.
typedef policy<promote_float<false>, promote_double<false>, max_root_iterations<400>,
               domain_error<ignore_error>, overflow_error<ignore_error>, underflow_error<ignore_error>,
               evaluation_error<ignore_error>, pole_error<ignore_error>> Pol;
static uint64_t bits(double v) { uint64_t u; memcpy(&u, &v, 8); return u; }
static double parse(const std::string& s) {
    if (s.rfind("0x", 0) == 0) { uint64_t u = std::stoull(s, nullptr, 16); double v; memcpy(&v, &u, 8); return v; }
    return std::stod(s);
}
int main() {
    std::string line;
    typedef boost::math::lanczos::lanczos13m53 L;
    while (std::getline(std::cin, line)) {
        std::istringstream ss(line); std::string fn, t; std::vector<double> a;
        ss >> fn; while (ss >> t) a.push_back(parse(t));
        double r;
        if (fn == "ibeta") r = boost::math::ibeta(a[0], a[1], a[2], Pol());
        else if (fn == "ibetac") r = boost::math::ibetac(a[0], a[1], a[2], Pol());
        else if (fn == "beta") r = boost::math::beta(a[0], a[1], Pol());
        else if (fn == "gamma_p") r = boost::math::gamma_p(a[0], a[1], Pol());
        else if (fn == "gamma_q") r = boost::math::gamma_q(a[0], a[1], Pol());
        else if (fn == "tgamma") r = boost::math::tgamma(a[0], Pol());
        else if (fn == "lgamma") r = boost::math::lgamma(a[0], Pol());
        else if (fn == "tgamma1pm1") r = boost::math::tgamma1pm1(a[0], Pol());
        else if (fn == "tgamma_delta_ratio") r = boost::math::tgamma_delta_ratio(a[0], a[1], Pol());
        else if (fn == "erf") r = boost::math::erf(a[0], Pol());
        else if (fn == "erfc") r = boost::math::erfc(a[0], Pol());
        else if (fn == "log1pmx") r = boost::math::log1pmx(a[0], Pol());
        else if (fn == "powm1") r = boost::math::powm1(a[0], a[1], Pol());
        else if (fn == "lanczos_sum") r = L::lanczos_sum(a[0]);
        else if (fn == "lanczos_sum_exp_g_scaled") r = L::lanczos_sum_expG_scaled(a[0]);
        else if (fn == "lanczos_sum_near_1") r = L::lanczos_sum_near_1(a[0]);
        else if (fn == "lanczos_sum_near_2") r = L::lanczos_sum_near_2(a[0]);
        else if (fn == "students_t_cdf") r = boost::math::cdf(boost::math::students_t_distribution<double, Pol>(a[0]), a[1]);
        else if (fn == "chi2_sf") r = boost::math::cdf(boost::math::complement(boost::math::chi_squared_distribution<double, Pol>(a[0]), a[1]));
        else { fprintf(stderr, "unknown fn %s\n", fn.c_str()); return 1; }
        printf("%s", fn.c_str());
        for (double v : a) printf(" %016llx", (unsigned long long)bits(v));
        printf(" %016llx\n", (unsigned long long)bits(r));
    }
}
