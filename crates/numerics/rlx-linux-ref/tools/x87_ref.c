// Reference for tests/fixtures/x87_ops.txt (rlx-libm `x87`): run on an x86-64 Linux host
// (x87 default precision control = extended):  cc -O0 x87_ref.c -o x87_ref && ./x87_ref > x87_ops.txt
// Inputs: the xorshift64 stream u = (s >> 11) / 2^53 * 2 - 1, scaled by 2^k (k in -40..40).
// Each line: "op a b <80-bit result as sign/exp hex + mantissa hex> <(double) result hex>".
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
static uint64_t s = 0x2545f4914f6cdd1dULL;
static double next(void) {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    double u = (double)(s >> 11) / 9007199254740992.0 * 2.0 - 1.0;
    int k = (int)((s >> 3) % 81) - 40;
    return ldexp(u, k);
}
static void put(const char *op, double a, double b, long double r) {
    unsigned char raw[16] = {0}; memcpy(raw, &r, 10);
    uint64_t m; uint16_t se; memcpy(&m, raw, 8); memcpy(&se, raw + 8, 2);
    double d = (double)r; uint64_t db; memcpy(&db, &d, 8);
    uint64_t ab, bb; memcpy(&ab, &a, 8); memcpy(&bb, &b, 8);
    printf("%s %016llx %016llx %04x%016llx %016llx\n", op, (unsigned long long)ab, (unsigned long long)bb,
           se, (unsigned long long)m, (unsigned long long)db);
}
int main(void) {
    for (int i = 0; i < 1500; i++) {
        double a = next(), b = next();
        volatile long double la = a, lb = b;
        put("add", a, b, la + lb);
        put("sub", a, b, la - lb);
        put("mul", a, b, la * lb);
        put("div", a, b, la / lb);
        put("sqrt", fabs(a), 0.0, sqrtl(fabsl(la)));
        // chained: a² + b² (two extended roundings) then its sqrt — dnrm2's shape
        volatile long double q = la * la + lb * lb;
        put("sumsq", a, b, q);
        put("hyp", a, b, sqrtl(q));
        // a/b + a·b (operands that are themselves extended)
        volatile long double t = la / lb; volatile long double u = la * lb;
        put("divmuladd", a, b, t + u);
    }
    return 0;
}
