// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

// On-GPU pad (constant/reflect/replicate/circular). One thread per OUTPUT
// element; maps each axis coordinate back to a source index (or the constant
// fill). f32 arena; non-f32 dtypes take the host-staged `PadHost` fallback.
//
// meta = [ out_dims[rank], in_dims[rank], before[rank], in_strides[rank] ] (u32).
// mode: 0=constant 1=reflect 2=replicate 3=circular.
extern "C" __global__ void pad(
    float* arena,
    unsigned int n,
    unsigned int src_off,
    unsigned int dst_off,
    unsigned int mode,
    float fill,
    unsigned int rank,
    const unsigned int* meta
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const unsigned int* odims = meta;
    const unsigned int* idims = meta + rank;
    const unsigned int* before = meta + 2u * rank;
    const unsigned int* istr = meta + 3u * rank;

    unsigned int rem = i;
    size_t iidx = 0;
    bool is_fill = false;
    for (int d = (int)rank - 1; d >= 0; --d) {
        unsigned int od = odims[d];
        int coord = (int)(rem % od);
        rem /= od;
        int nn = (int)idims[d];
        // Unpadded axis: identity (also avoids the reflect period==0 case).
        if ((unsigned int)nn == od) {
            iidx += (size_t)coord * istr[d];
            continue;
        }
        int p = coord - (int)before[d];
        int ic = 0;
        switch (mode) {
            case 0: // constant
                if (p < 0 || p >= nn) { is_fill = true; }
                else { ic = p; }
                break;
            case 2: // replicate (clamp to edge)
                ic = p < 0 ? 0 : (p >= nn ? nn - 1 : p);
                break;
            case 3: { // circular (wrap)
                int m = ((p % nn) + nn) % nn;
                ic = m;
                break;
            }
            default: { // 1 = reflect (exclude edge)
                int period = 2 * (nn - 1);
                int m = ((p % period) + period) % period;
                if (m >= nn) m = period - m;
                ic = m;
                break;
            }
        }
        if (is_fill) break;
        iidx += (size_t)ic * istr[d];
    }

    arena[dst_off + i] = is_fill ? fill : arena[src_off + iidx];
}
