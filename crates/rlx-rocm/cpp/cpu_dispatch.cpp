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

// rlx-rocm HIP-CPU validation TU.
//
// Compiled only when `cargo build --features hip-cpu-validate`.
//
// Kernel sources live in `rlx-gpu-kernels/kernels/`. Rather than duplicate the 358-line
// `launch_<kernel>` wrapper layer here, we just pull in rlx-cuda's
// `cpu_dispatch.cpp` directly. The wrappers it defines compile into
// `rlx_rocm_cpu_dispatch.a` exactly the same way they compile into
// `rlx_cuda_cpu_dispatch.a` — same HIP-CPU semantics, same kernels,
// same FFI surface. Any improvement to rlx-cuda's harness flows here
// automatically.
#include "../../rlx-cuda/cpp/cpu_dispatch.cpp"
