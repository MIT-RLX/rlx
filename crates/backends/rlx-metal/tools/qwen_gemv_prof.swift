// qwen_gemv_prof.swift — M4-native GPU-timing profiler for the qwen decode GEMV
// kernels (rlx-metal). Measures real per-kernel GPU execution time (via
// MTLCommandBuffer.gpuStartTime/gpuEndTime, min-of-N warmed) and derives
// effective memory bandwidth, so you can see whether a decode matmul is
// bandwidth-bound or launch/occupancy-bound WITHOUT any ISA reverse-engineering
// (applegpu can't decode M4/G16 yet).
//
// Extract a single-kernel .metal (e.g. q4k_mv_f32_sg from dequant_gguf.msl with
// its constants + dq_read_f16), then:
//   swift qwen_gemv_prof.swift path/to/q4ksg.metal
//
// Finding (qwen3-0.6B, M4 Pro): the kernel hits 87% of peak on a big GEMV
// (lm_head) but only 8% on a small q_proj — the small per-projection m=1 GEMVs
// are launch/occupancy-bound (~26us fixed overhead each). Fix = fewer + bigger
// GEMVs (fuse q/k/v, gate/up; raise small-n occupancy). See memory
// project_metal_decode_perf.
import Metal
import Foundation

guard let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue()
else { fatalError("no Metal") }
let src = try! String(contentsOfFile: CommandLine.arguments[1], encoding: .utf8)
let lib = try! dev.makeLibrary(source: src, options: nil)
let pso = try! dev.makeComputePipelineState(function: lib.makeFunction(name: "q4k_mv_f32_sg")!)

// M4 Pro peak unified-memory bandwidth
let PEAK_GBPS = 273.0

func roundUp(_ x: Int, _ a: Int) -> Int { (x + a - 1) / a * a }

// Profile one Q4_K GEMV of shape (k, n): dst[n] = dequant(W[n,k]) . x[k]
func profile(k: Int, n: Int, label: String) {
    let blk = 144, QKK = 256
    let xBytes = roundUp(k * 4, 256)
    let wBytes = roundUp(n * (k / QKK) * blk, 256)
    let dBytes = roundUp(n * 4, 256)
    let total = xBytes + wBytes + dBytes
    let arena = dev.makeBuffer(length: total, options: .storageModeShared)!
    var xOff: UInt64 = 0
    var wOff: UInt64 = UInt64(xBytes)
    var dOff: UInt64 = UInt64(xBytes + wBytes)
    var kd = UInt32(k), nd = UInt32(n)

    let NSG = 4, NR0 = 8
    let tgW = (n + NSG*NR0 - 1) / (NSG*NR0)         // threadgroups
    let tptg = MTLSize(width: NSG*32, height: 1, depth: 1)  // 128 threads
    let grid = MTLSize(width: tgW, height: 1, depth: 1)

    // warm + measure: min GPU time over many single-dispatch buffers
    var best = Double.greatestFiniteMagnitude
    let iters = 300
    for it in 0..<iters {
        let cb = q.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pso)
        enc.setBuffer(arena, offset: 0, index: 0)
        enc.setBytes(&xOff, length: 8, index: 1)
        enc.setBytes(&wOff, length: 8, index: 2)
        enc.setBytes(&dOff, length: 8, index: 3)
        enc.setBytes(&kd, length: 4, index: 4)
        enc.setBytes(&nd, length: 4, index: 5)
        enc.dispatchThreadgroups(grid, threadsPerThreadgroup: tptg)
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        if it >= 20 { best = min(best, cb.gpuEndTime - cb.gpuStartTime) }
    }
    let us = best * 1e6
    let readBytes = Double(wBytes + xBytes)          // weights dominate
    let gbps = readBytes / best / 1e9
    print(String(format: "%-22@ k=%5d n=%6d | GPU %.2f us | %.3f MB read | %.1f GB/s = %.0f%% peak",
                 label as NSString, k, n, us, readBytes/1e6, gbps, gbps/PEAK_GBPS*100))
}

print("device: \(dev.name)  peak≈\(PEAK_GBPS) GB/s\n")
profile(k: 1024, n: 1024,   label: "q_proj")
profile(k: 1024, n: 2048,   label: "qkv(fused)")
profile(k: 1024, n: 3072,   label: "gate/up")
profile(k: 3072, n: 1024,   label: "down_proj")
profile(k: 1024, n: 151936, label: "lm_head(big)")
