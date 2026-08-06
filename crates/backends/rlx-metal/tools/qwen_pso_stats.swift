import Metal
import Foundation

guard let device = MTLCreateSystemDefaultDevice() else { fatalError("no Metal device") }
print("device: \(device.name)")
print("device.maxThreadsPerThreadgroup: \(device.maxThreadsPerThreadgroup.width)")
print("")

for path in CommandLine.arguments.dropFirst() {
    let src = try! String(contentsOfFile: path, encoding: .utf8)
    let lib: MTLLibrary
    do { lib = try device.makeLibrary(source: src, options: nil) }
    catch { print("\(path): compile error: \(error)"); continue }
    for name in lib.functionNames {
        guard let fn = lib.makeFunction(name: name) else { continue }
        do {
            let pso = try device.makeComputePipelineState(function: fn)
            let simd = pso.threadExecutionWidth
            let maxT = pso.maxTotalThreadsPerThreadgroup
            let tgmem = pso.staticThreadgroupMemoryLength
            // Apple GPU register file: ~ maxThreads at simd occupancy.
            // maxTotalThreadsPerThreadgroup < 1024 ⇒ register-limited (lower occupancy).
            print("kernel \(name):")
            print("  threadExecutionWidth (SIMD)      : \(simd)")
            print("  maxTotalThreadsPerThreadgroup    : \(maxT)   (1024 = no register limit; lower = register-bound)")
            print("  staticThreadgroupMemoryLength    : \(tgmem) bytes")
            print("  → occupancy ceiling vs 1024      : \(String(format: "%.0f%%", Double(maxT)/1024.0*100))")
        } catch { print("  \(name): pipeline error: \(error)") }
    }
}
