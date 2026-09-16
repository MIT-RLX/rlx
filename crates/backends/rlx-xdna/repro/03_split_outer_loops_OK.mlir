module {
  aie.device(npu1_1col) {
    %core = aie.tile(0, 2)
    %shim = aie.tile(0, 0)
    aie.objectfifo @a0(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<64xf32>>
    aie.objectfifo @b0(%shim, {%core}, 2 : i32) : !aie.objectfifo<memref<64xf32>>
    aie.objectfifo @o0(%core, {%shim}, 2 : i32) : !aie.objectfifo<memref<64xf32>>
    %0 = aie.core(%core) {
      %z0 = arith.constant 0 : index
      %zmax = arith.constant 9223372036854775807 : index
      %one = arith.constant 1 : index
      %nN = arith.constant 8 : index
      %kN = arith.constant 8 : index
      %zerof = arith.constant 0.0 : f32
      %sc = arith.constant 0.05 : f32
      %ninf_i = arith.constant -8388608 : i32
      %ninf = arith.bitcast %ninf_i : i32 to f32
      scf.for %iter = %z0 to %zmax step %one {
        %a_sv = aie.objectfifo.acquire @a0(Consume, 1) : !aie.objectfifosubview<memref<64xf32>>
        %A = aie.objectfifo.subview.access %a_sv[0] : !aie.objectfifosubview<memref<64xf32>> -> memref<64xf32>
        %b_sv = aie.objectfifo.acquire @b0(Consume, 1) : !aie.objectfifosubview<memref<64xf32>>
        %B = aie.objectfifo.subview.access %b_sv[0] : !aie.objectfifosubview<memref<64xf32>> -> memref<64xf32>
        %o_sv = aie.objectfifo.acquire @o0(Produce, 1) : !aie.objectfifosubview<memref<64xf32>>
        %O = aie.objectfifo.subview.access %o_sv[0] : !aie.objectfifosubview<memref<64xf32>> -> memref<64xf32>
        scf.for %i = %z0 to %nN step %one {
          %sum = scf.for %j = %z0 to %nN step %one iter_args(%s = %zerof) -> (f32) {
            %jb = arith.muli %j, %kN : index
            %dot = scf.for %k = %z0 to %kN step %one iter_args(%a = %zerof) -> (f32) {
              %av = memref.load %A[%k] : memref<64xf32>
              %bi = arith.addi %jb, %k : index
              %bv = memref.load %B[%bi] : memref<64xf32>
              %m2 = arith.mulf %av, %bv : f32
              %a2 = arith.addf %a, %m2 : f32
              scf.yield %a2 : f32
            }
            %s2 = arith.addf %s, %dot : f32
            scf.yield %s2 : f32
          }
          memref.store %sum, %O[%i] : memref<64xf32>
        }
        scf.for %i2 = %z0 to %nN step %one {
          %sum2 = scf.for %jj = %z0 to %nN step %one iter_args(%sb = %zerof) -> (f32) {
            %jbb = arith.muli %jj, %kN : index
            %dotb = scf.for %kk = %z0 to %kN step %one iter_args(%ab = %zerof) -> (f32) {
              %avb = memref.load %A[%kk] : memref<64xf32>
              %bib = arith.addi %jbb, %kk : index
              %bvb = memref.load %B[%bib] : memref<64xf32>
              %mb = arith.mulf %avb, %bvb : f32
              %a2b = arith.addf %ab, %mb : f32
              scf.yield %a2b : f32
            }
            %s2b = arith.addf %sb, %dotb : f32
            scf.yield %s2b : f32
          }
          %prev = memref.load %O[%i2] : memref<64xf32>
          %tot = arith.addf %prev, %sum2 : f32
          memref.store %tot, %O[%i2] : memref<64xf32>
        }
        aie.objectfifo.release @a0(Consume, 1)
        aie.objectfifo.release @b0(Consume, 1)
        aie.objectfifo.release @o0(Produce, 1)
      }
      aie.end
    }
    aie.runtime_sequence(%arg0: memref<64xf32>, %arg1: memref<64xf32>, %arg2: memref<64xf32>) {
      %ta = aiex.dma_configure_task_for @a0 {
        aie.dma_bd(%arg0 : memref<64xf32>, 0, 64, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ta)
      %tb = aiex.dma_configure_task_for @b0 {
        aie.dma_bd(%arg1 : memref<64xf32>, 0, 64, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tb)
      %to = aiex.dma_configure_task_for @o0 {
        aie.dma_bd(%arg2 : memref<64xf32>, 0, 64, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 64, stride = 1>]) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%to)
      aiex.dma_await_task(%to)
      aiex.dma_free_task(%ta)
      aiex.dma_free_task(%tb)
    }
  }
}
