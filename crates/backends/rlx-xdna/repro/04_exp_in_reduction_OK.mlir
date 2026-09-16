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
            %av = memref.load %A[%j] : memref<64xf32>
            %xm = arith.mulf %av, %sc : f32
            %log2e_e = arith.constant 1.4426950408889634 : f32
            %ln2_e = arith.constant 0.6931471805599453 : f32
            %half_e = arith.constant 0.5 : f32
            %one_e = arith.constant 1.0 : f32
            %h5_e = arith.constant 0.008333333 : f32
            %h4_e = arith.constant 0.041666668 : f32
            %h3_e = arith.constant 0.16666667 : f32
            %c127_e = arith.constant 127 : i32
            %c23_e = arith.constant 23 : i32
            %cm1_e = arith.constant -1 : i32
            %c0_e = arith.constant 0 : i32
            %t_e = arith.mulf %xm, %log2e_e : f32
            %th_e = arith.addf %t_e, %half_e : f32
            %kt_e = arith.fptosi %th_e : f32 to i32
            %ktf_e = arith.sitofp %kt_e : i32 to f32
            %over_e = arith.cmpf ogt, %ktf_e, %th_e : f32
            %adj_e = arith.select %over_e, %cm1_e, %c0_e : i32
            %k_e = arith.addi %kt_e, %adj_e : i32
            %kf_e = arith.sitofp %k_e : i32 to f32
            %kln2_e = arith.mulf %kf_e, %ln2_e : f32
            %r_e = arith.subf %xm, %kln2_e : f32
            %pa_e = arith.mulf %h5_e, %r_e : f32
            %pa2_e = arith.addf %pa_e, %h4_e : f32
            %pb_e = arith.mulf %pa2_e, %r_e : f32
            %pb2_e = arith.addf %pb_e, %h3_e : f32
            %pc_e = arith.mulf %pb2_e, %r_e : f32
            %pc2_e = arith.addf %pc_e, %half_e : f32
            %pd_e = arith.mulf %pc2_e, %r_e : f32
            %pd2_e = arith.addf %pd_e, %one_e : f32
            %pe_e = arith.mulf %pd2_e, %r_e : f32
            %er_e = arith.addf %pe_e, %one_e : f32
            %k127_e = arith.addi %k_e, %c127_e : i32
            %ksh_e = arith.shli %k127_e, %c23_e : i32
            %p2k_e = arith.bitcast %ksh_e : i32 to f32
            %e = arith.mulf %p2k_e, %er_e : f32
            %s2 = arith.addf %s, %e : f32
            scf.yield %s2 : f32
          }
          memref.store %sum, %O[%i] : memref<64xf32>
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
