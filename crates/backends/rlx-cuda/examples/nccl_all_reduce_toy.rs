// RLX — versatile ML compiler + runtime.
//! NCCL all-reduce toy (device-resident).
//!
//! **One rank (any single NVIDIA GPU):**
//! ```sh
//! export LD_LIBRARY_PATH=$HOME/nccl-shim:$LD_LIBRARY_PATH   # if libnccl.so is not on the default path
//! RANK=0 WORLD=1 cargo run -p rlx-cuda --features nccl --example nccl_all_reduce_toy
//! ```
//!
//! **Two+ ranks** need distinct CUDA devices (NCCL rejects duplicate GPUs):
//! ```sh
//! ID=/tmp/rlx_nccl_id.bin; rm -f "$ID"
//! RANK=0 WORLD=2 ID_FILE=$ID CUDA_VISIBLE_DEVICES=0 cargo run -p rlx-cuda --features nccl --example nccl_all_reduce_toy &
//! RANK=1 WORLD=2 ID_FILE=$ID CUDA_VISIBLE_DEVICES=1 cargo run -p rlx-cuda --features nccl --example nccl_all_reduce_toy
//! wait
//! ```

use cudarc::driver::CudaContext;
use std::fs;
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

fn main() {
    let rank: usize = std::env::var("RANK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let world: usize = std::env::var("WORLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let id_file =
        PathBuf::from(std::env::var("ID_FILE").unwrap_or_else(|_| "/tmp/rlx_nccl_id.bin".into()));
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let group_id: u64 = 42;

    if world < 1 || rank >= world {
        eprintln!("bad RANK={rank} WORLD={world}");
        std::process::exit(2);
    }

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    let id = if rank == 0 {
        let id = rlx_cuda::distributed::new_nccl_id().expect("new_nccl_id");
        let bytes = rlx_cuda::distributed::id_to_bytes(&id);
        fs::write(&id_file, bytes).expect("write id file");
        id
    } else {
        let deadline = Instant::now() + Duration::from_secs(30);
        loop {
            if id_file.exists() {
                let bytes = fs::read(&id_file).expect("read id file");
                if bytes.len() == 128 {
                    let mut arr = [0u8; 128];
                    arr.copy_from_slice(&bytes);
                    break rlx_cuda::distributed::id_from_bytes(&arr);
                }
            }
            if Instant::now() > deadline {
                panic!("timeout waiting for {}", id_file.display());
            }
            thread::sleep(Duration::from_millis(20));
        }
    };

    rlx_cuda::distributed::init_and_register(group_id, stream.clone(), rank, world, id)
        .expect("init_and_register");

    let host: Vec<f32> = (0..n)
        .map(|i| (rank as f32 + 1.0) * (i as f32 + 1.0))
        .collect();
    let mut buf = stream.alloc_zeros::<f32>(n).expect("alloc");
    stream.memcpy_htod(&host, &mut buf).expect("htod");

    let attrs = group_id.to_le_bytes().to_vec(); // Sum (kind byte absent)
    rlx_cuda::distributed::try_all_reduce_f32(&mut buf, 0, n, &attrs)
        .expect("try_all_reduce")
        .then_some(())
        .expect("no nccl comm registered");

    stream.synchronize().expect("sync");
    let out = stream.clone_dtoh(&buf).expect("dtoh");

    // Sum across ranks of (rank+1)*(i+1) = (world*(world+1)/2) * (i+1)
    let scale = (world * (world + 1) / 2) as f32;
    let mut ok = true;
    for (i, &v) in out.iter().enumerate() {
        let exp = scale * (i as f32 + 1.0);
        if (v - exp).abs() > 1e-4 {
            eprintln!("rank {rank}: out[{i}]={v} expected {exp}");
            ok = false;
        }
    }
    if ok {
        println!(
            "rank {rank}/{world}: all_reduce ok n={n} out[0]={:.3}",
            out[0]
        );
    } else {
        std::process::exit(1);
    }

    if rank == 0 {
        let _ = fs::remove_file(&id_file);
    }
    rlx_cuda::distributed::unregister_nccl_comm(group_id);
}
