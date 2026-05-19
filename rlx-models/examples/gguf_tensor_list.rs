//! List every tensor + shape + dtype in a GGUF file. Used during
//! qwen35 weight-loader development to enumerate the tensor inventory
//! of a hybrid-arch file. Usage:
//!
//! ```text
//! cargo run --release -p rlx-models --example gguf_tensor_list -- <file.gguf>
//! ```

use rlx_gguf::GgufFile;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("usage: gguf_tensor_list <path>"))?;
    let raw = GgufFile::from_path(&path)?;
    let mut keys: Vec<&String> = raw.tensors.keys().collect();
    keys.sort();
    for k in keys {
        let t = &raw.tensors[k];
        println!("{:60} {:?} {:?}", k, t.shape, t.dtype);
    }
    Ok(())
}
