// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Optional password seals for cold / sidecar blobs (`encrypt` feature).
//!
//! Format: magic `RLXSEAL1` + salt + nonce + ciphertext (ChaCha20-Poly1305).

use anyhow::{Result, bail};

/// Magic prefix for sealed blobs.
pub const SEAL_MAGIC: &[u8; 8] = b"RLXSEAL1";

/// Seal plaintext bytes with a password (Argon2id + ChaCha20-Poly1305).
#[cfg(feature = "encrypt")]
pub fn seal_bytes(plaintext: &[u8], password: &str) -> Result<Vec<u8>> {
    use argon2::{Algorithm, Argon2, Params, Version};
    use chacha20poly1305::aead::{Aead, KeyInit};
    use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};
    use rand_core::RngCore;

    let mut salt = [0u8; 16];
    let mut nonce_bytes = [0u8; 12];
    rand_core::OsRng.fill_bytes(&mut salt);
    rand_core::OsRng.fill_bytes(&mut nonce_bytes);

    let params =
        Params::new(19_456, 2, 1, Some(32)).map_err(|e| anyhow::anyhow!("argon2 params: {e}"))?;
    let argon = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let mut key_bytes = [0u8; 32];
    argon
        .hash_password_into(password.as_bytes(), &salt, &mut key_bytes)
        .map_err(|e| anyhow::anyhow!("argon2: {e}"))?;

    let cipher = ChaCha20Poly1305::new(Key::from_slice(&key_bytes));
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = cipher
        .encrypt(nonce, plaintext)
        .map_err(|e| anyhow::anyhow!("seal encrypt: {e}"))?;

    let mut out = Vec::with_capacity(8 + 16 + 12 + ciphertext.len());
    out.extend_from_slice(SEAL_MAGIC);
    out.extend_from_slice(&salt);
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

/// Open a sealed blob.
#[cfg(feature = "encrypt")]
pub fn unseal_bytes(sealed: &[u8], password: &str) -> Result<Vec<u8>> {
    use argon2::{Algorithm, Argon2, Params, Version};
    use chacha20poly1305::aead::{Aead, KeyInit};
    use chacha20poly1305::{ChaCha20Poly1305, Key, Nonce};

    if sealed.len() < 8 + 16 + 12 + 16 {
        bail!("sealed blob too short");
    }
    if &sealed[..8] != SEAL_MAGIC {
        bail!("not an RLXSEAL1 blob");
    }
    let salt = &sealed[8..24];
    let nonce_bytes = &sealed[24..36];
    let ciphertext = &sealed[36..];

    let params =
        Params::new(19_456, 2, 1, Some(32)).map_err(|e| anyhow::anyhow!("argon2 params: {e}"))?;
    let argon = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let mut key_bytes = [0u8; 32];
    argon
        .hash_password_into(password.as_bytes(), salt, &mut key_bytes)
        .map_err(|e| anyhow::anyhow!("argon2: {e}"))?;

    let cipher = ChaCha20Poly1305::new(Key::from_slice(&key_bytes));
    let nonce = Nonce::from_slice(nonce_bytes);
    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| anyhow::anyhow!("unseal failed (wrong password or corrupt)"))
}

#[cfg(not(feature = "encrypt"))]
pub fn seal_bytes(_plaintext: &[u8], _password: &str) -> Result<Vec<u8>> {
    bail!("rlx-pkg was built without the `encrypt` feature")
}

#[cfg(not(feature = "encrypt"))]
pub fn unseal_bytes(_sealed: &[u8], _password: &str) -> Result<Vec<u8>> {
    bail!("rlx-pkg was built without the `encrypt` feature")
}

/// True if `bytes` begin with the seal magic.
pub fn is_sealed(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && &bytes[..8] == SEAL_MAGIC
}
