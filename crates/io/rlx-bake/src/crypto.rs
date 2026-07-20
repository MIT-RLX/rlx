// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.

//! Full-file encryption for `*.rlx` artifacts.
//!
//! On-disk layout (entire plaintext `RLXBAKE1` blob is the AEAD payload):
//!
//! ```text
//! RLXENC01          // 8 bytes magic
//! u32 LE version    // 1
//! u32 LE m_kib      // Argon2 memory (KiB)
//! u32 LE t_cost     // Argon2 iterations
//! u32 LE p_cost     // Argon2 parallelism
//! salt[16]
//! nonce[12]
//! ciphertext || tag // ChaCha20-Poly1305
//! ```

use anyhow::{Context, Result, bail};
use argon2::{Algorithm, Argon2, Params, Version};
use chacha20poly1305::aead::{Aead, KeyInit};
use chacha20poly1305::{ChaCha20Poly1305, Nonce};
use rand_core::{OsRng, RngCore};

/// Magic for an encrypted `*.rlx` file.
pub const RLX_ENC_MAGIC: &[u8; 8] = b"RLXENC01";

/// Encrypted container schema version.
pub const RLX_ENC_VERSION: u32 = 1;

const SALT_LEN: usize = 16;
const NONCE_LEN: usize = 12;
const KEY_LEN: usize = 32;

/// Header size before ciphertext: magic(8) + ver(4) + m/t/p(12) + salt(16) + nonce(12).
const HEADER_LEN: usize = 8 + 4 + 12 + SALT_LEN + NONCE_LEN;

/// Default Argon2id memory (KiB) — ~19 MiB (OWASP-ish).
pub const DEFAULT_M_KIB: u32 = 19_456;
/// Default Argon2id time cost.
pub const DEFAULT_T_COST: u32 = 2;
/// Default Argon2id parallelism.
pub const DEFAULT_P_COST: u32 = 1;

/// True when `bytes` start with the encrypted-container magic.
pub fn is_encrypted(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && &bytes[..8] == RLX_ENC_MAGIC
}

/// Encrypt a full plaintext `*.rlx` blob (must start with `RLXBAKE1`).
pub fn encrypt_bytes(plaintext: &[u8], password: &str) -> Result<Vec<u8>> {
    encrypt_bytes_with_params(
        plaintext,
        password,
        DEFAULT_M_KIB,
        DEFAULT_T_COST,
        DEFAULT_P_COST,
    )
}

/// Encrypt with explicit Argon2id parameters (for tests / tuning).
pub fn encrypt_bytes_with_params(
    plaintext: &[u8],
    password: &str,
    m_kib: u32,
    t_cost: u32,
    p_cost: u32,
) -> Result<Vec<u8>> {
    if password.is_empty() {
        bail!("encryption password must not be empty");
    }
    if plaintext.len() < 8 {
        bail!("plaintext too short to be a *.rlx blob");
    }

    let mut salt = [0u8; SALT_LEN];
    let mut nonce_bytes = [0u8; NONCE_LEN];
    OsRng.fill_bytes(&mut salt);
    OsRng.fill_bytes(&mut nonce_bytes);

    let key = derive_key(password.as_bytes(), &salt, m_kib, t_cost, p_cost)?;
    let cipher = ChaCha20Poly1305::new_from_slice(&key)
        .map_err(|e| anyhow::anyhow!("chacha20 key: {e}"))?;
    let nonce = Nonce::from_slice(&nonce_bytes);
    let ciphertext = cipher
        .encrypt(nonce, plaintext)
        .map_err(|e| anyhow::anyhow!("encrypt failed: {e}"))?;

    let mut out = Vec::with_capacity(HEADER_LEN + ciphertext.len());
    out.extend_from_slice(RLX_ENC_MAGIC);
    out.extend_from_slice(&RLX_ENC_VERSION.to_le_bytes());
    out.extend_from_slice(&m_kib.to_le_bytes());
    out.extend_from_slice(&t_cost.to_le_bytes());
    out.extend_from_slice(&p_cost.to_le_bytes());
    out.extend_from_slice(&salt);
    out.extend_from_slice(&nonce_bytes);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

/// Decrypt an encrypted `*.rlx` blob back to the plaintext `RLXBAKE1` bytes.
pub fn decrypt_bytes(encrypted: &[u8], password: &str) -> Result<Vec<u8>> {
    if password.is_empty() {
        bail!("decryption password must not be empty");
    }
    if !is_encrypted(encrypted) {
        bail!(
            "not an encrypted *.rlx (expected RLXENC01, got {:?})",
            encrypted
                .get(..8)
                .map(|b| String::from_utf8_lossy(b).into_owned())
                .unwrap_or_default()
        );
    }
    if encrypted.len() < HEADER_LEN + 16 {
        bail!("encrypted *.rlx too short");
    }
    let ver = u32::from_le_bytes(encrypted[8..12].try_into().unwrap());
    if ver != RLX_ENC_VERSION {
        bail!("encrypted *.rlx version {ver} != {RLX_ENC_VERSION}");
    }
    let m_kib = u32::from_le_bytes(encrypted[12..16].try_into().unwrap());
    let t_cost = u32::from_le_bytes(encrypted[16..20].try_into().unwrap());
    let p_cost = u32::from_le_bytes(encrypted[20..24].try_into().unwrap());
    let salt = &encrypted[24..24 + SALT_LEN];
    let nonce_bytes = &encrypted[24 + SALT_LEN..HEADER_LEN];
    let ciphertext = &encrypted[HEADER_LEN..];

    let key = derive_key(password.as_bytes(), salt, m_kib, t_cost, p_cost)?;
    let cipher = ChaCha20Poly1305::new_from_slice(&key)
        .map_err(|e| anyhow::anyhow!("chacha20 key: {e}"))?;
    let nonce = Nonce::from_slice(nonce_bytes);
    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| anyhow::anyhow!("decrypt failed (wrong password or corrupt file)"))
}

fn derive_key(
    password: &[u8],
    salt: &[u8],
    m_kib: u32,
    t_cost: u32,
    p_cost: u32,
) -> Result<[u8; KEY_LEN]> {
    let params = Params::new(m_kib, t_cost, p_cost, Some(KEY_LEN))
        .map_err(|e| anyhow::anyhow!("argon2 params: {e}"))?;
    let argon2 = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let mut key = [0u8; KEY_LEN];
    argon2
        .hash_password_into(password, salt, &mut key)
        .map_err(|e| anyhow::anyhow!("argon2 derive: {e}"))
        .context("deriving encryption key")?;
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let plain = b"RLXBAKE1\x02\x00\x00\x00hello-payload-bytes";
        // Fast params for unit tests.
        let enc =
            encrypt_bytes_with_params(plain, "s3cret", 8, 1, 1).expect("encrypt");
        assert!(is_encrypted(&enc));
        assert_ne!(&enc[..8], b"RLXBAKE1");
        let back = decrypt_bytes(&enc, "s3cret").expect("decrypt");
        assert_eq!(back, plain);
    }

    #[test]
    fn wrong_password_fails() {
        let plain = b"RLXBAKE1\x02\x00\x00\x00data";
        let enc = encrypt_bytes_with_params(plain, "right", 8, 1, 1).unwrap();
        assert!(decrypt_bytes(&enc, "wrong").is_err());
    }
}
