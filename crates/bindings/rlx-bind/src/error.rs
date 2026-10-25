// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The one error type every binding maps onto its own exceptions.
//!
//! The kind decides the exception class, the message is shown as-is:
//!
//! | kind      | JS              | Python         | C status          |
//! |-----------|-----------------|----------------|-------------------|
//! | `Type`    | `TypeError`     | `ValueError`   | `RLX_ERR_TYPE`    |
//! | `Range`   | `RangeError`    | `ValueError`   | `RLX_ERR_RANGE`   |
//! | `Runtime` | `InternalError` | `RuntimeError` | `RLX_ERR_RUNTIME` |
//! | `Panic`   | `InternalError` | `RuntimeError` | `RLX_ERR_PANIC`   |
//! | `NotFound`| `InternalError` | `FileNotFoundError` | `RLX_ERR_NOT_FOUND` |

use std::fmt;

/// Which exception class an [`ApiError`] becomes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// A wrong argument: unknown name, bad shape, wrong arity.
    Type,
    /// An index or value outside what the graph or dtype holds.
    Range,
    /// The call was well-formed but could not be carried out (I/O, a backend
    /// not compiled in, a consumed graph).
    Runtime,
    /// A Rust assertion inside rlx, caught at the boundary.
    Panic,
    /// A file (or archive entry) that does not exist.
    NotFound,
}

impl ErrorKind {
    /// Stable code for the C ABI (`RLX_ERR_*`). `0` is success.
    pub fn code(self) -> i32 {
        match self {
            ErrorKind::Type => -1,
            ErrorKind::Range => -2,
            ErrorKind::Runtime => -3,
            ErrorKind::Panic => -4,
            ErrorKind::NotFound => -5,
        }
    }
}

/// An error raised by a binding call, with the message the caller sees.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ApiError {
    pub kind: ErrorKind,
    pub message: String,
}

impl ApiError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub fn type_(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Type, message)
    }

    pub fn range(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Range, message)
    }

    pub fn runtime(message: impl Into<String>) -> Self {
        Self::new(ErrorKind::Runtime, message)
    }

    /// An I/O failure: `NotFound` when the cause chain holds a missing file,
    /// `Runtime` otherwise. `what` names the call (`load_gguf('m.gguf')`).
    pub fn io(what: &str, e: &(dyn std::error::Error + 'static)) -> Self {
        let mut missing = false;
        let mut cause: Option<&(dyn std::error::Error + 'static)> = Some(e);
        while let Some(c) = cause {
            if c.downcast_ref::<std::io::Error>()
                .is_some_and(|io| io.kind() == std::io::ErrorKind::NotFound)
            {
                missing = true;
            }
            cause = c.source();
        }
        let text = e.to_string();
        let mut message = format!("{what}: {text}");
        let mut source = e.source();
        while let Some(s) = source {
            message.push_str(&format!(": {s}"));
            source = s.source();
        }
        if missing || text.contains("No such file") {
            Self::new(ErrorKind::NotFound, message)
        } else {
            Self::runtime(message)
        }
    }
}

impl fmt::Display for ApiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ApiError {}

/// Every binding call returns this.
pub type ApiResult<T> = Result<T, ApiError>;

/// `return Err(ApiError::type_(format!(…)))`.
#[macro_export]
macro_rules! bail {
    ($($fmt:tt)*) => {
        return Err($crate::ApiError::type_(format!($($fmt)*)))
    };
}
