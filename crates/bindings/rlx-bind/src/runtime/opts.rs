// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Reading structured options from a [`serde_json::Value`].
//!
//! Every binding hands its options object over as JSON — a JS object, a
//! Python dict, a C string, a Swift `Codable`. Keys match ignoring case, `_`
//! and `-`, so JS's `dataMovement` and Python's `data_movement` are the same
//! option. [`Opts::finish`] rejects anything not read, so a misspelt option is
//! an error rather than a setting that silently never applied.

use std::cell::RefCell;

use serde_json::Value;

use crate::enums::key;
use crate::{ApiError, ApiResult};

/// A read-tracking view of an options object (or of nothing).
///
/// `what` is a noun naming the options — `"sampling"`, `"optimizer"`,
/// `"fusion"` — so errors read "unknown sampling option 'temprature'".
pub struct Opts<'a> {
    what: &'a str,
    map: Option<&'a serde_json::Map<String, Value>>,
    read: RefCell<Vec<String>>,
}

impl<'a> Opts<'a> {
    /// `v` must be an object or null.
    pub fn new(what: &'a str, v: &'a Value) -> ApiResult<Self> {
        let map = match v {
            Value::Null => None,
            Value::Object(m) => Some(m),
            other => {
                return Err(ApiError::type_(format!(
                    "{what} options: expected an object, got {}",
                    type_name(other)
                )));
            }
        };
        Ok(Self {
            what,
            map,
            read: RefCell::new(Vec::new()),
        })
    }

    /// The raw value of option `name`, if given (and not null).
    pub fn get(&self, name: &str) -> Option<&'a Value> {
        let want = key(name);
        self.read.borrow_mut().push(want.clone());
        self.map?
            .iter()
            .find(|(k, _)| key(k) == want)
            .map(|(_, v)| v)
            .filter(|v| !v.is_null())
    }

    pub fn bool(&self, name: &str, default: bool) -> ApiResult<bool> {
        match self.get(name) {
            None => Ok(default),
            Some(Value::Bool(b)) => Ok(*b),
            Some(other) => Err(self.wrong(name, "a boolean", other)),
        }
    }

    pub fn f64(&self, name: &str) -> ApiResult<Option<f64>> {
        match self.get(name) {
            None => Ok(None),
            Some(Value::Number(n)) => Ok(n.as_f64()),
            Some(other) => Err(self.wrong(name, "a number", other)),
        }
    }

    pub fn f32(&self, name: &str, default: f32) -> ApiResult<f32> {
        Ok(self.f64(name)?.map(|v| v as f32).unwrap_or(default))
    }

    pub fn u64(&self, name: &str) -> ApiResult<Option<u64>> {
        match self.get(name) {
            None => Ok(None),
            Some(Value::Number(n)) => match n.as_u64() {
                Some(v) => Ok(Some(v)),
                None => match n.as_f64() {
                    Some(f) if f >= 0.0 && f.fract() == 0.0 && f < 2f64.powi(53) => {
                        Ok(Some(f as u64))
                    }
                    _ => Err(self.wrong(name, "a non-negative integer", &Value::Number(n.clone()))),
                },
            },
            Some(other) => Err(self.wrong(name, "a non-negative integer", other)),
        }
    }

    pub fn usize(&self, name: &str, default: usize) -> ApiResult<usize> {
        Ok(self.u64(name)?.map(|v| v as usize).unwrap_or(default))
    }

    pub fn str(&self, name: &str) -> ApiResult<Option<&'a str>> {
        match self.get(name) {
            None => Ok(None),
            Some(Value::String(s)) => Ok(Some(s)),
            Some(other) => Err(self.wrong(name, "a string", other)),
        }
    }

    /// Error on any key that no getter asked for.
    pub fn finish(self) -> ApiResult<()> {
        let Some(map) = self.map else {
            return Ok(());
        };
        let read = self.read.into_inner();
        let unknown: Vec<&str> = map
            .keys()
            .filter(|k| !read.contains(&key(k)))
            .map(String::as_str)
            .collect();
        if unknown.is_empty() {
            return Ok(());
        }
        let mut known = read;
        known.sort();
        known.dedup();
        Err(ApiError::type_(format!(
            "unknown {} option{} {} (known: {})",
            self.what,
            if unknown.len() == 1 { "" } else { "s" },
            unknown
                .iter()
                .map(|k| format!("'{k}'"))
                .collect::<Vec<_>>()
                .join(", "),
            known.join(", ")
        )))
    }

    fn wrong(&self, name: &str, expected: &str, got: &Value) -> ApiError {
        ApiError::type_(format!(
            "{} option '{name}': expected {expected}, got {}",
            self.what,
            type_name(got)
        ))
    }
}

pub fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn keys_match_across_naming_conventions() {
        let v = json!({"dataMovement": "f16", "skip_fusion": true});
        let o = Opts::new("t", &v).unwrap();
        assert_eq!(o.str("data_movement").unwrap(), Some("f16"));
        assert!(o.bool("skipFusion", false).unwrap());
        o.finish().unwrap();
    }

    #[test]
    fn an_unread_option_is_an_error() {
        let v = json!({"lr": 0.1, "learning_rate": 0.1});
        let o = Opts::new("optimizer", &v).unwrap();
        o.f32("lr", 0.0).unwrap();
        let e = o.finish().unwrap_err();
        assert!(
            e.message
                .contains("unknown optimizer option 'learning_rate'"),
            "{e}"
        );
    }
}
