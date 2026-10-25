// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! The op table as data, for generators and drift checks.
//!
//! [`GRAPH_OPS`] is built from the same [`for_each_graph_op!`](crate::for_each_graph_op) rows as
//! [`crate::ops`], so it cannot disagree with what the bindings expose.
//! `rlx-bindgen` prints the C ABI, Swift and `.pyi` from it; the per-language
//! name rules live here so every generator and runtime installer agrees.

/// One argument of an op.
#[derive(Clone, Copy, Debug)]
pub struct ArgSpec {
    pub name: &'static str,
    /// The kind (see [`crate::kinds`]).
    pub kind: &'static str,
    /// The table default as written (`"-1"`, `"\"causal\""`, `"[1, 1]"`, `"None"`).
    pub default: Option<&'static str>,
    /// Part of the trailing options group.
    pub option: bool,
    /// Legacy option keys JS also reads.
    pub aliases: &'static [&'static str],
}

/// One op-table row.
#[derive(Clone, Copy, Debug)]
pub struct OpSpec {
    /// Canonical snake_case name.
    pub name: &'static str,
    pub doc: &'static str,
    /// `(language, name)` renames: `js`, `py`, `c`, `swift`, `js_also`, `py_also`.
    pub renames: &'static [(&'static str, &'static str)],
    pub args: &'static [ArgSpec],
    /// Return kind: `node`, `pair`, `nodes`, `unit`, `shape`, `strings`, `usize`, `string`.
    pub ret: &'static str,
}

impl OpSpec {
    fn rename(&self, lang: &str) -> Option<&'static str> {
        self.renames
            .iter()
            .find(|(l, _)| *l == lang)
            .map(|(_, n)| *n)
    }

    /// Positional (non-option) arguments.
    pub fn positional(&self) -> impl Iterator<Item = &ArgSpec> {
        self.args.iter().filter(|a| !a.option)
    }

    /// The trailing options group.
    pub fn options(&self) -> impl Iterator<Item = &ArgSpec> {
        self.args.iter().filter(|a| a.option)
    }

    pub fn has_options(&self) -> bool {
        self.args.iter().any(|a| a.option)
    }

    /// The JS method name(s): a `js` rename or the camelCased name, plus `js_also`.
    pub fn js_names(&self) -> Vec<String> {
        let mut out = vec![
            self.rename("js")
                .map(str::to_string)
                .unwrap_or_else(|| camel(self.name)),
        ];
        out.extend(
            self.renames
                .iter()
                .filter(|(l, _)| *l == "js_also")
                .map(|(_, n)| n.to_string()),
        );
        out
    }

    /// Extra Python names (the canonical name is always exposed too).
    pub fn py_aliases(&self) -> Vec<&'static str> {
        self.renames
            .iter()
            .filter(|(l, _)| *l == "py" || *l == "py_also")
            .map(|(_, n)| *n)
            .collect()
    }

    /// The C symbol suffix: `rlx_graph_<c_name>`.
    pub fn c_name(&self) -> String {
        self.rename("c")
            .map(str::to_string)
            .unwrap_or_else(|| self.name.trim_end_matches('_').to_string())
    }

    /// The Swift method name.
    pub fn swift_name(&self) -> String {
        self.rename("swift")
            .or_else(|| self.rename("js"))
            .map(str::to_string)
            .unwrap_or_else(|| camel(self.name))
    }
}

/// `attention_kind_vdim` → `attentionKindVdim`; a trailing `_` (a Python
/// keyword escape such as `where_`) is dropped.
pub fn camel(snake: &str) -> String {
    let mut out = String::with_capacity(snake.len());
    let mut upper = false;
    for c in snake.trim_end_matches('_').chars() {
        if c == '_' {
            upper = true;
        } else if upper {
            out.extend(c.to_uppercase());
            upper = false;
        } else {
            out.push(c);
        }
    }
    out
}

macro_rules! __opt_str {
    () => {
        None
    };
    ($d:expr) => {
        Some(stringify!($d))
    };
}

macro_rules! __ret_name {
    () => {
        "node"
    };
    ($r:ident) => {
        stringify!($r)
    };
}

macro_rules! manifest_rows {
    ( [$($extra:tt)*]
      $(
        $(#[doc = $doc:literal])*
        $name:ident $( [ $( $rk:ident = $rv:literal ),* $(,)? ] )?
        ( $( $arg:ident : $kind:ident $( = $def:expr )? ),* $(,)?
          $( ; { $( $oarg:ident $( | $oal:ident )* : $okind:ident $( = $odef:expr )? ),* $(,)? } )? )
        $( -> $ret:ident )?
        => | $g:ident | $body:block ;
      )*
    ) => {
        /// Every op-table row, in table order.
        pub static GRAPH_OPS: &[OpSpec] = &[
            $(
                OpSpec {
                    name: stringify!($name),
                    doc: concat!($( $doc, "\n", )*),
                    renames: &[ $( $( (stringify!($rk), $rv) ),* )? ],
                    args: &[
                        $( ArgSpec {
                            name: stringify!($arg),
                            kind: stringify!($kind),
                            default: __opt_str!($($def)?),
                            option: false,
                            aliases: &[],
                        }, )*
                        $( $( ArgSpec {
                            name: stringify!($oarg),
                            kind: stringify!($okind),
                            default: __opt_str!($($odef)?),
                            option: true,
                            aliases: &[ $( stringify!($oal) ),* ],
                        }, )* )?
                    ],
                    ret: __ret_name!($($ret)?),
                },
            )*
        ];
    };
}

crate::for_each_graph_op!(manifest_rows);

/// Look a row up by canonical name.
pub fn op(name: &str) -> Option<&'static OpSpec> {
    GRAPH_OPS.iter().find(|o| o.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique_in_every_language() {
        use std::collections::HashSet;
        let mut canon = HashSet::new();
        let mut js = HashSet::new();
        let mut c = HashSet::new();
        for o in GRAPH_OPS {
            assert!(canon.insert(o.name), "duplicate op {}", o.name);
            for n in o.js_names() {
                assert!(js.insert(n.clone()), "duplicate JS name {n}");
            }
            assert!(c.insert(o.c_name()), "duplicate C name {}", o.c_name());
        }
    }

    #[test]
    fn camel_case_matches_the_published_js_names() {
        assert_eq!(camel("attention_kind_vdim"), "attentionKindVdim");
        assert_eq!(camel("resize_nearest_2x"), "resizeNearest2x");
        assert_eq!(camel("where_"), "where");
        assert_eq!(
            op("batch_norm_inference").unwrap().js_names(),
            ["batchNorm"]
        );
        assert_eq!(op("rope").unwrap().py_aliases(), ["rope_n"]);
    }

    #[test]
    fn defaults_follow_required_arguments() {
        // Python (and Swift) reject a required parameter after a defaulted one.
        for o in GRAPH_OPS {
            let mut seen_default = false;
            for a in o.args {
                if a.default.is_some() {
                    seen_default = true;
                } else {
                    assert!(
                        !seen_default,
                        "{}: '{}' has no default but follows one that does",
                        o.name, a.name
                    );
                }
            }
        }
    }

    #[test]
    fn every_kind_is_known() {
        let known = [
            "node",
            "opt_node",
            "nodes",
            "f32",
            "f64",
            "i32",
            "i64",
            "u32",
            "u64",
            "usize",
            "bool",
            "usizes",
            "i64s",
            "f32s",
            "bytes",
            "str",
            "opt_usize",
            "opt_f32",
            "opt_usizes",
            "opt_str",
            "opt_activation",
            "pair",
            "opt_pair",
            "triple",
            "f32x3",
            "pads",
            "bands",
            "sos",
            "graph",
            "opt_graph",
            "activation",
            "binop",
        ];
        let enums: Vec<&str> = crate::enums::ALL_KINDS.iter().map(|(k, _)| *k).collect();
        for o in GRAPH_OPS {
            for a in o.args {
                assert!(
                    known.contains(&a.kind) || enums.contains(&a.kind),
                    "{}: unknown kind '{}' for '{}'",
                    o.name,
                    a.kind,
                    a.name
                );
            }
        }
    }
}
