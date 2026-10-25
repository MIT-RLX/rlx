// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Checks on what a caller feeds a [`CompiledGraph`](crate::CompiledGraph).
//!
//! Every one of these used to return plausible numbers instead of an error:
//! a misspelled input name, a buffer one row short, a weight never uploaded,
//! no inputs at all — each read the arena's zeros where the data should have
//! been. The graph is known at compile time, so the boundary can say what is
//! wrong, by name:
//!
//! * **errors** (a `run` panics, [`try_run`](crate::CompiledGraph::try_run)
//!   returns `Err`): an input name the graph does not declare, and a buffer
//!   whose element count differs from a static input's or param's shape;
//! * **warning, once per compiled graph**: declared inputs not supplied and
//!   params never set. Some callers rely on those reading zero, so this names
//!   them rather than refusing;
//!   [`check_bindings`](crate::CompiledGraph::check_bindings) returns the same
//!   list as an error.
//!
//! Unknown *param* names are accepted: loading a whole checkpoint into a
//! graph that uses part of it is ordinary. A *param* name passed to `run` is
//! uploaded with `set_param` (backends used to ignore it silently, so a
//! training loop that fed its weights through `run` never changed them).

use std::collections::{HashMap, HashSet};

use rlx_ir::{DType, Graph, Op};

/// What is wrong with a call's inputs or a graph's bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IoError {
    /// An input name the graph does not declare.
    UnknownInput {
        name: String,
        /// Declared inputs with the most similar names.
        closest: Vec<String>,
    },
    /// A buffer whose element count is not the declared shape's.
    WrongLength {
        name: String,
        kind: &'static str,
        got: usize,
        want: usize,
        shape: Vec<usize>,
    },
    /// An output name the graph does not give any output.
    UnknownOutput {
        name: String,
        /// Output names most like it.
        closest: Vec<String>,
    },
    /// Declared inputs not supplied and params never set.
    Unbound {
        inputs: Vec<String>,
        params: Vec<String>,
    },
}

impl std::fmt::Display for IoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IoError::UnknownInput { name, closest } => {
                write!(f, "rlx: the graph has no input `{name}`")?;
                if !closest.is_empty() {
                    write!(f, " (did you mean {}?)", quote_list(closest))?;
                }
                Ok(())
            }
            IoError::UnknownOutput { name, closest } => {
                write!(f, "rlx: the graph has no output named `{name}`")?;
                if !closest.is_empty() {
                    write!(f, " (did you mean {}?)", quote_list(closest))?;
                }
                Ok(())
            }
            IoError::WrongLength {
                name,
                kind,
                got,
                want,
                shape,
            } => write!(
                f,
                "rlx: {kind} `{name}` got {got} values; its shape {shape:?} holds {want}"
            ),
            IoError::Unbound { inputs, params } => {
                write!(f, "rlx: unbound")?;
                if !inputs.is_empty() {
                    write!(f, " inputs {}", quote_list(inputs))?;
                }
                if !params.is_empty() {
                    if !inputs.is_empty() {
                        write!(f, ",")?;
                    }
                    write!(f, " params {} (never set)", quote_list(params))?;
                }
                write!(f, " — they read zeros")
            }
        }
    }
}

impl std::error::Error for IoError {}

fn quote_list(xs: &[String]) -> String {
    const SHOW: usize = 6;
    let mut s: Vec<String> = xs.iter().take(SHOW).map(|x| format!("`{x}`")).collect();
    if xs.len() > SHOW {
        s.push(format!("… {} more", xs.len() - SHOW));
    }
    s.join(", ")
}

/// Edit distance, for "did you mean".
fn distance(a: &str, b: &str) -> usize {
    let (a, b): (Vec<char>, Vec<char>) = (a.chars().collect(), b.chars().collect());
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, cb) in b.iter().enumerate() {
            let sub = prev[j] + usize::from(!ca.eq_ignore_ascii_case(cb));
            cur[j + 1] = sub.min(prev[j + 1] + 1).min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Up to three of `names` close to `name`, nearest first.
fn closest<'a>(name: &str, names: impl IntoIterator<Item = &'a String>) -> Vec<String> {
    let mut c: Vec<(usize, &String)> = names
        .into_iter()
        .map(|k| (distance(name, k), k))
        .filter(|(d, k)| *d <= 2 + k.len() / 4)
        .collect();
    c.sort();
    c.into_iter().take(3).map(|(_, k)| k.clone()).collect()
}

/// A declared input, param or output of a compiled graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Empty for an output the graph does not name.
    pub name: String,
    pub dtype: DType,
    /// Static shape; `None` when a dim is symbolic.
    pub shape: Option<Vec<usize>>,
}

#[derive(Debug, Clone)]
struct Tensor {
    dtype: DType,
    /// Static shape; `None` when any dim is symbolic.
    shape: Option<Vec<usize>>,
    /// For a cache that an `Op::KvAppend` writes into in place: the elements
    /// of one row along its axis. Such an input declares spare rows for the
    /// write and is fed only its filled rows, so any whole number of rows up
    /// to the declared size is a valid length.
    row: Option<usize>,
}

/// The declared boundary of one compiled graph, and what has been bound.
#[derive(Debug, Clone, Default)]
pub(crate) struct IoSpec {
    inputs: HashMap<String, Tensor>,
    params: HashMap<String, Tensor>,
    /// Outputs in order, named or not.
    outputs: Vec<TensorInfo>,
    /// Params set (any setter) or bound at compile time.
    set_params: HashSet<String>,
    /// Inputs supplied by a persistent handle.
    bound_inputs: HashSet<String>,
    warned: bool,
}

impl IoSpec {
    pub(crate) fn from_graph(graph: &Graph) -> Self {
        let mut s = Self::default();
        let tensor = |shape: &rlx_ir::Shape| Tensor {
            dtype: shape.dtype(),
            shape: shape
                .dims()
                .iter()
                .map(|d| d.is_static().then(|| d.unwrap_static()))
                .collect(),
            row: None,
        };
        // Inputs a `KvAppend` appends to, and the row size along its axis.
        let mut rows: HashMap<rlx_ir::NodeId, usize> = HashMap::new();
        for n in graph.nodes() {
            if let (Op::KvAppend { axis, .. }, Some(&cache)) = (&n.op, n.inputs.first()) {
                let dims = graph.shape(cache).dims();
                let row = dims.iter().skip(axis + 1).try_fold(1usize, |acc, d| {
                    d.is_static().then(|| acc * d.unwrap_static())
                });
                if let Some(row) = row {
                    rows.insert(cache, row);
                }
            }
        }
        s.outputs = graph
            .outputs
            .iter()
            .enumerate()
            .map(|(i, &o)| {
                let t = tensor(graph.shape(o));
                TensorInfo {
                    name: graph.output_name(i).unwrap_or_default().to_string(),
                    dtype: t.dtype,
                    shape: t.shape,
                }
            })
            .collect();
        for n in graph.nodes() {
            let t = || tensor(&n.shape);
            match &n.op {
                Op::Input { name } => {
                    let row = rows.get(&n.id).copied();
                    s.inputs.insert(name.clone(), Tensor { row, ..t() });
                }
                Op::Param { name } => {
                    s.params.insert(name.clone(), t());
                }
                _ => {}
            }
        }
        s
    }

    fn infos(m: &HashMap<String, Tensor>) -> Vec<TensorInfo> {
        let mut v: Vec<TensorInfo> = m
            .iter()
            .map(|(n, t)| TensorInfo {
                name: n.clone(),
                dtype: t.dtype,
                shape: t.shape.clone(),
            })
            .collect();
        v.sort_by(|a, b| a.name.cmp(&b.name));
        v
    }

    /// `name` is a param and not also an input.
    pub(crate) fn is_param_only(&self, name: &str) -> bool {
        self.params.contains_key(name) && !self.inputs.contains_key(name)
    }

    /// Check a param upload's length without recording it.
    pub(crate) fn check_param_len(&self, name: &str, elems: usize) -> Result<(), IoError> {
        let mut probe = IoSpec {
            params: self.params.clone(),
            ..Default::default()
        };
        probe.note_param(name, Some(elems))
    }

    pub(crate) fn input_infos(&self) -> Vec<TensorInfo> {
        Self::infos(&self.inputs)
    }

    pub(crate) fn param_infos(&self) -> Vec<TensorInfo> {
        Self::infos(&self.params)
    }

    pub(crate) fn output_infos(&self) -> &[TensorInfo] {
        &self.outputs
    }

    /// The position of the output named `name`.
    pub(crate) fn output_index(&self, name: &str) -> Result<usize, IoError> {
        self.outputs
            .iter()
            .position(|o| o.name == name && !name.is_empty())
            .ok_or_else(|| IoError::UnknownOutput {
                name: name.to_string(),
                closest: closest(
                    name,
                    self.outputs
                        .iter()
                        .map(|o| &o.name)
                        .filter(|n| !n.is_empty()),
                ),
            })
    }

    /// Params bound at compile time (`CompileOptions::param_bindings`).
    pub(crate) fn mark_params<'a>(&mut self, names: impl IntoIterator<Item = &'a String>) {
        self.set_params.extend(names.into_iter().cloned());
    }

    pub(crate) fn mark_all_params(&mut self) {
        self.set_params.extend(self.params.keys().cloned());
    }

    pub(crate) fn mark_input_bound(&mut self, name: &str) {
        self.bound_inputs.insert(name.to_string());
    }

    /// A param upload of `elems` elements (`None`: not checkable, e.g. packed
    /// bytes).
    pub(crate) fn note_param(&mut self, name: &str, elems: Option<usize>) -> Result<(), IoError> {
        if let (Some(t), Some(got)) = (self.params.get(name), elems)
            && let Some(shape) = &t.shape
        {
            let want: usize = shape.iter().product();
            if got != want {
                return Err(IoError::WrongLength {
                    name: name.to_string(),
                    kind: "param",
                    got,
                    want,
                    shape: shape.clone(),
                });
            }
        }
        self.set_params.insert(name.to_string());
        Ok(())
    }

    /// Check one call's inputs: `(name, element count or None)`.
    pub(crate) fn check_inputs<'a>(
        &self,
        inputs: impl IntoIterator<Item = (&'a str, Option<usize>)>,
    ) -> Result<(), IoError> {
        for (name, elems) in inputs {
            let Some(t) = self.inputs.get(name) else {
                // A param passed to `run` is uploaded with `set_param` by
                // `CompiledGraph` before this check; anything else is a typo.
                if self.params.contains_key(name) {
                    continue;
                }
                return Err(IoError::UnknownInput {
                    name: name.to_string(),
                    closest: closest(name, self.inputs.keys()),
                });
            };
            if let (Some(shape), Some(got)) = (&t.shape, elems) {
                let want: usize = shape.iter().product();
                let filled_rows = t
                    .row
                    .is_some_and(|row| row > 0 && got <= want && got % row == 0);
                if got != want && !filled_rows {
                    return Err(IoError::WrongLength {
                        name: name.to_string(),
                        kind: "input",
                        got,
                        want,
                        shape: shape.clone(),
                    });
                }
            }
        }
        Ok(())
    }

    /// Element count of an `&[f32]` buffer of `floats` values for input
    /// `name`: a complex element is two `f32`s (re, im), anything else one.
    pub(crate) fn f32_elems(&self, name: &str, floats: usize) -> usize {
        match self.inputs.get(name).map(|t| t.dtype) {
            Some(DType::C64 | DType::C128) if floats.is_multiple_of(2) => floats / 2,
            _ => floats,
        }
    }

    /// Element count of `bytes` for input `name` when they are in its
    /// declared dtype.
    pub(crate) fn typed_elems(&self, name: &str, bytes: usize, dtype: DType) -> Option<usize> {
        let t = self.inputs.get(name)?;
        (t.dtype == dtype && dtype.size_bytes() > 0).then(|| bytes / dtype.size_bytes())
    }

    /// Inputs a call left unsupplied and params never set.
    pub(crate) fn unbound<'a>(&self, supplied: impl IntoIterator<Item = &'a str>) -> IoError {
        let supplied: HashSet<&str> = supplied.into_iter().collect();
        let mut inputs: Vec<String> = self
            .inputs
            .keys()
            .filter(|k| !supplied.contains(k.as_str()) && !self.bound_inputs.contains(*k))
            .cloned()
            .collect();
        let mut params: Vec<String> = self
            .params
            .keys()
            .filter(|k| !self.set_params.contains(*k))
            .cloned()
            .collect();
        inputs.sort();
        params.sort();
        IoError::Unbound { inputs, params }
    }

    /// Print the unbound names once per compiled graph.
    pub(crate) fn warn_unbound_once<'a>(&mut self, supplied: impl IntoIterator<Item = &'a str>) {
        if self.warned {
            return;
        }
        if let IoError::Unbound { inputs, params } = self.unbound(supplied)
            && (!inputs.is_empty() || !params.is_empty())
        {
            self.warned = true;
            eprintln!(
                "{} (warned once per compiled graph; `allow_unbound` marks names that \
                 should read zeros)",
                IoError::Unbound { inputs, params }
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rlx_ir::{Dim, Shape};

    fn spec() -> IoSpec {
        let mut g = Graph::new("t");
        g.input("tokens", Shape::new(&[2, 4], DType::F32));
        g.input(
            "x",
            Shape::from_dims(&[Dim::Dynamic(1), Dim::Static(3)], DType::F32),
        );
        g.param("w", Shape::new(&[4, 3], DType::F32));
        IoSpec::from_graph(&g)
    }

    #[test]
    fn unknown_input_suggests_the_closest_name() {
        let e = spec().check_inputs([("token", Some(8))]).unwrap_err();
        assert_eq!(
            e.to_string(),
            "rlx: the graph has no input `token` (did you mean `tokens`?)"
        );
    }

    #[test]
    fn wrong_length_names_the_shape() {
        let e = spec().check_inputs([("tokens", Some(3))]).unwrap_err();
        assert_eq!(
            e.to_string(),
            "rlx: input `tokens` got 3 values; its shape [2, 4] holds 8"
        );
    }

    /// A decode step's cache declares a spare row for `KvAppend` to write
    /// in place and is fed only the rows it holds (`rlx_flow::blocks::
    /// kv_spare_rows`); any other input must be fed whole.
    #[test]
    fn a_kv_append_cache_takes_its_filled_rows() {
        let mut g = Graph::new("decode");
        let s = |d: &[usize]| Shape::new(d, DType::F32);
        let past = g.input("past_k_0", s(&[1, 5, 16]));
        let row = g.input("new_k", s(&[1, 1, 16]));
        g.add_node(
            Op::KvAppend { axis: 1, pos: 4 },
            vec![past, row],
            s(&[1, 5, 16]),
        );
        let spec = IoSpec::from_graph(&g);
        for rows in [0, 3, 4, 5] {
            assert!(spec.check_inputs([("past_k_0", Some(rows * 16))]).is_ok());
        }
        // Past the declared size, or not a whole row.
        assert!(spec.check_inputs([("past_k_0", Some(96))]).is_err());
        assert!(spec.check_inputs([("past_k_0", Some(70))]).is_err());
        // The appended row is an ordinary input.
        assert!(spec.check_inputs([("new_k", Some(8))]).is_err());
    }

    #[test]
    fn symbolic_inputs_and_params_by_name_pass() {
        let s = spec();
        assert!(s.check_inputs([("x", Some(999)), ("w", Some(12))]).is_ok());
    }

    #[test]
    fn short_param_is_rejected_and_unknown_param_accepted() {
        let mut s = spec();
        assert!(s.note_param("w", Some(5)).is_err());
        assert!(s.note_param("not_in_graph", Some(5)).is_ok());
        assert!(s.note_param("w", Some(12)).is_ok());
    }

    #[test]
    fn describes_the_boundary() {
        let s = spec();
        let names: Vec<String> = s.input_infos().into_iter().map(|t| t.name).collect();
        assert_eq!(names, ["tokens", "x"]);
        let x = &s.input_infos()[1];
        assert_eq!(x.shape, None, "symbolic");
        assert_eq!(s.param_infos()[0].shape, Some(vec![4, 3]));
    }

    #[test]
    fn unbound_lists_missing_inputs_and_params() {
        let mut s = spec();
        assert_eq!(
            s.unbound(["tokens"]).to_string(),
            "rlx: unbound inputs `x`, params `w` (never set) — they read zeros"
        );
        s.note_param("w", None).unwrap();
        s.mark_input_bound("x");
        let IoError::Unbound { inputs, params } = s.unbound(["tokens"]) else {
            unreachable!()
        };
        assert!(inputs.is_empty() && params.is_empty());
    }
}
