// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Lexical scan of Rust source for **operand claims** inside `Op::…` match arms.
//!
//! Backends consume a node's operands by indexing (`node.inputs[2]`) or by
//! asserting a count (`node.inputs.len() == 3`). Either way they are declaring
//! how many operands the op has. Comparing those claims against the IR's
//! declared arity is the only check that tests that table against reality
//! rather than against itself — see the `backend_operand_indexing` gate.
//!
//! This module is the *parser*; the policy (which arities, which exceptions)
//! lives with the gate.
//!
//! # Precision
//!
//! Lexical, not semantic. It resolves literal indices inside a match arm it
//! can attribute to an op; it does not follow helper functions, non-literal
//! indices, or operands rebound through a slice. It finds a class of
//! disagreement, not all of them — a clean scan is not a proof.
//!
//! Every early version of this scanner reported nothing at all while being
//! quietly broken, so the unit tests below pin each failure mode rather than
//! only testing the happy path.

/// What a piece of source implies about an op's operand count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClaimKind {
    /// `node.inputs[k]` / `.get(k)` — needs `k + 1` operands.
    Index(usize),
    /// `node.inputs.len() == n` and friends — asserts `n` are reachable.
    Length(usize),
}

/// A claim about the operand count of whatever op the enclosing arm matched.
#[derive(Debug, Clone)]
pub struct Claim {
    /// 1-based line in the original source.
    pub line: usize,
    /// Ops named by the enclosing arm's pattern. More than one for a shared
    /// arm (`Op::A | Op::B => …`), whose body branches on which it got.
    pub ops: Vec<String>,
    /// Operands this claim implies the op can have.
    pub needs: usize,
    pub kind: ClaimKind,
}

/// Replace comments, string literals and char literals with spaces,
/// **preserving byte offsets** so line numbers computed on the result still
/// refer to the original.
///
/// Handles raw strings (`r"…"`, `r#"…"#`) and distinguishes a char literal
/// (`'x'`, `'\n'`, `'"'`) from a lifetime (`'a`). Getting either wrong is not
/// a cosmetic issue: a stray `'"'` would otherwise open a string that runs to
/// the next quote, blanking real code and making the scan silently blind for
/// the rest of the file.
pub fn blank_noncode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = vec![b' '; b.len()];
    let keep_newlines = |out: &mut Vec<u8>, from: usize, to: usize| {
        for (k, item) in out.iter_mut().enumerate().take(to).skip(from) {
            if b[k] == b'\n' {
                *item = b'\n';
            }
        }
    };
    let mut i = 0;
    while i < b.len() {
        if b[i..].starts_with(b"//") {
            let end = s[i..].find('\n').map_or(b.len(), |p| i + p);
            keep_newlines(&mut out, i, end);
            i = end;
        } else if b[i..].starts_with(b"/*") {
            // Rust block comments nest.
            let mut depth = 0usize;
            let mut j = i;
            while j < b.len() {
                if b[j..].starts_with(b"/*") {
                    depth += 1;
                    j += 2;
                } else if b[j..].starts_with(b"*/") {
                    depth -= 1;
                    j += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    j += 1;
                }
            }
            keep_newlines(&mut out, i, j.min(b.len()));
            i = j.min(b.len());
        } else if let Some(j) = raw_string_end(b, i) {
            keep_newlines(&mut out, i, j);
            i = j;
        } else if b[i] == b'"' {
            let mut j = i + 1;
            while j < b.len() {
                if b[j] == b'\\' {
                    j += 2;
                    continue;
                }
                if b[j] == b'"' {
                    j += 1;
                    break;
                }
                j += 1;
            }
            let j = j.min(b.len());
            keep_newlines(&mut out, i, j);
            i = j;
        } else if let Some(j) = char_literal_end(b, i) {
            i = j; // char literals never span lines
        } else {
            out[i] = b[i];
            i += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// End offset of a raw string starting at `i`, if one starts there.
fn raw_string_end(b: &[u8], i: usize) -> Option<usize> {
    if b[i] != b'r' {
        return None;
    }
    // An identifier char before `r` means this is the tail of a word, not a
    // raw-string prefix (`for`, `iter`, …).
    if i > 0 && (b[i - 1].is_ascii_alphanumeric() || b[i - 1] == b'_') {
        return None;
    }
    let mut j = i + 1;
    let mut hashes = 0usize;
    while j < b.len() && b[j] == b'#' {
        hashes += 1;
        j += 1;
    }
    if j >= b.len() || b[j] != b'"' {
        return None;
    }
    j += 1;
    // Terminator is `"` followed by the same number of `#`.
    while j < b.len() {
        if b[j] == b'"' {
            let mut k = j + 1;
            let mut seen = 0usize;
            while k < b.len() && b[k] == b'#' && seen < hashes {
                seen += 1;
                k += 1;
            }
            if seen == hashes {
                return Some(k);
            }
        }
        j += 1;
    }
    Some(b.len())
}

/// End offset of a char literal starting at `i`, if one starts there.
///
/// `'a` is a lifetime, not a literal: without the closing quote the scan would
/// treat everything up to the next `'` as literal text.
fn char_literal_end(b: &[u8], i: usize) -> Option<usize> {
    if b[i] != b'\'' {
        return None;
    }
    if b.get(i + 1) == Some(&b'\\') {
        // `'\n'`, `'\''`, `'\u{1F600}'` — scan to the closing quote.
        let mut j = i + 2;
        while j < b.len() && b[j] != b'\'' && b[j] != b'\n' {
            j += 1;
        }
        return (b.get(j) == Some(&b'\'')).then_some(j + 1);
    }
    // A single (possibly multi-byte) char followed by `'`.
    let mut j = i + 1;
    if j >= b.len() {
        return None;
    }
    j += 1;
    while j < b.len() && (b[j] & 0xC0) == 0x80 {
        j += 1; // UTF-8 continuation bytes
    }
    (b.get(j) == Some(&b'\'')).then_some(j + 1)
}

/// A `match` arm whose pattern names at least one `Op::…`.
#[derive(Debug, Clone)]
pub struct Arm {
    /// Byte range of the arm body.
    pub start: usize,
    pub end: usize,
    pub ops: Vec<String>,
}

/// Every `Op::…` match arm in `src`, with its brace-matched body range.
///
/// `src` should already be [`blank_noncode`]d.
pub fn arms(src: &str) -> Vec<Arm> {
    let b = src.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(p) = src[i..].find("=>") {
        let at = i + p;
        i = at + 2;
        let pat = &src[pattern_start(b, at)..at];
        let ops = op_names(pat);
        if ops.is_empty() {
            continue;
        }
        let mut k = at + 2;
        while k < b.len() && (b[k] as char).is_whitespace() {
            k += 1;
        }
        if k >= b.len() {
            continue;
        }
        let end = if b[k] == b'{' {
            balanced_end(b, k)
        } else {
            expr_arm_end(b, k)
        };
        out.push(Arm { start: k, end, ops });
    }
    out
}

/// Walk left from an arm's `=>` to where its pattern begins.
///
/// Skips **balanced** groups: a struct pattern `Op::X { a, b } =>` carries its
/// own braces and commas, so a naive backwards search stops inside them and
/// loses the op name — which silently makes the whole scan find nothing. A
/// previous arm's `=>` also ends the pattern; without that stop the walk runs
/// back through earlier arms and collects their ops too.
fn pattern_start(b: &[u8], arrow: usize) -> usize {
    let mut j = arrow as isize - 1;
    let mut depth = 0i32;
    while j >= 0 {
        let c = b[j as usize];
        match c {
            b')' | b']' | b'}' => depth += 1,
            b'(' | b'[' | b'{' => {
                if depth == 0 {
                    break;
                }
                depth -= 1;
            }
            b',' | b';' if depth == 0 => break,
            b'>' if depth == 0 && j > 0 && b[j as usize - 1] == b'=' => break,
            _ => {}
        }
        j -= 1;
    }
    (j + 1) as usize
}

/// Op names in an arm pattern.
///
/// `ReduceOp::Sum` and `BinaryOp::Add` both contain the substring `Op::`;
/// reading those as op names inflates the arm's permitted operand count and
/// hides findings, so a word boundary is required.
fn op_names(pat: &str) -> Vec<String> {
    let pb = pat.as_bytes();
    pat.match_indices("Op::")
        .filter(|(o, _)| *o == 0 || !(pb[o - 1].is_ascii_alphanumeric() || pb[o - 1] == b'_'))
        .map(|(o, _)| {
            pat[o + 4..]
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect::<String>()
        })
        .filter(|n: &String| n.chars().next().is_some_and(char::is_uppercase))
        .collect()
}

fn balanced_end(b: &[u8], open: usize) -> usize {
    let mut d = 0i32;
    let mut e = open;
    while e < b.len() {
        if b[e] == b'{' {
            d += 1;
        } else if b[e] == b'}' {
            d -= 1;
            if d == 0 {
                return e;
            }
        }
        e += 1;
    }
    b.len()
}

/// An expression arm ends at the next top-level comma.
fn expr_arm_end(b: &[u8], from: usize) -> usize {
    let mut d = 0i32;
    let mut e = from;
    while e < b.len() {
        match b[e] {
            b'(' | b'[' | b'{' => d += 1,
            b')' | b']' | b'}' => {
                if d == 0 {
                    return e;
                }
                d -= 1;
            }
            b',' if d == 0 => return e,
            _ => {}
        }
        e += 1;
    }
    b.len()
}

/// Literal operand indices read off **the current node**.
///
/// `graph.node(other).inputs[0]` is a different node's operand list and says
/// nothing about this op's arity, so the `node.` receiver is required.
fn operand_reads(src: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (o, _) in src.match_indices("node.inputs") {
        let rest = &src[o + "node.inputs".len()..];
        let digits: String = if let Some(t) = rest.strip_prefix(".get(") {
            t.chars().take_while(char::is_ascii_digit).collect()
        } else if let Some(t) = rest.strip_prefix('[') {
            t.chars().take_while(char::is_ascii_digit).collect()
        } else {
            continue;
        };
        if let Ok(k) = digits.parse::<usize>() {
            out.push((o, k));
        }
    }
    out
}

/// Operand counts a backend *asserts* rather than indexes.
///
/// Only the asserting forms are used. `len() < n` and `<= n` are guards — they
/// say what the code refuses, not what it expects — so reading them as claims
/// would invent counts the source never asserted.
fn operand_count_claims(src: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    for (o, _) in src.match_indices("node.inputs.len()") {
        let rest = src[o + "node.inputs.len()".len()..].trim_start();
        let (op, tail) = if let Some(t) = rest.strip_prefix("==") {
            ("==", t)
        } else if let Some(t) = rest.strip_prefix(">=") {
            (">=", t)
        } else if let Some(t) = rest.strip_prefix('>') {
            (">", t)
        } else {
            continue;
        };
        let digits: String = tail
            .trim_start()
            .chars()
            .take_while(char::is_ascii_digit)
            .collect();
        let Ok(n) = digits.parse::<usize>() else {
            continue;
        };
        out.push((o, if op == ">" { n + 1 } else { n }));
    }
    out
}

/// Every operand claim in `src`, attributed to the innermost enclosing
/// `Op::…` arm. Claims outside any such arm are dropped — there is no op to
/// hold them to.
pub fn scan(src: &str) -> Vec<Claim> {
    let blanked = blank_noncode(src);
    let arm_list = arms(&blanked);
    let indexed = operand_reads(&blanked)
        .into_iter()
        .map(|(o, k)| (o, k + 1, ClaimKind::Index(k)));
    let asserted = operand_count_claims(&blanked)
        .into_iter()
        .map(|(o, n)| (o, n, ClaimKind::Length(n)));

    indexed
        .chain(asserted)
        .filter_map(|(off, needs, kind)| {
            let arm = arm_list
                .iter()
                .filter(|a| a.start <= off && off < a.end)
                .min_by_key(|a| a.end - a.start)?;
            Some(Claim {
                line: blanked[..off].matches('\n').count() + 1,
                ops: arm.ops.clone(),
                needs,
                kind,
            })
        })
        .collect()
}

/// Number of `Op::…` arms found, for anti-vacuity assertions.
pub fn arm_count(src: &str) -> usize {
    arms(&blank_noncode(src)).len()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ops_at(src: &str, needle: &str) -> Vec<String> {
        let blanked = blank_noncode(src);
        let off = blanked.find(needle).expect("needle survives blanking");
        arms(&blanked)
            .into_iter()
            .filter(|a| a.start <= off && off < a.end)
            .min_by_key(|a| a.end - a.start)
            .map(|a| a.ops)
            .unwrap_or_default()
    }

    #[test]
    fn multi_line_struct_pattern_keeps_its_op_name() {
        let src = "match &node.op {\n Op::Conv {\n kernel_size,\n stride,\n } => {\n \
                   let b = node.inputs.get(2);\n }\n}";
        assert_eq!(ops_at(src, "node.inputs.get(2)"), ["Conv"]);
    }

    #[test]
    fn a_previous_arm_does_not_leak_into_the_next_pattern() {
        let src = "match &node.op {\n Op::Attention { .. } => { let _ = 1; }\n \
                   Op::Cast { .. } => { let a = node.inputs[0]; }\n}";
        assert_eq!(ops_at(src, "node.inputs[0]"), ["Cast"]);
    }

    #[test]
    fn enum_variants_containing_op_are_not_op_names() {
        let src = "match &node.op {\n Op::Reduce { op: ReduceOp::Sum, .. } => \
                   { let a = node.inputs[0]; }\n}";
        assert_eq!(ops_at(src, "node.inputs[0]"), ["Reduce"]);
    }

    #[test]
    fn a_nested_arm_owns_its_reads() {
        let src = "match &node.op {\n Op::Softmax { .. } => {\n match &other {\n \
                   Op::Binary(_) => { let a = node.inputs[1]; }\n }\n }\n}";
        assert_eq!(ops_at(src, "node.inputs[1]"), ["Binary"]);
    }

    #[test]
    fn a_shared_arm_reports_every_op_it_matches() {
        let src = "match &node.op {\n Op::Gru { .. } | Op::Rnn { .. } => \
                   { let a = node.inputs[4]; }\n}";
        let mut got = ops_at(src, "node.inputs[4]");
        got.sort();
        assert_eq!(got, ["Gru", "Rnn"]);
    }

    #[test]
    fn only_the_current_nodes_operands_are_read() {
        let got = operand_reads("let a = graph.node(o).inputs[0]; let b = node.inputs[1];");
        assert_eq!(got.len(), 1, "{got:?}");
        assert_eq!(got[0].1, 1);
    }

    #[test]
    fn only_asserting_length_comparisons_count() {
        let claims = operand_count_claims(
            "if node.inputs.len() == 3 {} if node.inputs.len() > 1 {} \
             if node.inputs.len() < 9 {} if node.inputs.len() <= 7 {}",
        );
        let implied: Vec<usize> = claims.iter().map(|c| c.1).collect();
        assert_eq!(implied, [3, 2], "== 3 implies 3; > 1 implies 2");
    }

    #[test]
    fn comments_and_strings_are_ignored_without_shifting_offsets() {
        let src = "// node.inputs[9]\nlet m = \"node.inputs[8]\";\nlet a = node.inputs[0];";
        let blanked = blank_noncode(src);
        assert_eq!(blanked.len(), src.len(), "offsets must be preserved");
        let got = operand_reads(&blanked);
        assert_eq!(got.len(), 1, "{got:?}");
        assert_eq!(blanked[..got[0].0].matches('\n').count() + 1, 3);
    }

    /// 28 backend files use raw strings. Treating `r#"…"#` as a plain string
    /// ends it at the first inner quote, so the rest of the literal is read as
    /// code — and worse, real code after it can be swallowed instead.
    #[test]
    fn raw_strings_are_blanked_whole() {
        let src = "let s = r#\"node.inputs[7] and a \" quote\"#;\nlet a = node.inputs[0];";
        let blanked = blank_noncode(src);
        assert_eq!(blanked.len(), src.len());
        let got = operand_reads(&blanked);
        assert_eq!(got.len(), 1, "raw-string content leaked: {got:?}");
        assert_eq!(got[0].1, 0);
    }

    /// A `'"'` char literal would otherwise open a string that runs to the
    /// next quote, blanking code and blinding the scan for the rest of a file.
    #[test]
    fn a_quote_char_literal_does_not_open_a_string() {
        let src = "let q = '\"'; let a = node.inputs[0]; let s = \"x\";";
        let blanked = blank_noncode(src);
        let got = operand_reads(&blanked);
        assert_eq!(
            got.len(),
            1,
            "code after a quote char was swallowed: {got:?}"
        );
    }

    /// `'a` is a lifetime. Treating it as a char literal consumes to the next
    /// `'`, which is typically another lifetime far away.
    #[test]
    fn lifetimes_are_not_char_literals() {
        let src = "fn f<'a>(x: &'a str) -> &'a str { let _ = node.inputs[0]; x }";
        let blanked = blank_noncode(src);
        assert_eq!(operand_reads(&blanked).len(), 1);
    }

    #[test]
    fn nested_block_comments_close_correctly() {
        let src = "/* outer /* inner */ still comment node.inputs[9] */\nlet a = node.inputs[0];";
        let blanked = blank_noncode(src);
        let got = operand_reads(&blanked);
        assert_eq!(got.len(), 1, "{got:?}");
        assert_eq!(got[0].1, 0);
    }

    #[test]
    fn scan_attributes_both_claim_kinds() {
        let src = "match &node.op {\n Op::Conv { .. } => {\n \
                   if node.inputs.len() == 3 { let a = node.inputs[2]; }\n }\n}";
        let claims = scan(src);
        assert_eq!(claims.len(), 2, "{claims:?}");
        assert!(claims.iter().all(|c| c.ops == ["Conv"]));
        assert!(
            claims
                .iter()
                .any(|c| c.kind == ClaimKind::Index(2) && c.needs == 3)
        );
        assert!(
            claims
                .iter()
                .any(|c| c.kind == ClaimKind::Length(3) && c.needs == 3)
        );
    }
}
