// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Whether text so far can begin one JSON value, a character at a time:
//! what JSON-constrained decoding keeps a candidate token by.

/// What the parser is in the middle of.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Before a value (whitespace allowed).
    Value,
    /// Just after `[`: a value, or `]`.
    ArrayStart,
    /// After a value inside a container: `,` or the container's close.
    AfterValue,
    /// Inside an object, before a key (`"`) or, if `first`, `}`.
    Key { first: bool },
    /// After a key: `:`.
    Colon,
    /// In a string; `key` when it is an object's key. `escape`: after a
    /// backslash; `hex`: `\u` digits still to come.
    Str { key: bool, escape: bool, hex: u8 },
    /// In a number: what may come next.
    Num(NumState),
    /// In `true` / `false` / `null`: the rest of the word.
    Word { word: &'static str, at: u8 },
    /// The top-level value is done; only whitespace may follow.
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumState {
    /// After `-`: a digit.
    Sign,
    /// After a leading `0`: `.`, `e` or the end.
    Zero,
    /// In the integer digits.
    Int,
    /// After `.`: a digit.
    Dot,
    /// In the fraction digits.
    Frac,
    /// After `e`: a sign or digit.
    Exp,
    /// After `e+` / `e-`: a digit.
    ExpSign,
    /// In the exponent digits.
    ExpDigits,
}

impl NumState {
    /// Whether the number may end here.
    fn complete(self) -> bool {
        matches!(self, Self::Zero | Self::Int | Self::Frac | Self::ExpDigits)
    }
}

/// A JSON prefix, fed a character at a time. Clone it to try a candidate.
#[derive(Debug, Clone)]
pub struct JsonPrefix {
    mode: Mode,
    /// Open containers, innermost last: `true` for an object.
    stack: Vec<bool>,
}

impl Default for JsonPrefix {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonPrefix {
    pub fn new() -> Self {
        Self {
            mode: Mode::Value,
            stack: Vec::new(),
        }
    }

    /// Whether the value is over: nothing but whitespace may follow (a
    /// number may yet go on, so a top-level one is [`Self::complete`] but
    /// not over).
    pub fn done(&self) -> bool {
        self.mode == Mode::Done
    }

    /// Whether one whole value has been read.
    pub fn complete(&self) -> bool {
        match self.mode {
            Mode::Done => true,
            Mode::Num(n) => self.stack.is_empty() && n.complete(),
            _ => false,
        }
    }

    /// A value ended: the container's next, or done.
    fn value_done(&mut self) {
        self.mode = if self.stack.is_empty() {
            Mode::Done
        } else {
            Mode::AfterValue
        };
    }

    /// Read `c`; false (and the state unchanged in meaning) when no JSON
    /// value continues this way.
    pub fn feed(&mut self, c: char) -> bool {
        // A number ends at the first character that is not part of it.
        if let Mode::Num(n) = self.mode {
            if let Some(next) = self.number(n, c) {
                self.mode = Mode::Num(next);
                return true;
            }
            if !n.complete() {
                return false;
            }
            self.value_done();
        }
        match self.mode {
            Mode::Value => self.value(c),
            Mode::ArrayStart => match c {
                c if c.is_whitespace() => true,
                ']' => self.close(),
                c => self.value(c),
            },
            Mode::AfterValue => match c {
                c if c.is_whitespace() => true,
                ',' => {
                    self.mode = if self.stack.last() == Some(&true) {
                        Mode::Key { first: false }
                    } else {
                        Mode::Value
                    };
                    true
                }
                '}' if self.stack.last() == Some(&true) => self.close(),
                ']' if self.stack.last() == Some(&false) => self.close(),
                _ => false,
            },
            Mode::Key { first } => match c {
                c if c.is_whitespace() => true,
                '"' => {
                    self.mode = Mode::Str {
                        key: true,
                        escape: false,
                        hex: 0,
                    };
                    true
                }
                '}' if first => self.close(),
                _ => false,
            },
            Mode::Colon => match c {
                c if c.is_whitespace() => true,
                ':' => {
                    self.mode = Mode::Value;
                    true
                }
                _ => false,
            },
            Mode::Str { key, escape, hex } => {
                if hex > 0 {
                    if !c.is_ascii_hexdigit() {
                        return false;
                    }
                    self.mode = Mode::Str {
                        key,
                        escape: false,
                        hex: hex - 1,
                    };
                    return true;
                }
                if escape {
                    let hex = match c {
                        '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' => 0,
                        'u' => 4,
                        _ => return false,
                    };
                    self.mode = Mode::Str {
                        key,
                        escape: false,
                        hex,
                    };
                    return true;
                }
                match c {
                    '"' => {
                        if key {
                            self.mode = Mode::Colon;
                        } else {
                            self.value_done();
                        }
                        true
                    }
                    '\\' => {
                        self.mode = Mode::Str {
                            key,
                            escape: true,
                            hex: 0,
                        };
                        true
                    }
                    c if (c as u32) < 0x20 => false,
                    _ => true,
                }
            }
            Mode::Word { word, at } => {
                if word.as_bytes().get(at as usize) != Some(&(c as u8)) || !c.is_ascii() {
                    return false;
                }
                if at as usize + 1 == word.len() {
                    self.value_done();
                } else {
                    self.mode = Mode::Word { word, at: at + 1 };
                }
                true
            }
            Mode::Done => c.is_whitespace(),
            Mode::Num(_) => unreachable!("handled above"),
        }
    }

    /// The start of a value.
    fn value(&mut self, c: char) -> bool {
        match c {
            c if c.is_whitespace() => true,
            '{' => {
                self.stack.push(true);
                self.mode = Mode::Key { first: true };
                true
            }
            '[' => {
                self.stack.push(false);
                self.mode = Mode::ArrayStart;
                true
            }
            '"' => {
                self.mode = Mode::Str {
                    key: false,
                    escape: false,
                    hex: 0,
                };
                true
            }
            '-' => {
                self.mode = Mode::Num(NumState::Sign);
                true
            }
            '0' => {
                self.mode = Mode::Num(NumState::Zero);
                true
            }
            '1'..='9' => {
                self.mode = Mode::Num(NumState::Int);
                true
            }
            't' | 'f' | 'n' => {
                let word = match c {
                    't' => "true",
                    'f' => "false",
                    _ => "null",
                };
                self.mode = Mode::Word { word, at: 1 };
                true
            }
            _ => false,
        }
    }

    /// The next number state after `c`, if `c` continues the number.
    fn number(&self, n: NumState, c: char) -> Option<NumState> {
        use NumState::*;
        Some(match (n, c) {
            (Sign, '0') => Zero,
            (Sign, '1'..='9') => Int,
            (Int, '0'..='9') => Int,
            (Zero | Int, '.') => Dot,
            (Dot | Frac, '0'..='9') => Frac,
            (Zero | Int | Frac, 'e' | 'E') => Exp,
            (Exp, '+' | '-') => ExpSign,
            (Exp | ExpSign | ExpDigits, '0'..='9') => ExpDigits,
            _ => return None,
        })
    }

    fn close(&mut self) -> bool {
        self.stack.pop();
        self.value_done();
        true
    }

    /// Feed all of `s`; false at the first character that does not fit.
    pub fn feed_str(&mut self, s: &str) -> bool {
        s.chars().all(|c| self.feed(c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn accepts(s: &str) -> bool {
        JsonPrefix::new().feed_str(s)
    }

    fn whole(s: &str) -> bool {
        let mut p = JsonPrefix::new();
        p.feed_str(s) && p.complete()
    }

    #[test]
    fn whole_values_complete() {
        for s in [
            "{}",
            "[]",
            r#"{"a": [1, -2.5e3, true, null, "x\"yé"], "b": {}}"#,
            "  42 ",
            "0",
            r#""s""#,
            "[[],[{}]]",
        ] {
            assert!(whole(s), "{s}");
        }
    }

    #[test]
    fn prefixes_are_accepted_but_not_complete() {
        for s in [r#"{"a": "#, "[1, 2", r#"{"a"#, "tr", "-", "1.", "[1, 2.5e"] {
            assert!(accepts(s), "{s}");
            assert!(!whole(s), "{s}");
        }
    }

    #[test]
    fn what_no_value_continues_is_refused() {
        for s in [
            "}",
            "[1,]x",
            r#"{"a" 1}"#,
            "{1: 2}",
            "01",
            "tx",
            "[1 2]",
            r#"{"a": 1,}"#,
            "{} {}",
            "\"a\nb\"",
        ] {
            assert!(!accepts(s), "{s}");
        }
    }
}
