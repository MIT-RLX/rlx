// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

//! Record/replay middleware (plan #63).
//!
//! Borrowed from MAX's `serve/recordreplay/{middleware, replay,
//! jsonl, schema}.py`. Capture (request, response) pairs to a
//! JSONL file as production traffic flows through; replay them
//! against a new build to detect regressions.
//!
//! Pure data layer:
//!   - [`RecordingWriter`] appends [`RecordedExchange`] rows to a
//!     file (one JSON object per line).
//!   - [`ReplayReader`] reads them back in declaration order.
//!   - The actual middleware that wraps a request handler is left
//!     to the future serving crate — we ship the storage format +
//!     reader/writer; the wiring is two function calls.

use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// One captured request/response pair plus metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecordedExchange {
    /// Wall-clock nanoseconds since UNIX epoch when the request
    /// was received. Replay tools use this to compute relative
    /// timing if they want to reproduce arrival rates.
    pub recv_ts_ns: u128,
    /// Latency of the response in nanoseconds (from request
    /// receive to response send). `None` if the request didn't
    /// complete (timeout / cancel / crash).
    pub latency_ns: Option<u64>,
    /// Free-form protocol tag (e.g. `"openai.chat"`,
    /// `"kserve.embedding"`). Replay readers can filter on this.
    pub protocol: String,
    /// Wire-format request as JSON.
    pub request: serde_json::Value,
    /// Wire-format response as JSON, or `None` for errored / open
    /// exchanges captured mid-flight.
    pub response: Option<serde_json::Value>,
}

impl RecordedExchange {
    /// Convenience constructor with the current wall clock.
    pub fn now(
        protocol: impl Into<String>,
        request: serde_json::Value,
        response: Option<serde_json::Value>,
        latency_ns: Option<u64>,
    ) -> Self {
        let recv_ts_ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        Self {
            recv_ts_ns,
            latency_ns,
            protocol: protocol.into(),
            request,
            response,
        }
    }
}

/// Append-only JSONL writer. One file = one stream of exchanges.
/// Cheap drop semantics: flush is best-effort on drop, explicit
/// `flush` available for callers who need durability.
pub struct RecordingWriter {
    out: BufWriter<File>,
    written: usize,
}

impl RecordingWriter {
    /// Open `path`, creating it (or appending if it exists).
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let f = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            out: BufWriter::new(f),
            written: 0,
        })
    }

    /// Append one exchange. Errors propagate so callers can
    /// downgrade to "log on failure, don't crash the request".
    pub fn write(&mut self, ex: &RecordedExchange) -> std::io::Result<()> {
        let line = serde_json::to_string(ex)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        self.out.write_all(line.as_bytes())?;
        self.out.write_all(b"\n")?;
        self.written += 1;
        Ok(())
    }

    pub fn flush(&mut self) -> std::io::Result<()> {
        self.out.flush()
    }

    /// Number of exchanges written this session.
    pub fn count(&self) -> usize {
        self.written
    }
}

impl Drop for RecordingWriter {
    fn drop(&mut self) {
        let _ = self.out.flush();
    }
}

/// Iterator-style replay reader. Yields exchanges in file order.
pub struct ReplayReader {
    inner: BufReader<File>,
}

impl ReplayReader {
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let f = File::open(path)?;
        Ok(Self {
            inner: BufReader::new(f),
        })
    }
}

impl Iterator for ReplayReader {
    type Item = std::io::Result<RecordedExchange>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut line = String::new();
        match self.inner.read_line(&mut line) {
            Ok(0) => None,
            Ok(_) => {
                if line.trim().is_empty() {
                    return self.next();
                }
                Some(
                    serde_json::from_str(line.trim())
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e)),
                )
            }
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_path(label: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir();
        dir.join(format!(
            "rlx-rr-{label}-{}.jsonl",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    #[test]
    fn round_trip_single_exchange() {
        let path = temp_path("single");
        {
            let mut w = RecordingWriter::open(&path).unwrap();
            let ex = RecordedExchange::now(
                "openai.chat",
                json!({ "model": "gpt-4o-mini", "messages": [] }),
                Some(json!({ "id": "abc", "choices": [] })),
                Some(123_456),
            );
            w.write(&ex).unwrap();
            assert_eq!(w.count(), 1);
            w.flush().unwrap();
        }

        let mut iter = ReplayReader::open(&path).unwrap();
        let got = iter.next().unwrap().unwrap();
        assert_eq!(got.protocol, "openai.chat");
        assert_eq!(got.latency_ns, Some(123_456));
        assert_eq!(got.request["model"], "gpt-4o-mini");
        assert!(got.response.is_some());
        assert!(iter.next().is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn writes_and_reads_in_order() {
        let path = temp_path("order");
        {
            let mut w = RecordingWriter::open(&path).unwrap();
            for i in 0..5 {
                w.write(&RecordedExchange::now(
                    "test",
                    json!({ "i": i }),
                    Some(json!({ "i": i })),
                    Some(i as u64 * 1000),
                ))
                .unwrap();
            }
        }

        let exchanges: Vec<RecordedExchange> = ReplayReader::open(&path)
            .unwrap()
            .collect::<std::io::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(exchanges.len(), 5);
        for (i, ex) in exchanges.iter().enumerate() {
            assert_eq!(ex.request["i"], i);
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn append_mode_preserves_existing() {
        let path = temp_path("append");
        {
            let mut w = RecordingWriter::open(&path).unwrap();
            w.write(&RecordedExchange::now(
                "first",
                json!({"a": 1}),
                Some(json!({})),
                Some(1),
            ))
            .unwrap();
        }
        {
            // Re-open: must append, not truncate.
            let mut w = RecordingWriter::open(&path).unwrap();
            w.write(&RecordedExchange::now(
                "second",
                json!({"a": 2}),
                Some(json!({})),
                Some(2),
            ))
            .unwrap();
        }

        let exchanges: Vec<_> = ReplayReader::open(&path)
            .unwrap()
            .collect::<std::io::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(exchanges.len(), 2);
        assert_eq!(exchanges[0].protocol, "first");
        assert_eq!(exchanges[1].protocol, "second");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn missing_response_round_trips() {
        let path = temp_path("missing-resp");
        {
            let mut w = RecordingWriter::open(&path).unwrap();
            w.write(&RecordedExchange::now(
                "abandoned",
                json!({"q": "hello"}),
                None,
                None,
            ))
            .unwrap();
        }
        let ex = ReplayReader::open(&path).unwrap().next().unwrap().unwrap();
        assert!(ex.response.is_none());
        assert!(ex.latency_ns.is_none());
        let _ = std::fs::remove_file(&path);
    }
}
