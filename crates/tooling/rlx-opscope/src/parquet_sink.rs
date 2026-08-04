// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

//! Parquet sink for the tidy sketch schema — a columnar, compressed alternative
//! to the CSV [`crate::Recorder`], for large sweeps that pandas/polars/DuckDB
//! read directly. Behind the `parquet` cargo feature (arrow/parquet pull ~30
//! crates, so the base build stays light). Same columns as the CSV.

use crate::StatSpec;
use arrow_array::{ArrayRef, Float32Array, RecordBatch, StringArray, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use parquet::arrow::ArrowWriter;
use std::sync::Arc;

/// Accumulates rows, then writes a single compressed Parquet file.
#[derive(Default)]
pub struct ParquetRecorder {
    run_id: Vec<u64>,
    step: Vec<u64>,
    backend: Vec<String>,
    dist: Vec<String>,
    m: Vec<u64>,
    k: Vec<u64>,
    n: Vec<u64>,
    numel: Vec<u64>,
    site: Vec<String>,
    role: Vec<String>,
    stat: Vec<String>,
    idx: Vec<u64>,
    value: Vec<f32>,
}

impl ParquetRecorder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Same signature as [`crate::Recorder::record`] — one row per sketch element.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        run_id: u64,
        step: u64,
        backend: &str,
        dist: &str,
        m: usize,
        k: usize,
        n: usize,
        specs: &[StatSpec],
        outs: &[Vec<f32>],
    ) {
        for spec in specs {
            let data = &outs[spec.out_idx];
            for (i, &v) in data.iter().enumerate() {
                self.run_id.push(run_id);
                self.step.push(step);
                self.backend.push(backend.into());
                self.dist.push(dist.into());
                self.m.push(m as u64);
                self.k.push(k as u64);
                self.n.push(n as u64);
                self.numel.push(spec.numel as u64);
                self.site.push(spec.site.clone());
                self.role.push(spec.role.into());
                self.stat.push(spec.stat.into());
                self.idx.push(i as u64);
                self.value.push(v);
            }
        }
    }

    pub fn rows(&self) -> usize {
        self.value.len()
    }

    fn schema() -> Arc<Schema> {
        let u = |n: &str| Field::new(n, DataType::UInt64, false);
        let s = |n: &str| Field::new(n, DataType::Utf8, false);
        Arc::new(Schema::new(vec![
            u("run_id"),
            u("step"),
            s("backend"),
            s("dist"),
            u("M"),
            u("K"),
            u("N"),
            u("numel"),
            s("site"),
            s("role"),
            s("stat"),
            u("idx"),
            Field::new("value", DataType::Float32, false),
        ]))
    }

    /// Write the accumulated rows to `path` as compressed Parquet.
    pub fn finish(self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let schema = Self::schema();
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(UInt64Array::from(self.run_id)) as ArrayRef,
                Arc::new(UInt64Array::from(self.step)),
                Arc::new(StringArray::from(self.backend)),
                Arc::new(StringArray::from(self.dist)),
                Arc::new(UInt64Array::from(self.m)),
                Arc::new(UInt64Array::from(self.k)),
                Arc::new(UInt64Array::from(self.n)),
                Arc::new(UInt64Array::from(self.numel)),
                Arc::new(StringArray::from(self.site)),
                Arc::new(StringArray::from(self.role)),
                Arc::new(StringArray::from(self.stat)),
                Arc::new(UInt64Array::from(self.idx)),
                Arc::new(Float32Array::from(self.value)),
            ],
        )?;
        let file = std::fs::File::create(path)?;
        let mut writer = ArrowWriter::try_new(file, schema, None)?;
        writer.write(&batch)?;
        writer.close()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StatSpec;

    #[test]
    fn parquet_roundtrip() {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        let specs = vec![StatSpec {
            out_idx: 0,
            site: "matmul#2".into(),
            role: "lhs",
            stat: "mean",
            len: 1,
            numel: 4096,
            flops: 1024,
        }];
        let outs = vec![vec![0.5f32]];
        let mut rec = ParquetRecorder::new();
        rec.record(0, 0, "cpu", "test", 64, 64, 64, &specs, &outs);
        assert_eq!(rec.rows(), 1);
        let path = std::env::temp_dir().join("opscope_pq_test.parquet");
        let ps = path.to_str().unwrap().to_string();
        rec.finish(&ps).unwrap();

        // Read it back.
        let file = std::fs::File::open(&ps).unwrap();
        let reader = ParquetRecordBatchReaderBuilder::try_new(file)
            .unwrap()
            .build()
            .unwrap();
        let total: usize = reader.map(|b| b.unwrap().num_rows()).sum();
        assert_eq!(total, 1);
    }
}
