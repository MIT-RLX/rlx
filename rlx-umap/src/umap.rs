// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.

//! High-level `Umap::fit` API (parametric UMAP on RLX).

use rlx_driver::Device;
use rlx_runtime::device_ext;

use crate::config::UmapConfig;
use crate::fitted::FittedUmap;
use crate::train::EpochProgress;
use crate::training::{FitOptions, fit_with_progress};

/// Parametric UMAP trainer (mirrors fast-umap `Umap<B>`).
pub struct Umap {
    config: UmapConfig,
    device: Device,
}

impl Umap {
    pub fn new(config: UmapConfig) -> Self {
        Self {
            config,
            device: Device::Cpu,
        }
    }

    pub fn with_device(config: UmapConfig, device: Device) -> Self {
        assert!(
            device_ext::is_available(device),
            "device {device:?} is not available"
        );
        Self { config, device }
    }

    pub fn config(&self) -> &UmapConfig {
        &self.config
    }

    /// Fit parametric UMAP on `data` (`n_samples × n_features`).
    pub fn fit(self, data: Vec<Vec<f64>>) -> FittedUmap {
        self.fit_with_signal(data, crate::interrupt::install_ctrlc_handler())
    }

    /// Fit with a progress callback (invoked when loss is read back).
    pub fn fit_with_progress(
        self,
        data: Vec<Vec<f64>>,
        on_progress: impl Fn(EpochProgress) + Send + 'static,
    ) -> FittedUmap {
        self.fit_with_signal_and_progress(
            data,
            crate::interrupt::install_ctrlc_handler(),
            on_progress,
        )
    }

    /// Fit with an external cancellation channel (e.g. Ctrl-C).
    pub fn fit_with_signal(
        self,
        data: Vec<Vec<f64>>,
        exit_rx: crossbeam_channel::Receiver<()>,
    ) -> FittedUmap {
        self.fit_with_signal_and_progress(data, exit_rx, |_| {})
    }

    fn fit_with_signal_and_progress(
        self,
        data: Vec<Vec<f64>>,
        exit_rx: crossbeam_channel::Receiver<()>,
        on_progress: impl Fn(EpochProgress) + Send + 'static,
    ) -> FittedUmap {
        let options = FitOptions {
            device: self.device,
            exit_rx: Some(exit_rx),
        };
        fit_with_progress(self.config, data, options, on_progress)
    }
}
