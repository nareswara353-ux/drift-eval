use std::time::Duration;

use indicatif::{ProgressBar, ProgressStyle};

#[derive(Clone)]
pub struct ProgressReporter {
    bar: Option<ProgressBar>,
}

impl ProgressReporter {
    pub fn new(enabled: bool, total: u64, label: &str) -> Self {
        if !enabled || total == 0 {
            return Self { bar: None };
        }
        let bar = ProgressBar::new(total);
        if let Ok(style) = ProgressStyle::with_template(
            "{spinner:.green} {msg} [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
        ) {
            bar.set_style(style.progress_chars("=>-"));
        }
        bar.set_message(label.to_string());
        bar.enable_steady_tick(Duration::from_millis(120));
        Self { bar: Some(bar) }
    }

    pub fn disabled() -> Self {
        Self { bar: None }
    }

    pub fn handle(&self) -> ProgressHandle<'_> {
        ProgressHandle { reporter: self }
    }

    pub fn set_position(&self, pos: u64) {
        if let Some(bar) = &self.bar {
            bar.set_position(pos);
        }
    }

    pub fn inc(&self, delta: u64) {
        if let Some(bar) = &self.bar {
            bar.inc(delta);
        }
    }

    pub fn set_message(&self, msg: impl Into<String>) {
        if let Some(bar) = &self.bar {
            bar.set_message(msg.into());
        }
    }

    pub fn finish(&self, msg: impl Into<String>) {
        if let Some(bar) = &self.bar {
            bar.finish_with_message(msg.into());
        }
    }

    pub fn abandon(&self) {
        if let Some(bar) = &self.bar {
            bar.abandon();
        }
    }
}

pub struct ProgressHandle<'a> {
    reporter: &'a ProgressReporter,
}

impl<'a> ProgressHandle<'a> {
    pub fn step(&self, label: &str) {
        self.reporter.set_message(label.to_string());
        self.reporter.inc(1);
    }

    pub fn set(&self, position: u64, label: &str) {
        self.reporter.set_position(position);
        self.reporter.set_message(label.to_string());
    }

    pub fn finish(&self, msg: &str) {
        self.reporter.finish(msg.to_string());
    }
}
