use std::path::Path;

use drift_core::{Config, DriftError};

use crate::cli::{Cli, Format};

fn read_file(path: &Path) -> drift_core::Result<String> {
    std::fs::read_to_string(path).map_err(|e| DriftError::io(path.display().to_string(), e))
}

fn parse_config(input: &str, format: Format) -> drift_core::Result<Config> {
    match format {
        Format::Toml => Config::from_toml_str(input),
        Format::Yaml => Config::from_yaml_str(input),
    }
}

fn apply_overrides(config: &mut Config, cli: &Cli) {
    if let Some(alpha) = cli.alpha {
        config.math.alpha = alpha;
    }
    if let Some(threads) = cli.threads {
        config.parallel.threads = Some(threads);
    }
    if let Some(output_dir) = &cli.output_dir {
        config.report.output_dir = output_dir.display().to_string();
    }
    if cli.no_progress {
        config.report.show_progress = false;
    }
}

pub fn resolve(cli: &Cli) -> drift_core::Result<Config> {
    let mut config = match &cli.config {
        Some(path) => {
            let content = read_file(path)?;
            parse_config(&content, cli.format)?
        }
        None => Config::default(),
    };
    apply_overrides(&mut config, cli);
    config.validate()?;
    Ok(config)
}
