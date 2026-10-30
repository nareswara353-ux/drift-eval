use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum Format {
    Toml,
    Yaml,
}

#[derive(Debug, Parser)]
#[command(
    name = "drift-eval",
    version,
    about = "High-Performance Statistical Validation & Distribution Drift Evaluator Engine",
    long_about = None
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,

    #[arg(long, global = true, env = "DRIFT_CONFIG")]
    pub config: Option<PathBuf>,

    #[arg(long, global = true, value_enum, default_value_t = Format::Toml)]
    pub format: Format,

    #[arg(long, global = true, env = "DRIFT_ALPHA")]
    pub alpha: Option<f64>,

    #[arg(long, global = true, env = "DRIFT_THREADS")]
    pub threads: Option<usize>,

    #[arg(long, global = true, env = "DRIFT_OUTPUT_DIR")]
    pub output_dir: Option<PathBuf>,

    #[arg(long, global = true)]
    pub no_progress: bool,

    #[arg(short, long, global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    Run,
    Validate,
    PrintConfig,
}

impl Cli {
    pub fn parse_args() -> Self {
        Self::parse()
    }
}
