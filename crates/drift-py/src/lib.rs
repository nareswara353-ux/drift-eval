
fn correlation_report_to_dict<'py>(py: pyo3::Python<'py>, report: &impl std::fmt::Debug) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyDict>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("report", format!("{:?}", report))?;
    Ok(dict)
}

use drift_core::{ColumnName, Config, DistributionSample, DriftError, TestOutcome};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn to_py_err(err: DriftError) -> PyErr {
    match err {
        DriftError::EmptySample(_)
        | DriftError::NonFiniteValue(_)
        | DriftError::InvalidAlpha(_)
        | DriftError::InvalidChunkSize(_)
        | DriftError::DimensionMismatch { .. }
        | DriftError::SampleSizeMismatch { .. }
        | DriftError::ConfigInvalid(_) => PyValueError::new_err(err.to_string()),
        _ => PyRuntimeError::new_err(err.to_string()),
    }
}

fn build_sample(
    py: Python<'_>,
    reference: Vec<f64>,
    target: Vec<f64>,
) -> PyResult<(DistributionSample, DistributionSample)> {
    let _ = py;
    let ref_sample =
        DistributionSample::new(ColumnName::new("reference").map_err(to_py_err)?, reference)
            .map_err(to_py_err)?;
    let tgt_sample = DistributionSample::new(ColumnName::new("target").map_err(to_py_err)?, target)
        .map_err(to_py_err)?;
    Ok((ref_sample, tgt_sample))
}

fn outcome_to_dict<'py>(py: Python<'py>, outcome: &TestOutcome) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("metric", outcome.metric.label())?;
    dict.set_item("statistic", outcome.statistic)?;
    dict.set_item("p_value", outcome.p_value)?;
    dict.set_item("critical", outcome.critical)?;
    dict.set_item(
        "verdict",
        match outcome.verdict {
            drift_core::DriftVerdict::Stable => "stable",
            drift_core::DriftVerdict::Drifted => "drifted",
            drift_core::DriftVerdict::Inconclusive => "inconclusive",
        },
    )?;
    dict.set_item("sample_size_reference", outcome.sample_size_reference)?;
    dict.set_item("sample_size_target", outcome.sample_size_target)?;
    Ok(dict)
}

#[pyfunction]
#[pyo3(signature = (reference, target, alpha = 0.05))]
fn ks_test(
    py: Python<'_>,
    reference: Vec<f64>,
    target: Vec<f64>,
    alpha: f64,
) -> PyResult<Bound<'_, PyDict>> {
    let (ref_sample, tgt_sample) = build_sample(py, reference, target)?;
    let outcome = drift_math::ks_two_sample(&ref_sample, &tgt_sample, alpha).map_err(to_py_err)?;
    correlation_report_to_dict(py, &outcome)
}

#[pyfunction]
#[pyo3(signature = (reference, target, chunk_size = 4096))]
fn wasserstein1(
    py: Python<'_>,
    reference: Vec<f64>,
    target: Vec<f64>,
    chunk_size: usize,
) -> PyResult<Bound<'_, PyDict>> {
    let (ref_sample, tgt_sample) = build_sample(py, reference, target)?;
    let outcome =
        drift_math::wasserstein1(&ref_sample, &tgt_sample, chunk_size).map_err(to_py_err)?;
    correlation_report_to_dict(py, &outcome)
}

#[pyfunction]
#[pyo3(signature = (reference, target, bins = 32, alpha = 0.05))]
fn chi_square_test(
    py: Python<'_>,
    reference: Vec<f64>,
    target: Vec<f64>,
    bins: usize,
    alpha: f64,
) -> PyResult<Bound<'_, PyDict>> {
    let (ref_sample, tgt_sample) = build_sample(py, reference, target)?;
    let outcome =
        drift_math::chi_square_test(&ref_sample, &tgt_sample, bins, alpha).map_err(to_py_err)?;
    correlation_report_to_dict(py, &outcome)
}

#[pyfunction]
#[pyo3(signature = (reference, target, alpha = 0.05))]
fn delta_correlation(
    py: Python<'_>,
    reference: Vec<f64>,
    target: Vec<f64>,
    alpha: f64,
) -> PyResult<Bound<'_, PyDict>> {
    let _ = alpha;
    let (ref_sample, tgt_sample) = build_sample(py, reference, target)?;
    let outcome =
        drift_math::delta_correlation(&[ref_sample.values.as_slice()], &[tgt_sample.values.as_slice()]).map_err(to_py_err)?;
    correlation_report_to_dict(py, &outcome)
}

#[pyfunction]
#[pyo3(signature = (reference, target, alpha = 0.05))]
fn run_all(
    py: Python<'_>,
    reference: Vec<f64>,
    target: Vec<f64>,
    alpha: f64,
) -> PyResult<Vec<Bound<'_, PyDict>>> {
    let (ref_sample, tgt_sample) = build_sample(py, reference, target)?;
    let mut config = Config::default();
    config.math.alpha = alpha;
    config.validate().map_err(to_py_err)?;
    let outcomes = drift_math::run_all(&ref_sample, &tgt_sample, &config).map_err(to_py_err)?;
    outcomes
        .iter()
        .map(|o| outcome_to_dict(py, o))
        .collect::<PyResult<Vec<_>>>()
}

#[pymodule]
fn drift_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ks_test, m)?)?;
    m.add_function(wrap_pyfunction!(wasserstein1, m)?)?;
    m.add_function(wrap_pyfunction!(chi_square_test, m)?)?;
    m.add_function(wrap_pyfunction!(delta_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(run_all, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
