//! Starter surfaces for the 20 exercises listed in `EXERCISES.md`.
//! Each function/type is intentionally minimal so you can evolve design yourself.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimError {
    Parse,
    UnitMismatch,
    NonConvergent,
    InvalidTransition,
    Overflow,
    MissingField,
    OutOfBounds,
}

// 1) ok-or-not-ok
pub fn parse_positive_i64(input: &str) -> Result<i64, SimError> {
    let value = input.parse::<i64>().map_err(|_| SimError::Parse)?;
    if value > 0 {
        Ok(value)
    } else {
        Err(SimError::Parse)
    }
}

// 2) option-result-flip
pub fn transpose_option_result<T, E>(v: Option<Result<T, E>>) -> Result<Option<T>, E> {
    match v {
        None => Ok(None),
        Some(Ok(t)) => Ok(Some(t)),
        Some(Err(e)) => Err(e),
    }
}

// 3) borrow-vs-own-api
pub fn normalize_name(input: &str) -> String {
    input.trim().to_lowercase()
}

// 4) clone-budget
pub fn sum_slice(values: &[i64]) -> i64 {
    values.iter().sum()
}

// 5) pure-model-vs-runner
pub fn euler_step(position: f64, velocity: f64, dt: f64) -> f64 {
    position + velocity * dt
}

// 6) state-machine-sim
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimState {
    Initialized,
    Configured,
    Running,
    Finished,
}

pub fn transition(state: SimState, next: SimState) -> Result<SimState, SimError> {
    use SimState::*;
    let valid = matches!((state, next),
        (Initialized, Configured)
            | (Configured, Running)
            | (Running, Finished)
    );
    if valid {
        Ok(next)
    } else {
        Err(SimError::InvalidTransition)
    }
}

// 7) adapter-two-representations
#[derive(Debug, Clone, Copy)]
pub struct Particle {
    pub x: f64,
    pub mass: f64,
}

pub fn particles_to_total_mass(particles: &[Particle]) -> f64 {
    particles.iter().map(|p| p.mass).sum()
}

// 8) dependency-direction-check
pub trait Integrator {
    fn integrate(&self, x: f64, v: f64, dt: f64) -> f64;
}

pub struct Euler;
impl Integrator for Euler {
    fn integrate(&self, x: f64, v: f64, dt: f64) -> f64 {
        x + v * dt
    }
}

// 9) one-system-or-many
#[derive(Debug, Clone)]
pub struct System {
    pub temperature_k: f64,
}

pub fn build_single_system(temp_k: f64) -> System {
    System {
        temperature_k: temp_k,
    }
}

// 10) typestate-builder (starter only)
pub struct SimulationBuilder {
    temperature_k: Option<f64>,
}

impl SimulationBuilder {
    pub fn new() -> Self {
        Self {
            temperature_k: None,
        }
    }

    pub fn temperature_k(mut self, value: f64) -> Self {
        self.temperature_k = Some(value);
        self
    }

    pub fn build(self) -> Result<System, SimError> {
        match self.temperature_k {
            Some(k) => Ok(System { temperature_k: k }),
            None => Err(SimError::Parse),
        }
    }
}

impl Default for SimulationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// 11) error-taxonomy
pub fn validate_temperature(k: f64) -> Result<f64, SimError> {
    if k > 0.0 {
        Ok(k)
    } else {
        Err(SimError::UnitMismatch)
    }
}

// 12) iterator-friendly-api
pub fn mean<I>(it: I) -> Option<f64>
where
    I: IntoIterator<Item = f64>,
{
    let mut n = 0usize;
    let mut sum = 0.0;
    for x in it {
        n += 1;
        sum += x;
    }
    (n > 0).then_some(sum / n as f64)
}

// 13) newtype-units
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Seconds(pub f64);
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Meters(pub f64);
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetersPerSecond(pub f64);

pub fn distance(v: MetersPerSecond, t: Seconds) -> Meters {
    Meters(v.0 * t.0)
}

// 14) dimensional-analysis-tests
pub fn kinetic_energy(mass_kg: f64, velocity_mps: f64) -> f64 {
    0.5 * mass_kg * velocity_mps * velocity_mps
}

// 15) conservation-laws
pub fn total_mass(masses: &[f64]) -> f64 {
    masses.iter().sum()
}

// 16) fault-injection-units
pub fn seconds_from_millis(ms: f64) -> Seconds {
    Seconds(ms / 1000.0)
}

// 17) mini-md-engine
pub fn integrate_position<I: Integrator>(integrator: &I, x: f64, v: f64, dt: f64) -> f64 {
    integrator.integrate(x, v, dt)
}

// 18) io-boundary-clean
pub fn parse_csv_line(line: &str) -> Result<Vec<f64>, SimError> {
    line.split(',')
        .map(|t| t.trim().parse::<f64>().map_err(|_| SimError::Parse))
        .collect()
}

// 19) refactor-for-clarity
pub fn clamp01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

// 20) capstone-review
pub fn run_steps<I: Integrator>(integrator: &I, mut x: f64, v: f64, dt: f64, n: usize) -> f64 {
    for _ in 0..n {
        x = integrator.integrate(x, v, dt);
    }
    x
}

// 21) pyo3-function-signatures
pub fn pyo3_scale(values: &[f64], scale: Option<f64>) -> Vec<f64> {
    let factor = scale.unwrap_or(1.0);
    values.iter().map(|v| v * factor).collect()
}

// 22) python-index-semantics
pub fn normalize_py_index(len: usize, index: isize) -> Result<usize, SimError> {
    if len == 0 {
        return Err(SimError::OutOfBounds);
    }
    let len_isize = len as isize;
    let normalized = if index < 0 { len_isize + index } else { index };
    if normalized >= 0 && normalized < len_isize {
        Ok(normalized as usize)
    } else {
        Err(SimError::OutOfBounds)
    }
}

// 23) overflow-to-pyerr
pub fn checked_sum_i64(values: &[i64]) -> Result<i64, SimError> {
    values.iter().try_fold(0_i64, |acc, &v| {
        acc.checked_add(v).ok_or(SimError::Overflow)
    })
}

// 24) kwargs-validation
pub fn required_positive_arg(
    kwargs: &std::collections::HashMap<String, f64>,
    key: &str,
) -> Result<f64, SimError> {
    let value = kwargs.get(key).copied().ok_or(SimError::MissingField)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(SimError::UnitMismatch)
    }
}

// 25) vectorized-bridge
pub fn batch_square(values: &[i64]) -> Vec<i64> {
    values.iter().map(|v| v * v).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn normalizes_python_index() {
        assert_eq!(normalize_py_index(4, 0), Ok(0));
        assert_eq!(normalize_py_index(4, -1), Ok(3));
        assert_eq!(normalize_py_index(4, -4), Ok(0));
        assert_eq!(normalize_py_index(4, 4), Err(SimError::OutOfBounds));
        assert_eq!(normalize_py_index(0, 0), Err(SimError::OutOfBounds));
    }

    #[test]
    fn catches_checked_sum_overflow() {
        assert_eq!(checked_sum_i64(&[1, 2, 3]), Ok(6));
        assert_eq!(checked_sum_i64(&[i64::MAX, 1]), Err(SimError::Overflow));
    }

    #[test]
    fn validates_required_positive_kwargs() {
        let mut kwargs = HashMap::new();
        kwargs.insert("temperature_k".to_string(), 300.0);
        assert_eq!(
            required_positive_arg(&kwargs, "temperature_k"),
            Ok(300.0)
        );
        assert_eq!(
            required_positive_arg(&kwargs, "pressure"),
            Err(SimError::MissingField)
        );
        kwargs.insert("temperature_k".to_string(), 0.0);
        assert_eq!(
            required_positive_arg(&kwargs, "temperature_k"),
            Err(SimError::UnitMismatch)
        );
    }
}
