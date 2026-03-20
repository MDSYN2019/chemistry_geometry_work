//! Starter surfaces for the 20 exercises listed in `EXERCISES.md`.
//! Each function/type is intentionally minimal so you can evolve design yourself.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimError {
    Parse,
    UnitMismatch,
    NonConvergent,
    InvalidTransition,
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
