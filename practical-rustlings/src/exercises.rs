//! Starter surfaces for the exercises listed in `EXERCISES.md`.
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
    BorrowConflict,
    Timeout,
    NullPtr,
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

// 66) returns-simple-vs-log
pub fn simple_return(prev: f64, next: f64) -> Result<f64, SimError> {
    if prev <= 0.0 || next <= 0.0 {
        return Err(SimError::OutOfBounds);
    }
    Ok(next / prev - 1.0)
}

pub fn log_return(prev: f64, next: f64) -> Result<f64, SimError> {
    if prev <= 0.0 || next <= 0.0 {
        return Err(SimError::OutOfBounds);
    }
    Ok((next / prev).ln())
}

// 67) rolling-volatility
pub fn rolling_sample_volatility(returns: &[f64], window: usize) -> Result<Vec<f64>, SimError> {
    if window < 2 || window > returns.len() {
        return Err(SimError::OutOfBounds);
    }

    let mut out = Vec::with_capacity(returns.len() - window + 1);
    for chunk in returns.windows(window) {
        let mean = chunk.iter().sum::<f64>() / window as f64;
        let var = chunk
            .iter()
            .map(|x| {
                let d = x - mean;
                d * d
            })
            .sum::<f64>()
            / (window as f64 - 1.0);
        out.push(var.sqrt());
    }
    Ok(out)
}

// 68) drawdown-tracker
pub fn max_drawdown(equity_curve: &[f64]) -> Result<f64, SimError> {
    if equity_curve.is_empty() {
        return Err(SimError::MissingField);
    }
    if equity_curve.iter().any(|x| *x <= 0.0) {
        return Err(SimError::OutOfBounds);
    }

    let mut peak = equity_curve[0];
    let mut worst = 0.0f64;
    for &value in equity_curve {
        if value > peak {
            peak = value;
        }
        let dd = (peak - value) / peak;
        if dd > worst {
            worst = dd;
        }
    }
    Ok(worst)
}

// 69) position-sizer
pub fn position_size(
    equity: f64,
    risk_fraction: f64,
    stop_distance: f64,
    point_value: f64,
) -> Result<u64, SimError> {
    if equity <= 0.0 || stop_distance <= 0.0 || point_value <= 0.0 {
        return Err(SimError::OutOfBounds);
    }
    if !(0.0..=1.0).contains(&risk_fraction) {
        return Err(SimError::OutOfBounds);
    }

    let risk_budget = equity * risk_fraction;
    let units = (risk_budget / (stop_distance * point_value)).floor();
    if !units.is_finite() || units < 0.0 {
        return Err(SimError::Overflow);
    }
    Ok(units as u64)
}

// 70) pnl-attribution
pub fn pnl(price_entry: f64, price_exit: f64, quantity: f64, multiplier: f64) -> f64 {
    (price_exit - price_entry) * quantity * multiplier
}

// 71) order-book-spread
pub fn spread_ticks(best_bid: f64, best_ask: f64, tick_size: f64) -> Result<f64, SimError> {
    if tick_size <= 0.0 || best_bid <= 0.0 || best_ask <= 0.0 {
        return Err(SimError::OutOfBounds);
    }
    if best_bid > best_ask {
        return Err(SimError::InvalidTransition);
    }
    Ok((best_ask - best_bid) / tick_size)
}

pub fn spread_bps(best_bid: f64, best_ask: f64) -> Result<f64, SimError> {
    if best_bid <= 0.0 || best_ask <= 0.0 {
        return Err(SimError::OutOfBounds);
    }
    if best_bid > best_ask {
        return Err(SimError::InvalidTransition);
    }
    let mid = 0.5 * (best_bid + best_ask);
    Ok((best_ask - best_bid) / mid * 10_000.0)
}

// 72) ewma-risk-model
pub fn ewma_variance_step(prev_variance: f64, return_t: f64, lambda: f64) -> Result<f64, SimError> {
    if prev_variance < 0.0 || !(0.0..1.0).contains(&lambda) {
        return Err(SimError::OutOfBounds);
    }
    Ok(lambda * prev_variance + (1.0 - lambda) * return_t * return_t)
}

pub fn ewma_variance_series(returns: &[f64], lambda: f64, seed_variance: f64) -> Result<Vec<f64>, SimError> {
    if returns.is_empty() {
        return Err(SimError::MissingField);
    }
    let mut out = Vec::with_capacity(returns.len());
    let mut var = seed_variance;
    for &r in returns {
        var = ewma_variance_step(var, r, lambda)?;
        out.push(var);
    }
    Ok(out)
}

// 73) black-scholes-baseline
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptionKind {
    Call,
    Put,
}

pub fn black_scholes_price(
    kind: OptionKind,
    spot: f64,
    strike: f64,
    rate: f64,
    vol: f64,
    time_years: f64,
) -> Result<f64, SimError> {
    if spot <= 0.0 || strike <= 0.0 || vol <= 0.0 || time_years <= 0.0 {
        return Err(SimError::OutOfBounds);
    }

    let sqrt_t = time_years.sqrt();
    let d1 = ((spot / strike).ln() + (rate + 0.5 * vol * vol) * time_years) / (vol * sqrt_t);
    let d2 = d1 - vol * sqrt_t;
    let nd1 = standard_normal_cdf(d1);
    let nd2 = standard_normal_cdf(d2);
    let discount = (-rate * time_years).exp();

    let price = match kind {
        OptionKind::Call => spot * nd1 - strike * discount * nd2,
        OptionKind::Put => strike * discount * standard_normal_cdf(-d2) - spot * standard_normal_cdf(-d1),
    };
    Ok(price)
}

fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

fn erf_approx(x: f64) -> f64 {
    let sign = x.signum();
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-ax * ax).exp();
    sign * y
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

// 26) interior-mutability-bus
pub fn push_event_safely(
    queue: &std::rc::Rc<std::cell::RefCell<Vec<String>>>,
    event: &str,
) -> Result<(), SimError> {
    let mut guard = queue.try_borrow_mut().map_err(|_| SimError::BorrowConflict)?;
    guard.push(event.to_string());
    Ok(())
}

// 27) concurrent-pipeline
pub fn concurrent_square_sum(values: Vec<i64>) -> Result<i64, SimError> {
    use std::sync::mpsc;
    use std::thread;

    let (tx_in, rx_in) = mpsc::channel::<i64>();
    let (tx_out, rx_out) = mpsc::channel::<i64>();

    let worker = thread::spawn(move || {
        while let Ok(v) = rx_in.recv() {
            if tx_out.send(v * v).is_err() {
                break;
            }
        }
    });

    for value in values {
        tx_in.send(value).map_err(|_| SimError::Parse)?;
    }
    drop(tx_in);

    let mut total = 0_i64;
    for squared in rx_out {
        total += squared;
    }

    worker.join().map_err(|_| SimError::Parse)?;
    Ok(total)
}

// 28) pin-and-self-reference
pub fn pin_vec_and_get_addr(values: Vec<i32>) -> (std::pin::Pin<Box<Vec<i32>>>, usize) {
    let pinned = Box::pin(values);
    let addr = pinned.as_ref().get_ref().as_ptr() as usize;
    (pinned, addr)
}

// 29) async-timeouts-retries
pub fn retry_with_limit<T, E, F>(attempts: usize, mut operation: F) -> Result<T, E>
where
    F: FnMut() -> Result<T, E>,
{
    assert!(attempts > 0, "attempts must be > 0");
    let mut remaining = attempts;
    loop {
        match operation() {
            Ok(v) => return Ok(v),
            Err(err) if remaining > 1 => {
                remaining -= 1;
            }
            Err(err) => return Err(err),
        }
    }
}

// 30) ffi-boundary-safety
pub fn c_string_len(ptr: *const std::os::raw::c_char) -> Result<usize, SimError> {
    if ptr.is_null() {
        return Err(SimError::NullPtr);
    }
    let c_str = unsafe { std::ffi::CStr::from_ptr(ptr) };
    c_str
        .to_str()
        .map(|s| s.len())
        .map_err(|_| SimError::Parse)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::ffi::CString;
    use std::rc::Rc;

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

    #[test]
    fn interior_mutability_reports_borrow_conflict() {
        let queue = Rc::new(RefCell::new(Vec::<String>::new()));
        let _hold = queue.borrow_mut();
        assert_eq!(
            push_event_safely(&queue, "event-a"),
            Err(SimError::BorrowConflict)
        );
    }

    #[test]
    fn concurrent_pipeline_squares_and_sums() {
        assert_eq!(concurrent_square_sum(vec![1, 2, 3, 4]), Ok(30));
        assert_eq!(concurrent_square_sum(vec![]), Ok(0));
    }

    #[test]
    fn pinned_vector_address_is_stable() {
        let (pinned, addr_before) = pin_vec_and_get_addr(vec![1, 2, 3]);
        let addr_after = pinned.as_ref().get_ref().as_ptr() as usize;
        assert_eq!(addr_before, addr_after);
    }

    #[test]
    fn retries_until_success_or_exhaustion() {
        let mut tries = 0usize;
        let ok = retry_with_limit(3, || {
            tries += 1;
            if tries < 3 { Err(SimError::Timeout) } else { Ok(42) }
        });
        assert_eq!(ok, Ok(42));

        let mut fails = 0usize;
        let err: Result<i32, SimError> = retry_with_limit(2, || {
            fails += 1;
            Err(SimError::Timeout)
        });
        assert_eq!(err, Err(SimError::Timeout));
        assert_eq!(fails, 2);
    }

    #[test]
    fn c_string_boundary_checks() {
        let msg = CString::new("rust").unwrap();
        assert_eq!(c_string_len(msg.as_ptr()), Ok(4));
        assert_eq!(c_string_len(std::ptr::null()), Err(SimError::NullPtr));
    }
}

// 26) interior-mutability-bus
use std::cell::RefCell;
use std::rc::Rc;

pub type SharedQueue<T> = Rc<RefCell<Vec<T>>>;

pub fn push_event<T>(queue: &SharedQueue<T>, event: T) {
    queue.borrow_mut().push(event);
}

// 27) concurrent-pipeline
pub fn parse_transform_aggregate(lines: &[&str]) -> Result<i64, SimError> {
    lines
        .iter()
        .map(|line| line.trim().parse::<i64>().map_err(|_| SimError::Parse))
        .map(|value| value.map(|v| v * 2))
        .try_fold(0_i64, |acc, value| {
            value.and_then(|v| acc.checked_add(v).ok_or(SimError::Overflow))
        })
}

// 28) pin-and-self-reference
pub fn pin_boxed_value<T>(value: T) -> std::pin::Pin<Box<T>> {
    Box::pin(value)
}

// 29) async-timeouts-retries
pub fn retry_with_budget<T, E, F>(attempts: usize, mut op: F) -> Result<T, E>
where
    F: FnMut() -> Result<T, E>,
{
    assert!(attempts > 0, "attempt budget must be > 0");
    let mut last_error = None;
    for _ in 0..attempts {
        match op() {
            Ok(value) => return Ok(value),
            Err(err) => last_error = Some(err),
        }
    }
    Err(last_error.expect("attempt budget guarantees at least one error"))
}

// 30) ffi-boundary-safety
pub fn c_string_ptr_len(ptr: *const std::os::raw::c_char) -> Result<usize, SimError> {
    if ptr.is_null() {
        return Err(SimError::NullPtr);
    }

    // SAFETY: pointer nullness is checked above; CStr validates termination/UTF-8 below.
    let cstr = unsafe { std::ffi::CStr::from_ptr(ptr) };
    let text = cstr.to_str().map_err(|_| SimError::Parse)?;
    Ok(text.len())
}

// 41) const-generics-window
pub fn moving_average_3(input: &[f64]) -> Vec<f64> {
    if input.len() < 3 {
        return Vec::new();
    }
    input.windows(3).map(|w| (w[0] + w[1] + w[2]) / 3.0).collect()
}

// 42) phantom-type-phase
pub struct Kelvin;
pub struct Celsius;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Temperature<Unit> {
    pub value: f64,
    _unit: std::marker::PhantomData<Unit>,
}

pub fn celsius(v: f64) -> Temperature<Celsius> {
    Temperature {
        value: v,
        _unit: std::marker::PhantomData,
    }
}

pub fn to_kelvin(t: Temperature<Celsius>) -> Temperature<Kelvin> {
    Temperature {
        value: t.value + 273.15,
        _unit: std::marker::PhantomData,
    }
}

// 43) iterator-chunking
pub fn chunk_sum(values: &[i64], chunk_size: usize) -> Vec<i64> {
    if chunk_size == 0 {
        return Vec::new();
    }
    values
        .chunks(chunk_size)
        .map(|chunk| chunk.iter().sum())
        .collect()
}

// 44) serde-boundary-plan
pub fn parse_key_value_line(line: &str) -> Result<(&str, &str), SimError> {
    line.split_once('=').ok_or(SimError::Parse)
}

// 45) deterministic-rng-injection
pub trait RngLike {
    fn next_u32(&mut self) -> u32;
}

pub fn random_in_range(rng: &mut impl RngLike, upper_exclusive: u32) -> Result<u32, SimError> {
    if upper_exclusive == 0 {
        return Err(SimError::OutOfBounds);
    }
    Ok(rng.next_u32() % upper_exclusive)
}

// 46) btreemap-vs-hashmap
pub fn sorted_word_counts(words: &[&str]) -> std::collections::BTreeMap<String, usize> {
    let mut counts = std::collections::BTreeMap::new();
    for word in words {
        *counts.entry((*word).to_owned()).or_insert(0) += 1;
    }
    counts
}

// 47) saturating-vs-checked
pub fn saturating_accumulate_u8(values: &[u8]) -> u8 {
    values
        .iter()
        .copied()
        .fold(0_u8, |acc, v| acc.saturating_add(v))
}

// 48) slice-pattern-matching
pub fn classify_triplet(values: &[i32]) -> &'static str {
    match values {
        [a, b, c] if a <= b && b <= c => "nondecreasing",
        [a, b, c] if a >= b && b >= c => "nonincreasing",
        [_, _, _] => "mixed",
        _ => "not-a-triplet",
    }
}

// 49) binary-search-contract
pub fn lower_bound(sorted: &[i64], target: i64) -> usize {
    match sorted.binary_search(&target) {
        Ok(idx) | Err(idx) => idx,
    }
}

// 50) small-dsl-evaluator
pub fn eval_add_mul(expr: &str) -> Result<i64, SimError> {
    let mut acc = 0_i64;
    for term in expr.split('+') {
        let product = term
            .split('*')
            .map(|p| p.trim().parse::<i64>().map_err(|_| SimError::Parse))
            .try_fold(1_i64, |acc, v| {
                v.and_then(|n| acc.checked_mul(n).ok_or(SimError::Overflow))
            })?;
        acc = acc.checked_add(product).ok_or(SimError::Overflow)?;
    }
    Ok(acc)
}


// 51) trait-object-dispatch
pub trait Metric {
    fn score(&self, x: f64) -> f64;
}

pub fn sum_metric(metric: &dyn Metric, xs: &[f64]) -> f64 {
    xs.iter().map(|&x| metric.score(x)).sum()
}

// 52) enum-driven-dispatch
#[derive(Debug, Clone, Copy)]
pub enum Transform {
    Square,
    Abs,
    Negate,
}

pub fn apply_transform(t: Transform, x: f64) -> f64 {
    match t {
        Transform::Square => x * x,
        Transform::Abs => x.abs(),
        Transform::Negate => -x,
    }
}

// 53) builder-default-overrides
#[derive(Debug, Clone, Copy)]
pub struct SolverConfig {
    pub dt: f64,
    pub steps: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self { dt: 0.01, steps: 100 }
    }
}

pub fn config_with_steps(steps: usize) -> Result<SolverConfig, SimError> {
    if steps == 0 {
        return Err(SimError::OutOfBounds);
    }
    Ok(SolverConfig { steps, ..SolverConfig::default() })
}

// 54) derive-more-manual-impl
#[derive(Debug, Clone, PartialEq)]
pub struct ParticleState {
    pub id: usize,
    pub energy: f64,
}

impl Eq for ParticleState {}

// 55) parsing-state-machine
pub fn parse_pair(line: &str) -> Result<(i64, i64), SimError> {
    let (a, b) = line.split_once(':').ok_or(SimError::Parse)?;
    let x = a.trim().parse::<i64>().map_err(|_| SimError::Parse)?;
    let y = b.trim().parse::<i64>().map_err(|_| SimError::Parse)?;
    Ok((x, y))
}

// 56) result-collect-partition
pub fn collect_ok_values(values: &[&str]) -> (Vec<i64>, usize) {
    let mut oks = Vec::new();
    let mut errs = 0usize;
    for v in values {
        match v.parse::<i64>() {
            Ok(n) => oks.push(n),
            Err(_) => errs += 1,
        }
    }
    (oks, errs)
}

// 57) lifetime-carrying-view
pub fn first_token<'a>(line: &'a str) -> Option<&'a str> {
    line.split_whitespace().next()
}

// 58) path-dependent-errors
pub fn ratio(numerator: f64, denominator: f64) -> Result<f64, SimError> {
    if denominator == 0.0 {
        return Err(SimError::NonConvergent);
    }
    Ok(numerator / denominator)
}

// 59) map-entry-api
pub fn increment_counter(map: &mut std::collections::HashMap<String, usize>, key: &str) {
    *map.entry(key.to_string()).or_insert(0) += 1;
}

// 60) mini-benchmark-harness
pub fn run_n_times<F>(n: usize, mut f: F)
where
    F: FnMut(),
{
    for _ in 0..n {
        f();
    }
}

// 61) question-mark-boundary
pub fn parse_and_double_result(input: &str) -> Result<i64, SimError> {
    let n = input.trim().parse::<i64>().map_err(|_| SimError::Parse)?;
    Ok(n * 2)
}

// 62) option-to-result-bridge
pub fn read_required_i64(
    values: &std::collections::HashMap<String, String>,
    key: &str,
) -> Result<i64, SimError> {
    let raw = values.get(key).ok_or(SimError::MissingField)?;
    let parsed = raw.parse::<i64>().map_err(|_| SimError::Parse)?;
    Ok(parsed)
}

// 63) result-to-option-bridge
pub fn parse_pair_option(input: &str) -> Option<(i64, i64)> {
    let (lhs, rhs) = input.split_once(':')?;
    let a = lhs.trim().parse::<i64>().ok()?;
    let b = rhs.trim().parse::<i64>().ok()?;
    Some((a, b))
}

// 64) error-conversion-with-from
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParserError {
    Empty,
    Parse,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineError {
    BadInput(ParserError),
    Overflow,
}

impl From<ParserError> for PipelineError {
    fn from(value: ParserError) -> Self {
        Self::BadInput(value)
    }
}

pub fn parse_nonempty_i64(input: &str) -> Result<i64, ParserError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(ParserError::Empty);
    }
    trimmed.parse::<i64>().map_err(|_| ParserError::Parse)
}

pub fn parse_and_square_pipeline(input: &str) -> Result<i64, PipelineError> {
    let n = parse_nonempty_i64(input)?;
    n.checked_mul(n).ok_or(PipelineError::Overflow)
}

// 65) main-return-result
pub fn cli_sum(args: &[String]) -> Result<i64, SimError> {
    let first = args.first().ok_or(SimError::MissingField)?;
    let second = args.get(1).ok_or(SimError::MissingField)?;
    let a = first.parse::<i64>().map_err(|_| SimError::Parse)?;
    let b = second.parse::<i64>().map_err(|_| SimError::Parse)?;
    a.checked_add(b).ok_or(SimError::Overflow)
}
