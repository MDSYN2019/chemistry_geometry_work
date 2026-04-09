// Exercise 88: cooperative-tick-scheduler
pub fn run_round_robin(tasks: &mut [usize], ticks: usize) {
    if tasks.is_empty() { return; }
    for turn in 0..ticks {
        let idx = turn % tasks.len();
        tasks[idx] += 1;
    }
}
