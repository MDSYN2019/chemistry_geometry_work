// Model a tiny typed state machine.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Initialized;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Running;
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Finished;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sim<S> {
    step: usize,
    state: S,
}

impl Sim<Initialized> {
    pub fn new() -> Self {
        Self {
            step: 0,
            state: Initialized,
        }
    }

    pub fn start(self) -> Sim<Running> {
        Sim {
            step: self.step,
            state: Running,
        }
    }
}

impl Sim<Running> {
    pub fn advance(mut self) -> Self {
        self.step += 1;
        self
    }

    pub fn finish(self) -> Sim<Finished> {
        Sim {
            step: self.step,
            state: Finished,
        }
    }
}

impl Sim<Finished> {
    pub fn steps(&self) -> usize {
        self.step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn typed_transitions_work() {
        let done = Sim::new().start().advance().advance().finish();
        assert_eq!(done.steps(), 2);
    }
}
