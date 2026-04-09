// Exercise 78: polling-state-machine
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PollState { Idle, Readable, Writable, Closed }

pub fn advance(state: PollState, event: &str) -> Option<PollState> {
    use PollState::*;
    match (state, event) {
        (Idle, "read") => Some(Readable),
        (Idle, "write") => Some(Writable),
        (Readable, "drain") | (Writable, "flush") => Some(Idle),
        (_, "close") => Some(Closed),
        _ => None,
    }
}
