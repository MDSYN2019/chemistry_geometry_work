// Basic pinning drill: pin a vector and verify its backing pointer is stable.

pub fn pin_and_address(values: Vec<u8>) -> (std::pin::Pin<Box<Vec<u8>>>, usize) {
    let pinned = Box::pin(values);
    let addr = pinned.as_ref().get_ref().as_ptr() as usize;
    (pinned, addr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn address_stays_same_while_pinned() {
        let (pinned, before) = pin_and_address(vec![1, 2, 3]);
        let after = pinned.as_ref().get_ref().as_ptr() as usize;
        assert_eq!(before, after);
    }
}
