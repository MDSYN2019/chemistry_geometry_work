// Retry a fallible operation up to `attempts` times.

pub fn retry_with_limit<T, E, F>(attempts: usize, mut op: F) -> Result<T, E>
where
    F: FnMut() -> Result<T, E>,
{
    assert!(attempts > 0, "attempts must be > 0");
    let mut remaining = attempts;
    loop {
        match op() {
            Ok(v) => return Ok(v),
            Err(_) if remaining > 1 => remaining -= 1,
            Err(e) => return Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retries_until_success() {
        let mut n = 0;
        let result = retry_with_limit(3, || {
            n += 1;
            if n < 3 { Err("not yet") } else { Ok(42) }
        });
        assert_eq!(result, Ok(42));
    }
}
