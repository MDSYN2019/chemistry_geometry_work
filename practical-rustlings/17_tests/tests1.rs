// Tests are important to ensure that your code does what you think it should
// do.

fn is_even(n: i64) -> bool {
    n % 2 == 0
}

pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[allow(dead_code)]
fn bad_code(a: i32, b: i32) -> i32 {
    a - b
}
fn main() {
    // You can optionally experiment here.
}

#[cfg(test)]
mod tests {
    // TODO: Import `is_even`. You can use a wildcard to import everything in
    // the outer module.

    use super::*;

    #[test]
    fn you_can_assert() {
        // TODO: Test the function `is_even` with some values.
        assert!(is_even(2), true);
        assert!(is_even(4), true);
    }
}
