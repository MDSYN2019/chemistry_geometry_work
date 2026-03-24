// Use scoped threads to borrow slices without cloning.

use std::thread;

pub fn scoped_sum(values: &[i64]) -> i64 {
    let mid = values.len() / 2;
    let (left, right) = values.split_at(mid);

    thread::scope(|scope| {
        let left_handle = scope.spawn(move || left.iter().sum::<i64>());
        let right_sum = right.iter().sum::<i64>();
        left_handle.join().expect("scoped thread panicked") + right_sum
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sums_in_parallel() {
        assert_eq!(scoped_sum(&[1, 2, 3, 4]), 10);
    }
}
