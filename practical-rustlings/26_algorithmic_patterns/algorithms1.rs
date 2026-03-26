fn lower_bound(sorted: &[i64], target: i64) -> usize {
    match sorted.binary_search(&target) {
        Ok(idx) | Err(idx) => idx,
    }
}

fn main() {
    assert_eq!(lower_bound(&[1, 3, 5], 4), 2);
    assert_eq!(lower_bound(&[1, 3, 5], 5), 2);
}
