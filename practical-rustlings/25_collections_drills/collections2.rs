fn chunk_sum(values: &[i64], chunk_size: usize) -> Vec<i64> {
    if chunk_size == 0 {
        return Vec::new();
    }

    values
        .chunks(chunk_size)
        .map(|chunk| chunk.iter().sum())
        .collect()
}

fn main() {
    let out = chunk_sum(&[1, 2, 3, 4, 5], 2);
    assert_eq!(out, vec![3, 7, 5]);
}
