fn moving_average_3(input: &[f64]) -> Vec<f64> {
    if input.len() < 3 {
        return Vec::new();
    }
    input
        .windows(3)
        .map(|w| (w[0] + w[1] + w[2]) / 3.0)
        .collect()
}

fn main() {
    let out = moving_average_3(&[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(out, vec![2.0, 3.0]);
}
