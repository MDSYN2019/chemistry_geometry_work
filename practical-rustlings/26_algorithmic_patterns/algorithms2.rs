fn saturating_accumulate_u8(values: &[u8]) -> u8 {
    values
        .iter()
        .copied()
        .fold(0_u8, |acc, v| acc.saturating_add(v))
}

fn main() {
    assert_eq!(saturating_accumulate_u8(&[250, 10]), u8::MAX);
}
