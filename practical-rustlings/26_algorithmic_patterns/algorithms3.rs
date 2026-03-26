fn eval_add_mul(expr: &str) -> Option<i64> {
    let mut acc = 0_i64;
    for term in expr.split('+') {
        let mut product = 1_i64;
        for factor in term.split('*') {
            let value = factor.trim().parse::<i64>().ok()?;
            product = product.checked_mul(value)?;
        }
        acc = acc.checked_add(product)?;
    }
    Some(acc)
}

fn main() {
    assert_eq!(eval_add_mul("2 * 3 + 4"), Some(10));
    assert_eq!(eval_add_mul("x + 1"), None);
}
