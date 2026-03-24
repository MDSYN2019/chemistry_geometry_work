fn fill_vec(vec: Vec<i32>) -> Vec<i32> {
    let mut vec = vec;
    vec.push(88);
    vec
}

fn main() {
    // You can optionally experiment here.
    let mut example_vec: Vec<i32> = vec![1, 2, 3, 4, 5];
    example_vec.push(50);
    let output = fill_vec(example_vec);
    println!("{:?}", output);
}

#[cfg(test)]
mod tests {
    use super::*;

    // TODO: Make both vectors `vec0` and `vec1` accessible at the same time to
    // fix the compiler error in the test.
    #[test]
    fn move_semantics2() {
        let vec0 = vec![22, 44, 66];
        let vec1 = fill_vec(vec0.clone());

        assert_eq!(vec0, [22, 44, 66]);
        assert_eq!(vec1, [22, 44, 66, 88]);
    }
}
