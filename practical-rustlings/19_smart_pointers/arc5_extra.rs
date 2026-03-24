use std::collections::HashMap;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

#[derive(Debug, PartialEq, Eq)]
enum DivisionError {
    DivideByZero,
    IntegerOverflow,
    NotDivisible,
}

fn divide(a: i64, b: i64) -> Result<i64, DivisionError> {
    match b {
        _ => Ok(a / b),
        0 => Err(DivisionError::DivideByZero),
    }
}

fn main() {
    let mut numbers = VecDeque::from([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    //let evens = numbers
    //    .extract_if(.., |x| *x % 2 == 0)
    //    .collect::<VecDeque<_>>();
    //let odds = numbers;
    let (evens, odds): (VecDeque<_>, VecDeque<_>) =
        numbers.clone().into_iter().partition(|x| x % 2 == 0);
    println!("evens: {:?}", evens);
    println!("odds: {:?}", odds);

    // iterator
    let odds_iterator_division_results = odds.into_iter().map(|n| divide(n, 3));
    println!("odds after iterator operation: {:?}", odds_iterator);
}
