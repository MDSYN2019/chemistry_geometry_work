/*
clone on write smart pointer - it can enclose
and provide immutable access to borrowed data and cloen the
data lazily when mutation or ownership is required.

Cow is for borrow if unchanged, own if modified

 */

use std::borrow::Cow;

fn example(input: &[i64]) -> Cow<'_, [i64]> {
    if input.len() == 4 {
        let mut v = input.to_vec();
        v.push(30);
        return Cow::Owned(v); // owned if modified
    } else {
        // do something else
        Cow::Borrowed(input) // borrowed if unchanged
    }
}

fn main() {
    let numbers = vec![27, 297, 38502, 81]; // this should be borrowed
    let numbers_2 = vec![26, 30, 10]; // this should be owned
    let numbers_modified = example(&numbers);
    let numbers_2_modified = example(&numbers_2);

    // Checking what kind of Cow objects we have

    println!("owned? {}", matches!(numbers_modified, Cow::Owned(_)));
    println!("owned? {}", matches!(numbers_2_modified, Cow::Owned(_)));
}
