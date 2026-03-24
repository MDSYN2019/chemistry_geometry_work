// This exercise explores the `Cow` (Clone-On-Write) smart pointer. It can
// enclose and provide immutable access to borrowed data and clone the data
// lazily when mutation or ownership is required. The type is designed to work
// with general borrowed data via the `Borrow` trait.

use std::borrow::Cow;

// cow - hold either a borrow view or an owned value

fn normalize_username(input: &str) -> Cow<'_, str> {
    /*
    Copy-on-write (COW) usually means: hold either a borrowed view (&T) or an owned value (T), and
    only allocate/clone into an owned form if/when you need to mutate or take ownership



     */
    // <'_, str> - not sure what this means - does this we are returning an owned pointer, or a string>
    // it it is already lowercase, we can return a borrowed slice
    if input.bytes().all(|b| !b.is_ascii_uppercase()) {
        return Cow::Borrowed(input);
    }

    // Otherwise, we must allocate a new string
    Cow::Owned(input.to_ascii_lowercase())
}

fn maybe_uppercase(input: &str) -> Cow<str> {
    if input.chars().any(|c| c.is_lowercase()) {
        Cow::Owned(input.to_uppercase())
    } else {
        Cow::Borrowed(input)
    }
}

fn abs_all(input: &mut Cow<[i32]>) {
    // a reference to a list of integers. But if you force it to become a mutable, then it will copy itself into an owned buffer
    for ind in 0..input.len() {
        let value = input[ind];
        if value < 0 {
            // Clones into a vector if not already owned.
            input.to_mut()[ind] = -value; // here we force the input to become mutable, so we can now allocate 
        }
    }
}

fn main() {
    let a = normalize_username("sang");
    let b = normalize_username("Sang");

    println!("a = {a}, owned? {}", matches!(a, Cow::Owned(_)));
    println!("b = {b}, owned? {}", matches!(b, Cow::Owned(_)));
    // You can optionally experiment here.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_mutation() {
        // Clone occurs because `input` needs to be mutated.
        let vec = vec![-1, 0, 1];
        let mut input = Cow::from(&vec);
        abs_all(&mut input);
        assert!(matches!(input, Cow::Owned(_)));
    }

    #[test]
    fn reference_no_mutation() {
        // No clone occurs because `input` doesn't need to be mutated.
        let vec = vec![0, 1, 2];
        let mut input = Cow::from(&vec);
        abs_all(&mut input);
        // TODO: Replace `todo!()` with `Cow::Owned(_)` or `Cow::Borrowed(_)`.
        assert!(matches!(input, Cow::Borrowed(_)));
    }

    #[test]
    fn owned_no_mutation() {
        // We can also pass `vec` without `&` so `Cow` owns it directly. In this
        // case, no mutation occurs (all numbers are already absolute) and thus
        // also no clone. But the result is still owned because it was never
        // borrowed or mutated.
        let vec = vec![0, 1, 2];
        let mut input = Cow::from(vec);
        abs_all(&mut input);
        // TODO: Replace `todo!()` with `Cow::Owned(_)` or `Cow::Borrowed(_)`.
        assert!(matches!(input, Cow::Owned(_)));
    }

    #[test]
    fn owned_mutation() {
        // Of course this is also the case if a mutation does occur (not all
        // numbers are absolute). In this case, the call to `to_mut()` in the
        // `abs_all` function returns a reference to the same data as before.
        let vec = vec![-1, 0, 1];
        let mut input = Cow::from(vec);
        abs_all(&mut input);
        // TODO: Replace `todo!()` with `Cow::Owned(_)` or `Cow::Borrowed(_)`.
        assert!(matches!(input, Cow::Owned(_)));
    }
}
