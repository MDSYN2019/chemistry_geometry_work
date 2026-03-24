// This powerful wrapper provides the ability to store a positive integer value.
// TODO: Rewrite it using a generic so that it supports wrapping ANY type.
struct Wrapper<T, A> {
    value: T,
    value2: A,
}

// TODO: Adapt the struct's implementation to be generic over the wrapped value.
impl<T, A> Wrapper<T, A> {
    fn new(value: T, value2: A) -> Self {
        Wrapper { value, value2 }
    }
}

fn main() {
    // You can optionally experiment here.

    let mut A = Wrapper {
        value: 1,
        value2: 30,
    };

    println!("{} {}", A.value, A.value2);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_u32_in_wrapper() {
        assert_eq!(Wrapper::new(42).value, 42);
    }

    #[test]
    fn store_str_in_wrapper() {
        assert_eq!(Wrapper::new("Foo").value, "Foo");
    }
}
