// The trait `AppendBar` has only one function which appends "Bar" to any object
// implementing this trait.

/*


*/

trait AppendBar {
    /*
    Define the methods in the trait appendbar that are
     */
    fn append_bar(self) -> Self;

    fn append_another_bar(self) -> Self;
}

impl AppendBar for String {
    // TODO: Implement `AppendBar` for the type `String`.
    fn append_bar(self) -> Self {
        self + " bar"
    }

    fn append_another_bar(self) -> Self {
        self + " another bar"
    }
}

fn main() {
    let s = String::from("Foo");
    let s = s.append_bar(); // we are simply appending a 'bar' string 
    let s = s.append_another_bar(); // we are appending another bar string 
    println!("s: {s}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_foo_bar() {
        assert_eq!(String::from("Foo").append_bar(), "FooBar");
    }

    #[test]
    fn is_bar_bar() {
        assert_eq!(String::from("").append_bar().append_bar(), "BarBar");
    }
}
