trait AppendBar {
    fn append_bar(self) -> Self;
}

impl AppendBar for Vec<String> {
    fn append_bar(mut self) -> Self {
        self.push("Bar".to_string());
        self
    }
}
// TODO: Implement the trait `AppendBar` for a vector of strings.
// `append_bar` should push the string "Bar" into the vector.

fn main() {
    // You can optionally experiment here.
    let test_args: Vec<String> = vec!["-w".to_string(), "60".to_string(), "arg".to_string()]; // should array of Strings
    let test_args = test_args.append_bar();
    println!("{:?}", test_args);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_vec_pop_eq_bar() {
        let mut foo = vec![String::from("Foo")].append_bar();
        assert_eq!(foo.pop().unwrap(), "Bar");
        assert_eq!(foo.pop().unwrap(), "Foo");
    }
}
