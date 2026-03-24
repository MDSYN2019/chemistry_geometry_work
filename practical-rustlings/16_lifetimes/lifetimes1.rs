// The Rust compiler needs to know how to check whether supplied references are
// valid, so that it can let the programmer know if a reference is at risk of
// going out of scope before it is used. Remember, references are borrows and do
// not own their own data. What if their owner goes out of scope?

/*
The rust compiler needs to know how to check whether supplied references
are valid, so that it can let the programmer know if a reference is at risk of
going out of scope before it is used

 */

// TODO: Fix the compiler error by updating the function signature.
fn longest<'a>(x: &'a str, y: &'a str) -> &'a str {
    // The returned reference must not outlive the thing it points to
    /*
    'a describes the relationship between input references and output reference

    both inputs must live at least 'a
    returned references lvies at most '

    'a is the overlapped lifetime

    There exists some lifetime 'a -  x is a &str that is valid for at least 'a,
    y is a &ste that is valid for at least 'a

    The returned &str will also be valid for at least 'a


     */
    if x.len() > y.len() { x } else { y }
}

fn main() {
    let r;
    {
        let x = 5;
        r = &x;
    }

    let s1 = String::from("hi");
    let s2 = String::from("longer");

    let result = longest(s1.as_str(), s2.as_str());
    println!("{result}");

    //println!("r: {r}");

    // You can optionally experiment here.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_longest() {
        assert_eq!(longest("abcd", "123"), "abcd");
        assert_eq!(longest("abc", "1234"), "1234");
    }
}
