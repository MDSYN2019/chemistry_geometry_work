/*
Cow - Borrow if possible, own if necessary
 */

use std::borrow::Cow;

fn make_uppercase(text: &mut Cow<'_, str>) -> () {
    /*
    This is where 'clone on write' part shows up
     */
    if text.chars().any(|c| c.is_lowercase()) {
        let owned = text.to_mut(); // clones if only borrowed - so this was owned
        owned.make_ascii_uppercase();
    }
}

fn main() {
    let mut a: Cow<'_, str> = Cow::Borrowed("HELLO");
    let mut b: Cow<'_, str> = Cow::Borrowed("hello");

    make_uppercase(&mut a);
    make_uppercase(&mut b);

    println!("a = {}", a);
    println!("b = {}", b);
}
