// At compile time, Rust needs to know how much space a type takes up. This
// becomes problematic for recursive types, where a value can have as part of
// itself another value of the same type. To get around the issue, we can use a
// `Box` - a smart pointer used to store data on the heap, which also allows us
// to wrap a recursive type.
//
// The recursive type we're implementing in this exercise is the "cons list", a
// data structure frequently found in functional programming languages. Each
// item in a cons list contains two elements: The value of the current item and
// the next item. The last item is a value called `Nil`.

// TODO: Use a `Box` in the enum definition to make the code compile.

/*

All values in Rust are stack allocated by default. Values can be boxed (allocated on the heap) by creating a Box<T>.

A box is a smart pointer to a heap allocated value of type T.
When a box goes out of scope, it's destructor is called, the inner object is destroyed, and the memory on the heap is freed.


A box is a smart pointer to a heap allocated value of type T.

When a box goes out of scope, it's destructor is called, the inner object is destroyed, and the memory on the heap is freed.




*/

use std::mem;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
struct Point {
    x: f64,
    y: f64,
}

struct Rectangle {
    top_left: Point,
    bottom_right: Point,
}

fn origin() -> Box<Point> {
    Box::new(Point { x: 0.0, y: 0.0 })
}

#[derive(PartialEq, Debug)]
enum List {
    Cons(i32, Box<List>), // takes the input of a box and List?
    Nil,
}

// TODO: Create an empty cons list.
fn create_empty_list() -> List {
    List::Nil
}

// TODO: Create a non-empty cons list.
fn create_non_empty_list() -> List {
    List::Cons(23, Box::new(List::Cons(32, Box::new(List::Nil))))
}

fn main() {
    // ---
    let point: Point = origin();
    let rectangle: Rectangle = Rectangle {
        top_left: origin(),
        bottom_right: Point { x: 3.0, y: -4.0 },
    };

    // Heap allocated rectangle
    let boxed_rectangle = Box::new(Rectangle {
        top_left: origin(),
        bottom_right: Point { x: 3.0, y: -4.0 },
    });

    // ---

    println!("This is an empty cons list: {:?}", create_empty_list());
    println!(
        "This is a non-empty cons list: {:?}",
        create_non_empty_list(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_empty_list() {
        assert_eq!(create_empty_list(), List::Nil);
    }

    #[test]
    fn test_create_non_empty_list() {
        assert_ne!(create_empty_list(), create_non_empty_list());
    }
}
