/*

Rc<T> in rust means 'Reference Counted shared ownership'

It lets multiple parts of a program own the same data, and the data is dropped only
when the last owner disappears



*/

// In this exercise, we want to express the concept of multiple owners via the
// `Rc<T>` type. This is a model of our solar system - there is a `Sun` type and
// multiple `Planet`s. The planets take ownership of the sun, indicating that
// they revolve around the sun.

use std::rc::Rc;

#[derive(Debug)]
struct Sun;

#[derive(Debug)]
enum Planet {
    Mercury(Rc<Sun>),
    Venus(Rc<Sun>),
    Earth(Rc<Sun>),
    Mars(Rc<Sun>),
    Jupiter(Rc<Sun>),
    Saturn(Rc<Sun>),
    Uranus(Rc<Sun>),
    Neptune(Rc<Sun>),
}

impl Planet {
    fn details(&self) {
        println!("Hi from {self:?}!");
    }
}

enum Expr {
    // expr is recursive - children are stored as Rc<Expr>
    Number(i32),
    Add(Rc<Expr>, Rc<Expr>), // both are reference counted pointers to Expr
    Multiply(Rc<Expr>, Rc<Expr>),
}

fn evaluate(expr: &Expr) -> i32 {
    match expr {
        Expr::Number(b) => *b,
        Expr::Add(a, b) => evaluate(a) + evaluate(b), // If these are numbers, then we reutnr the integer value and sum
        Expr::Multiply(a, b) => evaluate(a) * evaluate(b),
    }
}

fn main() {
    let five = Rc::new(Expr::Number(5));
    let three = Rc::new(Expr::Number(3));
    let sum = Rc::new(Expr::Add(Rc::clone(&five), Rc::clone(&three)));
    let two = Rc::new(Expr::Number(2));
    let expr = Expr::Multiply(Rc::clone(&sum), Rc::clone(&two));

    // =----
    let a = Rc::new(String::from("hello"));
    let b = Rc::clone(&a);
    let c = Rc::clone(&a);

    // You can optionally experiment here.
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rc1() {
        let sun = Rc::new(Sun);
        println!("reference count = {}", Rc::strong_count(&sun)); // 1 reference

        let mercury = Planet::Mercury(Rc::clone(&sun));
        println!("reference count = {}", Rc::strong_count(&sun)); // 2 references
        mercury.details();

        let venus = Planet::Venus(Rc::clone(&sun));
        println!("reference count = {}", Rc::strong_count(&sun)); // 3 references
        venus.details();

        let earth = Planet::Earth(Rc::clone(&sun));
        println!("reference count = {}", Rc::strong_count(&sun)); // 4 references
        earth.details();

        let mars = Planet::Mars(Rc::clone(&sun));
        println!("reference count = {}", Rc::strong_count(&sun)); // 5 references
        mars.details();

        let jupiter = Planet::Jupiter(Rc::clone(&sun));
        println!("reference count = {}", Rc::strong_count(&sun)); // 6 references
        jupiter.details();

        // TODO
        let saturn = Planet::Saturn(Rc::new(Sun));
        println!("reference count = {}", Rc::strong_count(&sun)); // 7 references
        saturn.details();

        // TODO
        let uranus = Planet::Uranus(Rc::new(Sun));
        println!("reference count = {}", Rc::strong_count(&sun)); // 8 references
        uranus.details();

        // TODO
        let neptune = Planet::Neptune(Rc::new(Sun));
        println!("reference count = {}", Rc::strong_count(&sun)); // 9 references
        neptune.details();

        assert_eq!(Rc::strong_count(&sun), 9);

        drop(neptune);
        println!("reference count = {}", Rc::strong_count(&sun)); // 8 references

        drop(uranus);
        println!("reference count = {}", Rc::strong_count(&sun)); // 7 references

        drop(saturn);
        println!("reference count = {}", Rc::strong_count(&sun)); // 6 references

        drop(jupiter);
        println!("reference count = {}", Rc::strong_count(&sun)); // 5 references

        drop(mars);
        println!("reference count = {}", Rc::strong_count(&sun)); // 4 references

        // TODO
        println!("reference count = {}", Rc::strong_count(&sun)); // 3 references

        // TODO
        println!("reference count = {}", Rc::strong_count(&sun)); // 2 references

        // TODO
        println!("reference count = {}", Rc::strong_count(&sun)); // 1 reference

        assert_eq!(Rc::strong_count(&sun), 1);
    }
}
