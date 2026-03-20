//! Compact reference implementations for selected topics.

use std::marker::PhantomData;

/// Reference for exercise 2: `Option<Result<T,E>>` -> `Result<Option<T>,E>`.
pub fn transpose<T, E>(input: Option<Result<T, E>>) -> Result<Option<T>, E> {
    match input {
        None => Ok(None),
        Some(Ok(v)) => Ok(Some(v)),
        Some(Err(e)) => Err(e),
    }
}

/// Reference for exercise 10: typestate builder.
pub struct Missing;
pub struct Present;

pub struct TypedBuilder<State> {
    temp_k: Option<f64>,
    _state: PhantomData<State>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuiltSystem {
    pub temp_k: f64,
}

impl TypedBuilder<Missing> {
    pub fn new() -> Self {
        Self {
            temp_k: None,
            _state: PhantomData,
        }
    }

    pub fn temperature_k(self, temp_k: f64) -> TypedBuilder<Present> {
        TypedBuilder {
            temp_k: Some(temp_k),
            _state: PhantomData,
        }
    }
}

impl TypedBuilder<Present> {
    pub fn build(self) -> BuiltSystem {
        BuiltSystem {
            temp_k: self.temp_k.expect("state guarantees temp is present"),
        }
    }
}

/// Reference for exercise 13: unit-safe newtypes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Seconds(pub f64);
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Meters(pub f64);
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetersPerSecond(pub f64);

impl std::ops::Mul<Seconds> for MetersPerSecond {
    type Output = Meters;

    fn mul(self, rhs: Seconds) -> Self::Output {
        Meters(self.0 * rhs.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transpose_examples() {
        assert_eq!(transpose::<i32, &'static str>(None), Ok(None));
        assert_eq!(transpose::<i32, &'static str>(Some(Ok(3))), Ok(Some(3)));
        assert_eq!(transpose::<i32, &'static str>(Some(Err("bad"))), Err("bad"));
    }

    #[test]
    fn typestate_builder_example() {
        let built = TypedBuilder::new().temperature_k(300.0).build();
        assert_eq!(built, BuiltSystem { temp_k: 300.0 });
    }

    #[test]
    fn units_example() {
        let d = MetersPerSecond(2.0) * Seconds(4.0);
        assert_eq!(d, Meters(8.0));
    }
}
