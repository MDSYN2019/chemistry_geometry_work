// Return borrowed text when no normalization is needed, owned text otherwise.

use std::borrow::Cow;

pub fn normalize_spaces(input: &str) -> Cow<'_, str> {
    if input.contains("  ") {
        Cow::Owned(input.split_whitespace().collect::<Vec<_>>().join(" "))
    } else {
        Cow::Borrowed(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avoids_alloc_when_possible() {
        let x = normalize_spaces("a b c");
        assert!(matches!(x, Cow::Borrowed(_)));
    }

    #[test]
    fn allocates_when_needed() {
        let x = normalize_spaces("a  b   c");
        assert_eq!(x, "a b c");
    }
}
