use std::collections::BTreeMap;

fn sorted_word_counts(words: &[&str]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for word in words {
        *counts.entry((*word).to_owned()).or_insert(0) += 1;
    }
    counts
}

fn main() {
    let counts = sorted_word_counts(&["beta", "alpha", "beta"]);
    assert_eq!(counts.get("alpha"), Some(&1));
    assert_eq!(counts.get("beta"), Some(&2));
}
