// Exercise 85: io-error-mapping
use std::io::ErrorKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DomainErr { Timeout, Parse, Missing, Other }

pub fn map_error(kind: ErrorKind) -> DomainErr {
    match kind {
        ErrorKind::TimedOut => DomainErr::Timeout,
        ErrorKind::InvalidData => DomainErr::Parse,
        ErrorKind::NotFound => DomainErr::Missing,
        _ => DomainErr::Other,
    }
}
