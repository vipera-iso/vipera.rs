//! Shared error type for every vipera crate.
//!
//! One enum, `Display` + `std::error::Error`, and an exit code the CLI maps directly. Library code
//! never panics; callers decide whether an error is fatal.

use std::fmt;

/// Every failure the engine can report, with the file offset when the failure came from reading a model.
#[derive(Debug)]
pub enum ViperaError {
    /// Filesystem failure while opening or reading a model.
    Io(std::io::Error),
    /// The file is structurally invalid. `offset` is where parsing stopped.
    Parse {
        /// Byte offset in the file where the problem was detected.
        offset: u64,
        /// What was wrong, phrased for a user.
        message: String,
    },
    /// The file is valid but uses a feature this build does not implement.
    Unsupported {
        /// What is not supported.
        message: String,
    },
    /// The command line itself was malformed: an unknown flag, a missing argument.
    Usage {
        /// What the caller should have written instead.
        message: String,
    },
    /// A caller-supplied value is out of range, such as a negative temperature.
    InvalidArgument {
        /// What was wrong with the value.
        message: String,
    },
    /// A placement request cannot be satisfied on this hardware.
    Infeasible {
        /// Which constraint failed and by how much.
        message: String,
    },
    /// A bug or an unexpected state. Always worth a report.
    Internal {
        /// What invariant was violated.
        message: String,
    },
}

impl ViperaError {
    /// A structural parse failure at a known offset.
    pub fn parse(offset: u64, message: impl Into<String>) -> Self {
        Self::Parse {
            offset,
            message: message.into(),
        }
    }

    /// A valid file using an unimplemented feature.
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::Unsupported {
            message: message.into(),
        }
    }

    /// A malformed command line.
    pub fn usage(message: impl Into<String>) -> Self {
        Self::Usage {
            message: message.into(),
        }
    }

    /// A caller-supplied value that is out of range.
    pub fn invalid_argument(message: impl Into<String>) -> Self {
        Self::InvalidArgument {
            message: message.into(),
        }
    }

    /// A request that this hardware cannot satisfy.
    pub fn infeasible(message: impl Into<String>) -> Self {
        Self::Infeasible {
            message: message.into(),
        }
    }

    /// An internal invariant violation.
    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal {
            message: message.into(),
        }
    }

    /// Process exit code for this error, per `contracts/cli.md`.
    pub const fn exit_code(&self) -> i32 {
        match self {
            Self::Usage { .. } => 1,
            Self::Io(_)
            | Self::Parse { .. }
            | Self::Unsupported { .. }
            | Self::InvalidArgument { .. } => 2,
            Self::Infeasible { .. } => 3,
            Self::Internal { .. } => 4,
        }
    }
}

impl fmt::Display for ViperaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "{e}"),
            Self::Parse { offset, message } => {
                write!(f, "invalid model at byte {offset}: {message}")
            }
            Self::Unsupported { message } => write!(f, "unsupported: {message}"),
            Self::Usage { message } => write!(f, "usage error: {message}; run `vipera --help`"),
            Self::InvalidArgument { message } => write!(f, "invalid argument: {message}"),
            Self::Infeasible { message } => write!(f, "infeasible placement: {message}"),
            Self::Internal { message } => write!(f, "internal error: {message}"),
        }
    }
}

impl std::error::Error for ViperaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ViperaError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_error_names_the_offset() {
        let e = ViperaError::parse(17, "bad magic");
        assert_eq!(e.to_string(), "invalid model at byte 17: bad magic");
        assert_eq!(e.exit_code(), 2);
    }

    #[test]
    fn infeasible_has_its_own_exit_code() {
        assert_eq!(ViperaError::infeasible("vram").exit_code(), 3);
    }

    #[test]
    fn a_usage_error_exits_one() {
        let e = ViperaError::usage("unknown flag \"--nope\"");
        assert_eq!(e.exit_code(), 1);
        assert!(e.to_string().contains("usage error"));
    }

    #[test]
    fn an_invalid_argument_is_a_usage_error() {
        let e = ViperaError::invalid_argument("temperature must not be negative");
        assert_eq!(e.exit_code(), 2);
        assert_eq!(
            e.to_string(),
            "invalid argument: temperature must not be negative"
        );
    }

    #[test]
    fn io_errors_convert_and_keep_their_source() {
        let io = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
        let e: ViperaError = io.into();
        assert_eq!(e.exit_code(), 2);
        assert!(std::error::Error::source(&e).is_some());
    }
}
