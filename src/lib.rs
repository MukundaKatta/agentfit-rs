//! Fit chat messages to an LLM context window.
//!
//! Token-aware truncation with pluggable tokenizers and multiple
//! strategies. Default ships with a `CharApproxTokenizer` (chars/4) for
//! zero-dep coarse counting; enable the `tiktoken` feature for accurate
//! BPE counting via [`tiktoken-rs`](https://crates.io/crates/tiktoken-rs).
//!
//! # Quick start
//!
//! ```
//! use agentfit::{Fitter, Message, Strategy};
//!
//! let messages = vec![
//!     Message::system("You are concise."),
//!     Message::user("First question..."),
//!     Message::assistant("First answer..."),
//!     Message::user("Second question, the one we actually care about."),
//! ];
//!
//! // 30 chars-as-tokens budget; oldest middle messages get dropped.
//! let fitter = Fitter::new(30);
//! let kept = fitter.fit(messages, Strategy::DropOldest);
//! assert!(!kept.is_empty());
//! // System message and the trailing user turn are always preserved.
//! assert_eq!(kept.first().unwrap().role, "system");
//! assert_eq!(kept.last().unwrap().role, "user");
//! ```
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

mod fitter;
mod tokenizer;

pub use crate::fitter::{Fitter, Message, Strategy};
pub use crate::tokenizer::{CharApproxTokenizer, Tokenizer};

#[cfg(feature = "tiktoken")]
pub use crate::tokenizer::TiktokenTokenizer;
