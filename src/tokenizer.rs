/// Counts tokens in a string. Implementations should be fast and stable.
pub trait Tokenizer: Send + Sync {
    /// Return an integer token count for `text`.
    fn count(&self, text: &str) -> usize;
}

/// Zero-dependency approximation: `chars.len() / 4`. Coarse but predictable.
///
/// Good enough for "fit to a budget with a safety margin"; for exact bills
/// or hard limits, enable the `tiktoken` feature and use
/// [`TiktokenTokenizer`] instead.
#[derive(Debug, Clone, Copy, Default)]
pub struct CharApproxTokenizer;

impl Tokenizer for CharApproxTokenizer {
    fn count(&self, text: &str) -> usize {
        let n = text.chars().count();
        // Round up so we never under-estimate.
        (n + 3) / 4
    }
}

#[cfg(feature = "tiktoken")]
pub use tiktoken_impl::TiktokenTokenizer;

#[cfg(feature = "tiktoken")]
mod tiktoken_impl {
    use super::Tokenizer;
    use tiktoken_rs::CoreBPE;

    /// Accurate BPE tokenizer backed by `tiktoken-rs`. Construct with one of
    /// the helpers below to pin a specific encoding.
    pub struct TiktokenTokenizer {
        bpe: CoreBPE,
    }

    impl TiktokenTokenizer {
        /// `cl100k_base` — used by GPT-3.5-turbo, GPT-4, and most OpenAI models.
        pub fn cl100k_base() -> Self {
            Self {
                bpe: tiktoken_rs::cl100k_base().expect("cl100k_base loads"),
            }
        }

        /// `o200k_base` — used by GPT-4o family.
        pub fn o200k_base() -> Self {
            Self {
                bpe: tiktoken_rs::o200k_base().expect("o200k_base loads"),
            }
        }

        /// Wrap any pre-built `CoreBPE`.
        pub fn from_bpe(bpe: CoreBPE) -> Self {
            Self { bpe }
        }
    }

    impl Tokenizer for TiktokenTokenizer {
        fn count(&self, text: &str) -> usize {
            self.bpe.encode_with_special_tokens(text).len()
        }
    }
}
