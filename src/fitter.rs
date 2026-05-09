use crate::tokenizer::{CharApproxTokenizer, Tokenizer};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// One chat message. The `role` is provider-specific (`system`, `user`,
/// `assistant`, `tool`, etc.).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Message {
    /// Role, e.g. `"system"`, `"user"`, `"assistant"`.
    pub role: String,
    /// Plain-text content. Multimodal not supported in v0.1.
    pub content: String,
}

impl Message {
    /// Construct a `system` message.
    pub fn system(text: impl Into<String>) -> Self {
        Self { role: "system".into(), content: text.into() }
    }
    /// Construct a `user` message.
    pub fn user(text: impl Into<String>) -> Self {
        Self { role: "user".into(), content: text.into() }
    }
    /// Construct an `assistant` message.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self { role: "assistant".into(), content: text.into() }
    }
}

/// Truncation strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Drop oldest non-anchor messages (after `system`, before final user) first.
    DropOldest,
    /// Drop messages from the middle outward, keeping the most recent.
    DropMiddle,
    /// Truncate the *content* of the largest message in place. Last resort.
    TruncateLargest,
}

/// Fits messages to a token budget.
#[derive(Clone)]
pub struct Fitter {
    max_tokens: usize,
    tokenizer: Arc<dyn Tokenizer>,
    /// Per-message overhead added by the provider (e.g. role/separator tokens).
    /// OpenAI uses ~4 tokens per message; tweak for your model.
    per_message_overhead: usize,
}

impl Fitter {
    /// Construct with a token budget; uses [`CharApproxTokenizer`] by default.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            tokenizer: Arc::new(CharApproxTokenizer),
            per_message_overhead: 4,
        }
    }

    /// Override the tokenizer.
    pub fn with_tokenizer<T: Tokenizer + 'static>(mut self, t: T) -> Self {
        self.tokenizer = Arc::new(t);
        self
    }

    /// Override the per-message overhead (defaults to 4).
    pub fn with_per_message_overhead(mut self, n: usize) -> Self {
        self.per_message_overhead = n;
        self
    }

    /// Compute the total token count of a message list under the current
    /// tokenizer + overhead.
    pub fn count(&self, messages: &[Message]) -> usize {
        messages
            .iter()
            .map(|m| self.tokenizer.count(&m.content) + self.per_message_overhead)
            .sum()
    }

    /// Fit `messages` to the budget using `strategy`.
    ///
    /// Anchors are always kept: the first `system` message (if any) and the
    /// trailing `user` message. If even those exceed the budget, returns
    /// them as-is — the caller decides how to handle that.
    pub fn fit(&self, mut messages: Vec<Message>, strategy: Strategy) -> Vec<Message> {
        if self.count(&messages) <= self.max_tokens {
            return messages;
        }

        // Identify anchors.
        let has_system = messages.first().map(|m| m.role == "system").unwrap_or(false);
        let last_idx = messages.len().checked_sub(1);
        let last_is_user = last_idx
            .map(|i| messages[i].role == "user")
            .unwrap_or(false);

        // Compute the inclusive index range that's eligible for dropping.
        let drop_start = if has_system { 1 } else { 0 };
        let drop_end_exclusive = if last_is_user {
            messages.len() - 1
        } else {
            messages.len()
        };

        match strategy {
            Strategy::DropOldest => {
                while self.count(&messages) > self.max_tokens && drop_start < messages.len() {
                    let cur_drop_end = if last_is_user {
                        messages.len() - 1
                    } else {
                        messages.len()
                    };
                    if drop_start >= cur_drop_end {
                        break;
                    }
                    messages.remove(drop_start);
                }
            }
            Strategy::DropMiddle => {
                while self.count(&messages) > self.max_tokens {
                    let lo = drop_start;
                    let hi = if last_is_user {
                        messages.len().saturating_sub(1)
                    } else {
                        messages.len()
                    };
                    if hi <= lo {
                        break;
                    }
                    let mid = lo + (hi - lo) / 2;
                    messages.remove(mid);
                }
            }
            Strategy::TruncateLargest => {
                const MARKER: &str = " …[truncated]";
                loop {
                    if self.count(&messages) <= self.max_tokens {
                        break;
                    }
                    // Find the largest content among non-anchor messages.
                    let candidate_indices: Vec<usize> = (drop_start..drop_end_exclusive).collect();
                    if candidate_indices.is_empty() {
                        break;
                    }
                    let (idx, cur_len) = candidate_indices
                        .into_iter()
                        .map(|i| (i, messages[i].content.chars().count()))
                        .max_by_key(|&(_, n)| n)
                        .unwrap();
                    if cur_len == 0 {
                        break;
                    }
                    // Halve and reattach a fixed marker.
                    let chars: Vec<char> = messages[idx].content.chars().collect();
                    let keep = chars.len() / 2;
                    let new_content: String =
                        chars.iter().take(keep).collect::<String>() + MARKER;
                    let new_len = new_content.chars().count();
                    // If shrinkage stalls (marker dominates), bail; we've
                    // hit the floor of what TruncateLargest can do.
                    if new_len >= cur_len {
                        break;
                    }
                    messages[idx].content = new_content;
                    if keep == 0 {
                        break;
                    }
                }
            }
        }

        messages
    }
}
