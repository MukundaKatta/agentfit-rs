# agentfit

[![crates.io](https://img.shields.io/crates/v/agentfit.svg)](https://crates.io/crates/agentfit)
[![docs.rs](https://docs.rs/agentfit/badge.svg)](https://docs.rs/agentfit)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Fit chat messages to an LLM context window. Token-aware truncation with multiple strategies and pluggable tokenizers.

```toml
[dependencies]
agentfit = "0.1"
# For accurate BPE counting:
agentfit = { version = "0.1", features = ["tiktoken"] }
```

## Why

Every agent loop eventually outgrows its context window. Naive truncation drops the wrong things — the system prompt, or the trailing user turn that's the actual question. `agentfit` is the small, focused primitive that does it correctly: anchors the system prompt and the final user turn, drops or shrinks the middle.

## Quick start

```rust
use agentfit::{Fitter, Message, Strategy};

let messages = vec![
    Message::system("You are concise."),
    Message::user("First question..."),
    Message::assistant("First answer..."),
    Message::user("Second question, the actual one."),
];

let fitter = Fitter::new(8_000); // budget in tokens
let kept = fitter.fit(messages, Strategy::DropOldest);
```

The default tokenizer is a zero-dep `chars / 4` approximation. Enable the `tiktoken` feature for accurate BPE counting:

```rust,ignore
use agentfit::{Fitter, TiktokenTokenizer, Strategy};

let fitter = Fitter::new(8_000)
    .with_tokenizer(TiktokenTokenizer::cl100k_base());  // GPT-4 family
```

## Strategies

- **`DropOldest`** — remove non-anchor messages from the start until the budget fits. Best default for chat assistants.
- **`DropMiddle`** — bisect the eligible range and drop messages from the middle outward, keeping early context and recent context.
- **`TruncateLargest`** — last resort: shrink the longest non-anchor message in place with a `…[truncated]` marker.

## Anchors

`agentfit` always preserves:

- The first `system` message (if any).
- The trailing `user` message — the one your model is actually answering.

Even if the budget is so tight that anchors alone exceed it, they're returned as-is. You decide what to do.

## What it doesn't do

- No streaming truncation (whole-message at a time).
- No multimodal (`Message::content` is `String`).
- No summarization-as-strategy in v0.1; that's a v0.2 candidate.
- Doesn't call any LLM.

## Sibling: JS `@mukundakatta/agentfit`

JS users: see [@mukundakatta/agentfit](https://www.npmjs.com/package/@mukundakatta/agentfit) on npm.

## License

MIT
