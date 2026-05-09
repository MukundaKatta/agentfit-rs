use agentfit::{Fitter, Message, Strategy, Tokenizer};

struct ExactTokenizer;
impl Tokenizer for ExactTokenizer {
    /// Each char = 1 token. Used to make tests deterministic without
    /// depending on the char-approx rounding.
    fn count(&self, text: &str) -> usize {
        text.chars().count()
    }
}

fn fitter(max_tokens: usize) -> Fitter {
    Fitter::new(max_tokens)
        .with_tokenizer(ExactTokenizer)
        .with_per_message_overhead(0)
}

#[test]
fn under_budget_returns_unchanged() {
    let msgs = vec![Message::user("hi")];
    let kept = fitter(100).fit(msgs.clone(), Strategy::DropOldest);
    assert_eq!(kept, msgs);
}

#[test]
fn drop_oldest_keeps_system_and_last_user() {
    let msgs = vec![
        Message::system("S"),                  // 1 token
        Message::user("AAAAAAAAAAAA"),         // 12
        Message::assistant("BBBBBBBBBBBB"),    // 12
        Message::user("Q?"),                   // 2
    ];
    // Total = 27. Budget = 5. DropOldest must remove A and B but keep S and Q?.
    let kept = fitter(5).fit(msgs, Strategy::DropOldest);
    assert_eq!(kept[0].role, "system");
    assert_eq!(kept.last().unwrap().role, "user");
    assert_eq!(kept.last().unwrap().content, "Q?");
    // Oldest dropped before the last; either both gone or just the oldest.
    assert!(kept.len() <= 3);
}

#[test]
fn drop_oldest_works_without_system() {
    let msgs = vec![
        Message::user("AAAAAAAAAAA"),
        Message::assistant("BBBBBBBBBBB"),
        Message::user("Q?"),
    ];
    let kept = fitter(5).fit(msgs, Strategy::DropOldest);
    assert_eq!(kept.last().unwrap().content, "Q?");
}

#[test]
fn drop_middle_preserves_recent_and_anchors() {
    let msgs = vec![
        Message::system("S"),
        Message::user("U1"),
        Message::assistant("A1"),
        Message::user("U2"),
        Message::assistant("A2"),
        Message::user("Q?"),
    ];
    // Force a drop by tight budget.
    let kept = fitter(8).fit(msgs, Strategy::DropMiddle);
    assert_eq!(kept[0].role, "system");
    assert_eq!(kept.last().unwrap().content, "Q?");
}

#[test]
fn truncate_largest_shrinks_a_message() {
    let big = "X".repeat(40);
    let msgs = vec![
        Message::system("S"),
        Message::user(big),
        Message::user("Q?"),
    ];
    let kept = fitter(15).fit(msgs, Strategy::TruncateLargest);
    assert_eq!(kept[0].role, "system");
    assert_eq!(kept.last().unwrap().content, "Q?");
    // The middle content should be shorter and contain the truncation marker.
    assert!(kept[1].content.len() < 40);
    assert!(kept[1].content.contains("[truncated]"));
}

#[test]
fn count_includes_per_message_overhead() {
    let f = Fitter::new(0)
        .with_tokenizer(ExactTokenizer)
        .with_per_message_overhead(3);
    let msgs = vec![Message::user("hi"), Message::user("ho")];
    // 2 chars + 3 overhead = 5 per msg, 2 msgs = 10
    assert_eq!(f.count(&msgs), 10);
}

#[test]
fn anchors_only_returned_if_below_anchors_budget() {
    let msgs = vec![
        Message::system("Lorem ipsum"),
        Message::user("very long question that exceeds budget"),
    ];
    // Budget too small even for anchors; we keep them as-is.
    let kept = fitter(5).fit(msgs.clone(), Strategy::DropOldest);
    assert_eq!(kept, msgs);
}

#[test]
fn char_approx_default_works() {
    // chars/4 rounded up.
    let f = Fitter::new(2);
    // "abcd" = 4 chars = 1 token under default; +4 overhead = 5 total > 2.
    let msgs = vec![Message::user("abcd"), Message::user("efgh")];
    let kept = f.fit(msgs, Strategy::DropOldest);
    // Either the last user is kept alone, or both still fit; in any case last-user is preserved.
    assert_eq!(kept.last().unwrap().content, "efgh");
}
