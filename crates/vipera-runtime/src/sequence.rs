//! One conversation's retained state: tokens, lifecycle, and restorable snapshots.
//!
//! A sequence survives across turns so its processed context can be reused (FR-009). It is
//! removed only on release, cancellation, or failure (FR-011); finishing a turn keeps the blocks.

use vipera_errors::ViperaError;

/// Where a sequence is in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequenceState {
    /// Accepted, nothing computed yet.
    Admitted,
    /// Computing prompt tokens.
    Prefill,
    /// Producing tokens one at a time.
    Decode,
    /// The current turn ended normally; the retained context stays for the next turn.
    Finished,
    /// The client gave up. Blocks are released.
    Cancelled,
    /// Generation failed. Blocks are released.
    Failed,
}

impl SequenceState {
    /// Name used in logs and JSON.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Admitted => "admitted",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::Finished => "finished",
            Self::Cancelled => "cancelled",
            Self::Failed => "failed",
        }
    }
}

/// Why a turn stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    /// A stop token was sampled.
    EndOfSequence,
    /// The token budget for this turn ran out.
    Length,
    /// The caller cancelled.
    Cancelled,
    /// The forward pass failed.
    Error,
}

impl FinishReason {
    /// Name used in logs and JSON.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EndOfSequence => "stop",
            Self::Length => "length",
            Self::Cancelled => "cancelled",
            Self::Error => "error",
        }
    }
}

/// A restorable point in a sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snapshot {
    /// Identifier used by restore.
    pub id: u32,
    /// Token position the sequence returns to.
    pub token_position: usize,
    /// Optional caller-supplied name.
    pub label: Option<String>,
}

/// The retained state of one conversation.
#[derive(Debug, Clone)]
pub struct Sequence {
    id: u64,
    state: SequenceState,
    tokens: Vec<u32>,
    processed: usize,
    max_new_tokens: u32,
    finish: Option<FinishReason>,
    snapshots: Vec<Snapshot>,
    next_snapshot: u32,
    pending_reused: usize,
}

impl Sequence {
    /// A sequence with no context yet.
    pub fn new(id: u64, max_new_tokens: u32) -> Self {
        Self {
            id,
            state: SequenceState::Admitted,
            tokens: Vec::new(),
            processed: 0,
            max_new_tokens,
            finish: None,
            snapshots: Vec::new(),
            next_snapshot: 0,
            pending_reused: 0,
        }
    }

    /// Identifier.
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// Lifecycle state.
    pub const fn state(&self) -> SequenceState {
        self.state
    }

    /// All retained tokens: prompt plus everything generated so far.
    pub fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// Number of retained tokens.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the sequence holds no tokens.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Tokens already sent through the forward pass.
    pub const fn processed(&self) -> usize {
        self.processed
    }

    /// Record that `count` tokens now have KV state.
    pub fn set_processed(&mut self, count: usize) {
        self.processed = count.min(self.tokens.len());
    }

    /// Per-turn token budget.
    pub const fn max_new_tokens(&self) -> u32 {
        self.max_new_tokens
    }

    /// Set the per-turn token budget.
    pub fn set_max_new_tokens(&mut self, value: u32) {
        self.max_new_tokens = value;
    }

    /// Why the last turn stopped.
    pub const fn finish_reason(&self) -> Option<FinishReason> {
        self.finish
    }

    /// Snapshots still valid, oldest first.
    pub fn snapshots(&self) -> &[Snapshot] {
        &self.snapshots
    }

    /// Prefix length reused by the turn about to run, and clears it.
    pub fn take_reused(&mut self) -> usize {
        std::mem::take(&mut self.pending_reused)
    }

    /// Append one generated token to the retained context, keeping the turn in decode.
    pub fn push_generated(&mut self, token: u32) {
        self.tokens.push(token);
    }

    /// Append prompt tokens to the retained context.
    pub fn push_prompt(&mut self, tokens: &[u32]) {
        self.tokens.extend_from_slice(tokens);
        self.finish = None;
        if self.state != SequenceState::Admitted {
            self.state = SequenceState::Admitted;
        }
    }

    /// How many leading tokens `tokens` shares with the retained context.
    pub fn common_prefix(&self, tokens: &[u32]) -> usize {
        self.tokens
            .iter()
            .zip(tokens)
            .take_while(|(left, right)| left == right)
            .count()
    }

    /// Cut the retained context back to `position`, dropping KV state past it as well.
    pub fn truncate(&mut self, position: usize) {
        self.tokens.truncate(position);
        self.processed = self.processed.min(position);
        self.snapshots
            .retain(|snapshot| snapshot.token_position <= position);
        self.finish = None;
        if self.state != SequenceState::Admitted {
            self.state = SequenceState::Admitted;
        }
    }

    /// Note that the next turn reused `reused` tokens.
    pub fn set_reused(&mut self, reused: usize) {
        self.pending_reused = reused;
    }

    /// Enter prefill.
    pub fn begin_prefill(&mut self) {
        self.state = SequenceState::Prefill;
        self.finish = None;
    }

    /// Enter decode.
    pub fn begin_decode(&mut self) {
        self.state = SequenceState::Decode;
    }

    /// End the turn normally. Blocks stay allocated for the next turn.
    pub fn finish(&mut self, reason: FinishReason) {
        self.state = SequenceState::Finished;
        self.finish = Some(reason);
    }

    /// The caller cancelled: the sequence will not be reused.
    pub fn cancel(&mut self) {
        self.state = SequenceState::Cancelled;
        self.finish = Some(FinishReason::Cancelled);
    }

    /// Generation failed: the sequence will not be reused.
    pub fn fail(&mut self) {
        self.state = SequenceState::Failed;
        self.finish = Some(FinishReason::Error);
    }

    /// Whether the sequence can still be used for another turn.
    pub const fn is_usable(&self) -> bool {
        matches!(
            self.state,
            SequenceState::Admitted
                | SequenceState::Prefill
                | SequenceState::Decode
                | SequenceState::Finished
        )
    }

    /// Record a restorable point at the current end of the context.
    pub fn snapshot(&mut self, label: Option<String>) -> Result<u32, ViperaError> {
        if !self.is_usable() {
            return Err(ViperaError::invalid_argument(format!(
                "sequence {} is {} and cannot be snapshotted",
                self.id,
                self.state.as_str()
            )));
        }
        let id = self.next_snapshot;
        self.next_snapshot += 1;
        self.snapshots.push(Snapshot {
            id,
            token_position: self.tokens.len(),
            label,
        });
        Ok(id)
    }

    /// Find a snapshot by id.
    pub fn snapshot_by_id(&self, id: u32) -> Option<&Snapshot> {
        self.snapshots.iter().find(|snapshot| snapshot.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncation_drops_later_snapshots_and_processed_state() {
        let mut sequence = Sequence::new(1, 8);
        sequence.push_prompt(&[1, 2, 3, 4]);
        sequence.set_processed(4);
        let early = sequence.snapshot(None).unwrap();
        sequence.push_prompt(&[5, 6]);
        let late = sequence.snapshot(None).unwrap();
        assert_eq!(sequence.snapshots().len(), 2);

        sequence.truncate(3);
        assert_eq!(sequence.tokens(), &[1, 2, 3]);
        assert_eq!(sequence.processed(), 3);
        assert!(
            sequence.snapshot_by_id(early).is_none(),
            "snapshot at 4 is invalid"
        );
        assert!(sequence.snapshot_by_id(late).is_none());
        assert_eq!(sequence.state(), SequenceState::Admitted);
    }

    #[test]
    fn a_cancelled_sequence_is_not_usable_and_cannot_be_snapshotted() {
        let mut sequence = Sequence::new(1, 8);
        sequence.push_prompt(&[1]);
        sequence.cancel();
        assert!(!sequence.is_usable());
        assert!(sequence.snapshot(None).is_err());
    }

    #[test]
    fn the_common_prefix_is_the_reusable_part() {
        let mut sequence = Sequence::new(1, 8);
        sequence.push_prompt(&[1, 2, 3]);
        assert_eq!(sequence.common_prefix(&[1, 2, 3, 4]), 3);
        assert_eq!(sequence.common_prefix(&[1, 9]), 1);
        assert_eq!(sequence.common_prefix(&[]), 0);
    }

    #[test]
    fn finishing_a_turn_keeps_the_context_usable() {
        let mut sequence = Sequence::new(1, 8);
        sequence.push_prompt(&[1, 2]);
        sequence.finish(FinishReason::Length);
        assert_eq!(sequence.state(), SequenceState::Finished);
        assert!(sequence.is_usable());
        sequence.push_prompt(&[3]);
        assert_eq!(sequence.tokens(), &[1, 2, 3]);
        assert_eq!(sequence.state(), SequenceState::Admitted);
    }
}
