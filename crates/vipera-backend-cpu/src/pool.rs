//! A thread pool that lives for one forward pass instead of one `matvec`.
//!
//! `matvec` used to spawn its workers and join them on every call. A token makes 197 calls, and the
//! spawn alone measured 53-74 us per call on an i5-11300H: 10-14 ms of a token's ~120 ms. The
//! threads here are started once per pass and reused.
//!
//! The workers are scoped (`std::thread::scope`) rather than detached, which is what lets a job
//! borrow the model's bytes for the length of the pass. A detached pool would need those bytes to
//! be `'static`, which means either leaking the file or the `unsafe` this workspace forbids outside
//! [`crate::simd`]. The pool therefore cannot outlive the pass that built it.
//!
//! Results come back over a channel rather than through a borrowed output slice: handing a worker
//! `&mut [f32]` would make that borrow last as long as the pool, and the caller could not touch its
//! own output buffer afterwards. Copying a chunk back costs microseconds against a `matvec` that
//! reads megabytes.

use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::Scope;

use vipera_core::gguf::GgmlType;
use vipera_errors::ViperaError;

use crate::quant::{self, OwnedActivation};

/// One vertical slice of one matvec: `count` rows starting at `first_row`.
struct Job<'d> {
    /// The whole tensor; the chunk is addressed by `first_row` and `stride`.
    data: &'d [u8],
    /// Bytes per row.
    stride: usize,
    ty: GgmlType,
    first_row: usize,
    count: usize,
    activation: Arc<OwnedActivation>,
}

/// Workers started for one pass, each with its own job queue.
pub(crate) struct Pool<'d> {
    workers: Vec<Sender<Job<'d>>>,
    replies: Receiver<(usize, Vec<f32>)>,
}

impl<'d> Pool<'d> {
    /// Start `threads` workers inside `scope`.
    pub(crate) fn new<'scope>(scope: &'scope Scope<'scope, 'd>, threads: usize) -> Self
    where
        'd: 'scope,
    {
        let (sender, replies) = mpsc::channel();
        let mut workers = Vec::with_capacity(threads);
        for _ in 0..threads {
            let (jobs, queue) = mpsc::channel::<Job<'d>>();
            workers.push(jobs);
            let sender = sender.clone();
            scope.spawn(move || run_worker(&queue, &sender));
        }
        Self { workers, replies }
    }

    /// Spread the rows of `out` over at most `threads` workers and fill them.
    ///
    /// A worker that has gone away is not fatal: the chunk it would have taken is computed on this
    /// thread instead, so a panic in one job degrades to the serial path rather than hanging on a
    /// reply that will never come.
    pub(crate) fn fill(
        &self,
        ty: GgmlType,
        data: &'d [u8],
        stride: usize,
        threads: usize,
        activation: &Arc<OwnedActivation>,
        out: &mut [f32],
    ) -> Result<(), ViperaError> {
        let threads = threads.min(self.workers.len()).max(1);
        if threads == 1 {
            quant::fill_rows(ty, data, stride, 0, activation.borrowed(), out);
            return Ok(());
        }

        let per_thread = out.len().div_ceil(threads);
        let mut dispatched = 0;
        for (chunk, slice) in out.chunks_mut(per_thread).enumerate() {
            let first_row = chunk * per_thread;
            let job = Job {
                data,
                stride,
                ty,
                first_row,
                count: slice.len(),
                activation: Arc::clone(activation),
            };
            match self.workers.get(chunk).map(|sender| sender.send(job)) {
                Some(Ok(())) => dispatched += 1,
                // No worker for this chunk, or it is gone: do its share here.
                _ => quant::fill_rows(ty, data, stride, first_row, activation.borrowed(), slice),
            }
        }

        for _ in 0..dispatched {
            let (first_row, values) = self
                .replies
                .recv()
                .map_err(|_| ViperaError::internal("a matvec worker stopped before replying"))?;
            let end = first_row + values.len();
            let slot = out.get_mut(first_row..end).ok_or_else(|| {
                ViperaError::internal("a matvec worker replied for rows outside the tensor")
            })?;
            slot.copy_from_slice(&values);
        }
        Ok(())
    }
}

/// Take jobs until the queue closes, answering each with the rows it produced.
fn run_worker<'d>(jobs: &Receiver<Job<'d>>, replies: &Sender<(usize, Vec<f32>)>) {
    for job in jobs {
        let mut values = vec![0.0_f32; job.count];
        quant::fill_rows(
            job.ty,
            job.data,
            job.stride,
            job.first_row,
            job.activation.borrowed(),
            &mut values,
        );
        if replies.send((job.first_row, values)).is_err() {
            return;
        }
    }
}
