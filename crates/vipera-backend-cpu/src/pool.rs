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

use crate::quant::{self, BatchActivation, OwnedActivation};

/// One vertical slice of one batched matvec: `count` rows starting at `first_row`, each dotted with
/// the same `B` activations.
struct Job<'d, const B: usize> {
    /// The whole tensor; the chunk is addressed by `first_row` and `stride`.
    data: &'d [u8],
    /// Bytes per row.
    stride: usize,
    ty: GgmlType,
    first_row: usize,
    count: usize,
    activations: Arc<[OwnedActivation; B]>,
}

/// Workers started for one pass, each with its own job queue.
///
/// `B` is the number of activations a job carries; a batch of one is the single-sequence path.
pub(crate) struct Pool<'d, const B: usize> {
    workers: Vec<Sender<Job<'d, B>>>,
    replies: Receiver<(usize, Vec<f32>)>,
}

impl<'d, const B: usize> Pool<'d, B> {
    /// Start `threads` workers inside `scope`.
    pub(crate) fn new<'scope>(scope: &'scope Scope<'scope, 'd>, threads: usize) -> Self
    where
        'd: 'scope,
    {
        let (sender, replies) = mpsc::channel();
        let mut workers = Vec::with_capacity(threads);
        for _ in 0..threads {
            let (jobs, queue) = mpsc::channel::<Job<'d, B>>();
            workers.push(jobs);
            let sender = sender.clone();
            scope.spawn(move || run_worker(&queue, &sender));
        }
        Self { workers, replies }
    }

    /// Spread the rows of `out` over at most `threads` participants and fill them.
    ///
    /// One of those participants is the calling thread, which computes the last chunk instead of
    /// blocking on the replies. That is not just saved idle time: a pass that wakes eight workers
    /// and waits leaves nine runnable threads for eight logical CPUs, and the work they lose is
    /// measured. On a 200 MB Q4_K read the workers-only form streamed 28.5 GB/s and handing the
    /// caller one chunk made it 31.1; the rest of the gap to the 42 GB/s the kernel reaches on its
    /// own came from the explicit prefetch, which `fill_rows_batch` no longer issues.
    ///
    /// `out` holds `B * rows` values, token-major: entry `entry`'s `rows` results start at
    /// `entry * rows`. That layout makes copying a worker's reply back `B` memcpys instead of one
    /// per row.
    ///
    /// A worker that has gone away is not fatal: the chunk it would have taken is computed on this
    /// thread instead, so a panic in one job degrades to the serial path rather than hanging on a
    /// reply that will never come.
    pub(crate) fn fill_batch(
        &self,
        ty: GgmlType,
        data: &'d [u8],
        stride: usize,
        threads: usize,
        activations: &Arc<[OwnedActivation; B]>,
        out: &mut [f32],
    ) -> Result<(), ViperaError> {
        if B == 0 {
            return Err(ViperaError::invalid_argument(
                "a matvec pool needs at least one entry",
            ));
        }
        if !out.len().is_multiple_of(B) {
            return Err(ViperaError::internal(format!(
                "{} outputs are not a whole number of {B}-entry batches",
                out.len()
            )));
        }
        let rows = out.len() / B;
        let threads = threads.min(self.workers.len()).max(1);
        if threads == 1 {
            let activation = BatchActivation::new(activations);
            quant::fill_rows_batch(ty, data, stride, 0, &activation, out);
            return Ok(());
        }

        let per_thread = rows.div_ceil(threads);
        let mut dispatched = 0;
        let mut chunk = 0;
        let mut first_row = 0;
        // Leave the final chunk for this thread: the workers start on theirs while it runs, so all
        // `threads` CPUs are doing kernel work instead of one waiting on channels.
        while first_row < rows && chunk + 1 < threads {
            let count = per_thread.min(rows - first_row);
            let job = Job {
                data,
                stride,
                ty,
                first_row,
                count,
                activations: Arc::clone(activations),
            };
            match self.workers.get(chunk).map(|sender| sender.send(job)) {
                Some(Ok(())) => dispatched += 1,
                // No worker for this chunk, or it is gone: do its share here.
                _ => {
                    let activation = BatchActivation::new(activations);
                    let mut values = vec![0.0_f32; count * B];
                    quant::fill_rows_batch(ty, data, stride, first_row, &activation, &mut values);
                    copy_back::<B>(out, rows, first_row, &values)?;
                }
            }
            chunk += 1;
            first_row += count;
        }
        if first_row < rows {
            let count = rows - first_row;
            let activation = BatchActivation::new(activations);
            let mut values = vec![0.0_f32; count * B];
            // The caller's chunk has to fail the same way a worker's does: a shape mismatch is a
            // panic in the kernel, and the contract is an error, not an unwind through the engine.
            let computed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                quant::fill_rows_batch(ty, data, stride, first_row, &activation, &mut values);
            }));
            if computed.is_err() {
                return Err(ViperaError::internal(
                    "a matvec chunk panicked; refusing to return partial results",
                ));
            }
            copy_back::<B>(out, rows, first_row, &values)?;
        }

        let mut worker_panicked = false;
        for _ in 0..dispatched {
            let (first_row, values) = self
                .replies
                .recv()
                .map_err(|_| ViperaError::internal("a matvec worker stopped before replying"))?;
            if first_row == usize::MAX {
                worker_panicked = true;
                continue;
            }
            copy_back::<B>(out, rows, first_row, &values)?;
        }
        if worker_panicked {
            return Err(ViperaError::internal(
                "a matvec worker panicked; refusing to return partial results",
            ));
        }
        Ok(())
    }
}

/// Copy one chunk's token-major results into the token-major output.
fn copy_back<const B: usize>(
    out: &mut [f32],
    rows: usize,
    first_row: usize,
    values: &[f32],
) -> Result<(), ViperaError> {
    let count = values.len() / B;
    if values.len() != count * B {
        return Err(ViperaError::internal(
            "a matvec worker replied for rows outside the tensor",
        ));
    }
    for entry in 0..B {
        let start = entry * rows + first_row;
        let slot = out.get_mut(start..start + count).ok_or_else(|| {
            ViperaError::internal("a matvec worker replied for rows outside the tensor")
        })?;
        let reply = &values[entry * count..(entry + 1) * count];
        slot.copy_from_slice(reply);
    }
    Ok(())
}

/// Take jobs until the queue closes, answering each with the rows it produced.
fn run_worker<'d, const B: usize>(
    jobs: &Receiver<Job<'d, B>>,
    replies: &Sender<(usize, Vec<f32>)>,
) {
    for job in jobs {
        let mut values = vec![0.0_f32; job.count * B];
        let activation = BatchActivation::new(&job.activations);
        // A panic inside one job must not hang the parent in replies.recv():
        // catch it, signal the parent with the sentinel, and keep the worker alive.
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            quant::fill_rows_batch(
                job.ty,
                job.data,
                job.stride,
                job.first_row,
                &activation,
                &mut values,
            );
        }))
        .is_err();
        if panicked {
            if replies.send((usize::MAX, Vec::new())).is_err() {
                return;
            }
            continue;
        }
        if replies.send((job.first_row, values)).is_err() {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::OwnedActivation;

    #[test]
    fn a_panicking_worker_returns_an_error_instead_of_hanging() {
        // Data far too short for the stride row_bytes implies: fill_rows_batch
        // would panic inside the worker (slice out of bounds).
        let data = vec![0u8; 10];
        let activation = OwnedActivation::new(vipera_core::gguf::GgmlType::F32, &[1.0f32; 32]);
        std::thread::scope(|scope| {
            let pool = Pool::<1>::new(scope, 2);
            let activations = Arc::new([activation]);
            let mut out = vec![0.0f32; 64];
            let result = pool.fill_batch(
                vipera_core::gguf::GgmlType::F32,
                &data,
                128,
                2,
                &activations,
                &mut out,
            );
            assert!(result.is_err(), "expected internal error, got {result:?}");
        });
    }
}
