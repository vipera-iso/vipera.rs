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
//! A worker takes two kinds of work. Rows of a product are the main one. The other is building the
//! owned activations of a batched product ([`Pool::prepare`]): quantising and summing 128 entries of
//! a prompt chunk is 0.16 ms at width 1536 and 1.1 ms at 8960, a chunk makes 196 such products, and
//! done on the calling thread it leaves every worker idle for that time.
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

use crate::quant::{self, BatchActivation, Fusion, OwnedActivation};
use crate::simd::SharedTile;

/// A run of entries to turn into owned activations. Each buffer holds the entry's values followed
/// by room for its group sums, so a worker finishes it in place instead of copying it again.
struct PrepJob {
    ty: GgmlType,
    first_entry: usize,
    values: usize,
    buffers: Vec<Vec<f32>>,
    /// The transform the caller would otherwise run on its own thread (the feed-forward norm, the
    /// SwiGLU multiply). Applied to each buffer here, before the sums and the quantiser.
    fusion: Fusion,
    /// The product-wide Q4_K activation tile this run also fills, when the caller prepared one.
    tile: Option<Arc<SharedTile>>,
}

/// What a worker is asked to do.
enum Task<'d, const B: usize> {
    Rows(Job<'d, B>),
    Prep(PrepJob),
}

/// The finished activations of a [`PrepJob`], keyed by its first entry; `usize::MAX` and no
/// activations when the job panicked.
type PrepReply = (usize, Vec<OwnedActivation>);

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
    /// The Q4_K activation tile built by the prep workers and shared by every participant.
    tile: Option<Arc<SharedTile>>,
}

/// Workers started for one pass, each with its own job queue.
///
/// `B` is the number of activations a job carries; a batch of one is the single-sequence path.
pub(crate) struct Pool<'d, const B: usize> {
    workers: Vec<Sender<Task<'d, B>>>,
    replies: Receiver<(usize, Vec<f32>)>,
    prepared: Receiver<PrepReply>,
    /// Activation-tile buffers handed back after their product, so a chunk's ~196 products do not
    /// allocate a megabyte each. A chunk alternates two sizes (width 1536 and 8960), and a buffer
    /// that has served the larger one keeps its capacity for both.
    tiles: std::sync::Mutex<Vec<Vec<i8>>>,
}

impl<'d, const B: usize> Pool<'d, B> {
    /// Start `threads` workers inside `scope`.
    pub(crate) fn new<'scope>(scope: &'scope Scope<'scope, 'd>, threads: usize) -> Self
    where
        'd: 'scope,
    {
        let (sender, replies) = mpsc::channel();
        let (prepared_sender, prepared) = mpsc::channel();
        let mut workers = Vec::with_capacity(threads);
        for _ in 0..threads {
            let (jobs, queue) = mpsc::channel::<Task<'d, B>>();
            workers.push(jobs);
            let sender = sender.clone();
            let prepared_sender = prepared_sender.clone();
            scope.spawn(move || run_worker(&queue, &sender, &prepared_sender));
        }
        Self {
            workers,
            replies,
            prepared,
            tiles: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Take a tile buffer for this product, or allocate one when the free list is empty.
    pub(crate) fn take_tile(&self, blocks: usize, halves: usize) -> Arc<SharedTile> {
        let buffer = self
            .tiles
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .pop();
        Arc::new(match buffer {
            Some(buffer) => SharedTile::from_buffer(buffer, blocks, halves),
            None => SharedTile::new(blocks, halves),
        })
    }

    /// Return a tile's buffer to the free list, if this was the last reference to it.
    pub(crate) fn recycle_tile(&self, tile: Arc<SharedTile>) {
        let Ok(tile) = Arc::try_unwrap(tile) else {
            return;
        };
        let buffer = tile.into_buffer();
        let mut free = self.tiles.lock().unwrap_or_else(|error| error.into_inner());
        // Two shapes per chunk; a handful of buffers covers them without holding memory forever.
        if free.len() < 4 {
            free.push(buffer);
        }
    }

    /// The owned activations of `B` entries, built by up to `threads` participants at once.
    ///
    /// Entry for entry the result is what `OwnedActivation::new` gives, bit for bit: the same
    /// sums and the same quantiser run on the same values, only on different threads. The calling
    /// thread copies the entries into buffers first, because a worker cannot borrow its stack,
    /// and keeps the last run for itself. A worker that has gone away does not fail the call: its
    /// run is built here instead, as [`Pool::fill_batch`] does for rows.
    pub(crate) fn prepare(
        &self,
        ty: GgmlType,
        activations: &[&[f32]; B],
        fusion: &Fusion,
        threads: usize,
        tile: Option<Arc<SharedTile>>,
    ) -> Result<Arc<[OwnedActivation; B]>, ViperaError> {
        let participants = threads.min(self.workers.len()).min(B).max(1);
        let values = activations.first().map_or(0, |x| x.len());
        let groups = values / 32;
        let mut buffers: Vec<Vec<f32>> = activations
            .iter()
            .map(|x| {
                let mut buffer = Vec::with_capacity(x.len() + groups);
                buffer.extend_from_slice(x);
                buffer
            })
            .collect();
        let per_participant = B.div_ceil(participants);
        let mut ready: Vec<(usize, Vec<OwnedActivation>)> = Vec::with_capacity(participants);
        let mut dispatched = 0;
        // Split from the back so each run is moved out without shifting the ones before it.
        let mut runs: Vec<(usize, Vec<Vec<f32>>)> = Vec::new();
        let mut end = B;
        while end > 0 {
            let start = end.saturating_sub(per_participant);
            runs.push((start, buffers.split_off(start)));
            end = start;
        }
        runs.reverse();
        let own = runs.pop();
        for (chunk, (first_entry, run)) in runs.into_iter().enumerate() {
            let job = PrepJob {
                ty,
                first_entry,
                values,
                buffers: run,
                fusion: fusion.clone(),
                tile: tile.clone(),
            };
            let rejected = match self.workers.get(chunk) {
                Some(sender) => match sender.send(Task::Prep(job)) {
                    Ok(()) => {
                        dispatched += 1;
                        None
                    }
                    Err(mpsc::SendError(Task::Prep(job))) => Some(job),
                    Err(mpsc::SendError(Task::Rows(_))) => None,
                },
                // No worker for this run: build it here.
                None => Some(job),
            };
            if let Some(job) = rejected {
                ready.push((job.first_entry, finish_prep(job)));
            }
        }
        if let Some((first_entry, run)) = own {
            let job = PrepJob {
                ty,
                first_entry,
                values,
                buffers: run,
                fusion: fusion.clone(),
                tile: tile.clone(),
            };
            let built = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| finish_prep(job)))
                .map_err(|_| ViperaError::internal("activation preparation panicked"))?;
            ready.push((first_entry, built));
        }
        for _ in 0..dispatched {
            let (first_entry, built) = self.prepared.recv().map_err(|_| {
                ViperaError::internal("a matvec worker stopped before preparing its activations")
            })?;
            if first_entry == usize::MAX {
                return Err(ViperaError::internal(
                    "a matvec worker panicked preparing activations",
                ));
            }
            ready.push((first_entry, built));
        }
        ready.sort_by_key(|(first_entry, _)| *first_entry);
        let all: Vec<OwnedActivation> = ready.into_iter().flat_map(|(_, built)| built).collect();
        let array: [OwnedActivation; B] = all.try_into().map_err(|_| {
            ViperaError::internal("prepared activations do not cover the batch exactly")
        })?;
        Ok(Arc::new(array))
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
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn fill_batch(
        &self,
        ty: GgmlType,
        data: &'d [u8],
        stride: usize,
        threads: usize,
        activations: &Arc<[OwnedActivation; B]>,
        out: &mut [f32],
        tile: Option<Arc<SharedTile>>,
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
            quant::fill_rows_batch(ty, data, stride, 0, &activation, out, tile.as_deref());
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
                tile: tile.clone(),
            };
            match self
                .workers
                .get(chunk)
                .map(|sender| sender.send(Task::Rows(job)))
            {
                Some(Ok(())) => dispatched += 1,
                // No worker for this chunk, or it is gone: do its share here.
                _ => {
                    let activation = BatchActivation::new(activations);
                    let mut values = vec![0.0_f32; count * B];
                    quant::fill_rows_batch(
                        ty,
                        data,
                        stride,
                        first_row,
                        &activation,
                        &mut values,
                        tile.as_deref(),
                    );
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
                quant::fill_rows_batch(
                    ty,
                    data,
                    stride,
                    first_row,
                    &activation,
                    &mut values,
                    tile.as_deref(),
                );
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

/// Turn a run of buffers into owned activations, in order.
fn finish_prep(job: PrepJob) -> Vec<OwnedActivation> {
    let PrepJob {
        ty,
        first_entry,
        values,
        buffers,
        fusion,
        tile,
    } = job;
    let built: Vec<OwnedActivation> = buffers
        .into_iter()
        .enumerate()
        .map(|(offset, buffer)| {
            OwnedActivation::from_buffer_fused(ty, buffer, values, &fusion, first_entry + offset)
        })
        .collect();
    // This participant's entries of the product-wide tile. Ranges are disjoint by construction
    // (`prepare` splits the batch into runs), and the row workers only read after every reply is in.
    if let Some(tile) = &tile {
        let q: Vec<&[i8]> = built
            .iter()
            .filter_map(|activation| activation.quantized().map(|x| x.q.as_slice()))
            .collect();
        if q.len() == built.len() {
            tile.fill_entries(first_entry, &q);
        }
    }
    built
}

/// Take jobs until the queue closes, answering each with the rows it produced.
fn run_worker<'d, const B: usize>(
    jobs: &Receiver<Task<'d, B>>,
    replies: &Sender<(usize, Vec<f32>)>,
    prepared: &Sender<PrepReply>,
) {
    for task in jobs {
        let job = match task {
            Task::Rows(job) => job,
            Task::Prep(job) => {
                let first_entry = job.first_entry;
                let reply =
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| finish_prep(job)))
                        .map_or((usize::MAX, Vec::new()), |built| (first_entry, built));
                if prepared.send(reply).is_err() {
                    return;
                }
                continue;
            }
        };
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
                job.tile.as_deref(),
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
                None,
            );
            assert!(result.is_err(), "expected internal error, got {result:?}");
        });
    }

    fn pseudo_random(seed: u64, len: usize) -> Vec<f32> {
        let mut state = seed;
        (0..len)
            .map(|at| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Whole groups of zeros and of negative zeros are the cases the quantiser and the
                // sums treat specially.
                match at / 32 % 7 {
                    3 => 0.0,
                    5 => -0.0,
                    _ => ((state >> 40) as f32 / (1_u64 << 24) as f32 - 0.5) * 6.0,
                }
            })
            .collect()
    }

    fn assert_same_activations(label: &str, a: &OwnedActivation, b: &OwnedActivation) {
        let bits = |values: &[f32]| values.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(a.x()), bits(b.x()), "{label}: x");
        assert_eq!(bits(a.sums()), bits(b.sums()), "{label}: sums");
        match (a.quantized(), b.quantized()) {
            (None, None) => {}
            (Some(a), Some(b)) => {
                assert_eq!(a.q, b.q, "{label}: quantised values");
                assert_eq!(bits(&a.d), bits(&b.d), "{label}: scales");
            }
            _ => panic!("{label}: one side quantised and the other did not"),
        }
    }

    /// The preparation a pool does is the one the calling thread does, entry for entry and bit for
    /// bit, whatever the split: entries that do not divide among the participants, one entry, more
    /// participants than entries, and every format that quantises and one that does not.
    #[test]
    fn prepared_activations_are_bit_identical_to_the_serial_ones() {
        fn check<const B: usize>(workers: usize, threads: usize) {
            for ty in [GgmlType::Q4K, GgmlType::Q6K, GgmlType::F32] {
                for values in [32_usize, 96, 1536] {
                    let inputs: Vec<Vec<f32>> = (0..B)
                        .map(|entry| pseudo_random(7 + entry as u64, values))
                        .collect();
                    let views: [&[f32]; B] = std::array::from_fn(|entry| inputs[entry].as_slice());
                    std::thread::scope(|scope| {
                        let pool = Pool::<B>::new(scope, workers);
                        let prepared = pool
                            .prepare(ty, &views, &Fusion::None, threads, None)
                            .expect("prepare");
                        for entry in 0..B {
                            let serial = OwnedActivation::new(ty, views[entry]);
                            assert_same_activations(
                                &format!("{ty:?} B={B} values={values} entry {entry}"),
                                &prepared[entry],
                                &serial,
                            );
                        }
                    });
                }
            }
        }
        check::<1>(3, 4);
        check::<2>(3, 4);
        check::<7>(3, 4);
        check::<32>(3, 4);
        check::<33>(7, 8);
        check::<16>(0, 1);
    }

    /// One Q4_K block with `d = 1`, `dmin = 0`, scale 1 and minimum 0 in every sub-block, and
    /// nibbles `0x21`: the pattern the quantiser tests use, so the kernel does its real arithmetic
    /// at representative magnitudes instead of multiplying zeros.
    fn q4k_block() -> Vec<u8> {
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        block[4..8].fill(1);
        block[12..16].fill(1);
        block[16..144].fill(0x21);
        block
    }

    /// Where a prefill product's wall clock goes at `B = 128`: the activation prepare (with and
    /// without the shared tile it also fills), the row fill on its own, and the call as the model
    /// makes it. Written because `ffn_down` (1536 rows x 8960 cols) streams 1 GB/s while
    /// `ffn_gate` (8960 x 1536) streams 3.3 GB/s at the same block count, and the kernel alone is
    /// equally fast on both shapes.
    ///
    /// ```text
    /// VIPERA_THREADS=8 cargo test -p vipera-backend-cpu --release --lib prefill_product_breakdown -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn prefill_product_breakdown() {
        const B: usize = 128;
        let participants = quant::max_threads();
        for &(rows, blocks) in &[(8960usize, 6usize), (1536, 35)] {
            let cols = blocks * 256;
            let stride = blocks * 144;
            let block_bytes = q4k_block();
            let matrix: Vec<u8> = (0..rows * blocks)
                .flat_map(|_| block_bytes.clone())
                .collect();
            let inputs: Vec<Vec<f32>> = (0..B)
                .map(|entry| pseudo_random(11 + entry as u64, cols))
                .collect();
            let views: [&[f32]; B] = std::array::from_fn(|entry| inputs[entry].as_slice());
            let gate: Arc<[f32]> = Arc::from(pseudo_random(7, B * cols));
            let silu = Fusion::Silu {
                gate: Arc::clone(&gate),
                width: cols,
            };
            let matrix_info = quant::Matrix {
                ty: vipera_core::gguf::GgmlType::Q4K,
                cols,
                rows,
            };
            let mut out = vec![0.0_f32; B * rows];
            std::thread::scope(|scope| {
                let pool = Pool::<B>::new(scope, participants.saturating_sub(1).max(1));
                // whole call, prepare with tile, prepare without tile, row fill alone
                let mut best = [f64::INFINITY; 4];
                for _ in 0..10 {
                    for (arm, slot) in best.iter_mut().enumerate() {
                        let started = std::time::Instant::now();
                        match arm {
                            0 => {
                                quant::matvec_pooled_batch(
                                    &pool,
                                    matrix_info,
                                    &matrix,
                                    &views,
                                    &silu,
                                    &mut out,
                                )
                                .expect("product");
                                std::hint::black_box(&out);
                            }
                            1 | 2 => {
                                let tile =
                                    (arm == 1).then(|| pool.take_tile(blocks, B.div_ceil(16)));
                                let owned = pool
                                    .prepare(
                                        vipera_core::gguf::GgmlType::Q4K,
                                        &views,
                                        &silu,
                                        participants,
                                        tile.clone(),
                                    )
                                    .expect("prepare");
                                std::hint::black_box(&owned);
                                if let Some(tile) = tile {
                                    pool.recycle_tile(tile);
                                }
                            }
                            _ => {
                                let owned: Arc<[OwnedActivation; B]> =
                                    Arc::new(std::array::from_fn(|entry| {
                                        OwnedActivation::new(
                                            vipera_core::gguf::GgmlType::Q4K,
                                            views[entry],
                                        )
                                    }));
                                let tile = pool.take_tile(blocks, B.div_ceil(16));
                                pool.fill_batch(
                                    vipera_core::gguf::GgmlType::Q4K,
                                    &matrix,
                                    stride,
                                    participants,
                                    &owned,
                                    &mut out,
                                    Some(tile.clone()),
                                )
                                .expect("fill");
                                std::hint::black_box(&out);
                                pool.recycle_tile(tile);
                            }
                        }
                        *slot = slot.min(started.elapsed().as_secs_f64() * 1e3);
                    }
                }
                let bytes = (rows * stride) as f64;
                println!(
                    "{rows:>5}x{cols:<5} {participants}T: whole {:>6.2} ms ({:>4.1} GB/s) | prep+tile {:>5.2} | prep {:>5.2} | fill {:>5.2} (prep {:.0}% of fill)",
                    best[0],
                    bytes / (best[0] * 1e6),
                    best[1],
                    best[2],
                    best[3],
                    100.0 * (best[0] - best[3]) / best[3]
                );
            });
        }
    }

    /// What building a batched product's activations costs on the calling thread against the pool,
    /// at the two widths a prefill chunk multiplies, 128 entries. The pool's gain is parallel, so
    /// it only shows on a machine with the cores for it. It is a measurement, not a check, so it
    /// is `#[ignore]`d:
    ///
    /// ```text
    /// cargo test -p vipera-backend-cpu --release --lib pool_prep_cost -- --ignored --nocapture
    /// ```
    #[test]
    #[ignore = "measurement, not a correctness check"]
    fn pool_prep_cost() {
        let participants = quant::max_threads();
        for values in [1536_usize, 8960] {
            let inputs: Vec<Vec<f32>> = (0..128)
                .map(|entry| pseudo_random(11 + entry as u64, values))
                .collect();
            let views: [&[f32]; 128] = std::array::from_fn(|entry| inputs[entry].as_slice());
            std::thread::scope(|scope| {
                let pool = Pool::<128>::new(scope, participants.saturating_sub(1).max(1));
                let mut best = [f64::INFINITY; 2];
                for _ in 0..30 {
                    for (arm, slot) in best.iter_mut().enumerate() {
                        let started = std::time::Instant::now();
                        if arm == 0 {
                            let serial: Arc<[OwnedActivation; 128]> =
                                Arc::new(std::array::from_fn(|entry| {
                                    OwnedActivation::new(GgmlType::Q4K, views[entry])
                                }));
                            std::hint::black_box(serial);
                        } else {
                            let pooled = pool
                                .prepare(GgmlType::Q4K, &views, &Fusion::None, participants, None)
                                .expect("prepare");
                            std::hint::black_box(pooled);
                        }
                        *slot = slot.min(started.elapsed().as_secs_f64() * 1e3);
                    }
                }
                println!(
                    "{values:>5} values, 128 entries, {participants} participants: calling thread {:.3} ms, pool {:.3} ms ({:.2}x)",
                    best[0],
                    best[1],
                    best[0] / best[1]
                );
            });
        }
    }
}
