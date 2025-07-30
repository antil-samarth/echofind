use crate::config::{FUZ_FACTOR, NUM_TARGETS_IN_HASH};

pub fn generate_constellation_hashes(
    peaks: &[(usize, usize)],
    min_time_delta: usize,
    max_time_delta: usize,
) -> Vec<(u64, usize)> {
    println!(
        "Generating constellation hashes (N={} targets)...",
        NUM_TARGETS_IN_HASH
    );
    let mut hashes: Vec<(u64, usize)> = Vec::new();

    for i in 0..peaks.len() {
        let anchor = peaks[i]; // (anchor_time, anchor_freq)
        let mut targets_for_this_anchor: Vec<(usize, usize)> = Vec::new(); // (target_freq, time_delta)

        for j in (i + 1)..peaks.len() {
            let target = peaks[j]; // (target_time, target_freq)
            let time_delta = target.0 - anchor.0;

            if time_delta > max_time_delta {
                break;
            }

            if time_delta >= min_time_delta {
                targets_for_this_anchor.push((target.1, time_delta)); // (freq_bin, delta)
                if targets_for_this_anchor.len() == NUM_TARGETS_IN_HASH {
                    break;
                }
            }
        }

        if targets_for_this_anchor.len() == NUM_TARGETS_IN_HASH {
            let hash = compute_constellation_hash(anchor.1, &targets_for_this_anchor);
            hashes.push((hash, anchor.0)); // Store hash and the anchor's time
        }
    }

    println!("Found {} initial constellations.", hashes.len());

    // Sort and remove duplicate hashes to reduce database size and improve matching speed.
    hashes.sort_unstable_by_key(|k| k.0);
    hashes.dedup_by_key(|k| k.0);

    println!("Found {} unique constellations.", hashes.len());

    hashes
}

fn compute_constellation_hash(
    anchor_freq: usize,
    targets: &[(usize, usize)], // (target_freq, time_delta)
) -> u64 {
    // --- Bit Packing Configuration ---
    // We pack 1 anchor_freq + N target_freqs + N time_deltas into a single u64.
    //
    // - Max freq_bin is 511 (from WINDOW_SIZE/2), which requires 9 bits (2^9 = 512).
    //   We apply a FUZ_FACTOR to reduce this, needing fewer bits.
    // - Max time_delta is configured (e.g., 45), which requires 6 bits (2^6 = 64).
    //
    // With FUZ_FACTOR=4, max fuzzed freq is 511/4 = 127, which needs 7 bits (2^7 = 128).
    const FREQ_BITS: u32 = 7;
    const DELTA_BITS: u32 = 6;

    // Verify that the total bits will fit in a u64.
    // For N=4: (1 anchor + 4 targets) * 7 bits_freq + 4 targets * 6 bits_delta = 35 + 24 = 59 bits.
    // This fits comfortably within a u64.
    const _: () = assert!(
        (1 + NUM_TARGETS_IN_HASH) as u32 * FREQ_BITS + (NUM_TARGETS_IN_HASH as u32 * DELTA_BITS)
            < 64,
        "Bit packing configuration exceeds u64 size"
    );


    let mut hash: u64 = 0;
    let mut current_shift = 0;

    // Pack the time deltas first
    for i in 0..NUM_TARGETS_IN_HASH {
        let delta = targets[i].1 as u64; // time_delta
        hash |= delta << current_shift;
        current_shift += DELTA_BITS;
    }

    // Pack the target frequencies
    for i in 0..NUM_TARGETS_IN_HASH {
        let fuzzed_freq = (targets[i].0 / FUZ_FACTOR) as u64; // target_freq
        hash |= fuzzed_freq << current_shift;
        current_shift += FREQ_BITS;
    }

    // Pack the anchor frequency
    let fuzzed_anchor_freq = (anchor_freq / FUZ_FACTOR) as u64;
    hash |= fuzzed_anchor_freq << current_shift;

    hash
}