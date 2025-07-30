pub fn generate_hashes(
    peaks: &[(usize, usize)],
    max_time_delta: usize,
    min_time_delta: usize,
    target_fanout: usize,
) -> Vec<(u64, usize)> {
    let mut hashes: Vec<(u64, usize)> = Vec::new();

    for i in 0..peaks.len() {
        let anchor = peaks[i];
        let mut targets_count = 0;

        for j in (i + 1)..peaks.len() {
            let target = peaks[j];
            let time_diff = target.0 - anchor.0;
            if time_diff > max_time_delta {
                break;
            }

            if time_diff >= min_time_delta && time_diff <= max_time_delta {
                let hash = match compute_hash(anchor.1, target.1, time_diff) {
                    Ok(hash) => hash,
                    Err(e) => {
                        eprintln!("Error creating hash: {}", e);
                        continue;
                    }
                };

                hashes.push((hash, anchor.0));

                targets_count += 1;

                if targets_count >= target_fanout {
                    break;
                }
            }
        }
    }
    hashes
}

fn compute_hash(
    anchor_freq_bin: usize,
    target_freq_bin: usize,
    time_diff: usize,
) -> Result<u64, String> {
    if anchor_freq_bin >= (1 << 22) {
        return Err(format!(
            "Anchor frequency bin exceeds the 22-bit limit: {}",
            anchor_freq_bin
        ));
    }
    if target_freq_bin >= (1 << 10) {
        return Err(format!(
            "Target frequency bin exceeds the 10-bit limit: {}",
            target_freq_bin
        ));
    }
    if time_diff >= (1 << 12) {
        return Err(format!(
            "Time difference exceeds the 12-bit limit: {}",
            time_diff
        ));
    }
    Ok(((anchor_freq_bin as u64) << 22) | ((target_freq_bin as u64) << 12) | (time_diff as u64))
}
