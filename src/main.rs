use hound;
use image;
use rusqlite::{Connection, params};
use rustfft::{Fft, FftPlanner, num_complex::Complex};
use std::{cmp::Ordering, collections::HashMap, f64::consts::PI, fs::read_dir};

mod config;

use config::{
    FREQ_BANDS, HOP_SIZE, MAX_TIME_DELTA, MIN_PEAK_FREQ_DISTANCE, MIN_PEAK_TIME_DISTANCE,
    MIN_TIME_DELTA, PEAK_THRESHOLD_FACTOR, TARGET_FANOUT, TARGET_SAMPLE_RATE, WINDOW_SIZE,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // constants
    const DB_PATH: &str = "src/media/audio_fingerprints.db";
    const AUDIO_DIR: &str = "src/media/wav/";
    const INPUT_FILE: &str = "src/media/recording/recording1.wav";

    const MODE: &str = "test"; // "test" or "train"

    if MODE == "train" {
        println!("\nConnecting to database: {}", DB_PATH);
        let mut conn = Connection::open(DB_PATH)?;
        println!("Connected to database.");

        conn.execute(
            "CREATE TABLE IF NOT EXISTS songs (
            song_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath    TEXT NOT NULL UNIQUE
        )",
            [],
        )?;

        conn.execute(
            "CREATE TABLE IF NOT EXISTS fingerprints (
                hash        INTEGER NOT NULL,
                time_offset INTEGER NOT NULL,
                song_id     INTEGER NOT NULL,
                FOREIGN KEY (song_id) REFERENCES songs (song_id)
            )",
            [],
        )?;

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_hash ON fingerprints (hash)",
            [],
        )?;

        println!("\nScanning directory: {}", AUDIO_DIR);
        let mut audio_files = Vec::new();
        for entry in read_dir(AUDIO_DIR)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "wav") {
                // Validate if the file is a valid audio file
                if hound::WavReader::open(&path).is_ok() {
                    audio_files.push(path.to_string_lossy().to_string());
                    println!("Found valid audio file: {}", path.display());
                } else {
                    println!(
                        "Skipping invalid or corrupted audio file: {}",
                        path.display()
                    );
                }
            }
        }
        println!("Found {} audio files.", audio_files.len());

        let mut audios_processed = 0;

        for filepath in &audio_files {
            println!("\n--- Processing: {} ---", filepath);

            let exists_check: Result<i64, rusqlite::Error> = conn.query_row(
                "SELECT song_id FROM songs WHERE filepath = ?1",
                rusqlite::params![&filepath], // Pass filepath by reference here
                |row| row.get::<usize, i64>(0),
            );
            match exists_check {
                Ok(existing_id) => {
                    println!(
                        "Song already exists in the database with ID: {}",
                        existing_id
                    );
                    continue;
                }
                Err(rusqlite::Error::QueryReturnedNoRows) => {
                    println!("Song not found in the database, proceeding to insert.");
                }
                Err(e) => {
                    eprintln!("Error checking DB for song existence: {}", e);
                    return Err(e.into());
                }
            }

            let song_id = match insert_song_record(&mut conn, &filepath) {
                Ok(id) => {
                    println!("Inserted new song record with ID: {}", id);
                    id
                }
                Err(e) => {
                    eprintln!("Error inserting song record: {}", e);
                    continue;
                }
            };
            println!("Using song_id: {}", song_id);

            let hashes = match process_audio_file(
                &filepath,
                TARGET_SAMPLE_RATE,
                FREQ_BANDS,
                WINDOW_SIZE,
                HOP_SIZE,
                MIN_TIME_DELTA,
                MAX_TIME_DELTA,
                TARGET_FANOUT,
            ) {
                Ok(hashes) => {
                    println!("Generated {} hashes for song ID: {}", hashes.len(), song_id);
                    hashes
                }
                Err(e) => {
                    eprintln!("Error processing audio file: {}", e);
                    continue;
                }
            };

            match insert_fingerprints(&mut conn, &hashes, song_id) {
                Ok(_) => {
                    println!(
                        "Inserted {} fingerprints for song ID: {}",
                        hashes.len(),
                        song_id
                    );
                }
                Err(e) => {
                    eprintln!("Error inserting fingerprints: {}", e);
                    continue;
                }
            }
            println!("{} Fingerprints inserted successfully.", hashes.len());

            audios_processed += 1;
        }
        println!(
            "\nProcessed {} audio files, skipped {} files.",
            audios_processed,
            audio_files.len() - audios_processed
        );
        println!("All done!");
    } else if MODE == "test" {
        println!("\n ------ TESTING ------ ");

        let snippet_hashes = process_audio_file(
            INPUT_FILE,
            TARGET_SAMPLE_RATE,
            FREQ_BANDS,
            WINDOW_SIZE,
            HOP_SIZE,
            MIN_TIME_DELTA,
            MAX_TIME_DELTA,
            TARGET_FANOUT,
        )?;
        if snippet_hashes.is_empty() {
            println!("No hashes generated for the test file.");
            return Ok(());
        }
        println!("Generated {} hashes for test file.", snippet_hashes.len());
        let hash_count = snippet_hashes.len() as u32;
        let conn = Connection::open(DB_PATH)?;
        println!("Connected to database.");

        let mut histogram: HashMap<i64, HashMap<i64, u32>> = HashMap::new();
        let mut stmt =
            conn.prepare_cached("SELECT song_id, time_offset FROM fingerprints WHERE hash = ?1")?;
        let mut total_matches = 0;

        // Process hashes in batches
        for (snippet_hash, snippet_anchor_time) in snippet_hashes.iter() {
            let db_matches = stmt.query_map(rusqlite::params![*snippet_hash as i64], |row| {
                Ok((row.get::<usize, i64>(0)?, row.get::<usize, i64>(1)?))
            })?;

            for result in db_matches {
                if let Ok((db_song_id, db_time_offset)) = result {
                    total_matches += 1;
                    let offset_diff = db_time_offset - (*snippet_anchor_time as i64);

                    *histogram
                        .entry(db_song_id)
                        .or_default()
                        .entry(offset_diff)
                        .or_default() += 1;
                }
            }
        }
        if total_matches == 0 {
            println!("No matches found in the database.");
            return Ok(());
        }
        println!("Total matches found: {}", total_matches);

        // Optimize match finding
        println!("\nAnalyzing matches...");
        let mut matches: Vec<_> = histogram
            .iter()
            .map(|(&song_id, offsets)| {
                let max_count_for_song = offsets.values().max().cloned().unwrap_or(0);
                (song_id, max_count_for_song)
            })
            .filter(|&(_, count)| count > 2) // Adjust threshold as needed
            .collect();

        // Sort by match count in descending order
        matches.sort_by(|a, b| b.1.cmp(&a.1));

        if !matches.is_empty() {
            println!("\nTop matches found:");

            for (i, &(song_id, match_count)) in matches.iter().take(3).enumerate() {
                println!("\nMatch #{}", i + 1);
                println!("Song ID: {}", song_id);
                println!("Match strength: {} matching points", match_count);

                // Calculate match confidence
                let confidence = (match_count as f64 / hash_count as f64) * 100.0;
                let confidence = confidence.min(100.0);
                println!("Confidence: {:.1}%", confidence);

                // Fetch and display song details
                if let Ok(filepath) = conn.query_row::<String, _, _>(
                    "SELECT filepath FROM songs WHERE song_id = ?1",
                    rusqlite::params![song_id],
                    |row| row.get(0),
                ) {
                    println!("Matched file: {}", filepath);
                }

                // Add match quality indicator
                let quality = if confidence > 15.0 {
                    "High"
                } else if confidence > 5.0 {
                    "Medium"
                } else {
                    "Low"
                };
                println!("Match quality: {}", quality);
            }
        } else {
            println!("\nNo matches found.");
        }
    } else {
        eprintln!("Invalid mode specified. Use 'index' or 'match'.");
    }

    Ok(())
}

fn insert_song_record(conn: &mut Connection, filepath: &str) -> Result<i64, rusqlite::Error> {
    let mut stmt = conn.prepare("INSERT INTO songs (filepath) VALUES (?1)")?;
    stmt.execute(params![filepath])?;

    let song_id: i64 = conn.query_row(
        "SELECT song_id FROM songs WHERE filepath = ?1",
        params![filepath],
        |row| row.get(0),
    )?;

    Ok(song_id)
}

fn process_audio_file(
    filepath: &str,
    target_sample_rate: u32,
    freq_bands: &[usize],
    window_size: usize,
    hop_size: usize,
    min_time_delta: usize,
    max_time_delta: usize,
    target_fanout: usize,
) -> Result<Vec<(u64, usize)>, Box<dyn std::error::Error>> {
    println!("\nLoading and preparing audio file: {}", filepath);
    let samples_vec = load_and_prepare_audio(filepath, target_sample_rate)?;
    let window_coefficients = hamming_window(window_size);
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(window_size);

    let spectrogram =
        compute_spectrogram(samples_vec, window_coefficients, window_size, hop_size, fft);

    let bin_ranges = freq_bands
        .windows(2)
        .map(|band| {
            (
                hz_to_bin(band[0], target_sample_rate, window_size),
                hz_to_bin(band[1], target_sample_rate, window_size),
            )
        })
        .collect::<Vec<_>>();

    /* let bin_ranges: Vec<(usize, usize)> = vec![(0, 5), (5, 80), (80, 160), (160, 320), (320, 640), (640, 1280), (1280, 4096)]; */
    let mut peaks: Vec<(usize, usize)> = Vec::new();

    for (time_slice_index, magnitudes) in spectrogram.iter().enumerate() {
        for &(low_bin, high_bin) in &bin_ranges {
            if high_bin > magnitudes.len() {
                continue;
            }
            let band_slice = &magnitudes[low_bin..high_bin];

            if let Some((max_index_in_band, _)) = band_slice
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            {
                let absolute_bin_index = low_bin + max_index_in_band;
                peaks.push((time_slice_index, absolute_bin_index));
            }
        }
    }

    println!("Found {} peaks.", peaks.len());

    let filtered_peaks = extract_significant_peaks(
        &spectrogram,
        &bin_ranges,
        PEAK_THRESHOLD_FACTOR,
        MIN_PEAK_TIME_DISTANCE,
        MIN_PEAK_FREQ_DISTANCE,
    );

    let hashes = generate_hashes(
        &filtered_peaks,
        min_time_delta,
        max_time_delta,
        target_fanout,
    );
    Ok(hashes)
}

fn insert_fingerprints(
    conn: &mut Connection,
    hashes: &[(u64, usize)],
    song_id: i64,
) -> Result<(), rusqlite::Error> {
    let tx = conn.transaction()?;
    {
        let mut stmt = tx
            .prepare("INSERT INTO fingerprints (hash, time_offset, song_id) VALUES (?1, ?2, ?3)")?;
        for (hash, time_offset) in hashes {
            stmt.execute(params![*hash as i64, *time_offset as i64, song_id])?;
        }
    }
    tx.commit()?;
    println!("Fingerprints inserted successfully.");
    Ok(())
}

fn load_and_prepare_audio(
    filepath: &str,
    target_sample_rate: u32,
) -> Result<Vec<i16>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(filepath)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let mut samples: Vec<i16> = reader.samples::<i16>().collect::<Result<_, _>>()?;

    println!(
        " File info:\n  ├ Sample Rate: {} Hz\n  ├ Channels: {}\n  ├ Bits: {}\n  ├ Format: {:?}\n  └ Samples: {}",
        spec.sample_rate,
        spec.channels,
        spec.bits_per_sample,
        spec.sample_format,
        reader.len()
    );

    if spec.channels == 2 {
        println!("Stereo audio detected. Converting to mono.");

        samples = samples
            .chunks_exact(2)
            .map(|chunk: &[i16]| ((chunk[0] as i32 + chunk[1] as i32) / 2) as i16)
            .collect();
    } else if spec.channels > 2 {
        eprintln!("Received audio has more than 2 channels, only stereo is supported");
        return Err("Only mono and stereo audio is supported".into());
    }

    if sample_rate > target_sample_rate {
        println!(
            "Downsampling: {} Hz → {} Hz",
            sample_rate, target_sample_rate
        );
        samples = downsample(&samples, sample_rate, target_sample_rate as u32);
    } else if sample_rate < target_sample_rate {
        eprintln!("Upsampling audio is not supported");
        return Err("Upsampling is not supported".into());
    }
    println!("Audio loaded and prepared successfully");

    Ok(samples)
}

fn generate_hashes(
    peaks: &[(usize, usize)],
    min_time_delta: usize,
    max_time_delta: usize,
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
                let hash = match create_hash(anchor.1, target.1, time_diff) {
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

fn compute_spectrogram(
    samples_vec: Vec<i16>,
    window_coefficients: Vec<f64>,
    window_size: usize,
    hop_size: usize,
    fft: std::sync::Arc<dyn Fft<f64>>,
) -> Vec<Vec<f64>> {
    let mut spectrogram: Vec<Vec<f64>> = Vec::new();

    println!("\nGenerating spectrogram...");

    for audio_chunk in samples_vec.windows(window_size).step_by(hop_size) {
        let mut complex_buffer: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); window_size];

        for (i, (sample, coeff)) in audio_chunk
            .iter()
            .zip(window_coefficients.iter())
            .enumerate()
        {
            complex_buffer[i] = Complex::new(*sample as f64 * coeff, 0.0);
        }

        fft.process(&mut complex_buffer);

        let num_freq_bins = window_size / 2;
        let mut magnitudes = Vec::with_capacity(num_freq_bins);

        for i in 0..num_freq_bins {
            let magnitude = complex_buffer[i].norm_sqr();
            magnitudes.push(magnitude);
        }

        spectrogram.push(magnitudes);
    }
    spectrogram
}

fn downsample(samples: &[i16], original_sample_rate: u32, target_sample_rate: u32) -> Vec<i16> {
    if target_sample_rate == original_sample_rate {
        return samples.to_vec();
    }

    if target_sample_rate > original_sample_rate {
        panic!("Up-sampling is not supported");
    }

    //println!("Downsampling to {} Hz", target_sample_rate);

    let mut downsampled_samples = Vec::new();

    let step = original_sample_rate / target_sample_rate;
    //println!("Step: {}", step);

    for sample in samples.iter().step_by(step as usize) {
        downsampled_samples.push(*sample);
    }

    return downsampled_samples;
}

fn hamming_window(window_len: usize) -> Vec<f64> {
    let mut window = Vec::with_capacity(window_len);
    for n in 0..window_len {
        // Hamming window formula
        // 0.54 - 0.46 * cos(2 * pi * n / (N - 1))
        // where N is the window length

        let value = 2.0 * PI * (n as f64) / (window_len as f64 - 1.0);
        window.push(0.54 - 0.46 * value.cos());
    }
    window
}

fn hz_to_bin(hz: usize, sample_rate: u32, window_size: usize) -> usize {
    let freq_res = sample_rate as f64 / window_size as f64;
    (hz as f64 / freq_res).round() as usize
}

fn create_hash(
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

fn extract_significant_peaks(
    spectrogram: &[Vec<f64>],
    bin_ranges: &[(usize, usize)],
    peak_threshold_factor: f64,
    min_time_distance: usize,
    min_freq_distance: usize,
) -> Vec<(usize, usize)> {
    // Returns Vec<(time_slice_idx, freq_bin_idx)>
    if spectrogram.is_empty() {
        return Vec::new();
    }

    let mut dynamically_thresholded_peaks: Vec<(usize, usize, f64)> = Vec::new(); // (time, freq, magnitude)

    for (time_slice_index, magnitudes) in spectrogram.iter().enumerate() {
        let mut candidates_this_slice: Vec<(usize, usize, f64)> = Vec::new(); // (time, freq_bin, mag)

        for &(low_bin, high_bin) in bin_ranges {
            if high_bin > magnitudes.len() || low_bin >= high_bin {
                continue;
            }

            let band_slice = &magnitudes[low_bin..high_bin];
            if let Some((max_idx_in_band, &max_magnitude_in_band)) = band_slice
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal))
            {
                if max_magnitude_in_band > 1e-6 {
                    candidates_this_slice.push((
                        time_slice_index,
                        low_bin + max_idx_in_band,
                        max_magnitude_in_band,
                    ));
                }
            }
        }

        if candidates_this_slice.is_empty() {
            continue;
        }

        let average_magnitude_this_slice: f64 = candidates_this_slice
            .iter()
            .map(|&(_, _, mag)| mag)
            .sum::<f64>()
            / candidates_this_slice.len() as f64;

        for &(time, freq_bin, magnitude) in &candidates_this_slice {
            if magnitude >= average_magnitude_this_slice * peak_threshold_factor {
                dynamically_thresholded_peaks.push((time, freq_bin, magnitude));
            }
        }
    }

    let mut final_peaks: Vec<(usize, usize)> = Vec::new();
    let mut last_added_peak: Option<(usize, usize)> = None;

    for (time_idx, freq_idx, _magnitude) in dynamically_thresholded_peaks {
        if let Some((last_time, last_freq)) = last_added_peak {
            let time_dist = time_idx.abs_diff(last_time);
            let freq_dist = freq_idx.abs_diff(last_freq);

            if time_dist >= min_time_distance || freq_dist >= min_freq_distance {
                final_peaks.push((time_idx, freq_idx));
                last_added_peak = Some((time_idx, freq_idx));
            }
        } else {
            // Always add the very first peak encountered
            final_peaks.push((time_idx, freq_idx));
            last_added_peak = Some((time_idx, freq_idx));
        }
    }

    println!(
        "Found {} final peaks after proximity filter.",
        final_peaks.len()
    ); // For debugging
    final_peaks
}

fn _spectrogram_to_image(spectrogram: &Vec<Vec<f64>>, output_path: &str) {
    if spectrogram.is_empty() || spectrogram[0].is_empty() {
        eprintln!("Spectrogram does not exist. Cannot visualize");
        return;
    }

    let width = spectrogram.len();
    let height = spectrogram[0].len();

    let mut min_log_mag = f64::MAX; // goes to min
    let mut max_log_mag = f64::MIN; // goes to max

    for time_slice in spectrogram {
        for &magnitude in time_slice {
            if magnitude < 1e-10 {
                continue;
            }

            let log_mag = (magnitude - 1e-6).log10();
            min_log_mag = min_log_mag.min(log_mag);
            max_log_mag = max_log_mag.max(log_mag);
        }
    }

    let log_mag_range = max_log_mag - min_log_mag;
    let log_mag_range = if log_mag_range < 1e-6 {
        1.0
    } else {
        log_mag_range
    };

    let mut img_buf = image::GrayImage::new(width as u32, height as u32);

    for (t, time_slice) in spectrogram.iter().enumerate() {
        for (f, magnitude) in time_slice.iter().enumerate() {
            let log_mag = (magnitude + 1e-6).log10();
            let scaled_val = (log_mag - min_log_mag) / log_mag_range;
            let intensity = (scaled_val.clamp(0.0, 1.0) * 255.0).round() as u8;

            let x = t as u32;
            let y = (height - 1 - f) as u32;

            img_buf.put_pixel(x, y, image::Luma([intensity]));
        }
    }

    match img_buf.save(output_path) {
        Ok(_) => println!("Spectrogram saved to {}", output_path),
        Err(e) => eprintln!("Error saving spectrogram: {}", e),
    }
}
