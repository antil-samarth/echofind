use hound;
use image;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;

fn main() {
    const TARGET_SAMPLE_RATE: u32 = 8192;
    const WINDOW_SIZE: usize = 1024;
    const HOP_SIZE: usize = 64;
    const FREQ_BANDS: &[usize] = &[5, 80, 160, 320, 640, 1280, 4096]; // Frequency bands in Hz
    const MIN_TIME_DELTA: usize = 15;
    const MAX_TIME_DELTA: usize = 45;
    const TARGET_FANOUT: usize = 4;

    let mut reader = hound::WavReader::open("src/media/wav/01_Genesis.wav")
        .expect("Failed to open the specified WAV file");
    println!("file name - 01_Genesis.wav");
    let spec = reader.spec();
    println!("Sample rate: {}", spec.sample_rate);
    println!("Channels: {}", spec.channels);
    println!("Bits per sample: {}", spec.bits_per_sample);
    println!("Sample format: {:?}", spec.sample_format);
    println!("Total samples: {}", reader.len());
    println!(
        "Duration: {} seconds",
        reader.duration() as f32 / spec.sample_rate as f32
    );

    let samples_result: Result<Vec<i16>, hound::Error> = reader.samples::<i16>().collect();
    let mut samples_vec = samples_result.expect("Failed to read all samples");

    println!("Number of samples read: {}", samples_vec.len());
    println!("First 10 samples: {:?}", &samples_vec[0..10]);

    if spec.channels == 2 {
        println!("This is a stereo file");
        println!("Converting to mono");

        samples_vec = samples_vec
            .chunks_exact(2)
            .map(|chunk: &[i16]| ((chunk[0] as i32 + chunk[1] as i32) / 2) as i16)
            .collect();

        println!("Number of samples read: {}", samples_vec.len());
        println!("First 10 samples: {:?}", &samples_vec[0..10]);
    }

    let downsampled_samples = downsample(&samples_vec, spec.sample_rate, TARGET_SAMPLE_RATE);
    println!("Number of samples read: {}", downsampled_samples.len());
    println!("First 10 samples: {:?}", &downsampled_samples[0..10]);

    let window_coefficients = hamming_window(WINDOW_SIZE);
    println!(
        "Generated Hamming window {} coefficients",
        window_coefficients.len()
    );

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(WINDOW_SIZE);

    let mut spectrogram: Vec<Vec<f64>> = Vec::new();

    println!("Generating spectrogram...");

    for audio_chunk in downsampled_samples.windows(WINDOW_SIZE).step_by(HOP_SIZE) {
        let mut complex_buffer: Vec<Complex<f64>> = vec![Complex::new(0.0, 0.0); WINDOW_SIZE];

        for (i, (sample, coeff)) in audio_chunk
            .iter()
            .zip(window_coefficients.iter())
            .enumerate()
        {
            complex_buffer[i] = Complex::new(*sample as f64 * coeff, 0.0);
        }

        fft.process(&mut complex_buffer);

        let num_freq_bins = WINDOW_SIZE / 2;
        let mut magnitudes = Vec::with_capacity(num_freq_bins);

        for i in 0..num_freq_bins {
            let magnitude = complex_buffer[i].norm_sqr();
            magnitudes.push(magnitude);
        }

        spectrogram.push(magnitudes);
    }

    println!(
        "Spectrogram generated: {} time slices, {} frequency bins",
        spectrogram.len(),
        if spectrogram.is_empty() {
            0
        } else {
            spectrogram[0].len()
        }
    );

    // Save the spectrogram as an image
    /* let output_path = "src/media/spectrogram.png";
    spectrogram_to_image(&spectrogram, output_path); */

    let bin_ranges: Vec<(usize, usize)> = FREQ_BANDS
        .windows(2)
        .map(|pair| {
            let low_bin = hz_to_bin(pair[0], TARGET_SAMPLE_RATE, WINDOW_SIZE);
            let high_bin = hz_to_bin(pair[1], TARGET_SAMPLE_RATE, WINDOW_SIZE);
            if high_bin <= low_bin {
                panic!(
                    "High bin is less than or equal to low bin for frequency band: {:?}",
                    pair
                );
            }
            (low_bin, high_bin)
        })
        .collect();

    println!("Frequency bands converted to bin ranges: {:?}", bin_ranges);
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

    println!("Found {} peaks", peaks.len());
    if peaks.len() > 15 {
        println!("First few peaks (time, freq_bin): {:?}", &peaks[0..14]);
    }

    let mut hashes: Vec<(u64, usize)> = Vec::new();

    for i in 0..peaks.len() {
        let anchor = peaks[i];
        let mut targets_count = 0;

        for j in (i + 1)..peaks.len() {
            let target = peaks[j];
            let time_diff = target.0 - anchor.0;
            if time_diff > MAX_TIME_DELTA {
                break;
            }

            if time_diff >= MIN_TIME_DELTA && time_diff <= MAX_TIME_DELTA {
                let hash = create_hash(anchor.1, target.1, time_diff);

                hashes.push((hash, anchor.0));

                targets_count += 1;

                if targets_count >= TARGET_FANOUT {
                    break;
                }
            }
        }
    }

    println!("Generated {} hashes.", hashes.len());
    println!("First few hashes (hash, time): {:?}", &hashes[0..10]);
}

fn downsample(samples: &[i16], orginal_sample_rate: u32, target_sample_rate: u32) -> Vec<i16> {
    if target_sample_rate == orginal_sample_rate {
        return samples.to_vec();
    }

    if target_sample_rate > orginal_sample_rate {
        panic!("Up-sampling is not supported");
    }

    println!("Downsampling to {} Hz", target_sample_rate);

    let mut downsampled_samples = Vec::new();

    let step = orginal_sample_rate / target_sample_rate;
    println!("Step: {}", step);

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

fn create_hash(anchor_freq_bin: usize, target_freq_bin: usize, time_diff: usize) -> u64 {
    if anchor_freq_bin >= (1 << 22) {
        panic!("Anchor frequency bin exceeds the 22-bit limit: {}", anchor_freq_bin);
    }
    if target_freq_bin >= (1 << 10) {
        panic!("Target frequency bin exceeds the 10-bit limit: {}", target_freq_bin);
    }
    if time_diff >= (1 << 12) {
        panic!("Time difference exceeds the 12-bit limit: {}", time_diff);
    }
    ((anchor_freq_bin as u64) << 22) | ((target_freq_bin as u64) << 12) | (time_diff as u64)
}

fn spectrogram_to_image(spectrogram: &Vec<Vec<f64>>, output_path: &str) {
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
