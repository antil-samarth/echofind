use rustfft::{Fft, num_complex::Complex};
use std::{cmp::Ordering, f64::consts::PI};

pub fn load_and_prepare_audio(
    filepath: &str,
    target_sample_rate: u32,
) -> Result<Vec<i16>, Box<dyn std::error::Error>> {
    let mut reader = hound::WavReader::open(filepath)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let mut samples: Vec<i16> = reader.samples::<i16>().collect::<Result<_, _>>()?;

    println!(
        "File info:\n  ├ Sample Rate: {} Hz\n  ├ Channels: {}\n  ├ Bits: {}\n  ├ Format: {:?}\n  └ Samples: {}",
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
        samples = downsample(&samples, sample_rate, target_sample_rate);
    } else if sample_rate < target_sample_rate {
        eprintln!("Upsampling audio is not supported");
        return Err("Upsampling is not supported".into());
    }
    println!("Audio loaded and prepared successfully");

    Ok(samples)
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

pub fn hamming_window(window_size: usize) -> Vec<f64> {
    let mut window = Vec::with_capacity(window_size);
    for n in 0..window_size {
        // Hamming window formula
        // 0.54 - 0.46 * cos(2 * pi * n / (N - 1))
        // where N is the window length

        let value = 2.0 * PI * (n as f64) / (window_size as f64 - 1.0);
        window.push(0.54 - 0.46 * value.cos());
    }
    window
}

pub fn hz_to_bin(hz: usize, window_size: usize, target_sample_rate: u32) -> usize {
    let freq_res = target_sample_rate as f64 / window_size as f64;
    (hz as f64 / freq_res).round() as usize
}

pub fn compute_spectrogram(
    samples_vec: Vec<i16>,
    window_coefficients: Vec<f64>,
    fft: std::sync::Arc<dyn Fft<f64>>,
    hop_size: usize,
    window_size: usize,
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

pub fn extract_significant_peaks(
    spectrogram: &[Vec<f64>],
    bin_ranges: &[(usize, usize)],
    min_peak_freq_distance: usize,
    min_peak_time_distance: usize,
    peak_threshold_factor: f64,
) -> Vec<(usize, usize)> {
    if spectrogram.is_empty() {
        return Vec::new();
    }

    let mut dynamically_thresholded_peaks: Vec<(usize, usize, f64)> = Vec::new();

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

            if time_dist >= min_peak_time_distance || freq_dist >= min_peak_freq_distance {
                final_peaks.push((time_idx, freq_idx));
                last_added_peak = Some((time_idx, freq_idx));
            }
        } else {
            final_peaks.push((time_idx, freq_idx));
            last_added_peak = Some((time_idx, freq_idx));
        }
    }

    println!(
        "Found {} final peaks after proximity filter.",
        final_peaks.len()
    );
    final_peaks
}
