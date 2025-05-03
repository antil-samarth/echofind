use hound;
use rustfft::{FftPlanner, num_complex::Complex};
use std::f64::consts::PI;

fn main() {
    const TARGET_SAMPLE_RATE: u32 = 8192;
    const WINDOW_SIZE: usize = 1024;
    const HOP_SIZE: usize = 64;

    println!("Hello, world!");
    let mut reader = hound::WavReader::open("src/media/wav/01_Genesis.wav")
        .expect("Failed to open file OONGA BOONGA");
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

    if samples_vec.len() != reader.len() as usize {
        panic!("Number of samples read does not match the total number of samples");
    }

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
