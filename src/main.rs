use hound;

fn main() {
    println!("Hello, world!");
    let mut reader = hound::WavReader::open("src/media/wav/01_Genesis.wav")
        .expect("Failed to open file");
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
}
