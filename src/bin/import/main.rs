/// This build script does the following:
/// 1. Loads PyTorch weights into a model record.
/// 2. Saves the model record to a file using the `NamedMpkFileRecorder`.
use std::path::Path;

use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder};
use burn_import::pytorch::{LoadArgs, PyTorchFileRecorder};
use burn_ndarray::NdArray;
use stablediffusion::model::stablediffusion::StableDiffusionRecord;

// Basic backend type (not used directly here).
type B = NdArray<f32>;

fn main() {
    let device = burn_ndarray::NdArrayDevice::Cpu;

    let load_args = LoadArgs::new("sd-v1-4.ckpt".into())
        .with_key_remap("model.diffusion_model", "diffusion")
        .with_top_level_key("state_dict")
        .with_debug_print();

    println!("Loading PyTorch weights into a model record...");
    // Load PyTorch weights into a model record.
    let record: StableDiffusionRecord<B> = PyTorchFileRecorder::<FullPrecisionSettings>::default()
        .load(load_args, &device)
        .expect("Failed to decode state");

    println!("Saving the model record to a file...");
    // Save the model record to a file.
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();

    // Save into the OUT_DIR directory so that the model can be loaded by the
    // let out_dir = std::env::var("OUT_DIR").unwrap();
    // let file_path = Path::new(&out_dir).join("model/mnist");
    let file_path = Path::new("sd-v1-4.bin");

    recorder
        .record(record, file_path.into())
        .expect("Failed to save model record");
}
