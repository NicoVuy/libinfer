//! # Basic Example
//!
//! Demonstrates basic functionality of `libinfer` by running inference
//! on a TensorRT engine with a synthetic input.
//!
//! ## Usage
//! ```bash
//! cargo run --example basic -- --path /path/to/your/model.engine
//! ```
//!
//! ## Engine Requirements
//! - You must provide your own TensorRT engine file (.engine)
//! - This example works with any TensorRT engine
//! - The example creates zero-filled synthetic input data with the correct dimensions
//! - To create engine files, use the TensorRT Python API or trtexec command-line tool

use clap::Parser;
use libinfer::ffi::InputTensor;
use libinfer::{Engine, Options, OutputTensor, TensorDataType};
use std::alloc::{alloc, dealloc, Layout};
use std::path::PathBuf;
use tracing::{error, info, info_span, Level};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

#[derive(Parser, Debug)]
#[clap(about = "Basic example for libinfer")]
struct Args {
    /// Path to the engine file
    #[arg(short, long, value_name = "PATH", value_parser)]
    path: PathBuf,

    /// Number of iterations to run
    #[arg(short, long, value_name = "ITERATIONS", default_value_t = 1 << 4)]
    iterations: usize,

    /// GPU device index to use
    #[arg(short, long, value_name = "DEVICE", default_value_t = 0)]
    device: u32,
}

/// Create a `Vec<u8>` of `count` elements, all initialised to `value`,
/// and guaranteed to be aligned to `align` bytes.
///
/// # Panics
///
/// Panics if `Layout::from_size_align(count, align)` would overflow or if
/// allocation fails.
pub fn aligned_vec_init(count: usize, value: u8, align: usize) -> Vec<u8> {
    // Safety: `count` and `align` must satisfy the layout rules.
    let layout =
        Layout::from_size_align(count, align).unwrap_or_else(|e| panic!("invalid layout: {e}"));

    // Allocate the raw memory.
    let ptr = unsafe { alloc(layout) };
    if ptr.is_null() {
        std::alloc::handle_alloc_error(layout);
    }

    // Initialise every byte with `value`.
    unsafe {
        std::ptr::write_bytes(ptr, value, count);
    }

    // Turn the raw pointer into a `Vec<u8>`.
    unsafe { Vec::from_raw_parts(ptr, count, count) }
}

/// Convenience wrapper that mimics `vec![0u8; n]` with a 4096-byte alignment.
pub fn aligned_zeros(count: usize) -> Vec<u8> {
    aligned_vec_init(count, 0u8, 4096)
}

fn main() {
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_thread_ids(true)
        .with_target(true)
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::CLOSE); // Logs span close events

    let filter_layer = tracing_subscriber::EnvFilter::new("info");

    tracing_subscriber::registry()
        .with(filter_layer) // Apply log level filter
        .with(fmt_layer)
        .init();

    let args = Args::parse();

    info!("Loading TensorRT engine from: {}", args.path.display());

    // Create engine options
    let options = Options {
        path: args.path.to_string_lossy().to_string(),
        device_index: args.device,
    };

    // Load the engine
    let mut engine = Engine::new(&options).unwrap_or_else(|e| {
        error!("Failed to load engine: {e}");
        std::process::exit(1);
    });

    info!("Engine loaded successfully");

    let input_infos = engine.get_input_dims();
    let output_infos = engine.get_output_dims();

    // Print model information
    info!("Engine loaded successfully");
    info!("Number of inputs: {}", input_infos.len());
    info!("Number of outputs: {}", output_infos.len());
    info!("Batch dimensions: {:?}", engine.get_batch_dims());
    
    // Print detailed information for all input tensors
    info!("Input tensors:");
    for input_info in &input_infos {
        info!(
            "  '{}': {:?} {:?}",
            input_info.name, input_info.dims, input_info.dtype
        );
    }

    // Print detailed information for all output tensors
    info!("Output tensors:");
    for output_info in &output_infos {
        info!(
            "  '{}': {:?} {:?}",
            output_info.name, output_info.dims, output_info.dtype
        );
    }

    // Create input tensors for all inputs
    let mut input_tensors = Vec::new();

    for input_info in &input_infos {
        // Calculate tensor size from dimensions
        let input_size = input_info.dims.iter().fold(1, |acc, &e| acc * e as usize);

        // Create appropriate input data based on data type
        let input_data = match input_info.dtype {
            TensorDataType::UINT8 => aligned_zeros(input_size),
            TensorDataType::FP32 => {
                // For FP32, we need 4 bytes per element
                aligned_zeros(input_size * 4)
            }
            TensorDataType::INT64 => {
                // For INT64, we need 8 bytes per element
                aligned_zeros(input_size * 8)
            }
            TensorDataType::BOOL => aligned_zeros(input_size),
            _ => {
                error!("Unsupported input data type");
                std::process::exit(1);
            }
        };

        info!(
            "input tensor: {} {} bytes",
            input_info.name,
            input_data.len()
        );
        input_tensors.push(InputTensor {
            name: input_info.name.clone(),
            data: input_data,
            dtype: input_info.dtype.clone(),
        });
    }

    info!("Running inference for {} iterations...", args.iterations);

    // Run inference for specified number of iterations
    for i in 0..args.iterations {
        if i % (args.iterations / 10).max(1) == 0 {
            info!("Iteration {}/{}", i, args.iterations);
        }

        let result = {
            let span = info_span!("infer");
            let _guard = span.enter();
            engine.pin_mut().infer(&input_tensors)
        };

        match result {
            Ok(outputs) => {
                if i == 0 {
                    // Print output information on first iteration
                    info!("Inference successful! Output tensors:");
                    for output in &outputs {
                        info!(
                            "  '{}' type {:?} : {} elements, {:?}",
                            output.name,
                            output.dtype,
                            output.data.len(),
                            &output.data[0..10]
                        );
                    }
                }
                let mut outputs2 = vec![];
                outputs.iter().for_each(|output| {
                    outputs2.push(OutputTensor {
                        name: output.name.clone(),
                        dtype: output.dtype,
                        data: aligned_zeros(output.data.len()),
                    });
                });
                if let Err(e) = {
                    let span = info_span!("infer");
                    let _guard = span.enter();
                    engine
                        .pin_mut()
                        .infer_zerocopy(&input_tensors, &mut outputs2)
                } {
                    error!("Zero copy inference error: {}", e);
                    break;
                }

                for i in 0..outputs.len() {
                    assert_eq!(outputs[i].data, outputs2[i].data, "Output data mismatch");
                }
                if i == 0 {
                    // Print output information on first iteration
                    info!("Inference successful! Output tensors:");
                    for output in &outputs2 {
                        info!(
                            "  '{}' type {:?} : {} elements, {:?}",
                            output.name,
                            output.dtype,
                            output.data.len(),
                            &output.data[0..10]
                        );
                    }
                }
            }
            Err(e) => {
                error!("Inference error: {e}");
                break;
            }
        }
    }

    info!("Inference complete!");
}
