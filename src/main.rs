use minifb::{Key, ScaleMode, Window, WindowOptions};
mod organism;
use core::f32;
use organism::Organism;
use rand::prelude::*;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU32, Ordering};

fn main() {
    const NUM_GENES: usize = 16;
    const NUM_NEURONS: usize = 4;
    const NUM_INPUTS: usize = 1;
    const NUM_OUTPUTS: usize = 1;
    const WIDTH: usize = 100;
    const HEIGHT: usize = 100;
    const NUM_ORGANISMS: usize = WIDTH * HEIGHT;

    let mut rng = thread_rng();
    let mut organisms: Box<[Organism<NUM_GENES, NUM_NEURONS, NUM_INPUTS, NUM_OUTPUTS>]> = (0
        ..NUM_ORGANISMS)
        .map(|_| Organism::new(&mut rng))
        .collect::<Vec<_>>()
        .into_boxed_slice();

    // Use a Vec of AtomicU32 instead of u32
    let buffer: Vec<AtomicU32> = (0..WIDTH * HEIGHT).map(|_| AtomicU32::new(0)).collect();

    let dt = 0.1;
    let total_time = f32::consts::PI * 2.0;
    let num_steps = (total_time / dt) as usize;

    let start = std::time::Instant::now();

    let mut window = Window::new(
        "Organism Simulation - Press ESC to exit",
        WIDTH,
        HEIGHT,
        WindowOptions {
            resize: true,
            scale_mode: ScaleMode::UpperLeft,
            ..WindowOptions::default()
        },
    )
    .expect("Unable to create the window");

    window.set_target_fps(0);

    let mut t: f32 = 0.0;

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let sin = t.sin();

        organisms
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, organism)| {
                organism.inputs[0] = (sin * 255.0);
                organism.update();

                // // Convert the output to a color
                let output = (organism.outputs[0] * 255.0) as u32;
                let color = (output << 16) | (output << 8) | output;

                // Use atomic operation to update the buffer
                buffer[i].store(color, Ordering::Relaxed);

                //for now just use the raw sin value to do a grayscale gradient from 0,0,0 to 255,255,255
                // let color = ((sin * 255.0) as u32) << 16
                //     | ((sin * 255.0) as u32) << 8
                //     | (sin * 255.0) as u32;
                // buffer[i].store(color, Ordering::Relaxed);
            });

        // Create a temporary buffer for minifb to use
        let temp_buffer: Vec<u32> = buffer
            .iter()
            .map(|atomic| atomic.load(Ordering::Relaxed))
            .collect();

        window
            .update_with_buffer(&temp_buffer, WIDTH, HEIGHT)
            .unwrap();

        t += dt;
        if t > total_time {
            t = 0.0;
        }
    }

    let elapsed = start.elapsed();

    println!("Executed {} steps in {:?}", num_steps, elapsed);
    let fps = num_steps as f32 / elapsed.as_secs_f32();
    println!("FPS: {}", fps);
}
