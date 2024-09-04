use minifb::{Key, ScaleMode, Window, WindowOptions};
mod organism;
use core::{f32, num};
use crossbeam::atomic::AtomicCell;
use organism::Organism;
use rand::prelude::*;
use rayon::prelude::*;
use std::sync::{Arc, RwLock};
use std::thread;
use std::time::{Duration, Instant};

fn main() {
    const NUM_GENES: usize = 128;
    const NUM_NEURONS: usize = 64;
    const NUM_INPUTS: usize = 1;
    const NUM_OUTPUTS: usize = 1;
    const WIDTH: usize = 100;
    const HEIGHT: usize = 10;
    const NUM_ORGANISMS: usize = WIDTH * HEIGHT;

    let mut rng = thread_rng();
    let organisms: Arc<RwLock<Box<[Organism<NUM_GENES, NUM_NEURONS, NUM_INPUTS, NUM_OUTPUTS>]>>> =
        Arc::new(RwLock::new(
            (0..NUM_ORGANISMS)
                .map(|_| Organism::new(&mut rng))
                .collect::<Vec<_>>()
                .into_boxed_slice(),
        ));

    let buffer: Arc<Vec<AtomicCell<u32>>> =
        Arc::new((0..WIDTH * HEIGHT).map(|_| AtomicCell::new(0)).collect());

    let dt = 0.01;
    let total_time = f32::consts::PI * 2.0;

    let update_rate = Arc::new(AtomicCell::new(0.0));

    // Spawn the update thread
    let update_organisms = Arc::clone(&organisms);
    let update_buffer = Arc::clone(&buffer);
    let thread_update_rate = Arc::clone(&update_rate);
    let update_thread = thread::spawn(move || {
        let mut t: f32 = 0.0;
        let mut last_time = Instant::now();
        let mut frames = 0;
        loop {
            let sin: f32 = t.sin();

            {
                let organisms = update_organisms.read().unwrap();
                organisms.par_iter().enumerate().for_each(|(i, organism)| {
                    let mut organism = organism.clone(); // Clone for thread-safe mutation
                    organism.inputs[0] = sin;
                    organism.update();

                    let output = organism.outputs[0];
                    let color = ((output * 255.0) as u32) << 16
                        | ((output * 255.0) as u32) << 8
                        | (output * 255.0) as u32;

                    update_buffer[i].store(color);
                });
            }

            t += dt;
            if t > total_time {
                t = 0.0;
            }

            frames += 1;
            let elapsed = last_time.elapsed();
            if elapsed >= Duration::from_secs(1) {
                let rate = frames as f32 / elapsed.as_secs_f32();
                thread_update_rate.store(rate);
                frames = 0;
                last_time = Instant::now();
            }
        }
    });

    // Main thread for rendering
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

    window.set_target_fps(60);

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mut buffer_snapshot: Vec<u32> = buffer.iter().map(|cell| cell.load()).collect();

        let update_speed = update_rate.load();
        //do a calculation based on how many organisms we have,
        //along with how many connections each organism has
        //to determine the nanoseconds per neuron firing
        let num_neurons = NUM_GENES * NUM_ORGANISMS;
        let nanoseconds_per_neuron = 1_000_000_000.0 / (num_neurons as f32 * update_speed);
        let fps_text = format!(
            "FPS: {:.2}, {:.2} ns/neuron",
            update_speed,
            nanoseconds_per_neuron
        );

        window.set_title(&fps_text);

        window
            .update_with_buffer(&buffer_snapshot, WIDTH, HEIGHT)
            .unwrap();
    }
}
