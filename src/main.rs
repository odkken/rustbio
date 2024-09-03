mod organism;
use organism::Organism;

//define a simple neural network
//"input sensors" are just u8s which will be set by the program (environment) every frame
//"output actuators" are just u8s which will be updated by the organism every frame, and read
//by the program (environment) every frame

//the organism will have a "brain" which is a neural network.  the brain will have a number of internal neurons stored in an array
//the connections between neurons will be determined by the organism's genome, which is an array of 32-bit integers

//each gene in the genome will be a 32-bit integer, with the following structure:
//bit 0: source type (0=input, 1=neuron)
//bits 1-7: source index (0-127)
//bit 8: sink type (0=neuron, 1=output)
//bits 9-15: sink index (0-127)
//bits 16-31: weight (0-65535)

//number of genes is hardcoded
//number of possible internal neurons is hardcoded

fn main() {
    //create a single organism.  for now, single input which is set to sin(x) every frame, and single output which is printed every frame

    //create a single environment.  for now, just a sin wave generator

    //declare all these as usize
    const NUM_GENES: usize = 32;
    const NUM_NEURONS: usize = 16;
    const NUM_INPUTS: usize = 1;
    const NUM_OUTPUTS: usize = 1;

    let mut organism = Organism::<NUM_GENES, NUM_NEURONS, NUM_INPUTS, NUM_OUTPUTS>::new();
    let dt = 0.001;
    let total_time = 10000.0;
    let num_steps = (total_time / dt) as usize;

    //time how long it takes to execute the organism
    let start = std::time::Instant::now();

    for t in 0..num_steps {
        let sin = (t as f32 * dt).sin();
        organism.inputs[0] = sin;

        organism.update();
    }
    let elapsed = start.elapsed();

    println!("Executed {} steps in {:?}", num_steps, elapsed);
}
