use rand::Rng;
use std::array;

#[derive(Clone, Copy)]
enum SourceType {
    Neuron(usize),
    Input(usize),
}

#[derive(Clone, Copy)]
enum SinkType {
    Neuron(usize),
    Output(usize),
}

#[derive(Clone, Copy)]
struct DecodedGene {
    source: SourceType,
    sink: SinkType,
    weight: f32,
}

pub struct Organism<
    const NUM_GENES: usize,
    const NUM_NEURONS: usize,
    const NUM_INPUTS: usize,
    const NUM_OUTPUTS: usize,
> {
    pub genome: [u32; NUM_GENES],
    pub neurons: [f32; NUM_NEURONS],
    pub inputs: [f32; NUM_INPUTS],
    pub outputs: [f32; NUM_OUTPUTS],
    decoded_genes: [DecodedGene; NUM_GENES],
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl<
        const NUM_GENES: usize,
        const NUM_NEURONS: usize,
        const NUM_INPUTS: usize,
        const NUM_OUTPUTS: usize,
    > Organism<NUM_GENES, NUM_NEURONS, NUM_INPUTS, NUM_OUTPUTS>
{
    pub fn new() -> Self {
        let mut rng = rand::thread_rng();
        let genome: [u32; NUM_GENES] = array::from_fn(|_| rng.gen());
        let neurons = [0.0; NUM_NEURONS];
        let inputs = [0.0; NUM_INPUTS];
        let outputs = [0.0; NUM_OUTPUTS];

        let decoded_genes = Self::decode_genes(&genome);

        Self {
            genome,
            neurons,
            inputs,
            outputs,
            decoded_genes,
        }
    }

    fn decode_genes(genome: &[u32; NUM_GENES]) -> [DecodedGene; NUM_GENES] {
        array::from_fn(|i| {
            let gene = genome[i];
            let source_type = gene & 1 != 0;
            let source_index = ((gene >> 1) & 0x7F) as usize;
            let sink_type = (gene >> 8) & 1 != 0;
            let sink_index = ((gene >> 9) & 0x7F) as usize;
            let weight = ((gene >> 16) & 0xFFFF) as f32 / 65535.0;

            let source = if source_type {
                SourceType::Neuron(source_index % NUM_NEURONS)
            } else {
                SourceType::Input(source_index % NUM_INPUTS)
            };

            let sink = if sink_type {
                SinkType::Output(sink_index % NUM_OUTPUTS)
            } else {
                SinkType::Neuron(sink_index % NUM_NEURONS)
            };

            DecodedGene {
                source,
                sink,
                weight,
            }
        })
    }
    //noinline for performance testing
    #[inline(never)]
    pub fn update(&mut self) {
        let mut neuron_updates = [0.0; NUM_NEURONS];
        let mut output_updates = [0.0; NUM_OUTPUTS];

        // Accumulate updates
        for gene in &self.decoded_genes {
            let input_value = match gene.source {
                SourceType::Neuron(idx) => self.neurons[idx],
                SourceType::Input(idx) => self.inputs[idx],
            };
            let update = input_value * gene.weight;
            match gene.sink {
                SinkType::Neuron(idx) => neuron_updates[idx] += update,
                SinkType::Output(idx) => output_updates[idx] += update,
            }
        }

        // Apply updates and activation function
        // for (neuron, update) in self.neurons.iter_mut().zip(neuron_updates.iter()) {
        //     *neuron = sigmoid(*neuron + *update);
        // }
        // for (output, update) in self.outputs.iter_mut().zip(output_updates.iter()) {
        //     *output = sigmoid(*output + *update);
        // }
    }
}
