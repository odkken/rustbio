use rand::Rng;
use std::array;

#[derive(Clone, Copy)]
pub enum SourceType {
    Neuron(usize),
    Input(usize),
}

#[derive(Clone, Copy)]
pub enum SinkType {
    Neuron(usize),
    Output(usize),
}

#[derive(Clone)]
pub struct DecodedGene {
    source: SourceType,
    sink: SinkType,
    weight: f32,
    // label: String,
}
//add the clone trait
#[derive(Clone)]
pub struct Organism<
    const NUM_GENES: usize,
    const NUM_NEURONS: usize,
    const NUM_INPUTS: usize,
    const NUM_OUTPUTS: usize,
> {
    pub neurons: [f32; NUM_NEURONS],
    pub inputs: [f32; NUM_INPUTS],
    pub outputs: [f32; NUM_OUTPUTS],
    pub decoded_genes: [DecodedGene; NUM_GENES],
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
    pub fn new<R: Rng>(rng: &mut R) -> Self {
        let genome: [u32; NUM_GENES] = array::from_fn(|_| rng.gen());
        let neurons = [0.0; NUM_NEURONS];
        let inputs = [0.0; NUM_INPUTS];
        let outputs = [0.0; NUM_OUTPUTS];

        let decoded_genes = Self::decode_genes(&genome);

        Self {
            neurons,
            inputs,
            outputs,
            decoded_genes,
        }
    }

    //gene encoding from MSB to LSB:
    //first (MSB) bit: source type (0 for neuron, 1 for input)
    //next 7 bits: source index (neuron index or input index)
    //next bit: sink type (0 for neuron, 1 for output)
    //next 7 bits: sink index (neuron index or output index)
    //last 16 bits: weight (0.0 to 1.0)
    pub fn decode_genes(genome: &[u32; NUM_GENES]) -> [DecodedGene; NUM_GENES] {
        array::from_fn(|i| {
            let gene = genome[i];
            let source_type = gene & 0x8000_0000 != 0;
            let source_index = ((gene >> 15) & 0x7F) as usize;
            let sink_type = gene & 0x0080_0000 != 0;
            let sink_index = ((gene >> 8) & 0x7F) as usize;
            //scale weight from -1.0 to 1.0
            let weight = (gene & 0xFFFF) as f32 / 0xFFFF as f32 * 2.0 - 1.0;

            let source = if source_type {
                SourceType::Input(source_index % NUM_INPUTS)
            } else {
                SourceType::Neuron(source_index % NUM_NEURONS)
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
                //label will be the index of the gene, the source type/index, the sink type/index, and the weight
                // label: format!(
                //     "{}: {} -> {} -> {}",
                //     i,
                //     match source {
                //         SourceType::Neuron(idx) => format!("N{}", idx),
                //         SourceType::Input(idx) => format!("I{}", idx),
                //     },
                //     match sink {
                //         SinkType::Neuron(idx) => format!("N{}", idx),
                //         SinkType::Output(idx) => format!("O{}", idx),
                //     },
                //     weight
                // ),
            }
        })
    }
    //noinline for performance testing
    #[inline(never)]
    pub fn update(&mut self) {
        // Accumulate updates
        for gene in &self.decoded_genes {
            let input_value = match gene.source {
                SourceType::Neuron(idx) => sigmoid(self.neurons[idx]),
                SourceType::Input(idx) => sigmoid(self.inputs[idx]),
            };
            let update = input_value * gene.weight;

            match gene.sink {
                SinkType::Neuron(idx) => self.neurons[idx] += update,
                SinkType::Output(idx) => self.outputs[idx] += update,
            }
        }
    }
}
