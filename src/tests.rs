use super::*;
use organism::sigmoid;
use organism::Organism;
use organism::SinkType;
use organism::SourceType;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn test_new_organism() {
    let mut rng = StdRng::seed_from_u64(42); // Use a seeded RNG for reproducibility
    let organism: Organism<4, 2, 1, 1> = Organism::new(&mut rng);

    assert_eq!(organism.neurons, [0.0; 2]);
    assert_eq!(organism.inputs, [0.0; 1]);
    assert_eq!(organism.outputs, [0.0; 1]);
    // We can't easily test the decoded_genes as they're randomly generated
}

#[test]
fn test_sigmoid() {
    assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    assert!((sigmoid(1.0) - 0.7310585786300049).abs() < 1e-6);
    assert!((sigmoid(-1.0) - 0.2689414213699951).abs() < 1e-6);
}

// Helper function to create an organism with specific genes
fn create_organism_with_genes<const G: usize, const N: usize, const I: usize, const O: usize>(
    genes: [u32; G],
) -> Organism<G, N, I, O> {
    let mut organism = Organism {
        neurons: [0.0; N],
        inputs: [0.0; I],
        outputs: [0.0; O],
        decoded_genes: unsafe { std::mem::zeroed() },
    };
    organism.decoded_genes = Organism::<G, N, I, O>::decode_genes(&genes);
    organism
}

//gene encoding from MSB to LSB:
//first (MSB) bit: source type (0 for neuron, 1 for input)
//next 7 bits: source index (neuron index or input index)
//next bit: sink type (0 for neuron, 1 for output)
//next 7 bits: sink index (neuron index or output index)
//last 16 bits: weight (-1.0 to 1.0)
fn create_gene(source: SourceType, sink: SinkType, weight: f32) -> u32 {
    let source_bits = match source {
        SourceType::Neuron(index) => index as u32,
        SourceType::Input(index) => index as u32 | 0x8000,
    };
    let sink_bits = match sink {
        SinkType::Neuron(index) => index as u32,
        SinkType::Output(index) => index as u32 | 0x8000,
    };
    let weight_bits = ((weight + 1.0) / 2.0 * 0xFFFF as f32) as u32;

    (source_bits << 16) | (sink_bits << 8) | weight_bits
}

#[test]
fn test_identity_neuron() {
    // Create an organism with a single internal connection from input -> internal -> output with weights 1.0
    let genes = [
        create_gene(SourceType::Input(0), SinkType::Neuron(0), 1.0),
        create_gene(SourceType::Neuron(0), SinkType::Output(0), 1.0),
    ];
    let mut organism: Organism<2, 1, 1, 1> = create_organism_with_genes(genes);

    organism.inputs[0] = 0.5;
    organism.update();
    let expected_output = sigmoid(sigmoid(0.5));
    println!(
        "Input: {}, Output: {}, Expected: {}",
        organism.inputs[0], organism.outputs[0], expected_output
    );
    assert!((organism.outputs[0] - expected_output).abs() < 1e-6);
}
#[test]
fn test_addition_network() {
    // Create an organism that adds two inputs
    let genes = [
        create_gene(SourceType::Input(0), SinkType::Neuron(0), 1.0),
        create_gene(SourceType::Input(1), SinkType::Neuron(0), 1.0),
        create_gene(SourceType::Neuron(0), SinkType::Output(0), 1.0),
    ];
    let mut organism: Organism<3, 1, 2, 1> = create_organism_with_genes(genes);

    organism.inputs = [0.5, 0.5];
    organism.update();
    // The output should be the sigmoid of the sum of the sigmoid of the inputs
    let expected_output = sigmoid(sigmoid(0.5) + sigmoid(0.5));
    println!(
        "Inputs: {:?}, Output: {}, Expected: {}",
        organism.inputs, organism.outputs[0], expected_output
    );
    assert!((organism.outputs[0] - expected_output).abs() < 1e-6);
}