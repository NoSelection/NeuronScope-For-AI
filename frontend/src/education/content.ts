/**
 * Educational content for every concept in NeuronScope.
 * Written for CS students / ML beginners — no assumed familiarity
 * with mechanistic interpretability.
 */

export interface EducationalEntry {
  short: string; // 1-sentence tooltip
  long: string; // 2-4 sentence explanation
  analogy?: string; // Optional real-world analogy
}

// ---------- Experiment Builder ----------

export const EDUCATION = {
  // -- Experiment concepts --
  experiment: {
    short: 'A controlled test where you break one part of the model and measure what changes.',
    long: 'An experiment runs the model twice on the same input: once normally (the "clean run") and once with a specific internal component disabled or modified (the "intervention run"). By comparing the two outputs, you can see exactly what that component was doing.',
    analogy: 'Like unplugging one wire in a circuit and checking which lights go off.',
  },

  base_input: {
    short: 'The text prompt the model will process.',
    long: 'This is the sentence you feed into the model. The model reads this text and tries to predict the next word. You\'ll see how the prediction changes when you intervene on the model\'s internals.',
    analogy: 'Think of it as the question you\'re asking the model — "The Eiffel Tower is in" naturally leads to "Paris".',
  },

  source_input: {
    short: 'A second prompt used only for activation patching.',
    long: 'When using activation patching, the model processes this second input separately. Then the internal activations (the numbers computed inside the model) from this source run are copied into the base run, letting you test what information flows through a specific component.',
    analogy: 'Like transplanting a memory from one brain into another and seeing if the answer changes.',
  },

  experiment_name: {
    short: 'A label to identify this experiment in your history.',
    long: 'Give your experiment a descriptive name so you can find it later. Good names describe what you\'re testing, e.g., "zero_L17_mlp_eiffel" tells you: zero ablation, layer 17, MLP, Eiffel Tower prompt.',
  },

  seed: {
    short: 'A number that ensures identical results if you run the same experiment again.',
    long: 'Neural networks can have small random variations between runs. The seed locks the random number generator so every run with the same seed produces the exact same output. This is critical for reproducibility — if you share your seed and config, anyone can replicate your result.',
  },

  // -- Interventions --
  intervention: {
    short: 'A modification you make to the model\'s internals during processing.',
    long: 'An intervention changes the activations (internal numbers) at a specific point in the model while it processes your input. This lets you test causal hypotheses: "Is this component responsible for the model\'s output?"',
    analogy: 'Like covering one eye and checking if you can still read — you\'re testing what each "eye" contributes.',
  },

  layer: {
    short: 'One processing step in the model\'s pipeline (0 = first, 33 = last).',
    long: 'A transformer model processes text through a series of layers, each building on the previous one. Early layers tend to handle syntax and basic patterns, middle layers handle semantic meaning, and late layers handle the final prediction. This model has 34 layers (numbered 0-33).',
    analogy: 'Like floors in a factory — raw materials enter at the ground floor, and each floor adds more refinement until the finished product comes out at the top.',
  },

  component: {
    short: 'Which sub-part of a layer to intervene on.',
    long: 'Each layer contains multiple components that do different things. The attention mechanism decides which words to focus on. The MLP (feed-forward network) transforms and stores knowledge. You can target each separately to understand their individual roles.',
  },

  component_embedding: {
    short: 'The initial conversion of words into numbers.',
    long: 'Before the model can process text, each word (token) is converted into a long list of numbers called an embedding vector. This is the model\'s "vocabulary" — similar words get similar numbers. Intervening here changes what the model "sees" at the very start.',
  },

  component_residual_pre: {
    short: 'The activation stream entering a layer, before processing.',
    long: 'The residual stream is the main "highway" of information flowing through the model. This captures it just before a layer processes it. Think of it as reading the mail before it reaches a department.',
  },

  component_residual_post: {
    short: 'The activation stream leaving a layer, after processing.',
    long: 'This captures the residual stream after a layer has added its contribution. Comparing pre and post tells you exactly what that layer added to the computation.',
  },

  component_attn_output: {
    short: 'The output of the attention mechanism — what the model decided to "focus on".',
    long: 'Attention lets the model look at all previous tokens and decide which ones are relevant for predicting the next word. The attention output is the result of that focusing process. Ablating it tests whether the model needs to attend to context here. You can also target individual attention heads within this component for more surgical analysis.',
  },

  attention_head: {
    short: 'One of 8 independent "focus patterns" within each layer\'s attention mechanism.',
    long: 'Each layer has multiple attention heads that independently decide which tokens to focus on. Different heads learn different patterns — one might track grammatical subjects, another might follow positional patterns, and another might look for semantic relationships. By ablating one head at a time, you can discover what specific focus pattern each head has learned.',
    analogy: 'Like a team of 8 readers each highlighting different parts of the same document. One highlights names, another highlights verbs, another highlights locations. Removing one reader\'s highlights tells you what information they were uniquely contributing.',
  },

  head_sweep: {
    short: 'Test each attention head individually to find which ones matter for the prediction.',
    long: 'A head sweep runs one experiment per attention head within a chosen layer, ablating each head separately while leaving the others intact. The result shows which heads carry information critical for the prediction. Heads with high KL divergence are attending to tokens that carry essential information.',
    analogy: 'Like testing each member of a band by muting them one at a time. If muting the guitarist kills the melody, the guitarist was carrying the tune.',
  },

  component_attn_pattern: {
    short: 'The attention weights — which tokens attend to which other tokens.',
    long: 'Attention patterns are matrices showing how strongly each token "looks at" every other token. High values mean the model is focusing heavily on that relationship. This is one of the most interpretable parts of a transformer.',
  },

  component_mlp_gate: {
    short: 'The gating mechanism that controls which neurons activate in the MLP.',
    long: 'Modern transformers use a "gated" MLP where one pathway decides which neurons to activate (the gate) and another provides the values. The gate effectively selects which stored knowledge to use for this input.',
  },

  component_mlp_output: {
    short: 'The final output of the MLP feed-forward network.',
    long: 'The MLP (Multi-Layer Perceptron) is where the model stores and retrieves factual knowledge. Research shows that specific facts like "Eiffel Tower → Paris" are often encoded in MLP layers. Ablating the MLP output tests whether that layer stores relevant knowledge.',
    analogy: 'Like removing one book from a library shelf and seeing if the librarian can still answer the question.',
  },

  component_final_logits: {
    short: 'The model\'s final prediction scores before choosing a word.',
    long: 'Logits are the raw scores the model assigns to every word in its vocabulary as potential next words. Higher logit = more likely to be chosen. These are converted to probabilities using the softmax function.',
  },

  // -- Intervention types --
  intervention_zero: {
    short: 'Completely silence a component by setting its output to zero.',
    long: 'Zero ablation replaces all activations at the target with zeros, completely removing that component\'s contribution. If the output changes dramatically, that component was important. This is the simplest and most common causal test.',
    analogy: 'Like muting one instrument in an orchestra — if the song sounds wrong, that instrument was essential.',
  },

  intervention_mean: {
    short: 'Replace activations with the average over all positions.',
    long: 'Mean ablation replaces the target\'s activations with the average (mean) activation computed over all token positions. This removes position-specific information while keeping the general "shape" of activity. It\'s a gentler test than zero ablation.',
    analogy: 'Like replacing a specific answer with "the average answer" — you lose the specifics but keep the general gist.',
  },

  intervention_patch: {
    short: 'Copy activations from a different input to test information flow.',
    long: 'Activation patching takes the activations computed on your source input and injects them into the base input\'s computation at a specific point. If the output changes to match the source, that component carries the distinguishing information between the two inputs. This is the gold standard for tracing information flow.',
    analogy: 'Like swapping a student\'s notes from one class into another exam — if their answers change, those notes contained the key difference.',
  },

  intervention_additive: {
    short: 'Add a perturbation (noise) to activations.',
    long: 'Additive perturbation adds a fixed value to the activations at the target. This lets you test how sensitive the model is to changes at that point — if a small perturbation causes a big output change, that component is in a critical position.',
  },

  token_position: {
    short: 'Which specific word (token) in the input to intervene on.',
    long: 'When left empty, the intervention applies to all tokens equally. When set to a specific number (e.g., 3), it only modifies the activations for that specific token, leaving the rest untouched. This lets you test whether a component\'s role is token-specific.',
    analogy: 'Like asking "which specific word in the sentence is this part of the model responding to?"',
  },

  // -- Results / Metrics --
  kl_divergence: {
    short: 'How much the intervention changed the model\'s prediction distribution.',
    long: 'KL divergence (Kullback-Leibler divergence) measures how different two probability distributions are. A KL of 0 means the intervention had zero effect. A KL above 1.0 usually indicates a meaningful causal effect. Above 5.0 suggests the component is critical for this prediction.',
    analogy: 'Like a "disturbance score" — 0 means nothing changed, bigger numbers mean bigger disruption.',
  },

  top_token_changed: {
    short: 'Whether the model\'s #1 prediction flipped to a different word.',
    long: 'This is the simplest test: did the intervention change what word the model would actually output? If yes, the component was directly responsible for the model choosing that specific word. This is a binary (yes/no) indicator of strong causal effect.',
  },

  clean_output: {
    short: 'What the model predicts normally, without any intervention.',
    long: 'The "clean" output is the model\'s natural prediction for your input text. This is the baseline — all intervention effects are measured against this. The probability (p) shows how confident the model was in this prediction.',
  },

  intervention_output: {
    short: 'What the model predicts after you modified its internals.',
    long: 'This is the model\'s prediction after your intervention was applied. Compare it with the clean output to see the effect. If the token changed, the intervention had a strong effect. If the probability dropped, the intervention weakened the model\'s confidence.',
  },

  logit: {
    short: 'The raw score a model assigns to a potential next word.',
    long: 'Before the model picks a word, it assigns a score (logit) to every word in its vocabulary. Higher logits mean the model considers that word more likely. Logits are turned into probabilities by the softmax function, which normalizes them so they add up to 100%.',
  },

  probability: {
    short: 'The model\'s confidence that a word is the right next word (0-100%).',
    long: 'Probability is the normalized version of the logit score. A probability of 90% means the model is very confident; 5% means it\'s considering many alternatives. After an intervention, watch how probabilities shift — even if the top word doesn\'t change, confidence changes reveal partial effects.',
  },

  rank_changes: {
    short: 'How words moved up or down in the model\'s preference ranking.',
    long: 'The model ranks all words from most to least likely. Rank changes show which words moved after the intervention. A word going from rank 1 to rank 50 means the intervention destroyed the evidence for that word. A word rising from rank 100 to rank 1 means the intervention removed something that was suppressing it.',
    analogy: 'Like a sports league table — see which "players" (words) gained or lost positions after you changed the rules.',
  },

  effect_size: {
    short: 'A normalized measure of how impactful the intervention was.',
    long: 'Effect size normalizes the KL divergence to make it comparable across different experiments. It accounts for the baseline distribution so you can meaningfully compare effects across different prompts and components.',
  },

  config_hash: {
    short: 'A fingerprint that uniquely identifies this exact experiment setup.',
    long: 'The config hash is computed from every parameter of your experiment. If you run the same experiment again and get the same hash, the results should be identical. This is how NeuronScope guarantees reproducibility — share the hash and anyone can verify your result.',
  },

  // -- Layer Sweep --
  layer_sweep: {
    short: 'Run the same intervention on every layer to find which ones matter most.',
    long: 'A layer sweep repeats your experiment 34 times — once per layer — measuring the causal effect at each. The resulting bar chart shows you which layers are critical for the prediction. Tall bars = important layers. This is often the first thing interpretability researchers do to map out a model\'s behavior.',
    analogy: 'Like testing each floor of the factory by shutting it down one at a time and seeing how the output changes.',
  },

  peak_kl: {
    short: 'The layer where the intervention had the largest effect.',
    long: 'The peak KL tells you which layer is most responsible for the prediction you\'re studying. In many cases, factual knowledge is concentrated in a small number of layers — the peak reveals where.',
  },

  // -- Model Info --
  architecture: {
    short: 'The model\'s design blueprint — what type of neural network it is.',
    long: 'This tells you the model family (e.g., Gemma 3). Different architectures have different internal structures, which affects where you should look for specific behaviors. NeuronScope automatically maps the architecture to the correct internal module paths.',
  },

  num_layers: {
    short: 'How many processing steps the model has.',
    long: 'More layers = more processing capacity. This model has 34 layers, meaning your input text passes through 34 sequential transformation stages. Each layer has its own attention heads and MLP, all of which can be individually targeted.',
  },

  hidden_size: {
    short: 'The width of the model\'s internal representations (2560 numbers per token).',
    long: 'At every layer, each token is represented as a vector of this many numbers. A hidden size of 2560 means each token\'s "state" is described by 2560 floating-point numbers. Larger hidden sizes allow the model to represent more complex concepts.',
  },

  intermediate_size: {
    short: 'The width of the MLP\'s hidden layer (10240 = 4x hidden size).',
    long: 'The MLP in each layer expands the representation to this larger size, applies non-linear transformations, then compresses back down. This expansion is where much of the model\'s "knowledge" is stored. The 4x ratio is standard in transformers.',
  },

  vocab_size: {
    short: 'How many unique words/tokens the model knows.',
    long: 'The vocabulary is the set of all tokens (word pieces) the model can read and produce. Each token has a unique ID. The final layer produces a score (logit) for every token in the vocabulary, and the highest-scoring one becomes the output.',
  },

  dtype: {
    short: 'The numerical precision used for computations.',
    long: 'bfloat16 means each number uses 16 bits of memory instead of 32. This halves memory usage with minimal accuracy loss. It\'s the standard for running large models on consumer GPUs.',
  },

  // -- Activations --
  activations: {
    short: 'The actual numbers computed inside the model as it processes your text.',
    long: 'Activations are the intermediate values at every point in the neural network. When you "capture" activations, you\'re recording what the model computed at a specific layer and component. These numbers are what interventions modify.',
    analogy: 'Like reading the voltages at every point in a circuit to understand what\'s happening.',
  },

  tensor_stats: {
    short: 'Summary statistics describing the captured activation values.',
    long: 'Mean tells you the average activation value. Std (standard deviation) tells you how spread out the values are. Min/max give the extremes. Norm measures the overall magnitude. These help you understand the activation landscape without examining every individual number.',
  },

  // -- Insights --
  insights: {
    short: 'Plain-language explanations of what your results mean.',
    long: 'NeuronScope automatically analyzes your experiment results and generates human-readable insights. Critical findings (red) indicate strong causal effects. Notable findings (amber) highlight interesting patterns. Info (blue) provides context. These help you interpret results without being an expert.',
  },
} as const satisfies Record<string, EducationalEntry>;

export type EducationKey = keyof typeof EDUCATION;
