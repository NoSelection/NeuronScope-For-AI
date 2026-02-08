# NeuronScope TODO

## Paper: "Should We Trust AI Self-Reports?"

Empirically investigating whether LLM self-referential outputs are grounded in internal representations or surface-level pattern matching, and what this means for AI literacy in education.

### Pilot Study (DONE)
- [x] 4 experiments (self_recognition, capability_awareness, training_knowledge, metacognition)
- [x] 2 ablation types (zero, mean) x 2 inputs (base, control) = 16 sweeps
- [x] 544 forward passes across 34 layers
- [x] Identified candidate self-model circuit: layers 5, 6, 8
- [x] Results in `results/self_model_circuits/`

### Additional Experiments Needed
- [ ] Expand to 15-20+ prompt pairs (diverse self-referential vs control)
- [ ] Factual knowledge baseline ("The Eiffel Tower is in") to confirm layers 5-8 are self-model specific, not general fact storage
- [ ] Attention head sweep on layers 5, 6, 8 to narrow the circuit
- [ ] Activation patching: transplant self-referential behavior between inputs via layers 5-8
- [ ] Bootstrap confidence intervals / permutation tests on differential KL
- [ ] Test on different model sizes (Gemma 2B, 9B) to see if circuit location scales

### Education Component
- [ ] Design student survey on AI self-report trust (pre/post interpretability demo)
- [ ] Develop simplified NeuronScope demo for non-technical educators
- [ ] Connect findings to AI literacy curriculum frameworks

### Writing
- [ ] Literature review: mechanistic interpretability + AI in education
- [ ] Methodology section
- [ ] Results + discussion
- [ ] Implications for AI literacy education
