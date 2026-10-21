# SignalDNA — Methods Draft

**Status:** Source-audited draft; pending evaluation-code audit and manuscript review.

## Dataset construction

SignalDNA uses a deterministic synthetic binary-classification dataset containing 12,000 waveforms. Each waveform comprises 1,024 float64 samples at a sampling frequency of 2,048 Hz, corresponding to a duration of 0.5 seconds.

Each dataset split is exactly class-balanced. Class 0 contains background waveforms; class 1 contains background waveforms with an added localized Gaussian transient. The splits contain 8,400 training examples, 1,800 validation examples, and 1,800 held-out test examples.

Separate pseudorandom streams are derived using NumPy SeedSequence.spawn(3), with parent seed 2026. This separates the streams but does not mathematically guarantee waveform uniqueness. The held-out test set has not been evaluated in the reported experiments.

## Background waveform generation

The fundamental component is:

    x_base(t) = A sin(2*pi*f*t + phi)

where f is sampled uniformly from 20 to 200 Hz, A from 0.5 to 1.5, and phi from 0 to 2*pi.

Zero to three optional harmonics, corresponding to orders 2 through 4, are added. Each harmonic has a strength sampled uniformly from 0 to 0.5, scaled by the fundamental amplitude. In the implemented generator, harmonic phases are zero rather than inherited from the fundamental phase.

Independent Gaussian noise is added with standard deviation A*r, where r is sampled uniformly from 0 to 0.15.

Sinusoidal baseline drift is then added, with amplitude uniformly sampled from 0 to 0.3, frequency from 0.1 to 5 Hz, and phase from 0 to 2*pi.

## Positive-class transient

For class 1, the waveform receives an additional Gaussian pulse:

    g(u) = s*M*exp(-0.5*((u-p)/w)^2)

where u = sample_index/1024 is normalized waveform position, s is randomly -1 or +1, M is sampled uniformly from 0.2 to 1.5, p from 0.1 to 0.9, and w from 0.005 to 0.05.

The transient width is a fraction of waveform duration, not a value in seconds. The generator accepts a wider range of widths, but the dataset builder uses the narrower interval stated above.

## Neural-network architecture

The classifier is a fully connected feedforward network with layer dimensions 1024-64-32-1. The two hidden layers use rectified linear unit activations. The output layer uses a sigmoid activation to estimate the positive-class probability.

Weights are initialized using a seeded standard normal distribution scaled by sqrt(2/fan_in), including the output layer. All biases are initialized to zero.

The reference implementation uses NumPy float64 arithmetic and a numerically stable sigmoid evaluation.

## Optimization

Training minimizes mean binary cross-entropy using manually implemented backpropagation and mini-batch stochastic gradient descent.

The learning rate is 0.01, the mini-batch size is 64, and training lasts 60 epochs. Initialization uses seed 2026. Each epoch uses a deterministic permutation generated with seed 3000 plus the zero-based epoch index.

Parameter updates follow:

    parameter_new = parameter_old - learning_rate*mean_batch_gradient

The NumPy implementation computes gradients directly. The CUDA implementation uses PyTorch automatic differentiation and binary cross-entropy with logits.

## Computational equivalence

The NumPy and CUDA training implementations use identical initialization, data, mini-batch order, learning rate, and float64 precision.

The implementations were compared through forward predictions, gradients, SGD updates, individual epochs, and complete 60-epoch training runs. The recorded 61-checkpoint learning curves agreed within a maximum absolute metric difference of 2.2204460492503131e-16.

Both experiment harnesses use the NumPy forward pass and shared evaluation utilities to calculate recorded training and validation metrics. Therefore, the learning-curve comparison is not an independent verification of the evaluation-metric implementation.

## Performance experiments

Exploratory benchmarks measured repeated CPU and CUDA training epochs after warmups. Experiments varied dataset size, CUDA mini-batch size, and CUDA floating-point precision.

The reference configuration uses float64 precision and batch size 64. CUDA timing includes tensor setup and output transfer unless otherwise stated.

Reported measurements characterize these implementations on the tested hardware; they do not establish general CPU-versus-GPU performance superiority.

## Remaining methodology verification

- Audit exact accuracy and ROC-AUC implementations.
- Verify dependency versions and hardware configuration.
- Document model-selection decisions and validation reuse.
- Define and approve the final held-out test evaluation protocol.
- Complete a reproducibility review before publication.