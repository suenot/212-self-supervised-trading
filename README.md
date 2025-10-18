# Chapter 281: Self-Supervised Learning for Trading

## Introduction

Self-supervised learning (SSL) represents a paradigm shift in machine learning where models learn rich representations from unlabeled data by solving pretext tasks derived from the data itself. In financial markets, labeled data is expensive, noisy, and often scarce relative to the vast amount of raw market data available. Self-supervised learning addresses this fundamental limitation by extracting meaningful features from the abundant unlabeled price, volume, and order flow data, then fine-tuning on the smaller labeled datasets for downstream tasks such as return prediction, regime detection, and risk estimation.

The core insight behind SSL is that by forcing a model to predict or reconstruct parts of its own input, it must learn the underlying statistical structure of the data. In trading, this means learning temporal dependencies, cross-asset correlations, volatility clustering, and microstructure patterns without explicit supervision. These learned representations can then be transferred to tasks where labels are limited, offering substantial improvements over training from scratch.

## Mathematical Foundations

### Contrastive Learning Framework

Contrastive learning is one of the most successful SSL paradigms. Given an anchor sample $x$, we construct a positive pair $x^+$ (an augmented version of $x$) and negative pairs $\{x_i^-\}_{i=1}^{K}$ (other samples). The InfoNCE loss is:

$$\mathcal{L}_{\text{InfoNCE}} = -\log \frac{\exp(\text{sim}(z, z^+) / \tau)}{\exp(\text{sim}(z, z^+) / \tau) + \sum_{i=1}^{K} \exp(\text{sim}(z, z_i^-) / \tau)}$$

where $z = f_\theta(x)$ is the encoded representation, $\text{sim}(u, v) = \frac{u^\top v}{\|u\| \|v\|}$ is cosine similarity, and $\tau$ is a temperature hyperparameter that controls the sharpness of the distribution.

### Data Augmentation for Financial Time Series

Unlike images, financial time series require domain-specific augmentations that preserve market semantics:

1. **Temporal Cropping**: Extract overlapping subsequences from a window:
$$x_{\text{crop}} = x[t_0 : t_0 + L], \quad t_0 \sim \text{Uniform}(0, T - L)$$

2. **Gaussian Noise Injection**: Add calibrated noise to simulate market microstructure noise:
$$x_{\text{noisy}} = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0, \sigma^2 I)$$
where $\sigma$ is proportional to the asset's typical bid-ask spread.

3. **Temporal Jittering**: Small random perturbations to timestamps to handle asynchronous data:
$$x_{\text{jitter}}(t) = x(t + \delta_t), \quad \delta_t \sim \mathcal{N}(0, \sigma_t^2)$$

4. **Feature Masking**: Randomly zero out subsets of features to force learning of cross-feature dependencies:
$$x_{\text{masked}} = x \odot m, \quad m_i \sim \text{Bernoulli}(1 - p)$$

5. **Magnitude Scaling**: Scale the amplitude while preserving temporal structure:
$$x_{\text{scaled}} = \alpha \cdot x, \quad \alpha \sim \text{Uniform}(0.8, 1.2)$$

### Predictive Self-Supervised Objectives

Beyond contrastive methods, predictive SSL tasks learn by forecasting masked or future portions of the input:

**Masked Feature Prediction**: Given a multivariate time series $X \in \mathbb{R}^{T \times D}$, mask a fraction $p$ of the entries and train the model to reconstruct them:

$$\mathcal{L}_{\text{mask}} = \frac{1}{|\mathcal{M}|} \sum_{(t,d) \in \mathcal{M}} \left( X_{t,d} - \hat{X}_{t,d} \right)^2$$

where $\mathcal{M}$ is the set of masked positions.

**Next-Step Prediction**: Predict the next observation given a context window:

$$\mathcal{L}_{\text{next}} = \sum_{t=C}^{T-1} \| x_{t+1} - g_\phi(f_\theta(x_{t-C+1:t})) \|^2$$

### Barlow Twins Objective

An alternative to contrastive learning that avoids the need for negative samples:

$$\mathcal{L}_{\text{BT}} = \sum_i (1 - \mathcal{C}_{ii})^2 + \lambda \sum_i \sum_{j \neq i} \mathcal{C}_{ij}^2$$

where $\mathcal{C}$ is the cross-correlation matrix between the two augmented batch embeddings:

$$\mathcal{C}_{ij} = \frac{\sum_b z_{b,i}^A z_{b,j}^B}{\sqrt{\sum_b (z_{b,i}^A)^2} \sqrt{\sum_b (z_{b,j}^B)^2}}$$

The first term encourages the diagonal of $\mathcal{C}$ to be ones (invariance), while the second term pushes off-diagonal elements to zero (redundancy reduction).

### VICReg: Variance-Invariance-Covariance Regularization

VICReg combines three terms without requiring negative pairs or batch normalization tricks:

$$\mathcal{L}_{\text{VICReg}} = \lambda \cdot s(Z^A, Z^B) + \mu \cdot [v(Z^A) + v(Z^B)] + \nu \cdot [c(Z^A) + c(Z^B)]$$

where:
- $s(Z^A, Z^B) = \frac{1}{N}\sum_i \| z_i^A - z_i^B \|^2$ is the invariance term
- $v(Z) = \frac{1}{D}\sum_j \max(0, \gamma - \sqrt{\text{Var}(z_j) + \epsilon})$ is the variance term (prevents collapse)
- $c(Z) = \frac{1}{D}\sum_{i \neq j} [\text{Cov}(Z)]_{ij}^2$ is the covariance term (decorrelation)

## Application to Trading

### Pretext Tasks for Market Data

Self-supervised learning in trading uses pretext tasks aligned with market structure:

1. **Cross-Asset Prediction**: Mask one asset's features and predict from correlated assets, learning correlation structure.

2. **Temporal Context Prediction**: Given a price segment, predict whether it comes before or after another segment, learning temporal ordering.

3. **Volatility Regime Prediction**: Predict the volatility regime of a masked window, forcing the model to learn regime-dependent dynamics.

4. **Order Flow Reconstruction**: Mask portions of the order book and reconstruct them, learning microstructure patterns.

### Transfer Learning Pipeline

The typical SSL pipeline for trading:

1. **Pre-training Phase**: Train the encoder $f_\theta$ on a large unlabeled dataset using the SSL objective. This captures general market dynamics.

2. **Fine-tuning Phase**: Freeze or partially unfreeze $f_\theta$ and train a small prediction head $g_\phi$ on the labeled downstream task.

3. **Evaluation**: Test on held-out data, comparing SSL-pretrained models against randomly initialized baselines.

### Advantages in Low-Label Regimes

In trading, labeled data is inherently limited:
- Future returns are only known after the fact
- Regime labels require expert annotation
- Event labels (crashes, squeezes) are rare
- Strategy labels are proprietary

SSL addresses this by learning from the 99%+ of data that is unlabeled, then transferring to the 1% that has labels.

## Rust Implementation Walkthrough

Our Rust implementation provides the core building blocks for self-supervised learning on financial time series:

1. **`TimeSeriesAugmentor`**: Applies domain-specific augmentations (noise injection, temporal cropping, feature masking, magnitude scaling) to create positive pairs for contrastive learning.

2. **`ContrastiveEncoder`**: A simple feed-forward encoder that maps raw features to a lower-dimensional representation space. In production, this would be replaced by a Transformer or TCN.

3. **`BarlowTwinsLoss`**: Computes the Barlow Twins objective from two sets of embeddings, encouraging invariance while reducing redundancy.

4. **`SelfSupervisedTrainer`**: Orchestrates the pretraining loop, applying augmentations, computing encodings, and updating weights via gradient-free optimization (for simplicity; production would use autograd).

5. **`DownstreamClassifier`**: A simple linear classifier that takes frozen SSL representations and trains on labeled data for downstream tasks (e.g., return direction prediction).

The implementation demonstrates key concepts:
- How augmentations create meaningful positive pairs for financial data
- Cross-correlation computation for the Barlow Twins loss
- The pretrain-then-finetune pipeline
- Integration with Bybit API for live market data

## Bybit Integration

The Bybit integration fetches real-time and historical market data to serve as the unlabeled pretraining corpus. We use the public REST API to:

1. Fetch OHLCV kline data across multiple timeframes
2. Construct multivariate feature vectors (returns, volume ratios, volatility estimates)
3. Apply SSL pretraining on the raw features
4. Fine-tune for direction prediction

The key insight is that Bybit provides abundant free market data across hundreds of trading pairs, making it an ideal source for self-supervised pretraining. By pretraining on many pairs simultaneously, the model learns universal market patterns that transfer to individual pair prediction.

## Key Takeaways

1. **Self-supervised learning extracts value from unlabeled data**: In trading, the vast majority of market data has no explicit labels. SSL methods like contrastive learning and Barlow Twins learn useful representations without supervision.

2. **Domain-specific augmentations are critical**: Unlike image augmentations (rotation, color jitter), financial augmentations must preserve market semantics. Noise injection, temporal cropping, feature masking, and magnitude scaling are appropriate for time series.

3. **The Barlow Twins objective avoids collapse without negative samples**: By optimizing for invariance (diagonal of cross-correlation near 1) and redundancy reduction (off-diagonal near 0), it provides a stable training signal.

4. **Pre-trained representations improve low-label tasks**: When labeled data is scarce (regime detection, event prediction, alpha signal discovery), SSL-pretrained encoders significantly outperform random initialization.

5. **Multi-asset pretraining captures universal patterns**: Training on multiple instruments simultaneously forces the encoder to learn general market dynamics (volatility clustering, mean reversion, momentum) rather than asset-specific quirks.

6. **The pretrain-finetune paradigm works for trading**: Just as in NLP and vision, the two-phase approach (unsupervised pretraining, supervised fine-tuning) is effective for financial ML, especially when labeled data is limited.

7. **Augmentation diversity prevents shortcut learning**: Using multiple augmentation strategies prevents the model from learning trivial shortcuts and forces it to capture genuinely useful features of the data.

## References

- Chen, T. et al. (2020). "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR)
- Zbontar, J. et al. (2021). "Barlow Twins: Self-Supervised Learning via Redundancy Reduction"
- Bardes, A. et al. (2022). "VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning"
- Yue, Z. et al. (2022). "TS2Vec: Towards Universal Representation of Time Series"
- Eldele, E. et al. (2021). "Time-Series Representation Learning via Temporal and Contextual Contrasting"
- Oord, A. et al. (2018). "Representation Learning with Contrastive Predictive Coding"
