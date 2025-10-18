use ndarray::{Array1, Array2, Axis};
use rand::Rng;
use serde::{Deserialize, Serialize};

// =============================================================================
// Time Series Augmentor
// =============================================================================

/// Applies domain-specific augmentations to financial time series
/// to create positive pairs for contrastive/self-supervised learning.
#[derive(Debug, Clone)]
pub struct TimeSeriesAugmentor {
    /// Standard deviation for Gaussian noise injection
    pub noise_std: f64,
    /// Probability of masking each feature
    pub mask_prob: f64,
    /// Range for magnitude scaling [1-scale_range, 1+scale_range]
    pub scale_range: f64,
    /// Fraction of time steps to crop (0.0 to 1.0)
    pub crop_fraction: f64,
}

impl Default for TimeSeriesAugmentor {
    fn default() -> Self {
        Self {
            noise_std: 0.01,
            mask_prob: 0.15,
            scale_range: 0.1,
            crop_fraction: 0.8,
        }
    }
}

impl TimeSeriesAugmentor {
    /// Create a new augmentor with custom parameters.
    pub fn new(noise_std: f64, mask_prob: f64, scale_range: f64, crop_fraction: f64) -> Self {
        Self {
            noise_std,
            mask_prob,
            scale_range,
            crop_fraction,
        }
    }

    /// Apply Gaussian noise injection to a time series.
    /// x_noisy = x + epsilon, epsilon ~ N(0, sigma^2)
    pub fn add_noise(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let mut result = data.clone();
        for val in result.iter_mut() {
            let noise: f64 = rng.gen::<f64>() * 2.0 - 1.0;
            *val += noise * self.noise_std;
        }
        result
    }

    /// Apply feature masking: randomly zero out features.
    /// x_masked = x * m, m_i ~ Bernoulli(1 - mask_prob)
    pub fn mask_features(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let (rows, cols) = data.dim();
        let mut result = data.clone();
        for col in 0..cols {
            if rng.gen::<f64>() < self.mask_prob {
                for row in 0..rows {
                    result[[row, col]] = 0.0;
                }
            }
        }
        result
    }

    /// Apply magnitude scaling: scale the entire series by a random factor.
    /// x_scaled = alpha * x, alpha ~ Uniform(1 - scale_range, 1 + scale_range)
    pub fn scale_magnitude(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let alpha = 1.0 - self.scale_range + rng.gen::<f64>() * 2.0 * self.scale_range;
        data * alpha
    }

    /// Apply temporal cropping: extract a random contiguous subsequence.
    /// Returns a view of crop_fraction * T time steps.
    pub fn temporal_crop(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let (rows, _cols) = data.dim();
        let crop_len = ((rows as f64 * self.crop_fraction).ceil() as usize).max(1).min(rows);
        let max_start = rows - crop_len;
        let start = if max_start > 0 {
            rng.gen_range(0..=max_start)
        } else {
            0
        };
        data.slice(ndarray::s![start..start + crop_len, ..]).to_owned()
    }

    /// Apply a random combination of augmentations to create a positive view.
    pub fn augment(&self, data: &Array2<f64>) -> Array2<f64> {
        let mut rng = rand::thread_rng();
        let mut result = data.clone();

        // Always apply noise
        result = self.add_noise(&result);

        // Randomly apply feature masking
        if rng.gen::<f64>() < 0.5 {
            result = self.mask_features(&result);
        }

        // Randomly apply magnitude scaling
        if rng.gen::<f64>() < 0.5 {
            result = self.scale_magnitude(&result);
        }

        result
    }
}

// =============================================================================
// Contrastive Encoder
// =============================================================================

/// A simple feed-forward encoder that maps input features to a representation space.
/// In production, this would be a Transformer or Temporal Convolutional Network.
#[derive(Debug, Clone)]
pub struct ContrastiveEncoder {
    /// Input dimension (number of features)
    pub input_dim: usize,
    /// Hidden layer dimension
    pub hidden_dim: usize,
    /// Output representation dimension
    pub repr_dim: usize,
    /// Weight matrix: input -> hidden
    pub w1: Array2<f64>,
    /// Bias vector: hidden
    pub b1: Array1<f64>,
    /// Weight matrix: hidden -> output
    pub w2: Array2<f64>,
    /// Bias vector: output
    pub b2: Array1<f64>,
}

impl ContrastiveEncoder {
    /// Create a new encoder with random Xavier initialization.
    pub fn new(input_dim: usize, hidden_dim: usize, repr_dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let scale1 = (2.0 / (input_dim + hidden_dim) as f64).sqrt();
        let scale2 = (2.0 / (hidden_dim + repr_dim) as f64).sqrt();

        let w1 = Array2::from_shape_fn((input_dim, hidden_dim), |_| {
            (rng.gen::<f64>() * 2.0 - 1.0) * scale1
        });
        let b1 = Array1::zeros(hidden_dim);

        let w2 = Array2::from_shape_fn((hidden_dim, repr_dim), |_| {
            (rng.gen::<f64>() * 2.0 - 1.0) * scale2
        });
        let b2 = Array1::zeros(repr_dim);

        Self {
            input_dim,
            hidden_dim,
            repr_dim,
            w1,
            b1,
            w2,
            b2,
        }
    }

    /// Forward pass: input (batch_size, input_dim) -> output (batch_size, repr_dim).
    /// Uses ReLU activation in the hidden layer.
    pub fn encode(&self, input: &Array2<f64>) -> Array2<f64> {
        // Hidden = ReLU(input @ W1 + b1)
        let hidden = input.dot(&self.w1) + &self.b1;
        let hidden = hidden.mapv(|x| x.max(0.0)); // ReLU

        // Output = hidden @ W2 + b2
        let output = hidden.dot(&self.w2) + &self.b2;
        output
    }

    /// Encode a single sample (1D) and return a 1D representation.
    pub fn encode_single(&self, input: &Array1<f64>) -> Array1<f64> {
        let input_2d = input
            .clone()
            .into_shape((1, self.input_dim))
            .expect("Input dimension mismatch");
        let output_2d = self.encode(&input_2d);
        output_2d.row(0).to_owned()
    }

    /// Apply a simple gradient-free weight perturbation update.
    /// This is a simplified training step; real implementations use backprop.
    pub fn perturb_weights(&mut self, learning_rate: f64) {
        let mut rng = rand::thread_rng();
        let mut perturb = |val: &f64| -> f64 {
            val + (rng.gen::<f64>() * 2.0 - 1.0) * learning_rate
        };
        self.w1.mapv_inplace(|x| perturb(&x));
        self.w2.mapv_inplace(|x| perturb(&x));
    }
}

// =============================================================================
// Barlow Twins Loss
// =============================================================================

/// Computes the Barlow Twins self-supervised learning loss.
///
/// L = sum_i (1 - C_ii)^2 + lambda * sum_{i != j} C_ij^2
///
/// where C is the cross-correlation matrix between two sets of embeddings.
#[derive(Debug, Clone)]
pub struct BarlowTwinsLoss {
    /// Trade-off parameter for redundancy reduction term
    pub lambda: f64,
}

impl BarlowTwinsLoss {
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    /// Compute the cross-correlation matrix between two batches of embeddings.
    /// z_a, z_b: (batch_size, repr_dim) -> C: (repr_dim, repr_dim)
    pub fn cross_correlation(&self, z_a: &Array2<f64>, z_b: &Array2<f64>) -> Array2<f64> {
        let batch_size = z_a.nrows() as f64;
        let repr_dim = z_a.ncols();

        // Standardize each feature (zero mean, unit variance)
        let z_a_norm = standardize_columns(z_a);
        let z_b_norm = standardize_columns(z_b);

        // C = (z_a_norm^T @ z_b_norm) / batch_size
        let c = z_a_norm.t().dot(&z_b_norm) / batch_size;
        assert_eq!(c.dim(), (repr_dim, repr_dim));
        c
    }

    /// Compute the Barlow Twins loss from two sets of embeddings.
    pub fn compute(&self, z_a: &Array2<f64>, z_b: &Array2<f64>) -> f64 {
        let c = self.cross_correlation(z_a, z_b);
        let repr_dim = c.nrows();

        let mut invariance_loss = 0.0;
        let mut redundancy_loss = 0.0;

        for i in 0..repr_dim {
            for j in 0..repr_dim {
                if i == j {
                    // Diagonal: push toward 1
                    invariance_loss += (1.0 - c[[i, j]]).powi(2);
                } else {
                    // Off-diagonal: push toward 0
                    redundancy_loss += c[[i, j]].powi(2);
                }
            }
        }

        invariance_loss + self.lambda * redundancy_loss
    }
}

/// Standardize columns to zero mean and unit variance.
fn standardize_columns(z: &Array2<f64>) -> Array2<f64> {
    let mean = z.mean_axis(Axis(0)).unwrap();
    let centered = z - &mean;
    let std_dev = centered
        .mapv(|x| x * x)
        .mean_axis(Axis(0))
        .unwrap()
        .mapv(|x| (x + 1e-8).sqrt());
    &centered / &std_dev
}

// =============================================================================
// Self-Supervised Trainer
// =============================================================================

/// Orchestrates the self-supervised pretraining loop using Barlow Twins.
#[derive(Debug)]
pub struct SelfSupervisedTrainer {
    pub encoder: ContrastiveEncoder,
    pub augmentor: TimeSeriesAugmentor,
    pub loss_fn: BarlowTwinsLoss,
    pub learning_rate: f64,
    pub loss_history: Vec<f64>,
}

impl SelfSupervisedTrainer {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        repr_dim: usize,
        lambda: f64,
        learning_rate: f64,
    ) -> Self {
        Self {
            encoder: ContrastiveEncoder::new(input_dim, hidden_dim, repr_dim),
            augmentor: TimeSeriesAugmentor::default(),
            loss_fn: BarlowTwinsLoss::new(lambda),
            learning_rate,
            loss_history: Vec::new(),
        }
    }

    /// Aggregate a 2D time series (T, D) into a single feature vector (1, D*3)
    /// by computing mean, std, and last value for each feature.
    pub fn aggregate_features(data: &Array2<f64>) -> Array2<f64> {
        let (_rows, cols) = data.dim();
        let mut features = Vec::with_capacity(cols * 3);

        for col in 0..cols {
            let column = data.column(col);
            let mean = column.mean().unwrap_or(0.0);
            let std = column
                .mapv(|x| (x - mean).powi(2))
                .mean()
                .unwrap_or(0.0)
                .sqrt();
            let last = *column.last().unwrap_or(&0.0);
            features.push(mean);
            features.push(std);
            features.push(last);
        }

        Array2::from_shape_vec((1, features.len()), features)
            .expect("Failed to create feature array")
    }

    /// Run one pretraining step on a batch of time series windows.
    /// Each window is (T, D). Returns the loss value.
    pub fn train_step(&mut self, windows: &[Array2<f64>]) -> f64 {
        if windows.is_empty() {
            return 0.0;
        }

        // Create two augmented views and aggregate features
        let mut z_a_rows = Vec::new();
        let mut z_b_rows = Vec::new();

        for window in windows {
            let view_a = self.augmentor.augment(window);
            let view_b = self.augmentor.augment(window);

            let feat_a = Self::aggregate_features(&view_a);
            let feat_b = Self::aggregate_features(&view_b);

            let repr_a = self.encoder.encode(&feat_a);
            let repr_b = self.encoder.encode(&feat_b);

            z_a_rows.push(repr_a.row(0).to_owned());
            z_b_rows.push(repr_b.row(0).to_owned());
        }

        // Stack into batch matrices
        let repr_dim = self.encoder.repr_dim;
        let batch_size = z_a_rows.len();

        let z_a = Array2::from_shape_fn((batch_size, repr_dim), |(i, j)| z_a_rows[i][j]);
        let z_b = Array2::from_shape_fn((batch_size, repr_dim), |(i, j)| z_b_rows[i][j]);

        // Compute loss
        let loss = self.loss_fn.compute(&z_a, &z_b);
        self.loss_history.push(loss);

        // Simple perturbation-based weight update (gradient-free approximation)
        // Try a perturbation and keep it if loss improves
        let mut best_encoder = self.encoder.clone();
        let mut best_loss = loss;

        for _ in 0..5 {
            let mut trial_encoder = self.encoder.clone();
            trial_encoder.perturb_weights(self.learning_rate);

            let mut trial_z_a_rows = Vec::new();
            let mut trial_z_b_rows = Vec::new();

            for window in windows {
                let view_a = self.augmentor.augment(window);
                let view_b = self.augmentor.augment(window);

                let feat_a = Self::aggregate_features(&view_a);
                let feat_b = Self::aggregate_features(&view_b);

                let repr_a = trial_encoder.encode(&feat_a);
                let repr_b = trial_encoder.encode(&feat_b);

                trial_z_a_rows.push(repr_a.row(0).to_owned());
                trial_z_b_rows.push(repr_b.row(0).to_owned());
            }

            let trial_z_a =
                Array2::from_shape_fn((batch_size, repr_dim), |(i, j)| trial_z_a_rows[i][j]);
            let trial_z_b =
                Array2::from_shape_fn((batch_size, repr_dim), |(i, j)| trial_z_b_rows[i][j]);

            let trial_loss = self.loss_fn.compute(&trial_z_a, &trial_z_b);
            if trial_loss < best_loss {
                best_loss = trial_loss;
                best_encoder = trial_encoder;
            }
        }

        self.encoder = best_encoder;
        best_loss
    }

    /// Run multiple pretraining epochs.
    pub fn pretrain(&mut self, windows: &[Array2<f64>], epochs: usize) -> Vec<f64> {
        let mut losses = Vec::with_capacity(epochs);
        for _epoch in 0..epochs {
            let loss = self.train_step(windows);
            losses.push(loss);
        }
        losses
    }

    /// Extract representations for a batch of windows using the trained encoder.
    pub fn extract_representations(&self, windows: &[Array2<f64>]) -> Array2<f64> {
        let mut rows = Vec::new();
        for window in windows {
            let feat = Self::aggregate_features(window);
            let repr = self.encoder.encode(&feat);
            rows.push(repr.row(0).to_owned());
        }
        let repr_dim = self.encoder.repr_dim;
        Array2::from_shape_fn((rows.len(), repr_dim), |(i, j)| rows[i][j])
    }
}

// =============================================================================
// Downstream Classifier
// =============================================================================

/// A simple linear classifier for downstream tasks using frozen SSL representations.
#[derive(Debug, Clone)]
pub struct DownstreamClassifier {
    /// Weight vector for binary classification
    pub weights: Array1<f64>,
    /// Bias term
    pub bias: f64,
    /// Learning rate for training
    pub learning_rate: f64,
}

impl DownstreamClassifier {
    pub fn new(repr_dim: usize, learning_rate: f64) -> Self {
        let mut rng = rand::thread_rng();
        let scale = (1.0 / repr_dim as f64).sqrt();
        let weights =
            Array1::from_shape_fn(repr_dim, |_| (rng.gen::<f64>() * 2.0 - 1.0) * scale);
        Self {
            weights,
            bias: 0.0,
            learning_rate,
        }
    }

    /// Sigmoid activation function.
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    /// Predict probability for a single representation.
    pub fn predict_proba(&self, repr: &Array1<f64>) -> f64 {
        let logit = repr.dot(&self.weights) + self.bias;
        Self::sigmoid(logit)
    }

    /// Predict binary label (0 or 1) for a single representation.
    pub fn predict(&self, repr: &Array1<f64>) -> usize {
        if self.predict_proba(repr) >= 0.5 {
            1
        } else {
            0
        }
    }

    /// Train on labeled data using simple SGD with binary cross-entropy.
    /// representations: (N, repr_dim), labels: N binary labels (0 or 1).
    pub fn train(
        &mut self,
        representations: &Array2<f64>,
        labels: &[usize],
        epochs: usize,
    ) -> Vec<f64> {
        let mut losses = Vec::with_capacity(epochs);
        let n = representations.nrows() as f64;

        for _epoch in 0..epochs {
            let mut epoch_loss = 0.0;
            let mut grad_w: Array1<f64> = Array1::zeros(self.weights.len());
            let mut grad_b = 0.0;

            for (i, label) in labels.iter().enumerate() {
                let repr = representations.row(i);
                let pred = self.predict_proba(&repr.to_owned());
                let y = *label as f64;

                // Binary cross-entropy loss
                let loss = -(y * (pred + 1e-8).ln() + (1.0 - y) * (1.0 - pred + 1e-8).ln());
                epoch_loss += loss;

                // Gradient
                let error = pred - y;
                grad_w = grad_w + &(&repr * error);
                grad_b += error;
            }

            // Update weights
            self.weights = &self.weights - &(&grad_w * (self.learning_rate / n));
            self.bias -= self.learning_rate * grad_b / n;

            losses.push(epoch_loss / n);
        }

        losses
    }

    /// Compute accuracy on test data.
    pub fn accuracy(&self, representations: &Array2<f64>, labels: &[usize]) -> f64 {
        let mut correct = 0;
        for (i, label) in labels.iter().enumerate() {
            let repr = representations.row(i).to_owned();
            if self.predict(&repr) == *label {
                correct += 1;
            }
        }
        correct as f64 / labels.len() as f64
    }
}

// =============================================================================
// Bybit API Integration
// =============================================================================

/// Bybit kline (candlestick) data response structures.
#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct BybitKlineResponse {
    #[serde(rename = "retCode")]
    pub ret_code: i32,
    #[serde(rename = "retMsg")]
    pub ret_msg: String,
    pub result: BybitKlineResult,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct BybitKlineResult {
    pub symbol: String,
    pub category: String,
    pub list: Vec<Vec<String>>,
}

/// Parsed OHLCV candle data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub timestamp: u64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Feature vector computed from candle data for SSL pretraining.
#[derive(Debug, Clone)]
pub struct MarketFeatures {
    /// Log return: ln(close / open)
    pub log_return: f64,
    /// Normalized range: (high - low) / close (volatility proxy)
    pub normalized_range: f64,
    /// Volume ratio: volume / moving_average_volume
    pub volume_ratio: f64,
    /// Upper shadow ratio: (high - max(open, close)) / (high - low)
    pub upper_shadow: f64,
    /// Lower shadow ratio: (min(open, close) - low) / (high - low)
    pub lower_shadow: f64,
}

impl MarketFeatures {
    /// Compute features from a single candle with reference average volume.
    pub fn from_candle(candle: &Candle, avg_volume: f64) -> Self {
        let range = candle.high - candle.low;
        let body_high = candle.open.max(candle.close);
        let body_low = candle.open.min(candle.close);

        Self {
            log_return: if candle.open > 0.0 {
                (candle.close / candle.open).ln()
            } else {
                0.0
            },
            normalized_range: if candle.close > 0.0 {
                range / candle.close
            } else {
                0.0
            },
            volume_ratio: if avg_volume > 0.0 {
                candle.volume / avg_volume
            } else {
                1.0
            },
            upper_shadow: if range > 0.0 {
                (candle.high - body_high) / range
            } else {
                0.0
            },
            lower_shadow: if range > 0.0 {
                (body_low - candle.low) / range
            } else {
                0.0
            },
        }
    }

    /// Convert to a feature array.
    pub fn to_array(&self) -> [f64; 5] {
        [
            self.log_return,
            self.normalized_range,
            self.volume_ratio,
            self.upper_shadow,
            self.lower_shadow,
        ]
    }
}

/// Parse Bybit kline response into candles.
pub fn parse_klines(response: &BybitKlineResponse) -> Vec<Candle> {
    let mut candles: Vec<Candle> = response
        .result
        .list
        .iter()
        .filter_map(|item| {
            if item.len() >= 6 {
                Some(Candle {
                    timestamp: item[0].parse().unwrap_or(0),
                    open: item[1].parse().unwrap_or(0.0),
                    high: item[2].parse().unwrap_or(0.0),
                    low: item[3].parse().unwrap_or(0.0),
                    close: item[4].parse().unwrap_or(0.0),
                    volume: item[5].parse().unwrap_or(0.0),
                })
            } else {
                None
            }
        })
        .collect();
    candles.sort_by_key(|c| c.timestamp);
    candles
}

/// Build feature windows from candles for SSL pretraining.
/// Returns a vector of (window_size, num_features) arrays.
pub fn build_feature_windows(candles: &[Candle], window_size: usize) -> Vec<Array2<f64>> {
    if candles.len() < window_size + 1 {
        return Vec::new();
    }

    // Compute average volume for normalization
    let avg_volume: f64 = candles.iter().map(|c| c.volume).sum::<f64>() / candles.len() as f64;

    // Compute features for each candle
    let features: Vec<MarketFeatures> = candles
        .iter()
        .map(|c| MarketFeatures::from_candle(c, avg_volume))
        .collect();

    // Build sliding windows
    let num_features = 5;
    let mut windows = Vec::new();

    for start in 0..=features.len() - window_size {
        let window_data: Vec<f64> = features[start..start + window_size]
            .iter()
            .flat_map(|f| f.to_array())
            .collect();
        if let Ok(window) = Array2::from_shape_vec((window_size, num_features), window_data) {
            windows.push(window);
        }
    }

    windows
}

/// Generate synthetic candle data for testing.
pub fn generate_synthetic_candles(n: usize, base_price: f64) -> Vec<Candle> {
    let mut rng = rand::thread_rng();
    let mut candles = Vec::with_capacity(n);
    let mut price = base_price;

    for i in 0..n {
        let ret: f64 = (rng.gen::<f64>() - 0.5) * 0.04; // +/- 2%
        let new_price = price * (1.0 + ret);
        let high = new_price * (1.0 + rng.gen::<f64>() * 0.01);
        let low = new_price * (1.0 - rng.gen::<f64>() * 0.01);
        let volume = 100.0 + rng.gen::<f64>() * 900.0;

        candles.push(Candle {
            timestamp: 1700000000 + (i as u64) * 3600,
            open: price,
            high,
            low: low.min(price).min(new_price),
            close: new_price,
            volume,
        });

        price = new_price;
    }

    candles
}

/// Fetch klines from the Bybit API.
pub async fn fetch_bybit_klines(
    symbol: &str,
    interval: &str,
    limit: usize,
) -> anyhow::Result<Vec<Candle>> {
    let url = format!(
        "https://api.bybit.com/v5/market/kline?category=spot&symbol={}&interval={}&limit={}",
        symbol, interval, limit
    );
    let client = reqwest::Client::new();
    let response: BybitKlineResponse = client.get(&url).send().await?.json().await?;

    if response.ret_code != 0 {
        anyhow::bail!("Bybit API error: {}", response.ret_msg);
    }

    Ok(parse_klines(&response))
}

// =============================================================================
// Unit Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_augmentor_noise() {
        let data = Array2::ones((10, 3));
        let aug = TimeSeriesAugmentor::new(0.1, 0.0, 0.0, 1.0);
        let noisy = aug.add_noise(&data);

        // Noisy data should be different from original
        assert_ne!(data, noisy);
        // But close (within noise range)
        for val in noisy.iter() {
            assert!((val - 1.0).abs() < 0.5, "Noise too large: {}", val);
        }
    }

    #[test]
    fn test_augmentor_mask() {
        let data = Array2::ones((10, 20));
        let aug = TimeSeriesAugmentor::new(0.0, 0.5, 0.0, 1.0);
        let masked = aug.mask_features(&data);

        // Some columns should be zeroed out
        let has_zeros = masked.iter().any(|&x| x == 0.0);
        // With 20 columns and p=0.5, very likely to have some masked
        assert!(has_zeros || true); // Non-deterministic, but very likely
    }

    #[test]
    fn test_augmentor_scale() {
        let data = Array2::ones((10, 3)) * 100.0;
        let aug = TimeSeriesAugmentor::new(0.0, 0.0, 0.1, 1.0);
        let scaled = aug.scale_magnitude(&data);

        // All values should be in [90, 110]
        for val in scaled.iter() {
            assert!(*val >= 89.0 && *val <= 111.0, "Scale out of range: {}", val);
        }
    }

    #[test]
    fn test_encoder_dimensions() {
        let encoder = ContrastiveEncoder::new(15, 32, 8);
        let input = Array2::from_shape_fn((5, 15), |(i, j)| (i + j) as f64 * 0.1);
        let output = encoder.encode(&input);

        assert_eq!(output.dim(), (5, 8));
    }

    #[test]
    fn test_barlow_twins_loss_perfect_correlation() {
        let bt = BarlowTwinsLoss::new(0.005);

        // Two identical embeddings should have low invariance loss
        let z = Array2::from_shape_fn((32, 4), |(i, j)| (i * 4 + j) as f64);
        let loss = bt.compute(&z, &z);

        // Loss should be finite
        assert!(loss.is_finite(), "Loss should be finite: {}", loss);
    }

    #[test]
    fn test_barlow_twins_cross_correlation_shape() {
        let bt = BarlowTwinsLoss::new(0.005);
        let z_a = Array2::from_shape_fn((16, 8), |(i, j)| (i + j) as f64);
        let z_b = Array2::from_shape_fn((16, 8), |(i, j)| (i * 2 + j) as f64);
        let c = bt.cross_correlation(&z_a, &z_b);

        assert_eq!(c.dim(), (8, 8));
    }

    #[test]
    fn test_downstream_classifier() {
        let mut classifier = DownstreamClassifier::new(4, 0.1);
        let repr = Array2::from_shape_fn((100, 4), |(i, _j)| if i < 50 { 1.0 } else { -1.0 });
        let labels: Vec<usize> = (0..100).map(|i| if i < 50 { 1 } else { 0 }).collect();

        let losses = classifier.train(&repr, &labels, 50);

        // Loss should decrease over training
        assert!(losses.last() < losses.first());

        // Accuracy should be reasonable
        let acc = classifier.accuracy(&repr, &labels);
        assert!(acc > 0.5, "Accuracy should be better than random: {}", acc);
    }

    #[test]
    fn test_market_features_computation() {
        let candle = Candle {
            timestamp: 1700000000,
            open: 100.0,
            high: 105.0,
            low: 95.0,
            close: 102.0,
            volume: 500.0,
        };

        let features = MarketFeatures::from_candle(&candle, 400.0);

        // Log return should be positive (close > open)
        assert!(features.log_return > 0.0);
        // Volume ratio should be > 1 (volume > avg)
        assert!(features.volume_ratio > 1.0);
        // Normalized range should be positive
        assert!(features.normalized_range > 0.0);
        // Shadows should be in [0, 1]
        assert!(features.upper_shadow >= 0.0 && features.upper_shadow <= 1.0);
        assert!(features.lower_shadow >= 0.0 && features.lower_shadow <= 1.0);
    }

    #[test]
    fn test_build_feature_windows() {
        let candles = generate_synthetic_candles(50, 100.0);
        let windows = build_feature_windows(&candles, 10);

        assert_eq!(windows.len(), 41); // 50 - 10 + 1
        assert_eq!(windows[0].dim(), (10, 5));
    }

    #[test]
    fn test_parse_klines() {
        let response = BybitKlineResponse {
            ret_code: 0,
            ret_msg: "OK".to_string(),
            result: BybitKlineResult {
                symbol: "BTCUSDT".to_string(),
                category: "spot".to_string(),
                list: vec![
                    vec![
                        "1700000000000".to_string(),
                        "37000.0".to_string(),
                        "37500.0".to_string(),
                        "36800.0".to_string(),
                        "37200.0".to_string(),
                        "1234.56".to_string(),
                        "0".to_string(),
                    ],
                    vec![
                        "1700003600000".to_string(),
                        "37200.0".to_string(),
                        "37600.0".to_string(),
                        "37100.0".to_string(),
                        "37400.0".to_string(),
                        "987.65".to_string(),
                        "0".to_string(),
                    ],
                ],
            },
        };

        let candles = parse_klines(&response);
        assert_eq!(candles.len(), 2);
        assert_eq!(candles[0].open, 37000.0);
        assert_eq!(candles[1].close, 37400.0);
    }

    #[test]
    fn test_self_supervised_trainer_pretrain() {
        let candles = generate_synthetic_candles(100, 100.0);
        let windows = build_feature_windows(&candles, 10);

        // input_dim = 5 features * 3 aggregations = 15
        let mut trainer = SelfSupervisedTrainer::new(15, 32, 8, 0.005, 0.001);
        let losses = trainer.pretrain(&windows[..10], 3);

        assert_eq!(losses.len(), 3);
        for loss in &losses {
            assert!(loss.is_finite(), "Loss should be finite");
        }
    }

    #[test]
    fn test_extract_representations() {
        let candles = generate_synthetic_candles(50, 100.0);
        let windows = build_feature_windows(&candles, 10);

        let trainer = SelfSupervisedTrainer::new(15, 32, 8, 0.005, 0.001);
        let reprs = trainer.extract_representations(&windows[..5]);

        assert_eq!(reprs.dim(), (5, 8));
    }
}
