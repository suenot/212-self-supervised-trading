use anyhow::Result;
use self_supervised_trading::*;

/// Demonstrates the full self-supervised learning pipeline for trading:
/// 1. Fetch market data from Bybit
/// 2. Build feature windows for SSL pretraining
/// 3. Pretrain encoder using Barlow Twins
/// 4. Extract representations and train downstream classifier
#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Self-Supervised Learning for Trading ===\n");

    // -------------------------------------------------------------------------
    // Step 1: Fetch market data from Bybit
    // -------------------------------------------------------------------------
    println!("Step 1: Fetching market data from Bybit...");

    let candles = match fetch_bybit_klines("BTCUSDT", "60", 200).await {
        Ok(c) => {
            println!("  Fetched {} candles from Bybit (BTCUSDT, 1h)", c.len());
            c
        }
        Err(e) => {
            println!("  Could not fetch from Bybit: {}. Using synthetic data.", e);
            generate_synthetic_candles(200, 42000.0)
        }
    };

    if candles.len() < 20 {
        println!("  Not enough candles. Using synthetic data.");
        let candles = generate_synthetic_candles(200, 42000.0);
        run_pipeline(&candles)?;
    } else {
        run_pipeline(&candles)?;
    }

    // -------------------------------------------------------------------------
    // Step 5: Multi-symbol pretraining demonstration
    // -------------------------------------------------------------------------
    println!("\n--- Step 5: Multi-Symbol SSL Pretraining ---");

    let symbols = vec!["BTCUSDT", "ETHUSDT", "SOLUSDT"];
    let mut all_windows = Vec::new();

    for symbol in &symbols {
        let sym_candles = match fetch_bybit_klines(symbol, "60", 100).await {
            Ok(c) => {
                println!("  Fetched {} candles for {}", c.len(), symbol);
                c
            }
            Err(_) => {
                println!("  Using synthetic data for {}", symbol);
                generate_synthetic_candles(100, 1000.0)
            }
        };
        let windows = build_feature_windows(&sym_candles, 10);
        all_windows.extend(windows);
    }

    println!(
        "  Total windows across {} symbols: {}",
        symbols.len(),
        all_windows.len()
    );

    if !all_windows.is_empty() {
        let mut multi_trainer = SelfSupervisedTrainer::new(15, 64, 16, 0.005, 0.001);
        let sample_size = all_windows.len().min(30);
        let losses = multi_trainer.pretrain(&all_windows[..sample_size], 5);
        println!(
            "  Multi-symbol pretraining: initial loss = {:.4}, final loss = {:.4}",
            losses.first().unwrap_or(&0.0),
            losses.last().unwrap_or(&0.0)
        );
    }

    println!("\n=== Self-Supervised Learning Pipeline Complete ===");
    Ok(())
}

/// Run the full SSL pipeline on a set of candles.
fn run_pipeline(candles: &[Candle]) -> Result<()> {
    // -------------------------------------------------------------------------
    // Step 2: Build feature windows for SSL pretraining
    // -------------------------------------------------------------------------
    println!("\nStep 2: Building feature windows...");

    let window_size = 10;
    let windows = build_feature_windows(candles, window_size);
    println!(
        "  Created {} windows of size {} x 5 features",
        windows.len(),
        window_size
    );

    if windows.is_empty() {
        println!("  No windows created. Need more data.");
        return Ok(());
    }

    // Show sample features
    println!("  Sample feature window (first 3 rows):");
    let sample = &windows[0];
    for i in 0..3.min(sample.nrows()) {
        println!(
            "    t={}: return={:.6}, range={:.6}, vol_ratio={:.4}, upper_shadow={:.4}, lower_shadow={:.4}",
            i,
            sample[[i, 0]], sample[[i, 1]], sample[[i, 2]], sample[[i, 3]], sample[[i, 4]]
        );
    }

    // -------------------------------------------------------------------------
    // Step 3: Pretrain encoder using Barlow Twins
    // -------------------------------------------------------------------------
    println!("\nStep 3: SSL Pretraining with Barlow Twins...");

    // input_dim = 5 features * 3 aggregations (mean, std, last) = 15
    let mut trainer = SelfSupervisedTrainer::new(15, 64, 16, 0.005, 0.001);

    let batch_size = windows.len().min(32);
    let epochs = 10;
    let losses = trainer.pretrain(&windows[..batch_size], epochs);

    println!("  Pretraining losses over {} epochs:", epochs);
    for (i, loss) in losses.iter().enumerate() {
        let bar_len = (loss / losses[0] * 30.0).min(30.0) as usize;
        let bar: String = "#".repeat(bar_len);
        println!("    Epoch {:2}: loss = {:.4} |{}|", i + 1, loss, bar);
    }

    // -------------------------------------------------------------------------
    // Step 4: Downstream classification (return direction prediction)
    // -------------------------------------------------------------------------
    println!("\nStep 4: Downstream Fine-tuning...");

    // Extract representations using pretrained encoder
    let representations = trainer.extract_representations(&windows);
    println!(
        "  Extracted representations: {} samples x {} dims",
        representations.nrows(),
        representations.ncols()
    );

    // Create labels: 1 if next-window return is positive, 0 otherwise
    let mut labels = Vec::new();
    for i in 0..windows.len().saturating_sub(1) {
        let current_return: f64 = windows[i].column(0).mean().unwrap_or(0.0);
        let next_return: f64 = windows[(i + 1).min(windows.len() - 1)]
            .column(0)
            .mean()
            .unwrap_or(0.0);
        labels.push(if next_return > current_return { 1 } else { 0 });
    }
    // Add label for last window
    labels.push(0);

    let n_train = (labels.len() as f64 * 0.7) as usize;
    let train_repr = representations.slice(ndarray::s![..n_train, ..]).to_owned();
    let test_repr = representations.slice(ndarray::s![n_train.., ..]).to_owned();
    let train_labels = &labels[..n_train];
    let test_labels = &labels[n_train..];

    // Train downstream classifier
    let mut classifier = DownstreamClassifier::new(16, 0.05);
    let clf_losses = classifier.train(&train_repr, train_labels, 100);

    let train_acc = classifier.accuracy(&train_repr, train_labels);
    let test_acc = classifier.accuracy(&test_repr, test_labels);

    println!(
        "  Classifier training: initial loss = {:.4}, final loss = {:.4}",
        clf_losses.first().unwrap_or(&0.0),
        clf_losses.last().unwrap_or(&0.0)
    );
    println!(
        "  Train accuracy: {:.2}% ({} samples)",
        train_acc * 100.0,
        n_train
    );
    println!(
        "  Test accuracy:  {:.2}% ({} samples)",
        test_acc * 100.0,
        test_labels.len()
    );

    // Show sample predictions
    println!("\n  Sample predictions (test set, first 10):");
    for i in 0..10.min(test_repr.nrows()) {
        let repr = test_repr.row(i).to_owned();
        let prob = classifier.predict_proba(&repr);
        let pred = classifier.predict(&repr);
        let actual = test_labels[i];
        let mark = if pred == actual { "OK" } else { "MISS" };
        println!(
            "    Sample {:2}: prob={:.3}, pred={}, actual={} [{}]",
            i + 1,
            prob,
            pred,
            actual,
            mark
        );
    }

    Ok(())
}
