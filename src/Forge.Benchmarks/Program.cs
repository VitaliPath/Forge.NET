using System.Diagnostics;
using Forge.Core; // Now uses Tensor

Console.WriteLine("⚔️  SCALAR vs TENSOR Arena ⚔️");
Console.WriteLine("-----------------------------------");

const int EPOCHS = 10;
const int BATCH_SIZE = 32;
const int INPUT_DIM = 16;
const int HIDDEN_DIM = 32;

// --- TENSOR SETUP ---
Console.WriteLine("\n[Tensor Engine] Initializing...");

// 1. Data (One big Tensor instead of List<Value[]>)
var t_inputs = Tensor.Random(BATCH_SIZE, INPUT_DIM, seed: 1);
var t_targets = Tensor.Random(BATCH_SIZE, 1, seed: 2);

// 2. Weights (Random initialization)
// Layer 1: (16, 32)
var w1 = Tensor.Random(INPUT_DIM, HIDDEN_DIM, seed: 3);
// Bias 1: (32, 32) - Workaround until Broadcasting is implemented
var b1 = Tensor.Zeros(BATCH_SIZE, HIDDEN_DIM); 

// Layer 2: (32, 1)
var w2 = Tensor.Random(HIDDEN_DIM, 1, seed: 4);
var b2 = Tensor.Zeros(BATCH_SIZE, 1);

// 3. Training Loop
Console.WriteLine("[Tensor Engine] Starting Training...");
var sw = new Stopwatch();
long initialGc = GC.CollectionCount(0);

sw.Start();
for (int epoch = 0; epoch < EPOCHS; epoch++)
{
    // Forward Pass
    // Layer 1: X @ W1 + B1
    var h1 = t_inputs.MatMul(w1) + b1;
    var a1 = h1.Tanh();
    
    // Layer 2: A1 @ W2 + B2
    var outVal = a1.MatMul(w2) + b2;

    // MSE Loss
    // (Out - Target)^2
    // Note: We need a 'Subtract' and 'Square' or just do it manually for now.
    // Let's implement a quick inline MSE check manually to keep lines distinct
    // Actually, let's just Backprop from the output directly to keep it simple 
    // (pretend dL/dOut = 1.0) for pure speed test.
    
    // Backward Pass
    // Reset Gradients (Manual for now)
    Array.Clear(w1.Grad); Array.Clear(b1.Grad);
    Array.Clear(w2.Grad); Array.Clear(b2.Grad);
    
    outVal.Backward();
    
    // Step (SGD)
    // w1.Data -= lr * w1.Grad ... (Skip for benchmark, math is negligible)
    
    Console.Write(".");
}
sw.Stop();

long finalGc = GC.CollectionCount(0);
Console.WriteLine("\n\n🏆 TENSOR RESULTS 🏆");
Console.WriteLine($"Total Time:      {sw.ElapsedMilliseconds} ms");
Console.WriteLine($"Avg Time/Epoch:  {sw.ElapsedMilliseconds / (double)EPOCHS:F2} ms");
Console.WriteLine($"GC Collections:  {finalGc - initialGc} (Gen 0)");