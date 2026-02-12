using System.Diagnostics;
using Forge.Core;
using Forge.Neural;
using BenchmarkDotNet.Running;
using Forge.Benchmarks;

var summary = BenchmarkRunner.Run<GraphBenchmarks>();

Console.WriteLine("⚔️  FORGE.NEURAL INTEGRATION TEST  ⚔️");
Console.WriteLine("-----------------------------------");

var x_data = new double[] { -1, -1, -1, 1, 1, -1, 1, 1 };
var t_inputs = new Tensor(4, 2, x_data);

var y_data = new double[] { -1, 1, 1, -1 };
var t_targets = new Tensor(4, 1, y_data);

var model = new Sequential();
model.Add(new Linear(2, 8, seed: 100));
model.Add(new Tanh());
model.Add(new Linear(8, 1, seed: 200));
model.Add(new Tanh());

var optimizer = new SGD(0.2);

Console.WriteLine("[Forge.Neural] Training...");
var sw = Stopwatch.StartNew();

for (int epoch = 0; epoch < 1000; epoch++)
{
    var pred = model.Forward(t_inputs);

    double totalLoss = 0.0;
    
    optimizer.ZeroGrad(model.Parameters());
    Array.Clear(t_inputs.Grad);

    for(int i=0; i < pred.Data.Length; i++)
    {
        double diff = pred.Data[i] - t_targets.Data[i];
        totalLoss += diff * diff;
        pred.Grad[i] = 2.0 * diff; 
    }

    pred.Backward();

    optimizer.Step(model.Parameters());

    if (epoch % 100 == 0) Console.WriteLine($"Epoch {epoch}: Loss = {totalLoss:F4}");
}
sw.Stop();

Console.WriteLine($"\nFinal Loss: {GetLoss(model.Forward(t_inputs), t_targets):F4}");
Console.WriteLine($"Time: {sw.ElapsedMilliseconds} ms");

double GetLoss(Tensor p, Tensor t)
{
    double sum = 0;
    for(int i=0; i<p.Data.Length; i++) sum += Math.Pow(p.Data[i] - t.Data[i], 2);
    return sum;
}