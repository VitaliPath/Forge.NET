using BenchmarkDotNet.Running;
using Forge.Benchmarks;

// Using BenchmarkSwitcher allows you to pick which benchmark to run via the CLI 
// (e.g., "1" for Graph, "2" for Vector) or run all of them.
var summary = BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);

Console.WriteLine("\n🚀 Benchmarking Sequence Complete.");