using System;
using System.Text;

namespace Forge.Core;

public class Tensor
{
    public readonly double[] Data;
    public readonly double[] Grad;
    
    public readonly int[] Shape;
    public readonly int[] Strides;
    
    private List<Tensor> _children;
    private Action _backward;

    public Tensor(int rows, int cols, double[]? data = null)
    {
        Shape = new int[] { rows, cols };
        
        Strides = new int[] { cols, 1 };
        
        int length = rows * cols;
        Data = data ?? new double[length];
        Grad = new double[length];
        
        _children = new List<Tensor>();
        _backward = () => { };
    }

    public static Tensor Random(int rows, int cols, int seed = 0)
    {
        var rng = new Random(seed);
        var data = new double[rows * cols];
        for(int i = 0; i < data.Length; i++)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            data[i] = randStdNormal;
        }
        return new Tensor(rows, cols, data);
    }
    
    public static Tensor Zeros(int rows, int cols)
    {
        return new Tensor(rows, cols, new double[rows * cols]);
    }

    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"Tensor({Shape[0]}x{Shape[1]})");
        for (int i = 0; i < Shape[0]; i++)
        {
            sb.Append("[ ");
            for (int j = 0; j < Shape[1]; j++)
            {
                int index = i * Strides[0] + j * Strides[1];
                sb.Append($"{Data[index]:F4} ");
            }
            sb.AppendLine("]");
        }
        return sb.ToString();
    }

    public Tensor MatMul(Tensor other)
    {
        // 1. Shape Check: (N, M) @ (M, P) -> (N, P)
        if (this.Shape[1] != other.Shape[0])
        {
            throw new Exception($"Shape Mismatch: Cannot multiply {this.Shape[0]}x{this.Shape[1]} and {other.Shape[0]}x{other.Shape[1]}");
        }

        int n = this.Shape[0]; // A Rows
        int m = this.Shape[1]; // Shared Dimension
        int p = other.Shape[1]; // B Cols

        var result = new Tensor(n, p);
        result._children.Add(this);
        result._children.Add(other);

        // 2. The Triple Loop (Forward Pass)
        // Optimization Note: This is the "Naive" implementation. 
        // In the future, we will parallelize this loops or use BLAS.
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                double sum = 0.0;
                for (int k = 0; k < m; k++)
                {
                    // Stride Magic: Manual address calculation
                    // A[i, k]
                    int idxA = (i * this.Strides[0]) + (k * this.Strides[1]);
                    // B[k, j]
                    int idxB = (k * other.Strides[0]) + (j * other.Strides[1]);

                    sum += this.Data[idxA] * other.Data[idxB];
                }
                
                // C[i, j]
                int idxResult = (i * result.Strides[0]) + (j * result.Strides[1]);
                result.Data[idxResult] = sum;
            }
        }

        // 3. The Backward Pass (Your Drill)
        result._backward = () =>
        {
            // We need "Views" of the Gradients to perform MatMul on them.
            // We wrap the Grad arrays in temporary Tensors so we can reuse the MatMul function.
            
            // Out.Grad view
            var outGrad = new Tensor(result.Shape, result.Strides, result.Grad, null);

            // 1. Calculate A.Grad += Out.Grad @ B.T
            var bT = other.Transpose(); 
            var dA = outGrad.MatMul(bT); // This is the calculated gradient for A

            // Accumulate into A.Grad (Element-wise Add)
            for (int i = 0; i < this.Grad.Length; i++)
            {
                this.Grad[i] += dA.Data[i];
            }

            // 2. Calculate B.Grad += A.T @ Out.Grad
            var aT = this.Transpose();
            var dB = aT.MatMul(outGrad); // This is the calculated gradient for B

            // Accumulate into B.Grad
            for (int i = 0; i < other.Grad.Length; i++)
            {
                other.Grad[i] += dB.Data[i];
            }
        };

        return result;
    }

    public static Tensor operator +(Tensor a, Tensor b)
    {
        if (a.Shape[0] == b.Shape[0] && a.Shape[1] == b.Shape[1])
        {
            var result = new Tensor(a.Shape[0], a.Shape[1]);
            result._children.Add(a);
            result._children.Add(b);

            for (int i = 0; i < result.Data.Length; i++)
            {
                result.Data[i] = a.Data[i] + b.Data[i];
            }

            result._backward = () =>
            {
                for (int i = 0; i < result.Grad.Length; i++)
                {
                    a.Grad[i] += result.Grad[i];
                    b.Grad[i] += result.Grad[i];
                }
            };
            return result;
        }

        int outRows = Math.Max(a.Shape[0], b.Shape[0]);
        int outCols = Math.Max(a.Shape[1], b.Shape[1]);

        if ((a.Shape[0] != outRows && a.Shape[0] != 1) ||
            (b.Shape[0] != outRows && b.Shape[0] != 1) ||
            (a.Shape[1] != outCols && a.Shape[1] != 1) ||
            (b.Shape[1] != outCols && b.Shape[1] != 1))
        {
            throw new Exception($"Incompatible shapes for broadcasting: {a.Shape[0]}x{a.Shape[1]} vs {b.Shape[0]}x{b.Shape[1]}");
        }

        var bResult = new Tensor(outRows, outCols);
        bResult._children.Add(a);
        bResult._children.Add(b);

        for (int i = 0; i < outRows; i++)
        {
            for (int j = 0; j < outCols; j++)
            {
                int rA = a.Shape[0] == 1 ? 0 : i;
                int cA = a.Shape[1] == 1 ? 0 : j;
                int idxA = rA * a.Strides[0] + cA * a.Strides[1];

                int rB = b.Shape[0] == 1 ? 0 : i;
                int cB = b.Shape[1] == 1 ? 0 : j;
                int idxB = rB * b.Strides[0] + cB * b.Strides[1];

                int idxOut = i * bResult.Strides[0] + j * bResult.Strides[1];

                bResult.Data[idxOut] = a.Data[idxA] + b.Data[idxB];
            }
        }

        bResult._backward = () =>
        {
            for (int i = 0; i < outRows; i++)
            {
                for (int j = 0; j < outCols; j++)
                {
                    int rA = a.Shape[0] == 1 ? 0 : i;
                    int cA = a.Shape[1] == 1 ? 0 : j;
                    int idxA = rA * a.Strides[0] + cA * a.Strides[1];

                    int rB = b.Shape[0] == 1 ? 0 : i;
                    int cB = b.Shape[1] == 1 ? 0 : j;
                    int idxB = rB * b.Strides[0] + cB * b.Strides[1];

                    int idxOut = i * bResult.Strides[0] + j * bResult.Strides[1];
                    double grad = bResult.Grad[idxOut];

                    a.Grad[idxA] += grad;
                    b.Grad[idxB] += grad;
                }
            }
        };

        return bResult;
    }

    public Tensor Relu()
    {
        var result = new Tensor(this.Shape[0], this.Shape[1]);
        result._children.Add(this);

        // 1. Forward
        for (int i = 0; i < this.Data.Length; i++)
        {
            double val = this.Data[i];
            result.Data[i] = val > 0 ? val : 0;
        }

        // 2. Backward
        result._backward = () =>
        {
            for (int i = 0; i < this.Grad.Length; i++)
            {
                // Gradient flows only if input was > 0
                double localGrad = result.Data[i] > 0 ? 1.0 : 0.0;
                this.Grad[i] += localGrad * result.Grad[i];
            }
        };

        return result;
    }

    public Tensor Tanh()
    {
        var result = new Tensor(this.Shape[0], this.Shape[1]);
        result._children.Add(this);

        for (int i = 0; i < this.Data.Length; i++)
        {
            result.Data[i] = Math.Tanh(this.Data[i]);
        }

        result._backward = () =>
        {
            for (int i = 0; i < this.Grad.Length; i++)
            {
                double t = result.Data[i]; 
                double localGrad = (1.0 - t * t);
                this.Grad[i] += localGrad * result.Grad[i];
            }
        };

        return result;
    }

    public void Backward()
    {
        var topo = new List<Tensor>();
        var visited = new HashSet<Tensor>();

        void BuildTopo(Tensor t)
        {
            if (visited.Contains(t)) return;
            visited.Add(t);
            foreach (var child in t._children)
            {
                BuildTopo(child);
            }
            topo.Add(t);
        }

        BuildTopo(this);

        bool isZero = true;
        for (int i = 0; i < Grad.Length; i++)
        {
            if (Grad[i] != 0.0)
            {
                isZero = false;
                break;
            }
        }

        if (isZero)
        {
            for (int i = 0; i < Grad.Length; i++) Grad[i] = 1.0;
        }

        topo.Reverse();
        foreach (var t in topo)
        {
            t._backward();
        }
    }

    /// <summary>
    /// Applies an in-place exponential decay multiplier to all elements: x = x * exp(-lambda * t).
    /// Designed to age weights in the Knowledge Graph to prioritize recent interactions.
    /// </summary>
    /// <param name="lambda">The decay constant (e.g., ln(2)/half-life).</param>
    /// <param name="time">The time delta (must be >= 0).</param>
    public void ApplyDecay(double lambda, double time)
    {
        double t = Math.Max(0, time);

        double multiplier = Math.Exp(-lambda * t);

        if (multiplier < 1e-9) multiplier = 0.0;

        for (int i = 0; i < Data.Length; i++)
        {
            Data[i] *= multiplier;
        }
    }

    private Tensor(int[] shape, int[] strides, double[] data, double[]? grad)
    {
        Shape = shape;
        Strides = strides;
        Data = data;
        Grad = grad ?? new double[data.Length]; 
        
        _children = new List<Tensor>();
        _backward = () => { };
    }

    public Tensor Transpose()
    {
        var newShape = new int[] { Shape[1], Shape[0] };
        
        var newStrides = new int[] { Strides[1], Strides[0] };
        
        return new Tensor(newShape, newStrides, Data, Grad);
    }    
}