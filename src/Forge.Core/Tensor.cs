using System;
using System.Text;

namespace Forge.Core;

public class Tensor
{
    // Data-Oriented Storage
    // We use a flat array for cache locality.
    public readonly double[] Data;
    public readonly double[] Grad;
    
    // Metadata
    public readonly int[] Shape;
    public readonly int[] Strides;
    
    // Autograd Graph
    private List<Tensor> _children;
    private Action _backward;

    public Tensor(int rows, int cols, double[]? data = null)
    {
        Shape = new int[] { rows, cols };
        
        // Strides: How many steps in the flat array to move 1 unit in that dimension?
        // For shape (2, 3):
        // Row Stride = 3 (To go down a row, skip 3 elements)
        // Col Stride = 1 (To go right a col, skip 1 element)
        Strides = new int[] { cols, 1 };
        
        int length = rows * cols;
        Data = data ?? new double[length];
        Grad = new double[length];
        
        _children = new List<Tensor>();
        _backward = () => { };
    }

    // Helper to create random tensor (Gaussian initialization)
    public static Tensor Random(int rows, int cols, int seed = 0)
    {
        var rng = new Random(seed);
        var data = new double[rows * cols];
        for(int i = 0; i < data.Length; i++)
        {
            // Box-Muller transform for normal distribution
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            data[i] = randStdNormal;
        }
        return new Tensor(rows, cols, data);
    }
    
    // Helper to create zeros
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
        // 1. Shape Check (Strict for now)
        if (a.Data.Length != b.Data.Length)
            throw new Exception("Shape mismatch: Tensors must have same size for addition.");

        var result = new Tensor(a.Shape[0], a.Shape[1]);
        result._children.Add(a);
        result._children.Add(b);

        // 2. Forward (Linear Loop - Systems Speed)
        for (int i = 0; i < result.Data.Length; i++)
        {
            result.Data[i] = a.Data[i] + b.Data[i];
        }

        // 3. Backward (Gradients flow equally to both)
        result._backward = () =>
        {
            for (int i = 0; i < result.Grad.Length; i++)
            {
                a.Grad[i] += 1.0 * result.Grad[i];
                b.Grad[i] += 1.0 * result.Grad[i];
            }
        };

        return result;
    }

    public Tensor Tanh()
    {
        var result = new Tensor(this.Shape[0], this.Shape[1]);
        result._children.Add(this);

        // 1. Forward
        for (int i = 0; i < this.Data.Length; i++)
        {
            result.Data[i] = Math.Tanh(this.Data[i]);
        }

        // 2. Backward
        result._backward = () =>
        {
            for (int i = 0; i < this.Grad.Length; i++)
            {
                // derivative of tanh(x) is (1 - tanh(x)^2)
                double t = result.Data[i]; 
                double localGrad = (1.0 - t * t);
                this.Grad[i] += localGrad * result.Grad[i];
            }
        };

        return result;
    }
    
    public void Backward()
    {
        // 1. Topological Sort
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

        // 2. Initialize Gradient of the root (Loss) to 1.0 (vector of 1s)
        // Note: Usually loss is a scalar (1x1), so this fills it with 1.
        for (int i = 0; i < Grad.Length; i++) Grad[i] = 1.0;

        // 3. Reverse execution
        topo.Reverse();
        foreach (var t in topo)
        {
            t._backward();
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
        // 1. Swap Shapes: (2, 3) -> (3, 2)
        var newShape = new int[] { Shape[1], Shape[0] };
        
        // 2. Swap Strides: (3, 1) -> (1, 3)
        // Now "moving down a row" in the new view means 
        // "moving right a column" in the original data.
        var newStrides = new int[] { Strides[1], Strides[0] };
        
        // 3. SHARE the data (Zero Copy)
        return new Tensor(newShape, newStrides, Data, Grad);
    }    
}