using System;
using System.Numerics;
using System.Text;
using System.Collections.Generic;
using System.Linq;

namespace Forge.Core;

public class Tensor
{
    public readonly float[] Data;
    public readonly float[] Grad;
    
    public readonly int[] Shape;
    public readonly int[] Strides;
    
    private List<Tensor> _children;
    private Action _backward;

    public Tensor(int rows, int cols, float[]? data = null)
    {
        Shape = new int[] { rows, cols };
        Strides = new int[] { cols, 1 };
        
        int length = rows * cols;
        Data = data ?? new float[length];
        Grad = new float[length];
        
        _children = new List<Tensor>();
        _backward = () => { };
    }

    public static Tensor Random(int rows, int cols, int seed = 0)
    {
        var rng = new Random(seed);
        var data = new float[rows * cols];
        for (int i = 0; i < data.Length; i++)
        {
            float u1 = 1.0f - (float)rng.NextDouble();
            float u2 = 1.0f - (float)rng.NextDouble();
            // Box-Muller transform
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
            data[i] = (float)randStdNormal;
        }
        return new Tensor(rows, cols, data);
    }
    
    public static Tensor Zeros(int rows, int cols)
    {
        return new Tensor(rows, cols, new float[rows * cols]);
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
        if (this.Shape[1] != other.Shape[0])
        {
            throw new Exception($"Shape Mismatch: Cannot multiply {this.Shape[0]}x{this.Shape[1]} and {other.Shape[0]}x{other.Shape[1]}");
        }

        int n = this.Shape[0]; 
        int m = this.Shape[1]; 
        int p = other.Shape[1]; 

        var result = new Tensor(n, p);
        result._children.Add(this);
        result._children.Add(other);

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < p; j++)
            {
                float sum = 0.0f;
                for (int k = 0; k < m; k++)
                {
                    int idxA = (i * this.Strides[0]) + (k * this.Strides[1]);
                    int idxB = (k * other.Strides[0]) + (j * other.Strides[1]);

                    sum += this.Data[idxA] * other.Data[idxB];
                }
                
                int idxResult = (i * result.Strides[0]) + (j * result.Strides[1]);
                result.Data[idxResult] = sum;
            }
        }

        result._backward = () =>
        {
            var outGrad = new Tensor(result.Shape, result.Strides, result.Grad, null);
            var bT = other.Transpose(); 
            var dA = outGrad.MatMul(bT); 

            for (int i = 0; i < this.Grad.Length; i++)
            {
                this.Grad[i] += dA.Data[i];
            }

            var aT = this.Transpose();
            var dB = aT.MatMul(outGrad); 

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
                    float grad = bResult.Grad[idxOut];

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

        for (int i = 0; i < this.Data.Length; i++)
        {
            float val = this.Data[i];
            result.Data[i] = val > 0 ? val : 0.0f;
        }

        result._backward = () =>
        {
            for (int i = 0; i < this.Grad.Length; i++)
            {
                // Fixed: Added explicit cast to float
                float localGrad = result.Data[i] > 0 ? 1.0f : 0.0f; 
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
            // Fixed: Math.Tanh returns double, needs explicit cast to float
            result.Data[i] = (float)Math.Tanh(this.Data[i]); 
        }

        result._backward = () =>
        {
            for (int i = 0; i < this.Grad.Length; i++)
            {
                float t = result.Data[i]; 
                float localGrad = (1.0f - t * t); // Fixed: Typo 'flat' corrected to 'float'
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
            if (Grad[i] != 0.0f)
            {
                isZero = false;
                break;
            }
        }

        if (isZero)
        {
            for (int i = 0; i < Grad.Length; i++) Grad[i] = 1.0f; // Fixed: Added 'f' suffix
        }

        topo.Reverse();
        foreach (var t in topo)
        {
            t._backward();
        }
    }

    public void ApplyDecay(double lambda, double time)
    {
        float multiplier = (float)Math.Exp(-lambda * Math.Max(0, time));
        if (multiplier < 1e-7f) multiplier = 0.0f;

        int i = 0;
        int width = Vector<float>.Count;

        if (Vector.IsHardwareAccelerated && Data.Length >= width)
        {
            var vMultiplier = new Vector<float>(multiplier);
            for (; i <= Data.Length - width; i += width)
            {
                var vData = new Vector<float>(Data, i);
                (vData * vMultiplier).CopyTo(Data, i);
            }
        }
        for (; i < Data.Length; i++) Data[i] *= multiplier;
    }

    private Tensor(int[] shape, int[] strides, float[] data, float[]? grad)
    {
        Shape = shape;
        Strides = strides;
        Data = data;
        Grad = grad ?? new float[data.Length]; 
        _children = new List<Tensor>();
        _backward = () => { };
    }

    public Tensor Transpose()
    {
        return new Tensor(new int[] { Shape[1], Shape[0] }, new int[] { Strides[1], Strides[0] }, Data, Grad);
    }    
}