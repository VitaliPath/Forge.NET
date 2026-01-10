using System;
using System.Collections.Generic;
using System.Linq;

namespace Forge.Algorithms
{
    public class KnnClassifier
    {
        private List<VectorPoint> _dataSet = new List<VectorPoint>();

        public void Train(IEnumerable<VectorPoint> dataset)
        {
            _dataSet = dataset.ToList();
        }

        public IEnumerable<(VectorPoint Point, double Distance)> Predict(double[] input, int k)
        {
            var comparer = Comparer<double>.Create((a, b) => b.CompareTo(a));
            var bestCandidates = new PriorityQueue<VectorPoint, double>(comparer);

            foreach (var point in _dataSet)
            {
                double dist = GetEuclideanDistance(point.Coordinates, input);

                if (bestCandidates.Count == k)
                {
                    bestCandidates.TryPeek(out _, out double maxDist);
                    if (dist >= maxDist) continue;
                }

                bestCandidates.Enqueue(point, dist);

                if (bestCandidates.Count > k)
                {
                    bestCandidates.Dequeue();
                }
            }

            var results = new List<(VectorPoint, double)>();
            while (bestCandidates.TryDequeue(out VectorPoint p, out double d))
            {
                results.Add((p, d));
            }

            results.Reverse();
            return results;
        }

        private double GetEuclideanDistance(double[] vectorA, double[] vectorB)
        {
            if (vectorA.Length != vectorB.Length)
                throw new ArgumentException("Vectors must have the same length");

            double sum = 0.0;
            for (int i = 0; i < vectorA.Length; i++)
            {
                double diff = vectorA[i] - vectorB[i];
                sum += diff * diff;
            }
            return Math.Sqrt(sum);
        }
    }
}