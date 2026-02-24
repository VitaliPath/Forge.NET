using System.Numerics;
using Forge.Core;

namespace Forge.Algorithms;

public class BagOfWordsEncoder
{
    public readonly Dictionary<string, int> Vocab;
    private readonly float[] _idfWeights;
    private readonly string[] _indexToWord;
    private readonly int _numDocs;

    private static readonly HashSet<string> StopWords = new()
    {
        "the", "is", "and", "a", "of", "to", "in", "for", "with", "it", "on",
        "this", "that", "by", "from", "an", "be", "as", "are", "at", "has",
        "can", "will", "your", "our", "their", "all", "but", "not", "which",
        "was", "were", "been", "have", "had", "does", "did", "how", "where",
        "when", "why", "who"
    };
    
    public IReadOnlyList<float> IdfWeights => _idfWeights;

    public BagOfWordsEncoder(IEnumerable<string> corpus)
    {
        Vocab = new Dictionary<string, int>();
        var docFreq = new Dictionary<string, int>();

        var materializedCorpus = corpus.ToList();
        _numDocs = materializedCorpus.Count;

        int index = 0;
        foreach (var text in materializedCorpus)
        {
            var words = Tokenize(text).Distinct();
            foreach (var word in words)
            {
                if (!Vocab.ContainsKey(word)) Vocab[word] = index++;
                if (!docFreq.ContainsKey(word)) docFreq[word] = 0;
                docFreq[word]++;
            }
        }

        _idfWeights = new float[Vocab.Count];
        _indexToWord = new string[Vocab.Count];

        foreach (var entry in Vocab)
        {
            double df = docFreq[entry.Key];
            _idfWeights[entry.Value] = (float)Math.Log((double)_numDocs / df);
            _indexToWord[entry.Value] = entry.Key;
        }
    }

    /// <summary>
    /// FORGE-024: Transforms a raw TF vector into a TF-IDF weighted vector.
    /// Optimized for O(1) weight lookup and SIMD hardware acceleration.
    /// </summary>
    public void TransformTfIdf(Tensor tfTensor)
    {
        if (tfTensor.Data.Length != _idfWeights.Length)
            throw new ArgumentException("Tensor dimensions must match vocabulary size.");

        Span<float> data = tfTensor.Data;
        ReadOnlySpan<float> weights = _idfWeights;
        int i = 0;
        int width = Vector<float>.Count;

        if (Vector.IsHardwareAccelerated && data.Length >= width)
        {
            for (; i <= data.Length - width; i += width)
            {
                var vData = new Vector<float>(data.Slice(i));
                var vWeights = new Vector<float>(weights.Slice(i));
                (vData * vWeights).CopyTo(data.Slice(i));
            }
        }

        for (; i < data.Length; i++)
        {
            data[i] *= weights[i];
        }
    }

    public List<string> GetTopWords(float[] vector, int count)
    {
        if (vector.Length != Vocab.Count)
            throw new ArgumentException($"Vector size ({vector.Length}) must match Vocab size ({Vocab.Count}).");

        var topK = new PriorityQueue<int, float>();

        for (int i = 0; i < vector.Length; i++)
        {
            float weight = vector[i];

            if (weight <= 0.0001f) continue;

            if (topK.Count < count)
            {
                topK.Enqueue(i, weight);
            }
            else
            {
                if (topK.TryPeek(out _, out float minWeight) && weight > minWeight)
                {
                    topK.Dequeue();
                    topK.Enqueue(i, weight);
                }
            }
        }

        var results = new List<string>();
        while (topK.Count > 0)
        {
            results.Add(_indexToWord[topK.Dequeue()]);
        }

        results.Reverse();
        return results;
    }

    public Tensor Encode(string text)
    {
        var t = Tensor.Zeros(1, Vocab.Count);
        var words = Tokenize(text);
        foreach (var word in words)
        {
            if (Vocab.TryGetValue(word, out int idx)) t.Data[idx] += 1.0f;
        }
        return t;
    }

    /// <summary>
    /// KG-039 / FORGE-020: Bridge to SIMD-accelerated similarity.
    /// Utilizes the VectorMath MathBridge for hardware-optimized execution.
    /// </summary>
    public static float CosineSimilarity(Tensor a, Tensor b)
    {
        return VectorMath.CosineSimilarity(a.Data, b.Data);
    }

    private string[] Tokenize(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return Array.Empty<string>();

        // 1. Exhaustive Split: Added '#' to catch Markdown headers specifically.
        // We treat all these as "Glue" or "Separators" to be discarded.
        var rawTokens = text.ToLower().Split(new[] { 
            ' ', '-', '.', ',', '!', '?', ':', ';', '(', ')', '[', ']', 
            '{', '}', '"', '\'', '`', '/', '\\', '\t', '\n', '\r', '*', 
            '+', '=', '<', '>', '#', '@', '$', '^', '&', '|', '~' 
        }, StringSplitOptions.RemoveEmptyEntries);

        var filteredTokens = new List<string>();

        foreach (var token in rawTokens)
        {
            // 2. Minimum Length Constraint (FORGE-0007 Rule)
            // Discards 'x', 'i', '##', and single-letter artifacts.
            if (token.Length < 3) continue;

            // 3. StopWords filtration
            if (StopWords.Contains(token)) continue;

            // 4. Alphanumeric Sanitization
            // We ensure the token contains strictly meaningful characters.
            // This catches mixed-symbol artifacts that Split might have missed.
            if (!IsAlphanumeric(token)) continue;

            // 5. Numerical Filter
            if (double.TryParse(token, out _)) continue;

            filteredTokens.Add(token);
        }

        return filteredTokens.ToArray();
    }

    /// <summary>
    /// Helper to ensure the token doesn't contain lingering symbols.
    /// </summary>
    private bool IsAlphanumeric(string token)
    {
        foreach (char c in token)
        {
            if (!char.IsLetterOrDigit(c) && c != '_') return false;
        }
        return true;
    }
}