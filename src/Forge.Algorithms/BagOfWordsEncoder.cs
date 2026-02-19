using Forge.Core;

namespace Forge.Algorithms;

public class BagOfWordsEncoder
{
    public readonly Dictionary<string, int> Vocab;
    private readonly Dictionary<string, int> _docFreq;
    private readonly int _numDocs;

    private static readonly HashSet<string> StopWords = new()
    {
        "the", "is", "and", "a", "of", "to", "in", "for", "with", "it", "on",
        "this", "that", "by", "from", "an", "be", "as", "are", "at", "has",
        "can", "will", "your", "our", "their", "all", "but", "not", "which",
        "was", "were", "been", "have", "had", "does", "did", "how", "where",
        "when", "why", "who"
    };

    public BagOfWordsEncoder(IEnumerable<string> corpus)
    {
        Vocab = new Dictionary<string, int>();
        _docFreq = new Dictionary<string, int>();

        var materializedCorpus = corpus.ToList();
        _numDocs = materializedCorpus.Count;

        int index = 0;

        foreach (var text in materializedCorpus)
        {
            var words = Tokenize(text).Distinct();
            foreach (var word in words)
            {
                if (!Vocab.ContainsKey(word))
                {
                    Vocab[word] = index++;
                }

                if (!_docFreq.ContainsKey(word))
                {
                    _docFreq[word] = 0;
                }
                _docFreq[word]++;
            }
        }
    }

    /// <summary>
    /// Transforms a raw TF vector into a TF-IDF weighted vector.
    /// Formula: w = tf * log(N / df)
    /// </summary>
    public void TransformTfIdf(Tensor tfTensor)
    {
        if (tfTensor.Data.Length != Vocab.Count)
            throw new ArgumentException("Tensor dimensions must match vocabulary size.");

        for (int i = 0; i < Vocab.Count; i++)
        {
            string word = Vocab.ElementAt(i).Key;

            double df = _docFreq[word];
            double idf = Math.Log(_numDocs / df);

            tfTensor.Data[i] *= idf;
        }
    }

    public List<string> GetTopWords(double[] vector, int count)
    {
        if (vector.Length != Vocab.Count)
            throw new ArgumentException($"Vector size ({vector.Length}) must match Vocab size ({Vocab.Count}).");

        return vector
            .Select((weight, index) => new
            {
                Word = Vocab.ElementAt(index).Key,
                Weight = weight
            })
            .OrderByDescending(x => x.Weight)
            .Take(count)
            .Select(x => x.Word)
            .ToList();
    }

    public Tensor Encode(string text)
    {
        var t = Tensor.Zeros(1, Vocab.Count);

        var words = Tokenize(text);
        foreach (var word in words)
        {
            if (Vocab.TryGetValue(word, out int idx))
            {
                t.Data[idx] += 1.0;
            }
        }

        return t;
    }

    /// <summary>
    /// KG-039 / FORGE-020: Bridge to SIMD-accelerated similarity.
    /// Utilizes the VectorMath MathBridge for hardware-optimized execution.
    /// </summary>
    public static double CosineSimilarity(Tensor a, Tensor b)
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
            if (!char.IsLetterOrDigit(c)) return false;
        }
        return true;
    }
}