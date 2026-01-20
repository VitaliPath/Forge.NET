using Forge.Core;

namespace Forge.Algorithms;

public class BagOfWordsEncoder
{
    public readonly Dictionary<string, int> Vocab;

    public BagOfWordsEncoder(IEnumerable<string> corpus)
    {
        Vocab = new Dictionary<string, int>();
        int index = 0;
        
        foreach (var text in corpus)
        {
            var words = Tokenize(text);
            foreach (var word in words)
            {
                if (!Vocab.ContainsKey(word))
                {
                    Vocab[word] = index++;
                }
            }
        }
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

    public static double CosineSimilarity(Tensor a, Tensor b)
    {
        if (a.Data.Length != b.Data.Length) 
            throw new Exception($"Vector dimension mismatch: {a.Data.Length} vs {b.Data.Length}");

        double dotProduct = 0.0;
        double magA = 0.0;
        double magB = 0.0;

        for(int i = 0; i < a.Data.Length; i++)
        {
            double valA = a.Data[i];
            double valB = b.Data[i];
            
            dotProduct += valA * valB;
            magA       += valA * valA;
            magB       += valB * valB;
        }
        
        magA = Math.Sqrt(magA);
        magB = Math.Sqrt(magB);
        
        if (magA == 0 || magB == 0) return 0.0; 
        
        return dotProduct / (magA * magB);
    }

    private string[] Tokenize(string text)
    {
        return text.ToLower().Split(new[] { ' ', '.', ',', '!', '?', '\t', '\n', '\r' }, 
                                    StringSplitOptions.RemoveEmptyEntries);
    }
}