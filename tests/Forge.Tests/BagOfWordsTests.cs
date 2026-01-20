using Xunit;
using Forge.Algorithms;

namespace Forge.Tests;

public class BagOfWordsTests
{
    [Fact]
    public void SanityCheck_CosineSimilarity()
    {
        // Corpus defines the "Universe" of words
        var corpus = new[] { "apple banana" };
        var encoder = new BagOfWordsEncoder(corpus);

        // 1. Exact Match (Angle = 0 degrees, Cosine = 1.0)
        var t1 = encoder.Encode("apple banana");
        var t2 = encoder.Encode("apple banana");
        Assert.Equal(1.0, BagOfWordsEncoder.CosineSimilarity(t1, t2), 4);

        // 2. Orthogonal (Angle = 90 degrees, Cosine = 0.0)
        // "apple" is [1, 0], "banana" is [0, 1]
        var t3 = encoder.Encode("apple");
        var t4 = encoder.Encode("banana");
        Assert.Equal(0.0, BagOfWordsEncoder.CosineSimilarity(t3, t4), 4);
        
        // 3. Partial Overlap
        // "apple banana" [1, 1] vs "apple" [1, 0]
        // Dot = 1*1 + 1*0 = 1
        // MagA = sqrt(2), MagB = sqrt(1)
        // Cos = 1 / 1.414 = 0.707
        var t5 = encoder.Encode("apple banana");
        var t6 = encoder.Encode("apple");
        Assert.Equal(0.7071, BagOfWordsEncoder.CosineSimilarity(t5, t6), 4);
    }
}