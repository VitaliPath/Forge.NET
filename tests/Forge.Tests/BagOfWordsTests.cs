using Xunit;
using Forge.Algorithms;

namespace Forge.Tests;

public class BagOfWordsTests
{
    [Fact]
    public void Tokenizer_Purges_Markdown_And_Symbols()
    {
        // Arrange: String containing common Markdown and mathematical noise
        var noise = "## Header with *** and (a + b) symbols!";
        var corpus = new[] { noise };
        var encoder = new BagOfWordsEncoder(corpus);

        // Act: Check vocabulary
        // Expected: header, symbols (length > 2 and alphanumeric)
        // Purged: ##, ***, (, +, b, ), !
        var vocab = encoder.Vocab.Keys;

        // Assert
        Assert.Contains("header", vocab);
        Assert.Contains("symbols", vocab);
        Assert.DoesNotContain("##", vocab);
        Assert.DoesNotContain("***", vocab);
        Assert.DoesNotContain("b", vocab); // Length check
    }

    [Fact]
    public void Tokenizer_Minimum_Length_Constraint()
    {
        // Arrange: Technical terms vs. single-char noise
        var text = "Tensor x i weights";
        var corpus = new[] { text };
        var encoder = new BagOfWordsEncoder(corpus);

        // Act
        var vocab = encoder.Vocab.Keys;

        // Assert
        Assert.Contains("tensor", vocab);
        Assert.Contains("weights", vocab);
        Assert.DoesNotContain("x", vocab); // Single char variables purged
        Assert.DoesNotContain("i", vocab);
    }

    [Fact]
    public void Verify_Idf_Weighting_Effect()
    {
        // Arrange: "common" appears in 3/3 docs. "unique" appears in 1/3.
        var corpus = new[] {
            "common unique",
            "common",
            "common irrelevant"
        };
        var encoder = new BagOfWordsEncoder(corpus);

        // Act: Encode a document and apply IDF
        var tensor = encoder.Encode("common unique");
        encoder.TransformTfIdf(tensor);

        // Assert
        int commonIdx = encoder.Vocab["common"];
        int uniqueIdx = encoder.Vocab["unique"];

        // 1. "common" appears in all documents. log(3/3) = log(1) = 0.0.
        Assert.Equal(0.0, tensor.Data[commonIdx]);

        // 2. "unique" appears in 1/3 documents. log(3/1) = 1.0986...
        // tf is 1.0, so 1.0 * 1.0986
        Assert.True(tensor.Data[uniqueIdx] > 1.0);
    }

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

    [Fact]
    public void TransformTfIdf_CalculatesCorrectValues_LinearTime()
    {
        // Arrange: 2 documents, "apple" appears in both, "banana" only in one.
        var corpus = new List<string> { "apple banana", "apple" };
        var encoder = new BagOfWordsEncoder(corpus);
        var tensor = encoder.Encode("apple banana");

        // Act
        encoder.TransformTfIdf(tensor);

        // Assert
        int appleIdx = encoder.Vocab["apple"];
        int bananaIdx = encoder.Vocab["banana"];

        // IDF(apple) = log(2/2) = 0
        // IDF(banana) = log(2/1) = 0.693...
        Assert.Equal(0, tensor.Data[appleIdx], precision: 5);
        Assert.True(tensor.Data[bananaIdx] > 0.69);
    }
}