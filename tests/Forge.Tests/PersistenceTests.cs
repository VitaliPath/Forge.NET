using Xunit;
using Forge.Graph;
using System.IO;
using System.Text;

namespace Forge.Tests;

public class PersistenceTests
{
    [Fact]
    public void Load_Throws_RecordLoadException_On_MagicBytes_Mismatch()
    {
        // Arrange: Create a stream with "garbage" header bytes
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);
        writer.Write(0xDEADBEEF); // Wrong Magic Bytes
        writer.Write(1);          // Correct Version
        ms.Position = 0;

        // Act & Assert: Should throw based on Header Validation
        Assert.Throws<InvalidDataException>(() => GraphCsr.Load(ms));
    }

    [Fact]
    public void Load_Throws_On_Unsupported_Version()
    {
        // Arrange: Correct Magic, but Version from the "Future"
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true);
        writer.Write(0x46524745); // "FRGE"
        writer.Write(999);        // Future Version
        ms.Position = 0;

        // Act & Assert
        Assert.Throws<InvalidDataException>(() => GraphCsr.Load(ms));
    }

    [Fact]
    public void Save_Maintains_Stream_Position_For_Chained_Snapshots()
    {
        // Arrange: This simulates saving a Graph followed by "Neural Model" data
        var graph = new Graph<string>();
        graph.AddNode("A", "data");
        var csr = graph.CompileCsr();
        
        using var ms = new MemoryStream();

        // Act
        csr.Save(ms);
        long positionAfterGraph = ms.Position;
        
        // Write trailing metadata (simulating Pitfall check: Chained Snapshots)
        using (var writer = new BinaryWriter(ms, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write("NeuralWeights_V1");
        }

        // Assert: Verify we didn't close or reset the stream
        Assert.True(ms.Position > positionAfterGraph);
        
        ms.Position = positionAfterGraph;
        using var reader = new BinaryReader(ms);
        Assert.Equal("NeuralWeights_V1", reader.ReadString());
    }

    [Fact]
    public void RoundTrip_Integrity_Checksum_Parity()
    {
        // Arrange: Build a complex graph
        var graph = new Graph<string>();
        graph.AddNode("User_1", "meta");
        graph.AddNode("User_2", "meta");
        graph.AddEdge("User_1", "User_2", 0.95f);
        
        var originalCsr = graph.CompileCsr();
        var originalHash = originalCsr.GetTopologyHash();
        
        using var ms = new MemoryStream();

        // Act: Save -> Load
        originalCsr.Save(ms);
        ms.Position = 0;
        var loadedCsr = GraphCsr.Load(ms);

        // Assert
        // 1. Data Integrity: Check specific values
        Assert.Equal(0.95f, loadedCsr.Weights[0]);
        Assert.Equal("User_1", loadedCsr.IndexToId[0]);
        
        // 2. Behavioral Integrity: Checksum Parity (FORGE-062)
        var loadedHash = loadedCsr.GetTopologyHash();
        Assert.Equal(originalHash, loadedHash);
        
        // 3. Structural Integrity: Dictionary re-hydration
        Assert.True(loadedCsr.IdToIndex.ContainsKey("User_2"));
        Assert.Equal(1, loadedCsr.IdToIndex["User_2"]);
    }
}