namespace Forge.Algorithms
{
    public struct VectorPoint
    {
        public string Id;
        public double[] Coordinates;
        public string Label;

        public VectorPoint(string id, double[] coordinates, string label)
        {
            Id = id;
            Coordinates = coordinates;
            Label = label;
        }
    }
}