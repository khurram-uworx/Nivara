namespace NivaraChat;

public interface ITextModel
{
    string Name { get; }
    string Process(string input);
}
