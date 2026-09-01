namespace NivaraChat.Models;

public interface ITextModel
{
    string Name { get; }
    string Process(string input);
}
