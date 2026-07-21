namespace MicroGpt;

public class Tokenizer
{
    public List<string> Chars { get; }
    public Dictionary<string, int> Stoi { get; } = [];
    public Dictionary<int, string> Itos { get; } = [];
    public int VocabSize { get; }
    public int BOS { get; }
    public int EOS { get; }

    public Tokenizer(List<string> docs)
    {
        var allChars = new SortedSet<char>(string.Join("", docs).ToCharArray());
        Chars = ["<BOS>", "<EOS>"];
        Chars.AddRange(allChars.Select(c => c.ToString()));
        VocabSize = Chars.Count;

        for (int i = 0; i < Chars.Count; i++)
        {
            Stoi[Chars[i]] = i;
            Itos[i] = Chars[i];
        }

        BOS = Stoi["<BOS>"];
        EOS = Stoi["<EOS>"];
    }

    public List<int> Encode(string text)
    {
        var tokens = new List<int> { BOS };
        tokens.AddRange(text.Select(ch => Stoi[ch.ToString()]));
        tokens.Add(EOS);
        return tokens;
    }

    public string Decode(int tokenId) => Itos[tokenId];
}
