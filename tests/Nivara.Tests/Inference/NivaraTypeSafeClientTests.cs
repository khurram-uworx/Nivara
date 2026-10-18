using NUnit.Framework;
using System.Text.Json;
using System.Text.Json.Nodes;
using TypeSafeAI;
using NivaraInference;
using Nivara.AI;

namespace Nivara.Tests.Inference;

[TestFixture]
public class NivaraTypeSafeClientTests
{
    [Test]
    public void ContentToText_TextPassthrough()
    {
        var c = TypeSafeContent.FromText("hello");
        Assert.That(NivaraTypeSafeLayaMapping.ContentToText(c), Is.EqualTo("hello"));
    }

    [Test]
    public void ContentToText_NullContent_ReturnsEmpty()
    {
        Assert.That(NivaraTypeSafeLayaMapping.ContentToText(null), Is.EqualTo(string.Empty));
    }

    [Test]
    public void ContentToText_JsonNullState_ReturnsEmpty()
    {
        // The SDK treats a JSON null as empty content, so the serializer has nothing to emit.
        var c = TypeSafeContent.FromJson("null");
        Assert.That(NivaraTypeSafeLayaMapping.ContentToText(c), Is.EqualTo(string.Empty));
    }

    [Test]
    public void ContentToText_NestedObjectArray_UsesPythonSeparators()
    {
        var c = TypeSafeContent.FromJson("{\"a\":[1,2],\"b\":\"x\"}");
        Assert.That(NivaraTypeSafeLayaMapping.ContentToText(c),
            Is.EqualTo("{\"a\": [1, 2], \"b\": \"x\"}"));
    }

    [Test]
    public void ContentToText_NonAsciiRaw()
    {
        var c = TypeSafeContent.FromJson("{\"t\":\"café\"}");
        Assert.That(NivaraTypeSafeLayaMapping.ContentToText(c),
            Is.EqualTo("{\"t\": \"café\"}"));
    }

    [Test]
    public void ContentToText_NumbersPreservedAsWritten()
    {
        var c = TypeSafeContent.FromJson("{\"n\":1.50,\"m\":1e16}");
        Assert.That(NivaraTypeSafeLayaMapping.ContentToText(c),
            Is.EqualTo("{\"n\": 1.50, \"m\": 1e16}"));
    }

    [Test]
    public void ContentToText_EmptyObjectAndArray()
    {
        Assert.That(NivaraTypeSafeLayaMapping.ContentToText(TypeSafeContent.FromJson("{}")),
            Is.EqualTo("{}"));
        Assert.That(NivaraTypeSafeLayaMapping.ContentToText(TypeSafeContent.FromJson("[]")),
            Is.EqualTo("[]"));
    }

    [Test]
    public void ToLayaQuestion_ChoiceKeysAndOrder()
    {
        var cq = new ChoiceQuestion(TypeSafeContent.FromText("topic?"),
            new Dictionary<string, TypeSafeContent?>
            {
                ["model-release"] = TypeSafeContent.FromText("new model"),
                ["research"] = null,
            });

        var lq = NivaraTypeSafeLayaMapping.ToLayaQuestion("q1", cq);
        Assert.That(lq.Type, Is.EqualTo(LayaQuestionType.Choice));
        Assert.That(lq.Instructions, Is.EqualTo("topic?"));
        Assert.That(lq.Options, Has.Count.EqualTo(2));
        Assert.That(lq.Options[0].Key, Is.EqualTo("model-release"));
        Assert.That(lq.Options[0].Description, Is.EqualTo("new model"));
        Assert.That(lq.Options[1].Key, Is.EqualTo("research"));
        Assert.That(lq.Options[1].Description, Is.Null);
    }

    [Test]
    public void ToLayaQuestion_NoulPartialSpec_KeepsDefaults()
    {
        var nq = new NoulQuestion(TypeSafeContent.FromText("resolved?"),
            new NoulCriteria(TypeSafeContent.FromText("fixed"), null));

        var lq = NivaraTypeSafeLayaMapping.ToLayaQuestion("n", nq);
        Assert.That(lq.Type, Is.EqualTo(LayaQuestionType.Noul));
        Assert.That(lq.NoulTrueCriterion, Is.EqualTo("fixed"));
        Assert.That(lq.NoulFalseCriterion, Is.EqualTo(string.Empty));
    }

    [Test]
    public void ToLayaQuestion_NullInstructions_BecomesEmpty()
    {
        var sq = new ScoreQuestion(null, new TypeSafeContent?[] { TypeSafeContent.FromText("low"), TypeSafeContent.FromText("high") });
        var lq = NivaraTypeSafeLayaMapping.ToLayaQuestion("s", sq);
        Assert.That(lq.Instructions, Is.EqualTo(string.Empty));
        Assert.That(lq.ScoreCriteria, Has.Count.EqualTo(2));
        Assert.That(lq.ScoreCriteria[0], Is.EqualTo("low"));
    }

    [Test]
    public void ToAnswer_Choice_UsesKeyAndCalibratedConfidence()
    {
        var cq = new ChoiceQuestion(TypeSafeContent.FromText("topic?"),
            new Dictionary<string, TypeSafeContent?>
            {
                ["a"] = null,
                ["b"] = null,
            });
        var decision = new LayaDecision
        {
            QuestionId = "q",
            Type = LayaQuestionType.Choice,
            Choice = "b",
            Probabilities = new[] { 0.2, 0.8 },
            Confidence = 0.42,
            AnswerConfidence = 0.81,
            ActionProbability = 0.99,
            Temperature = new LayaTemperature(1.0, "choice:2", 1.0, true, false),
        };

        var answer = (ChoiceAnswer)NivaraTypeSafeLayaMapping.ToAnswer(cq, decision);
        Assert.That(answer.Choice, Is.EqualTo("b"));
        Assert.That(answer.Probabilities["a"], Is.EqualTo(0.2));
        Assert.That(answer.Probabilities["b"], Is.EqualTo(0.8));
        Assert.That(answer.Confidence, Is.EqualTo(0.81));
        Assert.That(answer.AdditionalData!["confidence"].GetDouble(), Is.EqualTo(0.42));
        Assert.That(answer.AdditionalData["act_probability"].GetDouble(), Is.EqualTo(0.99));
    }

    [Test]
    public void ToAnswer_Score_ExpectationAndLegend()
    {
        var sq = new ScoreQuestion(TypeSafeContent.FromText("sig?"),
            new TypeSafeContent?[] { TypeSafeContent.FromText("minor"), TypeSafeContent.FromText("major") });
        var decision = new LayaDecision
        {
            QuestionId = "s",
            Type = LayaQuestionType.Score,
            Score = 0.75,
            Probabilities = new[] { 0.25, 0.75 },
            Confidence = 0.5,
            AnswerConfidence = 0.75,
            ActionProbability = 1.0,
            Temperature = new LayaTemperature(1.0, "score:2", 1.0, true, false),
        };

        var answer = (ScoreAnswer)NivaraTypeSafeLayaMapping.ToAnswer(sq, decision);
        Assert.That(answer.Score, Is.EqualTo(0.75));
        Assert.That(answer.Probabilities[0], Is.EqualTo(0.25));
        Assert.That(answer.Probabilities[1], Is.EqualTo(0.75));
        Assert.That(answer.Confidence, Is.EqualTo(0.75));
        Assert.That(answer.Legend[0]!.Value.Text, Is.EqualTo("minor"));
    }

    [Test]
    public void ToAnswer_Noul_ProbabilityAndAdditionalData()
    {
        var nq = new NoulQuestion(TypeSafeContent.FromText("done?"),
            new NoulCriteria(TypeSafeContent.FromText("yes"), TypeSafeContent.FromText("no")));
        var decision = new LayaDecision
        {
            QuestionId = "n",
            Type = LayaQuestionType.Noul,
            NoulProbability = 0.3,
            Probabilities = new[] { 0.7, 0.3 },
            Confidence = 0.7,
            AnswerConfidence = 0.7,
            ActionProbability = 0.99,
            Temperature = new LayaTemperature(1.0, "noul:2", 1.0, true, false),
        };

        var answer = (NoulAnswer)NivaraTypeSafeLayaMapping.ToAnswer(nq, decision);
        Assert.That(answer.Probability, Is.EqualTo(0.3));
        Assert.That(answer.AdditionalData!.ContainsKey("answer_confidence"), Is.True);
    }
}
