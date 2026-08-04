using System.Text;

namespace NivaraChat.Data;

public static class IntentDataGenerator
{
    static readonly string[] PersonNames = ["John Smith", "Jane Doe", "Bob Wilson", "Alice Brown", "Charlie Davis", "Emma Johnson", "Michael Lee", "Sarah Garcia"];
    static readonly string[] OrgNames = ["Acme Corp", "TechStart Inc", "Global Industries", "MegaSoft", "InnovateLab", "DataFlow Systems", "CloudNine Ltd", "NetWave"];
    static readonly string[] Topics = ["machine learning", "artificial intelligence", "data science", "cloud computing", "cybersecurity", "blockchain", "quantum computing", "robotics"];
    static readonly string[] Locations = ["New York", "London", "Tokyo", "San Francisco", "Berlin", "Sydney", "Toronto", "Singapore"];
    static readonly string[] Dates = ["January 15", "March 3", "June 20", "September 8", "December 25", "next Monday", "end of month", "Q3 2026"];
    static readonly string[] Products = ["NivaraChat", "DataFlow Pro", "CloudNine Suite", "AI Assistant", "SmartAnalytics", "PredictHub", "DeepInsight", "NeuralForge"];

    static readonly string[] FactualTemplates = [
        "{person} from {org} reported that {topic} is advancing rapidly in {location}",
        "According to {org}, {topic} will transform {location} by {date}",
        "{topic} has been adopted by {org} in {location} since {date}",
        "The {topic} project at {org} in {location} is progressing well",
        "{person} at {org} confirmed {topic} milestones for {date}",
        "{org} announced a new {topic} initiative in {location}",
        "{topic} research at {org} shows promising results in {location}",
        "Experts at {org} predict {topic} growth in {location}",
        "{person} from {org} shared {topic} insights for {location}",
        "{topic} implementation at {org} began on {date}",
        "According to reports, {topic} adoption increased in {location}",
        "{org} published findings on {topic} in {location}",
        "{person} at {org} discussed {topic} trends for {date}",
        "{topic} standards were updated by {org} in {location}",
        "The {topic} team at {org} achieved a breakthrough in {location}",
        "{org} released a {topic} report for {location}",
        "{topic} statistics from {org} show growth in {location}",
        "{person} from {org} provided {topic} data for {date}",
        "{org} confirmed {topic} deployment in {location}",
        "{topic} analysis by {org} indicates progress in {location}"
    ];

    static readonly string[] QuestionTemplates = [
        "Can you explain how {topic} works?",
        "What are the benefits of {topic}?",
        "How does {person} use {topic} at {org}?",
        "Why is {topic} important for {location}?",
        "When will {org} adopt {topic}?",
        "What is the future of {topic} in {location}?",
        "How can I learn about {topic}?",
        "What tools does {org} use for {topic}?",
        "Who are the experts in {topic} at {org}?",
        "Where can I find {topic} resources in {location}?",
        "What are the challenges of {topic}?",
        "How long does it take to learn {topic}?",
        "What certifications are available for {topic}?",
        "How does {topic} compare to alternatives?",
        "What are the prerequisites for {topic}?",
        "Can you recommend {topic} courses?",
        "What industries use {topic}?",
        "How is {topic} applied in {location}?",
        "What are the costs of {topic} implementation?",
        "How does {org} approach {topic}?"
    ];

    static readonly string[] CommandTemplates = [
        "Analyze sentiment of this text about {topic}",
        "Summarize the {topic} report from {org}",
        "Create a presentation on {topic} for {location}",
        "Send the {topic} document to {person}",
        "Schedule a meeting about {topic} with {person}",
        "Generate a {topic} analysis for {org}",
        "Update the {topic} dashboard for {location}",
        "Review the {topic} proposal from {person}",
        "Compile {topic} data from {org} for {date}",
        "Draft a {topic} strategy for {location}",
        "Extract key insights from {topic} research",
        "Compare {topic} solutions from {org} and competitors",
        "Prepare a {topic} training session for {person}",
        "Monitor {topic} performance metrics at {org}",
        "Archive the {topic} documents from {date}",
        "Validate {topic} results against {org} benchmarks",
        "Translate the {topic} guide into local language",
        "Optimize {topic} workflows for {location}",
        "Back up {topic} data from {org} systems",
        "Deploy the {topic} update to production"
    ];

    static readonly string[] ComplaintTemplates = [
        "I'm unhappy with the {topic} service at {org}",
        "The {topic} support from {org} is terrible",
        "My {topic} experience with {org} has been awful",
        "I'm frustrated with {topic} issues at {org}",
        "The {topic} team at {org} is not responding",
        "I want to complain about {topic} performance at {org}",
        "Your {topic} product has been disappointing",
        "I'm dissatisfied with {topic} results from {org}",
        "The {topic} implementation at {org} failed",
        "I need to escalate {topic} problems at {org}",
        "Your {topic} service is unreliable",
        "I'm angry about {topic} delays at {org}",
        "The {topic} quality from {org} is unacceptable",
        "I'm tired of {topic} outages at {org}",
        "Your {topic} team missed the deadline again",
        "I'm considering switching {topic} providers from {org}",
        "The {topic} billing from {org} is incorrect",
        "I'm very disappointed with {topic} support",
        "Your {topic} product lacks basic features",
        "I want a refund for {topic} services at {org}"
    ];

    static readonly string[] ChitchatTemplates = [
        "Hello! How are you today?",
        "Hi there! What's new?",
        "Hey! How's it going?",
        "Good morning! Nice to meet you",
        "What's up? Anything interesting happening?",
        "Hi! How's your day going?",
        "Hello! Hope you're having a great day",
        "Hey there! How's everything?",
        "Hi! What have you been up to?",
        "Hello! How can I brighten your day?",
        "Hey! How's life treating you?",
        "Hi there! Any fun plans today?",
        "Hello! What's on your mind?",
        "Hey! How's the weather where you are?",
        "Hi! How was your weekend?",
        "Hello! What's your favorite thing to do?",
        "Hey there! How's work going?",
        "Hi! Did you do anything exciting recently?",
        "Hello! How's your family?",
        "Hey! How's the new project coming along?"
    ];

    static readonly string[] FactualShort = [
        "The capital of France is Paris",
        "Water boils at 100 degrees Celsius",
        "The Earth orbits the Sun",
        "DNA carries genetic information",
        "Light travels at 300,000 km/s",
        "The Great Wall is in China",
        "Python is a programming language",
        "The Sun is a star",
        "Humans have 46 chromosomes",
        "The Amazon is the largest river"
    ];

    static readonly string[] QuestionShort = [
        "What is machine learning?",
        "How does photosynthesis work?",
        "Why is the sky blue?",
        "What are black holes?",
        "How do computers process data?",
        "What is climate change?",
        "How does the internet work?",
        "What is quantum physics?",
        "Why do leaves change color?",
        "What is artificial intelligence?"
    ];

    static readonly string[] CommandShort = [
        "Analyze this document",
        "Summarize the report",
        "Create a presentation",
        "Send the email",
        "Schedule a meeting",
        "Generate a chart",
        "Update the database",
        "Review the proposal",
        "Compile the data",
        "Draft a strategy"
    ];

    static readonly string[] ComplaintShort = [
        "This service is terrible",
        "I'm very unhappy with the support",
        "The product doesn't work",
        "I want a refund",
        "Your service is unreliable",
        "I'm frustrated with the delays",
        "The quality is unacceptable",
        "I'm considering canceling",
        "Your team is unresponsive",
        "I'm disappointed with the results"
    ];

    static readonly string[] ChitchatShort = [
        "Hello!",
        "Hi there!",
        "How are you?",
        "What's up?",
        "Hey!",
        "Good morning!",
        "How's it going?",
        "Nice to meet you",
        "How's your day?",
        "Any fun plans?"
    ];

    public static (string[] texts, int[] labels) GenerateIntentData(int count, int seed = 42)
    {
        var rng = new Random(seed);
        var texts = new string[count];
        var labels = new int[count];

        for (int i = 0; i < count; i++)
        {
            int intent = rng.Next(5);
            string text;
            if (rng.Next(3) == 0)
                text = intent switch
                {
                    0 => Pick(rng, FactualShort),
                    1 => Pick(rng, QuestionShort),
                    2 => Pick(rng, CommandShort),
                    3 => Pick(rng, ComplaintShort),
                    _ => Pick(rng, ChitchatShort)
                };
            else
                text = intent switch
                {
                    0 => GenerateFactual(rng),
                    1 => GenerateQuestion(rng),
                    2 => GenerateCommand(rng),
                    3 => GenerateComplaint(rng),
                    _ => GenerateChitchat(rng)
                };
            texts[i] = text;
            labels[i] = intent;
        }
        return (texts, labels);
    }

    static string GenerateFactual(Random rng)
    {
        var template = Pick(rng, FactualTemplates);
        return FillTemplate(template, rng);
    }

    static string GenerateQuestion(Random rng)
    {
        var template = Pick(rng, QuestionTemplates);
        return FillTemplate(template, rng);
    }

    static string GenerateCommand(Random rng)
    {
        var template = Pick(rng, CommandTemplates);
        return FillTemplate(template, rng);
    }

    static string GenerateComplaint(Random rng)
    {
        var template = Pick(rng, ComplaintTemplates);
        return FillTemplate(template, rng);
    }

    static string GenerateChitchat(Random rng)
    {
        var template = Pick(rng, ChitchatTemplates);
        return FillTemplate(template, rng);
    }

    static string FillTemplate(string template, Random rng)
    {
        return template
            .Replace("{person}", Pick(rng, PersonNames))
            .Replace("{org}", Pick(rng, OrgNames))
            .Replace("{topic}", Pick(rng, Topics))
            .Replace("{location}", Pick(rng, Locations))
            .Replace("{date}", Pick(rng, Dates))
            .Replace("{product}", Pick(rng, Products));
    }

    static T Pick<T>(Random rng, T[] array) => array[rng.Next(array.Length)];

    public static void SaveIntentCsv(string path, string[] texts, int[] labels)
    {
        var sb = new StringBuilder();
        sb.AppendLine("text,label");
        for (int i = 0; i < texts.Length; i++)
            sb.AppendLine($"\"{texts[i].Replace("\"", "\"\"")}\",{labels[i]}");
        File.WriteAllText(path, sb.ToString());
    }
}