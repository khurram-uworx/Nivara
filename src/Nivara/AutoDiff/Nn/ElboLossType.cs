namespace Nivara.AutoDiff.Nn;

/// <summary>Selects how the KL-divergence term is weighted in a VAE ELBO loss.</summary>
public enum ElboLossType
{
    /// <summary>The KL term is multiplied by the VAE's β parameter.</summary>
    KldBeta,
    /// <summary>The KL term is left unweighted (suited to KL-annealing schedules).</summary>
    KldAnnealing
}
