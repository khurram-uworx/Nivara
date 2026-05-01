# Nivara Release Checklist

## Baseline Decisions (Recorded)

- **First public NuGet release**: Yes — both `Nivara` and `Nivara.Extensions` publish together for the first time.
- **Target framework**: `net10.0`-only — intentional for this release. Multi-targeting deferred to v1 if needed.
- **Extensions support stance**: `Nivara.Extensions` has a separate (weaker) stability promise than core. Evolves independently.

## Pre-Publish Checklist

- [ ] Task 2: NuGet packaging metadata added (SourceLink, symbols, deterministic build, RepositoryType)
- [ ] Task 3: Package descriptions differentiated; release notes added
- [ ] Task 4: Release workflow created; publish instructions documented
- [ ] Task 5: Release-facing docs verified
- [ ] `dotnet build --configuration Release` passes
- [ ] `dotnet test --configuration Release` passes
- [ ] Tag the release commit (`v0.9.0`)
- [ ] Push to GitHub; verify CI passes
- [ ] Run release workflow / manual publish

## Publish Commands (Manual)

```bash
dotnet pack src\Nivara\Nivara.csproj --configuration Release
dotnet pack src\Nivara.Extensions\Nivara.Extensions.csproj --configuration Release
dotnet nuget push src\Nivara\bin\Release\Nivara.0.9.0.nupkg --api-key <key> --source https://api.nuget.org/v3/index.json
dotnet nuget push src\Nivara\bin\Release\Nivara.0.9.0.snupkg --api-key <key> --source https://api.nuget.org/v3/index.json
dotnet nuget push src\Nivara.Extensions\bin\Release\Nivara.Extensions.0.9.0.nupkg --api-key <key> --source https://api.nuget.org/v3/index.json
dotnet nuget push src\Nivara.Extensions\bin\Release\Nivara.Extensions.0.9.0.snupkg --api-key <key> --source https://api.nuget.org/v3/index.json
```

## Publishing

Pack and publish both packages from the repository root:

```bash
dotnet pack src\Nivara\Nivara.csproj --configuration Release
dotnet pack src\Nivara.Extensions\Nivara.Extensions.csproj --configuration Release

dotnet nuget push src\Nivara\bin\Release\Nivara.0.9.0.nupkg --api-key <key> --source https://api.nuget.org/v3/index.json
dotnet nuget push src\Nivara\bin\Release\Nivara.0.9.0.snupkg --api-key <key> --source https://api.nuget.org/v3/index.json
dotnet nuget push src\Nivara.Extensions\bin\Release\Nivara.Extensions.0.9.0.nupkg --api-key <key> --source https://api.nuget.org/v3/index.json
dotnet nuget push src\Nivara.Extensions\bin\Release\Nivara.Extensions.0.9.0.snupkg --api-key <key> --source https://api.nuget.org/v3/index.json
```

The release workflow (`.github/workflows/cd.yml`) automates this on tag push.
