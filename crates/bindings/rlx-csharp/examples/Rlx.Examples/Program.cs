// RLX — versatile ML compiler + runtime.
// Copyright (C) 2026 Eugene Hauptmann, Nataliya Kosmyna.
// SPDX-License-Identifier: MIT OR Apache-2.0

// dotnet run -- <inference|training|loading|distributed|backends> [device]

using System.Globalization;

CultureInfo.DefaultThreadCurrentCulture = CultureInfo.InvariantCulture;
var name = args.Length > 0 ? args[0] : "inference";
var rest = args.Skip(1).ToArray();
switch (name)
{
    case "inference": Inference.Run(rest); break;
    case "training": Training.Run(rest); break;
    case "loading": Loading.Run(); break;
    case "distributed": Distributed.Run(); break;
    case "backends": Backends.Run(); break;
    case "models": Models.Run(); break;
    default: Console.Error.WriteLine($"unknown example '{name}'"); return 2;
}
return 0;
