#pragma once

#include "game.h"

#include <cstdint>

constexpr float WinLossUtilityScale       = 1.0f;
constexpr float DrawUtilityScale[SIDE_NB] = {0.0f, 0.0f};

constexpr bool  UseUniformPolicy        = false;
constexpr bool  UsePassMove             = false;
constexpr float DefaultPassPolicyLogits = -20.0f;

constexpr float MaxNewVisitsProp = 0.36f;

constexpr float CpuctExploration     = 0.35f;
constexpr float CpuctExplorationLog  = 1.02f;
constexpr float CpuctExplorationBase = 328;

constexpr float CpuctUtilityStdevScale     = 0.05f;
constexpr float CpuctUtilityVarPrior       = 0.15f;
constexpr float CpuctUtilityVarPriorWeight = 1.80f;

constexpr float FpuReductionMax    = 0.055f;
constexpr float FpuLossProp        = 0.0008f;
constexpr float FpuUtilityBlendPow = 0.75f;
constexpr float DrawUtilityPenalty = 0.10f;

constexpr uint32_t MaxNumVisitsPerPlayout     = 1;
constexpr uint32_t MinTranspositionSkipVisits = 11;
constexpr bool     ExpandWhenFirstEvaluate    = true;

constexpr bool  UseLCBForBestmoveSelection = true;
constexpr float LCBStdevs                  = 6.28f;
constexpr float LCBMinVisitProp            = 0.1f;

constexpr float PolicyTemperature     = 0.90f;
constexpr float RootPolicyTemperature = 1.05f;

constexpr int   NumNodeTableShardsPowerOfTwo = 2048;
constexpr int   MatchSpace                   = 21;
constexpr int   MatchSpaceMin                = 7;
constexpr int   TurnTimeReservedMilliseconds = 50;
constexpr int   MoveHorizon                  = 60;
constexpr float AdvancedStopRatio            = 0.9f;
constexpr float AverageBranchFactor          = 1.5f;

constexpr uint64_t PlayoutsToPrintMCTSRootmoves     = 10000;
constexpr Time     MillisecondsToPrintMCTSRootmoves = 1000;
constexpr uint64_t NumPlayoutsAfterSingularRoot     = 1000;
constexpr uint32_t MaxNonPVRootmovesToPrint         = 10;