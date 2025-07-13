#pragma once

#include "game.h"

#include <cstdint>

struct SearchParams
{
    // ----------------------------------------------------
    // Evaluation Parameters

    /// The utility scale for the win-loss rate in [-1,1]
    float WinLossUtilityScale = 1.0f;
    /// The utility scale for the draw rate in [0,1] of each side
    float DrawUtilityScale[SIDE_NB] = {0.0f, 0.0f};
    /// Should we always use uniform policy (can be used to test value networks)
    bool UseUniformPolicy = false;
    /// Should we allow pass move in the policy
    bool UsePassMove = false;
    /// Default logits for the pass move if the evaluator does not provide it
    float DefaultPassPolicyLogits = -20.0f;
    /// Temperature for the policy softmax
    float PolicyTemperature = 0.90f;
    /// Temperature for the policy softmax at the root node
    float RootPolicyTemperature = 1.05f;

    // ----------------------------------------------------
    // CPUCT Parameters for exploration and exploitation

    /// Cpuct exploration factor for the selection value
    float CpuctExploration = 0.35f;
    /// Cpuct exploration scale of log term for the selection value
    float CpuctExplorationLog = 1.02f;
    /// Cpuct exploration base of log term for the selection value
    float CpuctExplorationBase = 328;
    /// Cpuct exploration scale for the stddev of the utility value
    float CpuctUtilityStdevScale = 0.05f;
    /// Prior of the utility variance for the selection value
    float CpuctUtilityVarPrior = 0.15f;
    /// Weight of the utility variance prior for the selection value
    float CpuctUtilityVarPriorWeight = 1.80f;
    /// The maximum reduction of the first play urgency value
    float FpuReductionMax = 0.055f;
    /// The proportion of the first play urgency loss
    float FpuLossProp = 0.0008f;
    /// The exponent of the sum of explored policy for the first play urgency
    float FpuUtilityBlendPow = 0.75f;
    /// Utility penalty for drawish child nodes in the selection value
    float DrawUtilityPenalty = 0.10f;

    // ----------------------------------------------------
    // MCTS Parameters

    /// Number of desired new visits in one playout
    uint32_t NumNewVisitsPerPlayout = 1;
    /// Max proportion of new visits to parent visits in a single playout
    float MaxNewVisitsProp = 0.36f;
    /// Number of minimal edge visits to skip visiting transposition nodes
    uint32_t MinTranspositionSkipVisits = 11;
    /// Whether to expand the node edges when first evaluating it
    bool ExpandWhenFirstEvaluate = true;
    /// Whether to use LCB for best move selection
    bool UseLCBForBestmoveSelection = true;
    /// Scale of the stddev of the utility value for computing LCB radius
    float LCBStdevs = 6.28f;
    /// Minimum visit proportion to consider LCB to replace best move
    float LCBMinVisitProp = 0.1f;
    /// Exponent of the number of shards used in the shared node table
    uint32_t NumNodeTableShardsPowerOfTwo = 10;

    // ----------------------------------------------------
    // Time Control Parameters

    /// Maximum move count space reserved for the match time
    int MatchSpace = 21;
    /// Minimum move count space reserved for the match time
    int MatchSpaceMin = 7;
    /// Time reserved for each turn to avoid communication overhead timeout
    int TurnTimeReservedMilliseconds = 50;
    /// Stop new playout when time have been used to this ratio after last playout
    float LastPlayoutTimeCapRatio = 0.95f;
    /// Stop new playout when time have been used to this ratio for single root move
    float SingularRootTimeCapRatio = 0.1f;

    // ----------------------------------------------------
    // Output Parameters

    /// How many new visits to print the root moves
    uint64_t NewVisitsToPrintMCTSRootmoves = 0;
    /// How many playouts to print the root moves
    uint64_t PlayoutsToPrintMCTSRootmoves = 0;
    /// How many milliseconds to print the root moves
    uint64_t MillisecondsToPrintMCTSRootmoves = 1000;
    /// Maximal number of root moves to print in the output
    uint32_t MaxNumRootmovesToPrint = 10;
};