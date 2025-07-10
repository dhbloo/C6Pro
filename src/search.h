#pragma once

#include "game.h"
#include "node.h"
#include "nodetable.h"
#include "utils.h"

#include <atomic>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

class ThreadPool;        // forward declaration
class MainSearchThread;  // forward declaration

/// SearchOptions contains the search settings that can be configured by the user.
/// These options include the resource limits, search mode, and various other variables
/// that can be used to control the behavior of the search algorithm.
struct SearchOptions
{
    /// Enable time control to terminate the search
    bool useTimeLimit = false;
    /// The maximum turn time in milliseconds (0 for no limit)
    Time maxTurnTime = 0;
    /// The maximum match time in milliseconds (0 for no limit)
    Time maxMatchTime = 0;
    /// The match time left in milliseconds (0 for no limit)
    Time timeLeft = 0;
    /// The maximum number of new visits to search (0 for no limit)
    uint64_t maxNewVisits = 0;
    /// The maximum number of playouts to search (0 for no limit)
    uint64_t maxPlayouts = 0;

    /// Set time control config according to the rule:
    /// MatchTime (ms) |              TurnTime (ms)
    ///                | less than 0 | equal to 0 | more than 0
    ///    less than 0 |  Infinite   |  Infinite  | Turn only
    ///     equal to 0 |  Infinite   |  Infinite  | Turn only
    ///    more than 0 |  Match only | Match only | Match+Turn
    void setTimeControl(int64_t turnTime, int64_t matchTime);
};

/// RootMove is representing an action at the root of the search tree, which
/// contains the principle variation, and various searched values of the move.
/// It can be composed of a single move or two moves depending on the game stage.
struct RootMove
{
    RootMove(Move move1, Move move2) : movePair {move1, move2} { clearPV(); }
    void clearPV()
    {
        if (movePair.first != Move::None)
            pv = {movePair.first, movePair.second};
        else
            pv = {movePair.second};
    }
    bool operator==(const std::pair<Move, Move> &movePair) const
    {
        return movePair == this->movePair;
    }

    /// The initial moves at the root of the search tree.
    std::pair<Move, Move> movePair;
    /// The eval of the move, can be a mate or mated result.
    Eval value = EVAL_NONE;
    /// The maximal game ply reached in the search.
    int selDepth = 0;
    /// Winning rate of the move in the range [0, 1].
    float winRate = std::numeric_limits<float>::quiet_NaN();
    /// The draw rate of the move in the range [0, 1].
    float drawRate = std::numeric_limits<float>::quiet_NaN();
    /// The utility value of the move, normally in range [-1, 1].
    float utility = std::numeric_limits<float>::quiet_NaN();
    /// The standard variance of the utility value.
    float utilityStdev = std::numeric_limits<float>::quiet_NaN();
    /// The policy prior of the move produced by the  in the range [0, 1].
    float policyPrior = std::numeric_limits<float>::quiet_NaN();
    /// The lower confidence bound value of the utility.
    float lcbValue = std::numeric_limits<float>::quiet_NaN();
    /// The selection value the move to rank the child to descend.
    float selectionValue = std::numeric_limits<float>::quiet_NaN();
    /// The number of accumuated edge visits searched for this root move.
    uint64_t edgeVisits = 0;
    /// The principle variation of the move.
    std::vector<Move> pv;
};

/// SharedSearchState holds all the shared variables that are used by all threads.
struct SharedSearchState
{
    /// The reference to the thread pool that holds these threads.
    ThreadPool &pool;
    /// The search options that controls the search.
    SearchOptions options;
    /// The flag to indicate if the search should be terminated.
    ALIGN_CACHELINE std::atomic_bool terminate;
    /// The number of accmulated playouts searched by all threads.
    ALIGN_CACHELINE std::atomic<uint64_t> accumulatedPlayouts;
    /// The pointer to the root node of the search tree.
    Node *rootNode;
    /// The node table that holds all created nodes in the search.
    std::unique_ptr<NodeTable> nodeTable;
    /// The global node age to synchronize the node table.
    uint32_t globalNodeAge;
    /// The number of selectable root moves, set by updateRootMovesData().
    uint32_t numSelectableRootMoves;
    /// The searched position of last root node represented by a list of moves.
    std::vector<Move> previousPosition;

    /// Initialize the root node from the root state.
    void setupRootNode(State &state);
    /// Garbage collect all old nodes in the node table.
    void recycleOldNodes();
    /// Rank the root moves and update PV, then print all root moves
    void updateRootMovesData(MainSearchThread &th);
};
