#include "search.h"

#include "eval.h"
#include "thread.h"

#include <algorithm>
#include <mutex>
#include <print>
#include <sstream>

namespace {

/// NodeType is used to represent the type of a node in the search tree.
/// It indicates whether the node is a root node. This is used to identify
/// different move stage in the connect6 search tree.
enum class NodeType {
    NT_ROOT_FIRST,
    NT_ROOT_SECOND,
    NT_NONROOT_FIRST,
    NT_NONROOT_SECOND,
};

/// Get the node type in the next ply of the search tree.
constexpr NodeType nextNodeType(NodeType nt)
{
    switch (nt) {
    case NodeType::NT_ROOT_FIRST: return NodeType::NT_ROOT_SECOND;
    case NodeType::NT_ROOT_SECOND: return NodeType::NT_NONROOT_FIRST;
    case NodeType::NT_NONROOT_FIRST: return NodeType::NT_NONROOT_SECOND;
    case NodeType::NT_NONROOT_SECOND: return NodeType::NT_NONROOT_FIRST;
    }
    return NodeType::NT_ROOT_FIRST;  // Default case, should not happen
}

/// Compute the utility value from the win-loss rate and draw rate.
float utilityValue(const SearchParams &p, float winLossRate, float drawRate, Player currentSide)
{
    float utility = winLossRate * p.WinLossUtilityScale;
    utility += drawRate * p.DrawUtilityScale[currentSide];
    return utility;
}

/// Compute the Cpuct exploration factor for the given parent node visits.
float cpuctExplorationFactor(const SearchParams &p, uint32_t parentVisits)
{
    float cpuct = p.CpuctExploration;
    if (p.CpuctExplorationLog != 0)
        cpuct += p.CpuctExplorationLog * std::log(1.0f + parentVisits / p.CpuctExplorationBase);
    return cpuct * std::sqrt(parentVisits + 1e-2f);
}

/// Compute the initial utility value for unexplored children, considering first play urgency.
float fpuValue(const SearchParams &p,
               float               parentAvgUtility,
               float               parentRawUtility,
               float               exploredPolicySum)
{
    float blendWeight      = std::min(1.0f, std::pow(exploredPolicySum, p.FpuUtilityBlendPow));
    float parentUtilityFPU = blendWeight * parentAvgUtility + (1 - blendWeight) * parentRawUtility;
    float fpu              = parentUtilityFPU - p.FpuReductionMax * std::sqrt(exploredPolicySum);
    fpu -= (1 + fpu) * p.FpuLossProp;
    return fpu;
}

/// Compute PUCT selection value with the given child statistics.
float puctSelectionValue(const SearchParams &p,
                         float               childUtility,
                         float               childDraw,
                         float               parentDraw,
                         float               childPolicy,
                         uint32_t            childVisits,
                         uint32_t            childVirtualVisits,
                         float               cpuctExploration)
{
    float U = cpuctExploration * childPolicy / (1 + childVisits);
    float Q = childUtility;

    // Reduce utility value for drawish child nodes for PUCT selection
    // Encourage exploration for less drawish child nodes
    if (p.DrawUtilityPenalty != 0)
        Q -= p.DrawUtilityPenalty * childDraw * (1 - parentDraw);

    // Account for virtual losses
    if (childVirtualVisits > 0)
        Q = (Q * childVisits - childVirtualVisits) / (childVisits + childVirtualVisits);

    return Q + U;
}

/// allocateOrFindNode: allocate a new node if it does not exist in the node table
/// @param nodeTable The node table to allocate or find the node
/// @param hash The hash key of the node
/// @param globalNodeAge The global node age to synchronize the node table
/// @param currentSide The current side to move in this node
/// @return A pair of (the node pointer, whether the node is inserted by myself)
std::pair<Node *, bool>
allocateOrFindNode(NodeTable &nodeTable, uint64_t hash, uint32_t globalNodeAge, Player currentSide)
{
    // Try to find a transposition node with the state's zobrist hash
    Node *node         = nodeTable.findNode(hash);
    bool  didInsertion = false;

    // Allocate and insert a new child node if we do not find a transposition
    if (!node)
        std::tie(node, didInsertion) = nodeTable.tryEmplaceNode(hash, globalNodeAge, currentSide);

    return {node, didInsertion};
}

/// select: select the best child node according to the selection value
/// @param node The node to select child from, must be already expanded
/// @param th The current search thread, used for root move selection
/// @return A pair of (the non-null best child edge pointer, the child node pointer)
///   The child node pointer is nullptr if the edge is unexplored (has zero visit).
template <NodeType NT>
std::pair<Edge *, Node *> selectChild(Node &node, const SearchThread &th)
{
    const SearchParams &params = th.sss.searchParams;
    assert(!node.isLeaf());

    uint32_t parentVisits     = node.getVisits();
    float    parentDraw       = node.getD();
    float    cpuctExploration = cpuctExplorationFactor(params, parentVisits);

    // Apply dynamic cpuct scaling based on parent utility variance if needed
    if (params.CpuctUtilityStdevScale > 0) {
        float parentUtilityVar =
            node.getQVar(params.CpuctUtilityVarPrior, params.CpuctUtilityVarPriorWeight);
        float parentUtilityStdevProp = std::sqrt(parentUtilityVar / params.CpuctUtilityVarPrior);
        float parentUtilityStdevFactor =
            1.0f + params.CpuctUtilityStdevScale * (parentUtilityStdevProp - 1.0f);
        cpuctExploration *= parentUtilityStdevFactor;
    }

    float bestSelectionValue = -std::numeric_limits<float>::infinity();
    Edge *bestEdge           = nullptr;
    Node *bestNode           = nullptr;
    float exploredPolicySum  = 0.0f;

    // Iterate through all expanded children to find the best selection value
    EdgeArray &edges = *node.edges();
    uint32_t   edgeIndex;
    for (edgeIndex = 0; edgeIndex < edges.numEdges; edgeIndex++) {
        Edge &childEdge = edges[edgeIndex];
        Move  move      = childEdge.getMove();

        // TODO: Skip the edge if this move is not in the root move list
        if constexpr (NT == NodeType::NT_ROOT_SECOND) {
            auto movePair = std::pair {th.state->getRecentMove(), move};
            auto rm       = std::find(th.rootMoves.begin(), th.rootMoves.end(), movePair);
            if (rm == th.rootMoves.end())
                continue;
        }

        // If child nodes are not expanded, then the following edges must be unexpanded as well
        Node *childNode = childEdge.child();
        if (!childNode)
            break;

        // Accumulated explored policy sum
        float childPolicy = childEdge.getP();
        exploredPolicySum += childPolicy;

        // Compute selection value and update the best selection value
        uint32_t childVisits        = childEdge.getVisits();
        uint32_t childVirtualVisits = childNode->getVirtualVisits();
        float    childUtility       = -childNode->getQ();
        float    childDraw          = childNode->getD();
        float    selectionValue     = puctSelectionValue(params,
                                                  childUtility,
                                                  childDraw,
                                                  parentDraw,
                                                  childPolicy,
                                                  childVisits,
                                                  childVirtualVisits,
                                                  cpuctExploration);
        if (selectionValue > bestSelectionValue) {
            bestSelectionValue = selectionValue;
            bestEdge           = &childEdge;
            bestNode           = childNode;
        }
    }

    // Compute selection value of the first unexplored child (which will have the highest
    // policy among the rest unexplored children)
    if (edgeIndex < edges.numEdges) {
        float fpuUtility = fpuValue(params, node.getQ(), node.getEvalUtility(), exploredPolicySum);

        Edge    &childEdge          = edges[edgeIndex];
        float    childPolicy        = childEdge.getP();
        uint32_t childVisits        = 0;  // Unexplored edge must has zero edge visit
        uint32_t childVirtualVisits = 0;  // Unexplored edge must has zero virtual visit
        float    selectionValue     = puctSelectionValue(params,
                                                  fpuUtility,
                                                  parentDraw,
                                                  parentDraw,
                                                  childPolicy,
                                                  childVisits,
                                                  childVirtualVisits,
                                                  cpuctExploration);
        if (selectionValue > bestSelectionValue) {
            bestSelectionValue = selectionValue;
            bestEdge           = &childEdge;
            bestNode           = nullptr;  // No child node
        }
    }

    assert(bestEdge);
    return {bestEdge, bestNode};
}

/// expand: generate edges and evaluate the policy of this node
/// @return Whether this node has no valid move, which means this node is a terminal node.
bool expandNode(Node &node, const SearchThread &th)
{
    State              &state  = *th.state;
    const SearchParams &params = th.sss.searchParams;

    std::vector<Move>  legalMoves = state.getLegalMoves(params.UsePassMove);
    std::vector<float> policyValues;

    if (legalMoves.empty())
        return true;  // No legal moves, this node is terminal

    if (params.UseUniformPolicy) {
        // If we use uniform policy, we just fill the policy values with same value
        float policyValue = 1.0f / legalMoves.size();
        policyValues.resize(legalMoves.size(), policyValue);
    }
    else {
        PolicyBuffer policyBuf {state.boardSize()};

        // Set the compute flag for all legal moves
        for (const Move &move : legalMoves)
            policyBuf.setComputeFlag(move);

        // Fill the pass move with the default logits (usually a large negative value),
        // so that it will get a very low probability after softmax.
        policyBuf(Move::Pass) = params.DefaultPassPolicyLogits;

        // Evaluate the policy for the current side to move
        state.evaluator()->evaluatePolicy(state.currentSide(), policyBuf);
        policyBuf.applySoftmax();

        // Copy the policy values from the policy buffer
        policyValues.resize(legalMoves.size());
        for (size_t i = 0; i < legalMoves.size(); ++i)
            policyValues[i] = policyBuf(legalMoves[i]);
    }

    // Generate edges for all legal moves
    node.createEdges(legalMoves, policyValues);
    return false;
}

/// evaluate: evaluate the value of this node and make the first visit
void evaluateNode(Node &node, const SearchThread &th)
{
    State              &state  = *th.state;
    const SearchParams &params = th.sss.searchParams;

    // Check if the state has been filled (no legal moves).
    if (state.legalMoveCount() == 0) {
        float drawUtility = utilityValue(params, 0.0f, 1.0f, state.currentSide());
        node.setTerminal(drawUtility, EVAL_DRAW);
        return;
    }

    // TODO: Check for immediate winning
    // TODO: Search VCF

    // Evaluate value for new node that has not been visited
    Value value       = state.evaluator()->evaluateValue(state.currentSide());
    float winLossRate = value.winLossRate();
    float drawRate    = value.drawRate();
    float utility     = utilityValue(params, winLossRate, drawRate, state.currentSide());
    node.setNonTerminal(utility, winLossRate, drawRate);

    // If ExpandWhenFirstEvaluate mode is enabled, we expand the node immediately
    if (params.ExpandWhenFirstEvaluate)
        expandNode(node, th);
}

/// select and backpropagate: select the best child node and backpropagate the statistics
/// @param node The node to search, must been already allocated.
/// @param th The search thread that is searching this node.
/// @param state The game state of this node. The state's hash must be equal to the node's.
/// @param ply The current search ply. Root node is zero.
/// @param visits The number of desired new visits for this playout.
/// @return The number of actual new visits added to this node.
template <NodeType NT>
uint32_t searchNode(Node &node, SearchThread &th, int ply, uint32_t newVisits)
{
    State              &state  = *th.state;
    const SearchParams &params = th.sss.searchParams;
    assert(node.getHash() == state.hash());

    // Discard visits in this node if it is unevaluated
    uint32_t parentVisits = node.getVisits();
    if (parentVisits == 0)
        return 0;

    // Check if the search is being terminated
    if (th.sss.terminate.load(std::memory_order_relaxed))
        return 0;

    // Cap new visits so that we dont do too much at one time
    newVisits = std::min(newVisits, uint32_t(parentVisits * params.MaxNewVisitsProp) + 1);

    // Return directly if this node is a terminal node and not at root
    if (node.isTerminal()) {
        node.addVisits(newVisits);
        return newVisits;
    }

    // Make sure the parent node is expanded before we select a child
    if (node.isLeaf()) {
        bool noValidMove = expandNode(node, th);

        // If we found that there is no valid move, we mark this node as terminal
        // node the finish this visit.
        if (noValidMove) {
            node.addVisits(newVisits);
            return newVisits;
        }
    }

    bool     stopThisPlayout = false;
    uint32_t actualNewVisits = 0;
    while (!stopThisPlayout && newVisits > 0) {
        // Select the best edge to explore
        auto [childEdge, childNode] = selectChild<NT>(node, th);

        // Make the move to reach the child node
        Move move = childEdge->getMove();
        state.move(move);

        // Reaching a leaf node, expand it
        bool allocatedNode = false;
        if (!childNode) {
            std::tie(childNode, allocatedNode) = allocateOrFindNode(*th.sss.nodeTable,
                                                                    state.hash(),
                                                                    th.sss.globalNodeAge,
                                                                    state.currentSide());

            // Remember this child node in the edge
            childEdge->setChild(childNode);
        }

        // Evaluate the new child node if we are the one who really allocated the node
        if (allocatedNode) {
            // Mark that we are now starting to visit this node
            node.beginVisit(1);
            evaluateNode(*childNode, th);
            if (ply + 1 > th.selDepth)
                th.selDepth = ply + 1;

            // Increment child edge visit count
            childEdge->addVisits(1);
            node.updateStats();
            node.finishVisit(1, 1);
            actualNewVisits++;
            newVisits--;
        }
        else {
            // When transposition happens, we stop the playout if the child node has been
            // visited more times than the parent node. Only continue the playout if the
            // child node has been visited less times than the edge visits, or the absolute
            // child node visits is less than the given threshold.
            uint32_t childEdgeVisits = childEdge->getVisits();
            uint32_t childNodeVisits = childNode->getVisits();
            if (childEdgeVisits >= childNodeVisits
                || childNodeVisits < params.MinTranspositionSkipVisits) {
                node.beginVisit(newVisits);
                uint32_t actualChildNewVisits =
                    searchNode<nextNodeType(NT)>(*childNode, th, ply + 1, newVisits);
                assert(actualChildNewVisits <= newVisits);

                if (actualChildNewVisits > 0) {
                    childEdge->addVisits(actualChildNewVisits);
                    node.updateStats();
                    actualNewVisits += actualChildNewVisits;
                }
                // Discard this playout if we can not make new visits to the best child,
                // since some other thread is evaluating it
                else
                    stopThisPlayout = true;

                node.finishVisit(newVisits, actualChildNewVisits);
                newVisits -= actualChildNewVisits;
            }
            else {
                // Increment edge visits without search the node
                childEdge->addVisits(1);
                node.updateStats();
                node.addVisits(1);
                actualNewVisits++;
                newVisits--;
            }
        }

        // Undo the move
        state.undo();

        // Record root move's seldepth
        if constexpr (NT == NodeType::NT_ROOT_SECOND) {
            auto rmIt = std::find(th.rootMoves.begin(),
                                  th.rootMoves.end(),
                                  std::pair {state.getRecentMove(), move});
            if (rmIt != th.rootMoves.end()) {
                RootMove &rm = *rmIt;
                rm.selDepth  = std::max(rm.selDepth, th.selDepth);
            }
        }
    }

    return actualNewVisits;
}

/// Select best move to play for the given node.
/// @param node The node to compute selection value. Must be expanded.
/// @param edgeIndices[out] The edge indices of selectable children.
/// @param selectionValues[out] The selection values of selectable children.
/// @param lcbValues[out] The lower confidence bound values of selectable children.
///     Only filled when we are using LCB for selection.
/// @param allowDirectPolicyMove If true and the node has no explored children,
///     allow to select from those unexplored children by policy prior directly.
/// @return The index of the best move (with highest selection value) to select.
///     Returns -1 if there is no selectable children.
int selectBestmoveOfChildNode(const Node            &node,
                              const SearchThread    &th,
                              std::vector<uint32_t> &edgeIndices,
                              std::vector<float>    &selectionValues,
                              std::vector<float>    &lcbValues,
                              bool                   allowDirectPolicyMove)
{
    const SearchParams &params = th.sss.searchParams;
    assert(!node.isLeaf());
    edgeIndices.clear();
    selectionValues.clear();
    lcbValues.clear();

    int       bestmoveIndex          = -1;
    float     bestmoveSelectionValue = std::numeric_limits<float>::lowest();
    EvalBound maxBound {-EVAL_INFINITE};

    const EdgeArray &edges = *node.edges();
    for (uint32_t edgeIndex = 0; edgeIndex < edges.numEdges; edgeIndex++) {
        const Edge &childEdge = edges[edgeIndex];
        // Only select from expanded children in the first pass
        Node *childNode = childEdge.child();
        if (!childNode)
            continue;

        float childPolicy    = childEdge.getP();
        float selectionValue = 2.0f * childPolicy;

        assert(childNode->getVisits() > 0);
        uint32_t childVisits = childEdge.getVisits();
        // Skip zero visits children
        if (childVisits > 0) {
            // Discount the child visits by 1 and add small weight on raw policy
            float visitWeight = childVisits * float(childVisits - 1) / float(childVisits);
            selectionValue += visitWeight;
        }

        // Find the best edge with the highest selection value
        if (selectionValue > bestmoveSelectionValue) {
            bestmoveIndex          = edgeIndices.size();
            bestmoveSelectionValue = selectionValue;
        }
        edgeIndices.push_back(edgeIndex);
        selectionValues.push_back(selectionValue);

        // Update bound stats
        maxBound |= childNode->getBound();
    }

    // Compute lower confidence bound values if needed
    if (params.UseLCBForBestmoveSelection && !edgeIndices.empty()) {
        int   bestLCBIndex = -1;
        float bestLCBValue = std::numeric_limits<float>::lowest();

        std::vector<float> lcbRadius;

        // Compute LCB values for all selectable children and find highest LCB value
        for (size_t i = 0; i < edgeIndices.size(); i++) {
            uint32_t    edgeIndex = edgeIndices[i];
            const Edge &childEdge = edges[edgeIndex];
            Node       *childNode = childEdge.child();
            assert(childNode);

            float utilityMean = -childNode->getQ();

            // Only compute LCB for children with enough visits
            uint32_t childVisits = childNode->getVisits();
            if (childVisits < 2) {
                lcbValues.push_back(utilityMean - 1e5f);
                lcbRadius.push_back(1e4f);
            }
            else {
                float utilityVar = childNode->getQVar();
                float radius     = params.LCBStdevs * std::sqrt(utilityVar / childVisits);
                lcbValues.push_back(utilityMean - radius);
                lcbRadius.push_back(radius);

                if (selectionValues[i] > 0
                    && selectionValues[i] >= params.LCBMinVisitProp * bestmoveSelectionValue
                    && lcbValues[i] > bestLCBValue) {
                    bestLCBIndex = i;
                    bestLCBValue = lcbValues[i];
                }
            }
        }

        // Best LCB child gets a bonus on selection value
        if (bestLCBIndex >= 0) {
            float bestLCBSelectionValue = selectionValues[bestLCBIndex];
            for (size_t i = 0; i < edgeIndices.size(); i++) {
                if (i == bestLCBIndex)
                    continue;

                // Compute how much the best LCB value is better than the current child
                float lcbBonus = bestLCBValue - lcbValues[i];
                if (lcbBonus <= 0)
                    continue;

                // Compute how many times larger the radius can be before this LCB value is better
                float gain   = std::min(lcbBonus / lcbRadius[i] + 1.0f, 5.0f);
                float lbound = gain * gain * selectionValues[i];
                if (lbound > bestLCBSelectionValue)
                    bestLCBSelectionValue = lbound;
            }
            selectionValues[bestLCBIndex] = bestLCBSelectionValue;
            bestmoveIndex                 = bestLCBIndex;
        }
    }

    // Select best check mate move if possible
    if (maxBound.lower >= EVAL_MATE_IN_MAX_PLY) {
        bestmoveIndex          = -1;
        bestmoveSelectionValue = std::numeric_limits<float>::lowest();
        // Make best move the one with the maximum lower bound
        for (size_t i = 0; i < edgeIndices.size(); i++) {
            uint32_t    edgeIndex = edgeIndices[i];
            const Edge &childEdge = edges[edgeIndex];
            Node       *childNode = childEdge.child();
            assert(childNode);

            Eval childLowerBound = -childNode->getBound().upper;
            if (childLowerBound < EVAL_MATE_IN_MAX_PLY)  // Downweight non-proven mate moves
                selectionValues[i] *= 1e-9f;
            else if (childLowerBound < maxBound.lower)  // Downweight non-shorted mate moves
                selectionValues[i] *= 1e-3f * (1 + maxBound.lower - childLowerBound);

            // Find the best edge with the highest selection value
            if (selectionValues[i] > bestmoveSelectionValue) {
                bestmoveIndex          = i;
                bestmoveSelectionValue = selectionValues[i];
            }
        }
    }

    // If we have no expanded children for selection, try select by raw policy if allowed
    if (edgeIndices.empty() && allowDirectPolicyMove) {
        for (uint32_t edgeIndex = 0; edgeIndex < edges.numEdges; edgeIndex++) {
            const Edge &childEdge   = edges[edgeIndex];
            float       childPolicy = childEdge.getP();
            if (childPolicy > bestmoveSelectionValue) {
                bestmoveIndex          = edgeIndices.size();
                bestmoveSelectionValue = childPolicy;
            }
            edgeIndices.push_back(edgeIndex);
            selectionValues.push_back(childPolicy);
        }
        assert(!edgeIndices.empty());
    }

    return bestmoveIndex;
}

/// Extract PV of the given node recursively.
/// @param node The node to extract PV.
/// @param pv[out] The extracted PV will be appended to this array.
/// @param maxDepth Only extract PV within this depth.
void extractPVOfChildNode(const Node         &node,
                          const SearchThread &th,
                          std::vector<Move>  &pv,
                          int                 maxDepth = 100)
{
    const Node           *curNode = &node;
    std::vector<uint32_t> tempEdgeIndices;
    std::vector<float>    tempSelectionValues, tempLCBValues;
    for (int depth = 0; depth < maxDepth; depth++) {
        if (curNode->isLeaf())
            break;

        int bestmoveIndex = selectBestmoveOfChildNode(*curNode,
                                                      th,
                                                      tempEdgeIndices,
                                                      tempSelectionValues,
                                                      tempLCBValues,
                                                      true);
        if (bestmoveIndex < 0)
            break;

        uint32_t         bestEdgeIndex = tempEdgeIndices[bestmoveIndex];
        const EdgeArray &edges         = *curNode->edges();
        const Edge      &bestEdge      = edges[bestEdgeIndex];
        pv.push_back(bestEdge.getMove());

        curNode = bestEdge.child();
        if (!curNode)
            break;
    }
}

/// Apply a custom function to the node and its children recursively.
/// @param node The node to traverse recursively.
/// @param f The function to apply to the node and its children.
/// @param prng The PRNG to use for permuting the children for multi-thread visit.
/// @param globalNodeAge The global node age to synchronize the visits.
void recursiveApply(Node &node, std::function<void(Node &)> *f, PRNG *prng, uint32_t globalNodeAge)
{
    std::atomic<uint32_t> &nodeAge = node.getAgeRef();
    // If the node's age has been updated, then the node's traversal is done
    if (nodeAge.load(std::memory_order_acquire) == globalNodeAge)
        return;

    if (!node.isLeaf()) {
        EdgeArray &edges = *node.edges();

        // Get the edge indices of the children to visit
        std::vector<uint32_t> edgeIndices;
        for (uint32_t edgeIndex = 0; edgeIndex < edges.numEdges; edgeIndex++) {
            Edge &childEdge = edges[edgeIndex];
            Node *childNode = childEdge.child();
            if (childNode)
                edgeIndices.push_back(edgeIndex);
        }

        // Shuffle the indices for better multi-thread visit
        if (prng)
            std::shuffle(edgeIndices.begin(), edgeIndices.end(), *prng);

        // Recursively apply the function to the children
        for (uint32_t edgeIndex : edgeIndices) {
            Edge &childEdge = edges[edgeIndex];
            Node *childNode = childEdge.child();
            recursiveApply(*childNode, f, prng, globalNodeAge);
        }
    }

    // If we are the one who first set the node age, we call the function
    uint32_t oldAge = nodeAge.exchange(globalNodeAge, std::memory_order_acq_rel);
    if (f && oldAge != globalNodeAge)
        (*f)(node);
}

}  // namespace

void SharedSearchState::setupRootNode(MainSearchThread &th)
{
    State &state = *th.state;

    // Get the current root position
    std::vector<Move> rootPosition;
    for (int moveIndex = 0; moveIndex < state.moveCount(); moveIndex++) {
        Move move = state.getHistoryMove(moveIndex);
        rootPosition.push_back(move);
    }

    // If the root position has not changed, we do not need to update the root node
    if (rootNode && rootPosition == previousPosition)
        return;

    // Initialize the root node to expanded state
    std::tie(rootNode, std::ignore) =
        allocateOrFindNode(*nodeTable, state.hash(), globalNodeAge, state.currentSide());
    if (rootNode->getVisits() == 0)
        evaluateNode(*rootNode, th);
    if (rootNode->isLeaf())
        expandNode(*rootNode, th);
    assert(rootNode->edges()->numEdges > 0);

    // Garbage collect old nodes (only when we go forward, and not with singular root)
    if (rootPosition.size() >= previousPosition.size() && pool.main()->rootMoves.size() > 1)
        recycleOldNodes();

    // Update previous Position
    previousPosition = std::move(rootPosition);
}

void SharedSearchState::recycleOldNodes()
{
    globalNodeAge += 1;  // Increment global node age

    // Setup node counters
    std::atomic<uint32_t> numReachableNodes {0};
    std::atomic<uint32_t> numRecycledNodes {0};

    // Mark all reachable nodes from the root node and update their node age
    std::function<void(Node &)> f = [&](Node &node) {
        numReachableNodes.fetch_add(1, std::memory_order_relaxed);
    };
    pool.main()->runCustomTaskAndWait(
        [this, &f](SearchThread &th) {
            PRNG prng(LCHash(th.id()));
            recursiveApply(*this->rootNode, &f, th.id() ? &prng : nullptr, this->globalNodeAge);
        },
        true);

    // Remove all unreachable nodes using all threads
    std::atomic<size_t> numShardsProcessed {0};
    pool.main()->runCustomTaskAndWait(
        [this, &numRecycledNodes, &numShardsProcessed](SearchThread &th) {
            for (;;) {
                size_t shardIdx = numShardsProcessed.fetch_add(1, std::memory_order_relaxed);
                if (shardIdx >= this->nodeTable->getNumShards())
                    return;

                NodeTable::Shard shard = this->nodeTable->getShardByShardIndex(shardIdx);
                std::unique_lock lock(shard.mutex);
                for (auto it = shard.table.begin(); it != shard.table.end();) {
                    Node *node = std::addressof(const_cast<Node &>(*it));
                    if (node->getAgeRef().load(std::memory_order_relaxed) != this->globalNodeAge) {
                        it = shard.table.erase(it);
                        numRecycledNodes.fetch_add(1, std::memory_order_relaxed);
                    }
                    else
                        it++;
                }
            }
        },
        true);

    std::println("Note stats: {} reachable, {} recycled, root visits: {}",
                 numReachableNodes.load(),
                 numRecycledNodes.load(),
                 rootNode->getVisits());
}

uint32_t SharedSearchState::updateRootMovesData(MainSearchThread &th)
{
    const SearchParams &params = th.sss.searchParams;
    assert(rootNode != nullptr);
    assert(!rootNode->isLeaf());

    std::vector<uint32_t> edgeIndices;
    std::vector<float>    selectionValues, lcbValues;
    int                   bestChildIndex =
        selectBestmoveOfChildNode(*rootNode, th, edgeIndices, selectionValues, lcbValues, true);

    for (RootMove &rm : th.rootMoves) {
        rm.value          = EVAL_NONE;
        rm.selDepth       = 0;
        rm.winRate        = std::numeric_limits<float>::quiet_NaN();
        rm.drawRate       = std::numeric_limits<float>::quiet_NaN();
        rm.utility        = std::numeric_limits<float>::quiet_NaN();
        rm.utilityStdev   = std::numeric_limits<float>::quiet_NaN();
        rm.policyPrior    = std::numeric_limits<float>::quiet_NaN();
        rm.lcbValue       = std::numeric_limits<float>::quiet_NaN();
        rm.selectionValue = std::numeric_limits<float>::lowest();
        rm.edgeVisits     = 0;
        rm.clearPV();
    }

    EdgeArray &edges                  = *rootNode->edges();
    uint32_t   numSelectableRootMoves = 0;
    for (size_t i = 0; i < edgeIndices.size(); i++) {
        uint32_t edgeIndex = edgeIndices[i];
        Edge    &childEdge = edges[edgeIndex];
        Move     move      = childEdge.getMove();

        // Skip edges that are not in the root moves
        auto rm = std::find(th.rootMoves.begin(),
                            th.rootMoves.end(),
                            std::pair {th.state->getRecentMove(), move});
        if (rm == th.rootMoves.end())
            continue;

        // Record the root move's value for explored children
        Node *childNode = childEdge.child();
        if (childNode) {
            EvalBound childBound = childNode->getBound();
            if (childNode->getVisits() > 0) {
                float childUtility = -childNode->getQ();

                rm->winRate  = childUtility * 0.5f + 0.5f;
                rm->drawRate = childNode->getD();
                rm->utility  = childUtility;
                if (Eval lo = childBound.childLowerBound(); lo >= EVAL_MATE_IN_MAX_PLY)
                    rm->value = MateIn(std::max(MateStep(lo, 0) - th.state->moveCount(), 0));
                else if (Eval up = childBound.childUpperBound(); up <= EVAL_MATED_IN_MAX_PLY)
                    rm->value = MatedIn(std::max(MateStep(up, 0) - th.state->moveCount(), 0));
                else
                    rm->value = winLossRateToEval(rm->winRate);
                rm->utilityStdev = std::sqrt(childNode->getQVar());
                numSelectableRootMoves++;
            }
            rm->edgeVisits = childEdge.getVisits();
            extractPVOfChildNode(*childNode, th, rm->pv);
        }
        rm->policyPrior = childEdge.getP();
        if (i < lcbValues.size())
            rm->lcbValue = lcbValues[i];
        rm->selectionValue = selectionValues[i];
    }

    // If we do not have any visited children, display all of them to show policy
    if (numSelectableRootMoves == 0)
        numSelectableRootMoves = th.rootMoves.size();
    numSelectableRootMoves = std::min(numSelectableRootMoves, params.MaxNumRootmovesToPrint);

    // Sort the root moves in descending order by selection value
    std::stable_sort(th.rootMoves.begin(),
                     th.rootMoves.end(),
                     [](const RootMove &m1, const RootMove &m2) {
                         return m1.selectionValue > m2.selectionValue;
                     });

    return numSelectableRootMoves;
}

void SearchThread::clear(bool newGame)
{
    rootMoves.clear();
    numNewVisits = 0;
    numPlayouts  = 0;
    selDepth     = 0;
}

void MainSearchThread::clear(bool newGame)
{
    SearchThread::clear(newGame);
    bestMove_  = std::pair {Move::None, Move::None};
    startTime_ = optimumTime_ = maximumTime_ = 0;
    lastOutputNewVisits_                     = 0;
    lastOutputPlayouts_                      = 0;
    lastOutputTime_                          = 0;

    sss.terminate           = false;
    sss.accumulatedPlayouts = 0;
    sss.rootNode            = nullptr;
    if (!newGame)
        return;

    // Clear the node table using all threads, and wait for finish
    sss.previousPosition.clear();
    sss.globalNodeAge = 0;
    std::atomic<size_t> numShardsProcessed {0};
    runCustomTaskAndWait(
        [&numShardsProcessed](SearchThread &th) {
            for (;;) {
                size_t shardIdx = numShardsProcessed.fetch_add(1, std::memory_order_relaxed);
                if (shardIdx >= th.sss.nodeTable->getNumShards())
                    return;

                NodeTable::Shard shard = th.sss.nodeTable->getShardByShardIndex(shardIdx);
                std::unique_lock lock(shard.mutex);
                shard.table.clear();
            }
        },
        true);
    sss.pool.waitForIdle();

    // Reset node table num shards if needed
    const SearchParams &params = sss.searchParams;
    if (!sss.nodeTable || sss.nodeTable->getNumShards() != params.NumNodeTableShardsPowerOfTwo)
        sss.nodeTable = std::make_unique<NodeTable>(params.NumNodeTableShardsPowerOfTwo);
}

void MainSearchThread::search()
{
    // Init time management
    setupTimeControl();

    // Starts worker threads, then starts main thread
    sss.setupRootNode(*this);  // Setup root node and other stuffs
    runCustomTaskAndWait([](SearchThread &th) { th.SearchThread::search(); }, true);

    // Rank root moves and record best move
    printSearchOutput(true);
    bestMove_ = rootMoves[0].movePair;
}

void SearchThread::search()
{
    const SearchParams &params = sss.searchParams;
    assert(state->evaluator());
    assert(sss.rootNode != nullptr && !sss.rootNode->isLeaf());

    // Main search loop
    std::vector<Node *> selectedPath;
    while (!sss.terminate.load(std::memory_order_relaxed)) {
        // Strictly stop extra playouts if the accumulated playouts have reached the maximum
        if (sss.limits.maxPlayouts > 0) {
            // Acquire a new playout without exceeding the maximum playout limit
            uint64_t currentPlayouts    = sss.accumulatedPlayouts.load(std::memory_order_relaxed);
            bool     acquiredNewPlayout = false;
            while (true) {
                if (currentPlayouts >= sss.limits.maxPlayouts)
                    break;

                if (sss.accumulatedPlayouts.compare_exchange_weak(currentPlayouts,
                                                                  currentPlayouts + 1,
                                                                  std::memory_order_relaxed,
                                                                  std::memory_order_relaxed)) {
                    acquiredNewPlayout = true;
                    break;
                }
            }

            if (!acquiredNewPlayout)
                break;  // We failed to acquire a new playout, so we stop the search
        }

        // Determine the number of new visits to make in this playout
        uint32_t desiredNumNewVisits = params.NumNewVisitsPerPlayout;
        if (sss.limits.maxNewVisits) {
            uint64_t newVisitsSearched = sss.pool.newVisitsSearched();
            // Stop the search if we have reached the maximum number of new visits
            if (newVisitsSearched >= sss.limits.maxNewVisits)
                break;

            // Cap new number of new visits from search limits
            uint64_t maxNewVisitsToCap = sss.limits.maxNewVisits - newVisitsSearched;
            if (maxNewVisitsToCap < desiredNumNewVisits)
                desiredNumNewVisits = maxNewVisitsToCap;
        }

        // Clear the seldepth for this search thread
        selDepth = 0;

        // Search the tree for new nodes
        bool isFirstMove = (state->nonPassMoveCount() == 0) || (state->nonPassMoveCount() % 2 == 1);
        auto searchFn    = isFirstMove ? searchNode<NodeType::NT_ROOT_FIRST>
                                       : searchNode<NodeType::NT_NONROOT_SECOND>;
        uint32_t newVisits = searchFn(*sss.rootNode, *this, 0, desiredNumNewVisits);

        // Accumulate the new visits and playouts
        numNewVisits.fetch_add(newVisits, std::memory_order_relaxed);
        numPlayouts.fetch_add(1, std::memory_order_relaxed);

        if (isMainThread()) {
            MainSearchThread &mainTh = static_cast<MainSearchThread &>(*this);
            // Check if we should terminate the search
            mainTh.checkStopInSearch();
            // Print the search output with throttling
            mainTh.printSearchOutput(false);
        }
    }
}

void MainSearchThread::printSearchOutput(bool noThrottle)
{
    const SearchParams &params = sss.searchParams;

    // Check if we should print outputs by an amount of newVisits/playouts/time
    bool shouldPrintRootMoves = noThrottle;
    if (params.NewVisitsToPrintMCTSRootmoves > 0) {
        uint64_t currNumNewVisits = sss.pool.newVisitsSearched();
        uint64_t elapsedNewVisits = currNumNewVisits - lastOutputNewVisits_;
        if (elapsedNewVisits >= params.NewVisitsToPrintMCTSRootmoves) {
            lastOutputNewVisits_ = currNumNewVisits;
            shouldPrintRootMoves = true;
        }
    }
    if (params.PlayoutsToPrintMCTSRootmoves > 0) {
        uint64_t currNumPlayouts = sss.pool.playoutsSearched();
        uint64_t elapsedPlayouts = currNumPlayouts - lastOutputPlayouts_;
        if (elapsedPlayouts >= params.PlayoutsToPrintMCTSRootmoves) {
            lastOutputPlayouts_  = currNumPlayouts;
            shouldPrintRootMoves = true;
        }
    }
    if (params.MillisecondsToPrintMCTSRootmoves > 0) {
        Time currentTime = Now();
        Time elapsedTime = currentTime - lastOutputTime_;
        if (elapsedTime >= params.MillisecondsToPrintMCTSRootmoves) {
            lastOutputTime_      = currentTime;
            shouldPrintRootMoves = true;
        }
    }

    if (!shouldPrintRootMoves)
        return;

    // If we should print root moves, update the data and print them
    size_t numSelectableRootMoves = sss.updateRootMovesData(*this);

    Time     elapsed   = Now() - startTime_;
    uint64_t newVisits = sss.pool.newVisitsSearched();
    uint64_t playouts  = sss.pool.playoutsSearched();
    auto     pvText    = [](const std::vector<Move> &pv) -> std::string {
        std::string result;
        for (size_t i = 0; i < pv.size(); i++)
            std::format_to(std::back_inserter(result), "{}{}", i ? " " : "", pv[i]);
        return result;
    };

    for (size_t pvIdx = 0; pvIdx < std::min(numSelectableRootMoves, rootMoves.size()); pvIdx++) {
        RootMove &curMove = rootMoves[pvIdx];

        std::print("({}) {}", pvIdx + 1, curMove.value);
        std::print(" | (W {:.2f}, D {:.2f}, S {:.2f})",
                   curMove.winRate * 100,
                   curMove.drawRate * 100,
                   curMove.utilityStdev);
        std::print(" | V {}", nodesText(curMove.edgeVisits));
        std::print(" | SD {}", curMove.selDepth);
        std::print(" | {}\n", pvText(curMove.pv));
    }

    std::println("Speed {} | Visit {} | Playout {} | Time {}",
                 speedText(newVisits * 1000 / std::max(elapsed, (Time)1)),
                 nodesText(newVisits),
                 nodesText(playouts),
                 timeText(elapsed));
}

void MainSearchThread::setupTimeControl()
{
    lastOutputTime_ = startTime_ = Now();

    const SearchLimits &limits = sss.limits;
    const SearchParams &params = sss.searchParams;

    Time matchTimeLeft = limits.timeLeft;
    if (limits.maxMatchTime == 0)  // unlimited match time
        matchTimeLeft = std::numeric_limits<Time>::max();

    maximumTime_ =
        Time(matchTimeLeft
             / std::clamp(state->legalMoveCount(), params.MatchSpaceMin, params.MatchSpace));
    maximumTime_ = std::min(limits.maxTurnTime, maximumTime_);
    maximumTime_ = std::max(maximumTime_ - params.TurnTimeReservedMilliseconds, (Time)0);

    float timeCapRatio =
        (rootMoves.size() == 1) ? params.SingularRootTimeCapRatio : params.LastPlayoutTimeCapRatio;
    optimumTime_ = Time(maximumTime_ * timeCapRatio);
}

void SearchLimits::setTimeControl(int64_t turnTime, int64_t matchTime)
{
    if (turnTime <= 0 && matchTime <= 0) {  // Infinite time
        this->maxTurnTime  = 0;
        this->maxMatchTime = 0;
        this->useTimeLimit = false;
    }
    else if (turnTime > 0 && matchTime <= 0) {  // Turn time only
        this->maxTurnTime  = turnTime;
        this->maxMatchTime = 0;
        this->useTimeLimit = true;
    }
    else if (turnTime <= 0) {  // Match time only
        this->maxTurnTime  = matchTime;
        this->maxMatchTime = matchTime;
        this->useTimeLimit = true;
    }
    else {  // Match time + turn time
        this->maxTurnTime  = turnTime;
        this->maxMatchTime = matchTime;
        this->useTimeLimit = true;
    }

    this->timeLeft = this->maxMatchTime;
}