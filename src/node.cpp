#include "node.h"

#include "eval.h"
#include "nodetable.h"

// Default as a losing node for the parent's side of view
Node::Node(uint64_t hash, uint32_t age, Player currentSide)
    : hash_(hash)
    , edges_(nullptr)
    , n_(0)
    , nVirtual_(0)
    , q_(1.0f)
    , qSqr_(1.0f)
    , wl_(1.0f)
    , d_(0.0f)
    , age_(age)
    , bound_()
    , utility_(1.0f)
    , drawRate_(0.0f)
    , terminalEval_(EVAL_NONE)
    , side_(currentSide)
{}

Node::~Node()
{
    EdgeArray *edgeArray = edges_.exchange(nullptr, std::memory_order_relaxed);
    if (edgeArray)
        delete edgeArray;
}

void Node::setTerminal(float utility, Eval eval)
{
    assert(eval != EVAL_NONE);
    assert(-EVAL_INFINITE <= eval && eval <= EVAL_INFINITE);
    utility_      = utility;
    terminalEval_ = eval;

    Value v(eval);
    winLossRate_ = v.winLossRate();
    drawRate_    = v.drawRate();

    q_.store(utility_, std::memory_order_relaxed);
    qSqr_.store(utility_ * utility_, std::memory_order_relaxed);
    wl_.store(winLossRate_, std::memory_order_relaxed);
    d_.store(drawRate_, std::memory_order_relaxed);
    bound_.store(EvalBound {eval}, std::memory_order_relaxed);
    n_.store(1, std::memory_order_release);
}

void Node::setNonTerminal(float utility, float winLossRate, float drawRate)
{
    assert(winLossRate >= -1.0f && winLossRate <= 1.0f);
    assert(drawRate >= 0.0f && drawRate <= 1.0f);

    this->utility_      = utility;
    this->winLossRate_  = winLossRate;
    this->drawRate_     = drawRate;
    this->terminalEval_ = EVAL_NONE;

    q_.store(utility_, std::memory_order_relaxed);
    qSqr_.store(utility_ * utility_, std::memory_order_relaxed);
    wl_.store(winLossRate_, std::memory_order_relaxed);
    d_.store(drawRate_, std::memory_order_relaxed);
    n_.store(1, std::memory_order_release);
}

void Node::createEdges(const std::vector<Move> &moves, const std::vector<float> &policyValues)
{
    assert(moves.size() == policyValues.size());
    uint32_t numEdges = static_cast<uint32_t>(moves.size());

    using AllocType      = std::aligned_storage_t<sizeof(EdgeArray), alignof(EdgeArray)>;
    size_t     numAllocs = (sizeof(EdgeArray) + numEdges * sizeof(Edge)) / sizeof(AllocType);
    EdgeArray *tempEdges = reinterpret_cast<EdgeArray *>(new AllocType[numAllocs]);

    // Copy the move and policy array to the allocated edge array
    tempEdges->numEdges = numEdges;
    for (uint32_t i = 0; i < numEdges; i++)
        new (&tempEdges->edges[i]) Edge(moves[i], policyValues[i]);

    EdgeArray *expected = nullptr;
    bool       suc = edges_.compare_exchange_strong(expected, tempEdges, std::memory_order_release);
    // If we are not the one that sets the edge array, then we need to delete the temp edge array
    if (!suc)
        delete[] tempEdges;
}

float Node::getQVar(float priorVar, float priorWeight) const
{
    uint32_t visits = n_.load(std::memory_order_relaxed);
    if (visits < 2)
        return priorVar;

    float weight        = visits;
    float utilityAvg    = q_.load(std::memory_order_relaxed);
    float utilitySqrAvg = qSqr_.load(std::memory_order_relaxed);

    float sampleVariance = std::max(utilitySqrAvg - utilityAvg * utilityAvg, 0.0f);
    float regularizedVariance =
        (sampleVariance * weight + priorVar * priorWeight) / (weight + priorWeight - 1.0f);
    return regularizedVariance;
}

void Node::updateStats()
{
    const EdgeArray *edgeArray = edges();
    // If this node is not expanded, then we do not need to update any stats
    if (!edgeArray)
        return;

    uint32_t  nSum     = 1;
    float     qSum     = utility_;
    float     qSqrSum  = utility_ * utility_;
    float     wlSum    = winLossRate_;
    float     dSum     = drawRate_;
    EvalBound maxBound = EvalBound {-EVAL_INFINITE};
    for (uint32_t i = 0; i < edgeArray->numEdges; i++) {
        const Edge &edge   = (*edgeArray)[i];
        uint32_t    childN = edge.getVisits();

        // No need to read from child node if it has zero edge visits
        if (childN == 0) {
            // All unvisited children all have uninitialized bound (-inf, inf)
            maxBound |= EvalBound {};
            break;
        }

        Node *childNode = edge.child();
        assert(childNode);  // child node should be guaranteed to be non-null

        float childQ    = childNode->q_.load(std::memory_order_relaxed);
        float childQSqr = childNode->qSqr_.load(std::memory_order_relaxed);
        float childWL   = childNode->wl_.load(std::memory_order_relaxed);
        float childD    = childNode->d_.load(std::memory_order_relaxed);
        bool  flipSign  = (side_ != childNode->side_);  // Flip sign if side changes for this child
        nSum += childN;
        qSum += childN * (flipSign ? -childQ : childQ);
        qSqrSum += childN * childQSqr;
        wlSum += childN * (flipSign ? -childWL : childWL);
        dSum += childN * childD;
        maxBound |= childNode->bound_.load(std::memory_order_relaxed);
    }

    float norm = 1.0f / nSum;
    q_.store(qSum * norm, std::memory_order_relaxed);
    qSqr_.store(qSqrSum * norm, std::memory_order_relaxed);
    wl_.store(wlSum * norm, std::memory_order_relaxed);
    d_.store(dSum * norm, std::memory_order_relaxed);

    // Only update bound if this is not a terminal node
    if (!isTerminal())
        bound_.store(maxBound, std::memory_order_relaxed);
}
