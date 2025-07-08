#pragma once

#include "game.h"

#include <atomic>
#include <cassert>
#include <memory>

class Node;       // forward declaration
class NodeTable;  // forward declaration

/// Edge represents an edge in the MCTS graph.
/// It holds the move, policy prior and num visits of this edge.
class Edge
{
public:
    Edge(Move move, float policy) : move_(move), edgeVisits_(0), child_(nullptr) { setP(policy); }

    /// Get the move of this edge.
    Move getMove() const { return move_; }

    /// Converts a float32 policy in [0, 1] to a quantized 16bit policy.
    void setP(float p)
    {
        assert(0.0f <= p && p <= 1.0f);
        constexpr int32_t roundings = (1 << 11) - (3 << 28);
#if defined(__cpp_lib_bit_cast) && __cpp_lib_bit_cast >= 201806L
        int32_t bits = std::bit_cast<int32_t>(p);
#else
        int32_t bits;
        std::memcpy(&bits, &p, sizeof(float));
#endif
        bits += roundings;
        policy_ = (bits < 0) ? 0 : static_cast<uint16_t>(bits >> 12);
    }

    /// Get the normalized policy of this edge.
    float getP() const
    {
        uint32_t bits = (static_cast<uint32_t>(policy_) << 12) | (3 << 28);
#if defined(__cpp_lib_bit_cast) && __cpp_lib_bit_cast >= 201806L
        return std::bit_cast<float>(bits);
#else
        float p;
        std::memcpy(&p, &bits, sizeof(uint32_t));
        return p;
#endif
    }

    /// Get the number of edge visits of this edge.
    uint32_t getVisits() const { return edgeVisits_.load(std::memory_order_acquire); }

    /// Increment the number of edge visits by delta atomically.
    /// @param delta The number of edge visits to change.
    /// @return The number of edge visits after change.
    uint32_t addVisits(uint32_t delta)
    {
        return edgeVisits_.fetch_add(delta, std::memory_order_acq_rel);
    }

    /// Get the child node of this edge.
    /// @note when `getVisits() > 0`, the child node should be guaranteed to be non-null.
    Node *child() const { return child_.load(std::memory_order_acquire); }

    /// Set the child node of this edge.
    void setChild(Node *node) { child_.store(node, std::memory_order_release); }

private:
    /// Move of this edge in respect to the current side to move.
    Move move_;

    /// quantized and compressed 16bit policy value
    uint16_t policy_;

    /// Number of finished visits of this edge.
    std::atomic<uint32_t> edgeVisits_;

    /// Pointer to the child node.
    std::atomic<Node *> child_;
};

/// EdgeArray represents an allocated array of edges.
struct alignas(alignof(Edge)) EdgeArray
{
    uint32_t numEdges;
    Edge     edges[];

    /// Get the edge reference at the given index.
    Edge &operator[](uint32_t index)
    {
        assert(index < numEdges);
        return edges[index];
    }

    /// Get the const edge reference at the given index.
    const Edge &operator[](uint32_t index) const
    {
        assert(index < numEdges);
        return edges[index];
    }
};

static_assert(alignof(EdgeArray) == alignof(Edge), "Sanity check on EdgeArray's alignment");
static_assert(sizeof(Edge) % sizeof(EdgeArray) == 0, "Sanity check on Edge's size");

/// Eval represents the value bound of a node.
struct EvalBound
{
    Eval lower, upper;

    EvalBound() : lower(-EVAL_INFINITE), upper(EVAL_INFINITE) {};
    EvalBound(Eval terminalEval) : lower(terminalEval), upper(terminalEval) {}

    /// Returns whether this bound is terminal.
    bool isTerminal() const { return lower == upper; }
    /// Returns the lower bound of this child node's bound.
    Eval childLowerBound() const { return -upper; }
    /// Returns the upper bound of this child node's bound.
    Eval childUpperBound() const { return -lower; }
    /// Add a child node's bound to this parent node's bound.
    EvalBound &operator|=(EvalBound childBound)
    {
        lower = std::max<Eval>(lower, -childBound.upper);
        upper = std::max<Eval>(upper, -childBound.lower);
        return *this;
    }
};

static_assert(std::atomic<EvalBound>::is_always_lock_free,
              "std::atomic<ValueBound> should be a lock free atomic variable");

/// Node represents a node in the MCTS graph.
/// It holds the edges, result and various statistics of this node.
class Node
{
public:
    /// Constructs a new unevaluated node with no children edges.
    /// @param hash The graph hash key of this node.
    /// @param age The initial age of this node.
    /// @param currentSide The current side to move in this node.
    explicit Node(uint64_t hash, uint32_t age, Player currentSide);
    ~Node();

    // Disallow copy and move. We need node's address to have pointer stability.
    Node(const Node &rhs)            = delete;
    Node &operator=(const Node &rhs) = delete;
    Node(Node &&rhs)                 = delete;
    Node &operator=(Node &&rhs)      = delete;

    /// Set this node to be a terminal node and set num visits to 1.
    /// @param eval The terminal eval of this node.
    void setTerminal(Eval eval);

    /// Set this node to be a non-terminal node and set num visits to 1.
    /// @param utility The raw utility value of this node.
    /// @param drawRate The raw draw probability of this node.
    void setNonTerminal(float utility, float drawRate);

    /// Initializes the edges of this node from the given move picker.
    /// @param moves The moves to create edges for this node.
    /// @param policyValues The policy values for the moves, in the same order as `moves`.
    ///   The values should be in [0, 1] and sum to 1.0.
    void createEdges(const std::vector<Move> &moves, const std::vector<float> &policyValues);

    /// Returns the graph hash key of this node.
    uint64_t getHash() const { return hash_; }

    /// Returns whether this node has no children.
    bool isLeaf() const { return edges_.load(std::memory_order_relaxed) == nullptr; }

    /// Returns the edge array at the given index.
    /// If this node has no edges, returns nullptr.
    EdgeArray       *edges() { return edges_.load(std::memory_order_relaxed); }
    const EdgeArray *edges() const { return edges_.load(std::memory_order_relaxed); }

    /// Returns the average utility value of this node.
    float getQ() const { return q_.load(std::memory_order_relaxed); }

    /// Returns the estimated sample variance of utility value.
    float getQVar(float priorVar = 1.0f, float priorWeight = 1.0f) const;

    /// Returns the average draw rate of this node.
    float getD() const { return d_.load(std::memory_order_relaxed); }

    /// Returns the total visits of this node.
    uint32_t getVisits() const { return n_.load(std::memory_order_acquire); }

    /// Returns the total virtual visits of this node.
    uint32_t getVirtualVisits() const { return nVirtual_.load(std::memory_order_acquire); }

    /// Returns the reference to the age of this node.
    std::atomic<uint32_t> &getAgeRef() { return age_; }

    /// Returns the evaluated utility value of this node.
    float getEvalUtility() const { return utility_; }

    /// Returns the evaluated draw rate of this node.
    float getEvalDrawRate() const { return drawRate_; }

    /// Returns the propogated value bound of this node.
    EvalBound getBound() const { return bound_.load(std::memory_order_relaxed); }

    /// Returns if this node is terminal.
    bool isTerminal() const { return terminalEval_ != EVAL_NONE; }

    /// Update the average utility and the average draw rate from childrens.
    /// @param board The corresponding board of this node.
    /// @param nodeTable The node table for finding children nodes.
    void updateStats();

    /// Begin the visit of this node.
    void beginVisit(uint32_t newVisits)
    {
        nVirtual_.fetch_add(newVisits, std::memory_order_acq_rel);
    }

    /// Finish the visit of this node by incrementing the total visits of this node.
    void finishVisit(uint32_t newVisits, uint32_t actualNewVisits)
    {
        if (actualNewVisits)
            n_.fetch_add(actualNewVisits, std::memory_order_acq_rel);
        nVirtual_.fetch_add(-newVisits, std::memory_order_release);
    }

    /// Directly increment the total visits of this node by delta.
    /// Usually used for visiting a terminal node.
    void addVisits(uint32_t delta) { n_.fetch_add(delta, std::memory_order_acq_rel); }

private:
    /// The graph hash of this node.
    const uint64_t hash_;

    /// The edge array of this node, edges are sorted by normalized policy.
    std::atomic<EdgeArray *> edges_;

    /// Total visits under this node's subgraph.
    /// For leaf node, this indicates if the node's value has been evaluated.
    /// For non-leaf node, this is the sum of all children's edge visits plus 1.
    std::atomic<uint32_t> n_;

    /// Total started but not finished visits under this node's subgraph,
    /// mainly for computing virtual loss when multi-threading is used.
    std::atomic<uint32_t> nVirtual_;

    /// Average utility (from current side to move) of this node, in [-1,1].
    std::atomic<float> q_;

    /// Average squared utility of this node, for computing utility variance.
    std::atomic<float> qSqr_;

    /// Average draw rate of this node in [0,1]. Not flipped when changing side.
    std::atomic<float> d_;

    /// The age of this node, used to find and recycle unused nodes.
    std::atomic<uint32_t> age_;

    /// The propogated terminal value bound of this node.
    std::atomic<EvalBound> bound_;

    /// For non-terminal node, this stores the node's raw utility value in [-1,1].
    /// Higher values means better position from current side to move.
    float utility_;

    /// For non-terminal node, this stores the node's raw draw probability in [0,1].
    /// (from current side to move).
    float drawRate_;

    /// For terminal node, this stores the theoretical value (from current
    /// side to move), including the game ply to mate/mated. If this node is
    /// not a terminal node, this value is VALUE_NONE.
    Eval terminalEval_;

    /// The current side to move.
    Player side_;
};
