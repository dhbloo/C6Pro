#pragma once

#include "game.h"

#include <cstdint>
#include <format>

/// The scaling factor used to convert between winning rate and value.
static constexpr float ScalingFactor = 200.0f;

/// Convert a winning-loss rate in range [-1, 1] to an evaluation value.
Eval winLossRateToEval(float wlrate);

/// Value represents an evaluation of the game position from the side to move perspective.
/// It can be used for regression-like or categorical-like game result.
/// The categorical-like game result can contain the probabilities for a win, loss, or draw.
/// The regression-like game result can contain a single value.
class Value
{
public:
    /// Construct a categorical-like Value with the given win, loss, and draw logits.
    explicit Value(float winLogits, float lossLogits, float drawLogits, bool applySoftmax = true);
    /// Construct a regression-like Value.
    explicit Value(Eval value);

    /// Get the winning-loss rate in range [-1,1] for the side to move.
    float winLossRate() const { return winProb_ - lossProb_; }
    /// Get the winning-rate in range [0,1] for the side to move.
    float winningRate() const { return winLossRate() * 0.5f + 0.5f; }
    /// Get the draw rate in range [0,1] for the side to move.
    float drawRate() const { return drawProb_; }
    /// Get the winning probability in [0,1].
    float winProb() const { return winProb_; }
    /// Get the losing probability in [0,1].
    float lossProb() const { return lossProb_; }
    /// Get the evaluation value.
    Eval eval() const { return eval_; }

private:
    float winProb_, lossProb_, drawProb_;
    Eval  eval_;
};

/// Formatter for the Value type.
template <>
struct std::formatter<Value>
{
    std::string format_spec;

    constexpr auto parse(std::format_parse_context &ctx)
    {
        auto it    = ctx.begin();
        auto end   = ctx.end();
        auto start = it;
        while (it != end && *it != '}')
            ++it;
        format_spec = std::string_view(start, it);
        return it;
    }

    auto format(const Value &value, std::format_context &ctx) const
    {
        std::string fmt    = format_spec.empty() ? "{}" : "{:" + format_spec + "}";
        auto        wl     = value.winLossRate();
        auto        wr     = value.winningRate();
        auto        ev     = value.eval();
        auto        win    = value.winProb();
        auto        loss   = value.lossProb();
        auto        draw   = value.drawRate();
        auto        result = std::format("(WL: {}, WR: {}, EV: {}, W|L|D: {}|{}|{})",
                                  std::vformat(fmt, std::make_format_args(wl)),
                                  std::vformat(fmt, std::make_format_args(wr)),
                                  std::vformat(fmt, std::make_format_args(ev)),
                                  std::vformat(fmt, std::make_format_args(win)),
                                  std::vformat(fmt, std::make_format_args(loss)),
                                  std::vformat(fmt, std::make_format_args(draw)));

        return std::format_to(ctx.out(), "{}", result);
    }
};

// --------------------------------------------------------

/// PolicyBuffer represents an policy distribution for the game actions.
class PolicyBuffer
{
public:
    /// Construct an empty policy buffer for the given board size.
    PolicyBuffer(int boardSize);

    /// Get the policy value from the move index.
    /// The move index should be in the range [1, Move::MAX_MOVES].
    float &operator()(Move move) { return policy_[static_cast<int>(move)]; }
    /// Get the const policy value from the move index.
    /// The move index should be in the range [1, Move::MAX_MOVES].
    float operator()(Move move) const { return policy_[static_cast<int>(move)]; }
    /// Set the compute flag at the given move index.
    void setComputeFlag(Move move);
    /// Clear the compute flag at the given move index.
    void clearComputeFlag(Move move);
    /// Get the compute flag at the given move index.
    bool getComputeFlag(Move move) const;
    /// Applies softmax to all policy with enabled compute flag.
    void applySoftmax();

private:
    int      boardSize_;
    bool     computeFlagPass_;
    uint32_t computeFlagKey_[32];
    float    policy_[Move::MAX_MOVES + 1];
};

// --------------------------------------------------------

/// Evaluator is the base class for evaluation plugins.
/// It provides overridable hook over board move/undo update, and interface for doing value
/// evaluation and policy evaluation. Different evaluation implementation may inherit from
/// this class to replace the default classical evaluation builtin the board.
class Evaluator
{
public:
    virtual ~Evaluator() = default;

    /// Resets the evaluator state to empty board.
    virtual void reset() = 0;
    /// Update hook called when put a new stone on board (a non-pass move).
    /// @param color The color of the stone to put. Must be C_BLACK or C_WHITE or C_GAP.
    /// @param x The x coordinate in [0, boardSize-1] of the stone to put.
    /// @param y The y coordinate in [0, boardSize-1] of the stone to put.
    virtual void move(Color color, int x, int y) = 0;
    /// Update hook called when rollback a stone on board (a non-pass move).
    /// @param color The color of the stone we put before. Must be C_BLACK or C_WHITE or C_GAP.
    /// @param x The x coordinate in [0, boardSize-1] of the stone we put before.
    /// @param y The y coordinate in [0, boardSize-1] of the stone we put before.
    virtual void undo(Color color, int x, int y) = 0;

    /// Evaluates value for current side to move.
    virtual Value evaluateValue(Player side) = 0;
    /// Evaluates policy for current side to move.
    virtual void evaluatePolicy(Player side, PolicyBuffer &policyBuffer) = 0;
};
