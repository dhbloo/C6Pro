#include "eval.h"

#include "utils.h"

#include <algorithm>

Eval winLossRateToEval(float wlrate)
{
    float valueF32 = ScalingFactor * std::logf((1 + wlrate) / (1 - wlrate));
    valueF32       = std::clamp(valueF32, (float)EVAL_MIN, (float)EVAL_MAX);
    return static_cast<Eval>(valueF32);
}

Value::Value(float winLogits, float lossLogits, float drawLogits, bool applySoftmax)
{
    if (applySoftmax) {
        float maxValue = std::max(std::max(winLogits, lossLogits), drawLogits);
        winProb_       = std::exp(winLogits - maxValue);
        lossProb_      = std::exp(lossLogits - maxValue);
        drawProb_      = std::exp(drawLogits - maxValue);
    }
    else {
        winProb_  = winLogits;
        lossProb_ = lossLogits;
        drawProb_ = drawLogits;
    }

    // Make sure the probabilities sum to 1.
    float invSum = 1.0f / (winProb_ + lossProb_ + drawProb_);
    winProb_ *= invSum;
    lossProb_ *= invSum;
    drawProb_ *= invSum;

    // Compute the eval from the winning-loss rate.
    eval_ = winLossRateToEval(winLossRate());
}

Value::Value(Eval value) : eval_(value)
{
    if (value >= EVAL_MATE_IN_MAX_PLY) {
        winProb_  = 1.0f;
        lossProb_ = 0.0f;
        drawProb_ = 0.0f;
    }
    else if (value <= EVAL_MATED_IN_MAX_PLY) {
        winProb_  = 0.0f;
        lossProb_ = 1.0f;
        drawProb_ = 0.0f;
    }
    else {
        // No draws in this case. We can use a more advanced WLD model in future to
        // get a better estimation of the draw probability.
        winProb_  = 1.0f / (1.0f + expf(-static_cast<float>(value) / ScalingFactor));
        lossProb_ = 1.0f - winProb_;
        drawProb_ = 0.0f;
    }
}

std::ostream &operator<<(std::ostream &out, const Value value)
{
    out << "Value ("
        << "W|L|D: " << value.winProb() << "|"
        << "|" << value.lossProb() << "|" << value.drawRate() << ", "
        << "winloss: " << value.winLossRate() << ", "
        << "winrate: " << value.winningRate() << ", "
        << "eval: " << value.eval() << ")";
    return out;
}

PolicyBuffer::PolicyBuffer(int boardSize) : boardSize_(boardSize), computeFlagPass_(false)
{
    // Initialize the compute flag key to false as default
    std::fill(computeFlagKey_, computeFlagKey_ + 32, 0U);
}

void PolicyBuffer::setComputeFlag(Move move)
{
    if (move.isPass())
        computeFlagPass_ = true;
    else if (move)
        computeFlagKey_[move.y()] |= (1U << move.x());
}

void PolicyBuffer::clearComputeFlag(Move move)
{
    if (move.isPass())
        computeFlagPass_ = false;
    else if (move)
        computeFlagKey_[move.y()] &= ~(1U << move.x());
}

bool PolicyBuffer::getComputeFlag(Move move) const
{
    if (move.isPass())
        return computeFlagPass_;
    else if (move)
        return (computeFlagKey_[move.y()] & (1U << move.x())) != 0;
    return false;
}

void PolicyBuffer::applySoftmax()
{
    // Find max computed policy
    float maxPolicy =
        computeFlagPass_ ? policy_[Move::MAX_MOVES] : std::numeric_limits<float>::lowest();
    for (int y = 0; y < boardSize_; y++) {
        for (int x = 0; x < boardSize_; x++) {
            if (computeFlagKey_[y] & (1U << x)) {
                float p = policy_[(int)Move {x, y}];
                if (p > maxPolicy)
                    maxPolicy = p;
            }
        }
    }

    // Apply exponent function and sum
    float sumPolicy = 0;
    for (int y = 0; y < boardSize_; y++) {
        for (int x = 0; x < boardSize_; x++) {
            int i = (int)Move {x, y};
            if (computeFlagKey_[y] & (1U << x)) {
                policy_[i] = std::exp(policy_[i] - maxPolicy);
                sumPolicy += policy_[i];
            }
        }
    }
    if (computeFlagPass_) {
        policy_[Move::MAX_MOVES] = std::exp(policy_[Move::MAX_MOVES] - maxPolicy);
        sumPolicy += policy_[Move::MAX_MOVES];
    }

    // Divide sum policy to normalize
    float invSumPolicy = 1 / sumPolicy;
    for (int y = 0; y < boardSize_; y++) {
        for (int x = 0; x < boardSize_; x++) {
            int i = (int)Move {x, y};
            if (computeFlagKey_[y] & (1U << x))
                policy_[i] *= invSumPolicy;
            else
                policy_[i] = 0;  // Ensure non-computed moves are zero
        }
    }
    if (computeFlagPass_)
        policy_[Move::MAX_MOVES] *= invSumPolicy;
    else
        policy_[Move::MAX_MOVES] = 0;  // Ensure pass move is zero if not computed
}