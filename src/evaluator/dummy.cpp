#include "dummy.h"

#include <limits>

DummyEvaluator::DummyEvaluator(int boardSize) : boardSize_(boardSize) {}

void DummyEvaluator::reset() {}

void DummyEvaluator::move(Color color, int x, int y) {}

void DummyEvaluator::undo(Color color, int x, int y) {}

Value DummyEvaluator::evaluateValue(Player side)
{
    // A dummy value with equal win, loss, and draw probabilities
    return Value {0.0f, 0.0f, 0.0f, true};
}

void DummyEvaluator::evaluatePolicy(Player side, PolicyBuffer &policyBuffer)
{
    // Fill the policy buffer with uniform distribution
    for (int y = 0; y < boardSize_; ++y)
        for (int x = 0; x < boardSize_; ++x)
            policyBuffer(Move {x, y}) = 0.0f;
    policyBuffer(Move::Pass) = 0.0f;
}