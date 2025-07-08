#pragma once

#include "../eval.h"

class DummyEvaluator : public Evaluator
{
public:
    DummyEvaluator(int boardSize);
    virtual ~DummyEvaluator() = default;

    /// Resets the evaluator state to empty board.
    virtual void reset();
    /// Update hook called when put a new stone on board (a non-pass move).
    /// @param color The color of the stone to put. Must be C_BLACK or C_WHITE or C_GAP.
    /// @param x The x coordinate in [0, boardSize-1] of the stone to put.
    /// @param y The y coordinate in [0, boardSize-1] of the stone to put.
    virtual void move(Color color, int x, int y);
    /// Update hook called when rollback a stone on board (a non-pass move).
    /// @param color The color of the stone we put before. Must be C_BLACK or C_WHITE or C_GAP.
    /// @param x The x coordinate in [0, boardSize-1] of the stone we put before.
    /// @param y The y coordinate in [0, boardSize-1] of the stone we put before.
    virtual void undo(Color color, int x, int y);

    /// Evaluates value for current side to move.
    virtual Value evaluateValue(Player side);
    /// Evaluates policy for current side to move.
    virtual void evaluatePolicy(Player side, PolicyBuffer &policyBuffer);

private:
    /// The board size of the game.
    int boardSize_;
};