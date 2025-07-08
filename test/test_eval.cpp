#include "eval.h"
#include "evaluator/factory.h"
#include "game.h"

#include <doctest/doctest.h>
#include <iostream>

TEST_CASE("Value")
{
    SUBCASE("Uniform Value")
    {
        Value v {0.0f, 0.0f, 0.0f, true};
        CHECK_EQ(v.winLossRate(), 0.0f);
        CHECK_EQ(v.winningRate(), 0.5f);
        CHECK_EQ(v.winProb(), 1.0f / 3.0f);
        CHECK_EQ(v.lossProb(), 1.0f / 3.0f);
        CHECK_EQ(v.drawRate(), 1.0f / 3.0f);
        CHECK_EQ(v.eval(), 0);
        std::cout << v << std::endl;
    }

    SUBCASE("Zero Eval")
    {
        Value v {Eval(0)};
        CHECK_EQ(v.winLossRate(), 0.0f);
        CHECK_EQ(v.winningRate(), 0.5f);
        CHECK_EQ(v.winProb(), 0.5f);
        CHECK_EQ(v.lossProb(), 0.5f);
        CHECK_EQ(v.drawRate(), 0.0f);
        CHECK_EQ(v.eval(), 0);
        std::cout << v << std::endl;
    }
}

TEST_CASE("PolicyBuffer") {}

TEST_CASE("Dummy Evaluator")
{
    auto evaluator = CreateEvaluator(19, 0);
    CHECK(evaluator != nullptr);

    State initState {19};
    initState.reset();

    SUBCASE("Reset")
    {
        State state(initState, evaluator.get());
        state.reset();

        SUBCASE("Value")
        {
            Value v = evaluator->evaluateValue(state.currentSide());
            CHECK_EQ(v.winProb(), 1.0f / 3.0f);
            CHECK_EQ(v.lossProb(), 1.0f / 3.0f);
            CHECK_EQ(v.drawRate(), 1.0f / 3.0f);
            CHECK_EQ(v.winLossRate(), 0.0f);
            CHECK_EQ(v.winningRate(), 0.5f);
            CHECK_EQ(v.eval(), 0);
        }

        SUBCASE("Policy without Pass")
        {
            PolicyBuffer policyBuffer(19);
            for (Move move : state.getLegalMoves(false))
                policyBuffer.setComputeFlag(move);

            evaluator->evaluatePolicy(state.currentSide(), policyBuffer);
            policyBuffer.applySoftmax();

            float sum = 0.0f;
            for (int y = 0; y < state.boardSize(); ++y) {
                for (int x = 0; x < state.boardSize(); ++x) {
                    Move move {x, y};

                    float prob = policyBuffer(move);
                    sum += prob;
                    CHECK(prob >= 0.0f);
                    CHECK(prob <= 1.0f);
                }
            }
            CHECK(sum == doctest::Approx(1.0f));
        }

        SUBCASE("Policy with Pass")
        {
            PolicyBuffer policyBuffer(19);
            for (Move move : state.getLegalMoves(true))
                policyBuffer.setComputeFlag(move);

            evaluator->evaluatePolicy(state.currentSide(), policyBuffer);
            policyBuffer.applySoftmax();

            float sum = 0.0f;
            for (int y = 0; y < state.boardSize(); ++y) {
                for (int x = 0; x < state.boardSize(); ++x) {
                    Move move {x, y};

                    float prob = policyBuffer(move);
                    sum += prob;
                    CHECK(prob >= 0.0f);
                    CHECK(prob <= 1.0f);
                }
            }
            float prob = policyBuffer(Move::Pass);
            sum += prob;
            CHECK(prob >= 0.0f);
            CHECK(prob <= 1.0f);
            CHECK(sum == doctest::Approx(1.0f));
        }

        std::cout << state << std::endl;
    }
}