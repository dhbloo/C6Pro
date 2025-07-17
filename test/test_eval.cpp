#include "eval.h"
#include "evaluator/factory.h"
#include "game.h"

#include <doctest/doctest.h>
#include <print>

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
        std::println("Uniform Value: {}", v);
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
        std::println("Zero Eval: {}", v);
    }
}

TEST_CASE("PolicyBuffer")
{
    PolicyBuffer policy_buffer {19};

    SUBCASE("Initial PolicyBuffer")
    {
        bool is_all_empty = true;
        for (int i = 0; i < 19; i++) {
            for (int j = 0; j < 19; j++) {
                if (policy_buffer.getComputeFlag(Move {i, j})) {
                    is_all_empty = false;
                }
            }
        }
        CHECK_EQ(is_all_empty, true);
    }

    SUBCASE("Add score to PolicyBuffer")
    {
        // set (10, 10), (9, 10), (12, 11)
        policy_buffer(Move {10, 10}) = 3.0;
        policy_buffer.setComputeFlag(Move {10, 10});
        policy_buffer(Move {9, 10}) = 2.0;
        policy_buffer.setComputeFlag(Move {9, 10});
        policy_buffer(Move {12, 11}) = 1.0;
        policy_buffer.setComputeFlag(Move {12, 11});
        CHECK_EQ(policy_buffer(Move {10, 10}), 3.0);
        CHECK_EQ(policy_buffer.getComputeFlag(Move {10, 10}), true);
        CHECK_EQ(policy_buffer(Move(9, 10)), 2.0);
        CHECK_EQ(policy_buffer.getComputeFlag(Move {9, 10}), true);
        CHECK_EQ(policy_buffer(Move(12, 11)), 1.0);
        CHECK_EQ(policy_buffer.getComputeFlag(Move {12, 11}), true);
    }

    SUBCASE("Apply Softmax")
    {
        policy_buffer(Move {10, 10}) = 3.0;
        policy_buffer.setComputeFlag(Move {10, 10});
        policy_buffer(Move {9, 10}) = 2.0;
        policy_buffer.setComputeFlag(Move {9, 10});
        policy_buffer(Move {12, 11}) = 1.0;
        policy_buffer.setComputeFlag(Move {12, 11});
        policy_buffer.applySoftmax();
        float sum = 0.0f;
        for (int i = 0; i < 19; i++) {
            for (int j = 0; j < 19; j++) {
                if (policy_buffer.getComputeFlag(Move {i, j})) {
                    sum += policy_buffer(Move {i, j});
                }
            }
        }
        std::println("Policy Buffer (10, 10): {}", policy_buffer(Move {10, 10}));
        std::println("Policy Buffer (9, 10): {}", policy_buffer(Move {9, 10}));
        std::println("Policy Buffer (12, 11): {}", policy_buffer(Move {12, 11}));
        CHECK_EQ(sum, 1.0f);
    }
}

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

        SUBCASE("Tracing")
        {
            std::println("Initial state with evaluator:\n{:trace}", state);
        }

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
    }
}