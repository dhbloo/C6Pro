#include "game.h"

#include <doctest/doctest.h>
#include <iostream>

TEST_CASE("Game State 19x19")
{
    State state {19};
    CHECK_EQ(state.boardSize(), 19);
    state.reset();

    SUBCASE("Initial State")
    {
        CHECK_EQ(state.currentSide(), P_FIRST);
        CHECK_EQ(state.moveCount(), 0);
        CHECK_EQ(state.passMoveCount(P_FIRST), 0);
        CHECK_EQ(state.passMoveCount(P_SECOND), 0);
        CHECK_EQ(state.nonPassMoveCount(), 0);
        CHECK_EQ(state.legalMoveCount(false), 361);  // 19x19 board
        CHECK_EQ(state.legalMoveCount(true), 362);   // 19x19 board + 1 pass
        CHECK_EQ(state.terminalEval(), EVAL_NONE);
        std::cout << state << std::endl;
    }

    SUBCASE("Make Moves")
    {
        state.move(Move {5, 5});
        CHECK_EQ(state.currentSide(), P_SECOND);
        CHECK_EQ(state.moveCount(), 1);
        CHECK_EQ(state.passMoveCount(P_FIRST), 0);
        CHECK_EQ(state.passMoveCount(P_SECOND), 0);
        CHECK_EQ(state.nonPassMoveCount(), 1);
        CHECK_EQ(state.legalMoveCount(false), 360);  // One less legal move
        CHECK_EQ(state.legalMoveCount(true), 361);   // One less legal move + 1 pass
        CHECK_EQ(state.terminalEval(), EVAL_NONE);
        std::cout << state << std::endl;

        state.move(Move {6, 5});
        CHECK_EQ(state.currentSide(), P_SECOND);
        CHECK_EQ(state.moveCount(), 2);
        CHECK_EQ(state.passMoveCount(P_FIRST), 0);
        CHECK_EQ(state.passMoveCount(P_SECOND), 0);
        CHECK_EQ(state.nonPassMoveCount(), 2);
        CHECK_EQ(state.legalMoveCount(false), 359);  // Two less legal moves
        CHECK_EQ(state.legalMoveCount(true), 360);   // Two less legal moves + 1 pass
        CHECK_EQ(state.terminalEval(), EVAL_NONE);
        std::cout << state << std::endl;

        state.move(Move {6, 6});
        CHECK_EQ(state.currentSide(), P_FIRST);
        CHECK_EQ(state.moveCount(), 3);
        CHECK_EQ(state.passMoveCount(P_FIRST), 0);
        CHECK_EQ(state.passMoveCount(P_SECOND), 0);
        CHECK_EQ(state.nonPassMoveCount(), 3);
        CHECK_EQ(state.legalMoveCount(false), 358);  // Three less legal moves
        CHECK_EQ(state.legalMoveCount(true), 359);   // Three less legal moves
        CHECK_EQ(state.terminalEval(), EVAL_NONE);
        std::cout << state << std::endl;

        state.move(Move {5, 6});
        CHECK_EQ(state.currentSide(), P_FIRST);
        CHECK_EQ(state.moveCount(), 4);
        CHECK_EQ(state.passMoveCount(P_FIRST), 0);
        CHECK_EQ(state.passMoveCount(P_SECOND), 0);
        CHECK_EQ(state.nonPassMoveCount(), 4);
        CHECK_EQ(state.legalMoveCount(false), 357);  // Four less legal moves
        CHECK_EQ(state.legalMoveCount(true), 358);   // Four less legal moves
        CHECK_EQ(state.terminalEval(), EVAL_NONE);
        std::cout << state << std::endl;

        state.move(Move {5, 7});
        CHECK_EQ(state.currentSide(), P_SECOND);
        CHECK_EQ(state.moveCount(), 5);
        CHECK_EQ(state.passMoveCount(P_FIRST), 0);
        CHECK_EQ(state.passMoveCount(P_SECOND), 0);
        CHECK_EQ(state.nonPassMoveCount(), 5);
        CHECK_EQ(state.legalMoveCount(false), 356);  // Five less legal moves
        CHECK_EQ(state.legalMoveCount(true), 357);   // Five less legal moves
        CHECK_EQ(state.terminalEval(), EVAL_NONE);
        std::cout << state << std::endl;

        state.move(Move {6, 7});
        CHECK_EQ(state.currentSide(), P_SECOND);
        CHECK_EQ(state.moveCount(), 6);
        CHECK_EQ(state.passMoveCount(P_FIRST), 0);
        CHECK_EQ(state.passMoveCount(P_SECOND), 0);
        CHECK_EQ(state.nonPassMoveCount(), 6);
        CHECK_EQ(state.legalMoveCount(false), 355);  // Six less legal moves
        CHECK_EQ(state.legalMoveCount(true), 356);   // Six less legal moves
        CHECK_EQ(state.terminalEval(), EVAL_NONE);
        std::cout << state << std::endl;

        state.move(Move {6, 8});
        CHECK_EQ(state.currentSide(), P_FIRST);
        CHECK_EQ(state.moveCount(), 7);
        CHECK_EQ(state.passMoveCount(P_FIRST), 0);
        CHECK_EQ(state.passMoveCount(P_SECOND), 0);
        CHECK_EQ(state.nonPassMoveCount(), 7);
        CHECK_EQ(state.legalMoveCount(false), 354);  // Seven less legal moves
        CHECK_EQ(state.legalMoveCount(true), 355);   // Seven less legal moves
        CHECK_EQ(state.terminalEval(), EVAL_NONE);
        std::cout << state << std::endl;

        state.move(Move {6, 4});
        CHECK_EQ(state.currentSide(), P_FIRST);
        CHECK_EQ(state.moveCount(), 8);
        CHECK_EQ(state.passMoveCount(P_FIRST), 0);
        CHECK_EQ(state.passMoveCount(P_SECOND), 0);
        CHECK_EQ(state.nonPassMoveCount(), 8);
        CHECK_EQ(state.legalMoveCount(false), 353);  // Eight less legal moves
        CHECK_EQ(state.legalMoveCount(true), 354);   // Eight less legal moves
        CHECK_EQ(state.terminalEval(), EVAL_NONE);
        std::cout << state << std::endl;

        state.move(Move {6, 9});
        CHECK_EQ(state.currentSide(), P_SECOND);
        CHECK_EQ(state.moveCount(), 9);
        CHECK_EQ(state.passMoveCount(P_FIRST), 0);
        CHECK_EQ(state.passMoveCount(P_SECOND), 0);
        CHECK_EQ(state.nonPassMoveCount(), 9);
        CHECK_EQ(state.legalMoveCount(false), 352);  // Nine less legal moves
        CHECK_EQ(state.legalMoveCount(true), 353);   // Nine less legal moves
        CHECK_EQ(state.terminalEval(), EVAL_NONE);
        std::cout << state << std::endl;
    }

    SUBCASE("Pass Moves")
    {
        state.move(Move {5, 5});
        state.move(Move {6, 5});
        state.move(Move {6, 6});
        state.move(Move {5, 6});
        state.move(Move {5, 7});
        state.move(Move {6, 7});
        state.move(Move {6, 8});
        state.move(Move {6, 4});
        state.move(Move {6, 9});

        state.move(Move::Pass);
        CHECK_EQ(state.currentSide(), P_SECOND);
        CHECK_EQ(state.moveCount(), 10);
        CHECK_EQ(state.passMoveCount(P_FIRST), 0);
        CHECK_EQ(state.passMoveCount(P_SECOND), 1);
        CHECK_EQ(state.nonPassMoveCount(), 9);
        CHECK_EQ(state.legalMoveCount(false), 352);  // Still nine less legal moves
        CHECK_EQ(state.legalMoveCount(true), 353);   // Still nine less legal moves + 1 pass
        CHECK_EQ(state.terminalEval(), EVAL_NONE);

        state.move(Move::Pass);
        CHECK_EQ(state.currentSide(), P_FIRST);
        CHECK_EQ(state.moveCount(), 11);
        CHECK_EQ(state.passMoveCount(P_FIRST), 0);
        CHECK_EQ(state.passMoveCount(P_SECOND), 2);
        CHECK_EQ(state.nonPassMoveCount(), 9);
        CHECK_EQ(state.legalMoveCount(false), 352);  // Still nine less legal moves
        CHECK_EQ(state.legalMoveCount(true), 353);   // Still nine less legal moves + 1 pass
        CHECK_EQ(state.terminalEval(), EVAL_NONE);

        state.move(Move::Pass);
        CHECK_EQ(state.currentSide(), P_FIRST);
        CHECK_EQ(state.moveCount(), 12);
        CHECK_EQ(state.passMoveCount(P_FIRST), 1);
        CHECK_EQ(state.passMoveCount(P_SECOND), 2);
        CHECK_EQ(state.nonPassMoveCount(), 9);
        CHECK_EQ(state.legalMoveCount(false), 352);  // Still nine less legal
        CHECK_EQ(state.legalMoveCount(true), 353);   // Still nine less legal moves + 1 pass
        CHECK_EQ(state.terminalEval(), EVAL_NONE);

        state.move(Move {5, 4});
        CHECK_EQ(state.currentSide(), P_SECOND);
        CHECK_EQ(state.moveCount(), 13);
        CHECK_EQ(state.passMoveCount(P_FIRST), 1);
        CHECK_EQ(state.passMoveCount(P_SECOND), 2);
        CHECK_EQ(state.nonPassMoveCount(), 10);
        CHECK_EQ(state.legalMoveCount(false), 351);
        CHECK_EQ(state.legalMoveCount(true), 352);
        CHECK_EQ(state.terminalEval(), EVAL_NONE);

        state.move(Move::Pass);
        CHECK_EQ(state.currentSide(), P_SECOND);
        CHECK_EQ(state.moveCount(), 14);
        CHECK_EQ(state.passMoveCount(P_FIRST), 1);
        CHECK_EQ(state.passMoveCount(P_SECOND), 3);
        CHECK_EQ(state.nonPassMoveCount(), 10);
        CHECK_EQ(state.legalMoveCount(false), 351);
        CHECK_EQ(state.legalMoveCount(true), 352);
        CHECK_EQ(state.terminalEval(), EVAL_NONE);

        state.move(Move::Pass);
        CHECK_EQ(state.currentSide(), P_FIRST);
        CHECK_EQ(state.moveCount(), 15);
        CHECK_EQ(state.passMoveCount(P_FIRST), 1);
        CHECK_EQ(state.passMoveCount(P_SECOND), 4);
        CHECK_EQ(state.nonPassMoveCount(), 10);
        CHECK_EQ(state.legalMoveCount(false), 351);
        CHECK_EQ(state.legalMoveCount(true), 352);
        CHECK_EQ(state.terminalEval(), EVAL_NONE);
        std::cout << state << std::endl;
    }

    SUBCASE("Win Check 1")
    {
        state.move(Move {5, 5});
        state.move(Move {6, 5});
        state.move(Move {6, 6});
        state.move(Move {5, 6});
        state.move(Move {5, 7});
        state.move(Move {6, 7});
        state.move(Move {6, 8});
        state.move(Move {6, 4});
        state.move(Move {6, 9});
        state.move(Move::Pass);
        state.move(Move::Pass);
        state.move(Move::Pass);
        state.move(Move {5, 4});
        state.move(Move::Pass);
        state.move(Move::Pass);

        state.move(Move {5, 3});
        CHECK_EQ(state.currentSide(), P_FIRST);
        CHECK_EQ(state.moveCount(), 16);
        CHECK_EQ(state.passMoveCount(P_FIRST), 1);
        CHECK_EQ(state.passMoveCount(P_SECOND), 4);
        CHECK_EQ(state.nonPassMoveCount(), 11);
        CHECK_EQ(state.legalMoveCount(false), 350);
        CHECK_EQ(state.legalMoveCount(true), 351);
        CHECK_EQ(state.terminalEval(), EVAL_NONE);

        state.move(Move {5, 8});
        CHECK_EQ(state.currentSide(), P_SECOND);
        CHECK_EQ(state.moveCount(), 17);
        CHECK_EQ(state.passMoveCount(P_FIRST), 1);
        CHECK_EQ(state.passMoveCount(P_SECOND), 4);
        CHECK_EQ(state.nonPassMoveCount(), 12);
        CHECK_EQ(state.legalMoveCount(false), 349);
        CHECK_EQ(state.legalMoveCount(true), 350);
        CHECK_EQ(state.terminalEval(), MatedIn(0));
        std::cout << state << std::endl;
    }

    SUBCASE("Win Check 2")
    {
        state.move(Move {5, 5});
        state.move(Move {6, 5});
        state.move(Move {6, 6});
        state.move(Move {5, 6});
        state.move(Move {5, 7});
        state.move(Move {6, 7});
        state.move(Move {6, 8});
        state.move(Move {6, 4});
        state.move(Move {6, 9});
        state.move(Move::Pass);
        state.move(Move::Pass);
        state.move(Move::Pass);
        state.move(Move {5, 4});
        state.move(Move::Pass);
        state.move(Move::Pass);

        state.move(Move {5, 3});
        state.move(Move::Pass);
        CHECK_EQ(state.currentSide(), P_SECOND);
        CHECK_EQ(state.moveCount(), 17);
        CHECK_EQ(state.passMoveCount(P_FIRST), 2);
        CHECK_EQ(state.passMoveCount(P_SECOND), 4);
        CHECK_EQ(state.nonPassMoveCount(), 11);
        CHECK_EQ(state.legalMoveCount(false), 350);
        CHECK_EQ(state.legalMoveCount(true), 351);
        CHECK_EQ(state.terminalEval(), EVAL_NONE);

        state.move(Move::Pass);
        state.move(Move::Pass);
        CHECK_EQ(state.currentSide(), P_FIRST);
        CHECK_EQ(state.moveCount(), 19);
        CHECK_EQ(state.passMoveCount(P_FIRST), 2);
        CHECK_EQ(state.passMoveCount(P_SECOND), 6);
        CHECK_EQ(state.nonPassMoveCount(), 11);
        CHECK_EQ(state.legalMoveCount(false), 350);
        CHECK_EQ(state.legalMoveCount(true), 351);
        CHECK_EQ(state.terminalEval(), EVAL_NONE);

        state.move(Move {5, 8});
        CHECK_EQ(state.currentSide(), P_FIRST);
        CHECK_EQ(state.moveCount(), 20);
        CHECK_EQ(state.passMoveCount(P_FIRST), 2);
        CHECK_EQ(state.passMoveCount(P_SECOND), 6);
        CHECK_EQ(state.nonPassMoveCount(), 12);
        CHECK_EQ(state.legalMoveCount(false), 349);
        CHECK_EQ(state.legalMoveCount(true), 350);
        CHECK_EQ(state.terminalEval(), MateIn(0));
        std::cout << state << std::endl;
    }

    SUBCASE("History and recent moves") {}

    SUBCASE("Legal moves") {}
}