#include "game.h"

#include "eval.h"
#include "utils.h"

#include <cassert>
#include <iomanip>
#include <sstream>

namespace {

/// Check if there are at least 5 consecutive one bits in the bitkey.
bool check6(const uint32_t (&bitKeys)[2], Player side)
{
    uint32_t bitKey = ~bitKeys[side] & bitKeys[~side];
    uint32_t mask =
        bitKey & (bitKey << 1) & (bitKey << 2) & (bitKey << 3) & (bitKey << 4) & (bitKey << 5);
    return mask != 0;
}

}  // namespace

// --------------------------------------------------------

namespace ZobristHash {

uint64_t zobristInit;
uint64_t zobristCell[2][Move::MAX_MOVES + 1];
uint64_t zobristSide[2];

const auto initZobrish = []() {
    PRNG prng(0);

    zobristInit = prng();
    for (int i = 0; i < Move::MAX_MOVES + 1; i++) {
        zobristCell[0][i] = prng();
        zobristCell[1][i] = prng();
    }
    zobristSide[0] = prng();
    zobristSide[1] = prng();

    return true;
}();

};  // namespace ZobristHash

// --------------------------------------------------------

State::State(const int boardSize)
    : boardSize_ {boardSize}
    , moveCount_ {0}
    , passCount_ {0, 0}
    , curSide_ {P_FIRST}
    , curHash_ {0}
    , evaluator_ {nullptr}
{
    history_ = new Move[1 + 2 * boardSize_ * boardSize_];

    reset();
}

State::State(const State &other, Evaluator *evaluator)
    : boardSize_ {other.boardSize_}
    , moveCount_ {other.moveCount_}
    , passCount_ {other.passCount_[P_FIRST], other.passCount_[P_SECOND]}
    , curSide_ {other.curSide_}
    , curHash_ {other.curHash_}
    , evaluator_ {evaluator}
{
    history_ = new Move[1 + 2 * boardSize_ * boardSize_];

    // Directly copy the state data
    std::memcpy(bitKey0_, other.bitKey0_, sizeof(bitKey0_));
    std::memcpy(bitKey1_, other.bitKey1_, sizeof(bitKey1_));
    std::memcpy(bitKey2_, other.bitKey2_, sizeof(bitKey2_));
    std::memcpy(bitKey3_, other.bitKey3_, sizeof(bitKey3_));
    std::memcpy(history_, other.history_, sizeof(Move) * (1 + 2 * boardSize_ * boardSize_));

    // Sync the evaluator with the board state
    if (evaluator_) {
        evaluator_->reset();

        Player side = P_FIRST;
        for (int i = 0; i < moveCount_; i++) {
            Move move = history_[i];
            if (!move.isPass())
                evaluator_->move(static_cast<Color>(side), move.x(), move.y());
            side = ~side;
        }
    }
}

State::~State()
{
    delete[] history_;
}

void State::reset()
{
    std::memset(bitKey0_, 0, sizeof(bitKey0_));
    std::memset(bitKey1_, 0, sizeof(bitKey1_));
    std::memset(bitKey2_, 0, sizeof(bitKey2_));
    std::memset(bitKey3_, 0, sizeof(bitKey3_));

    moveCount_          = 0;
    passCount_[C_BLACK] = 0;
    passCount_[C_WHITE] = 0;
    curSide_            = P_FIRST;
    curHash_            = ZobristHash::zobristInit;

    for (int y = 0; y < boardSize_; y++) {
        for (int x = 0; x < boardSize_; x++) {
            flipBit(C_BLACK, x, y);
            flipBit(C_WHITE, x, y);
        }
    }

    std::memset(history_, 0, sizeof(Move) * (1 + 2 * boardSize_ * boardSize_));

    if (evaluator_)
        evaluator_->reset();
}

void State::move(Move move)
{
    assert(isLegal(move));

    if (move.isPass()) {
        assert(passMoveCount(P_FIRST) + passMoveCount(P_SECOND) < boardSize_ * boardSize_);
        passCount_[curSide_]++;
    }
    else {
        const int x = move.x();
        const int y = move.y();
        assert(getColorAt(x, y) == C_EMPTY);

        flipBit(static_cast<Color>(curSide_), x, y);
        curHash_ ^= ZobristHash::zobristCell[curSide_][static_cast<int>(move)];

        if (evaluator_)
            evaluator_->move(static_cast<Color>(curSide_), x, y);
    }

    history_[moveCount_++] = move;
    curSide_               = moveCount_ % 2 ? ~curSide_ : curSide_;
}

void State::undo()
{
    Move lastMove = getRecentMove();
    assert(lastMove);

    curSide_ = moveCount_ % 2 ? ~curSide_ : curSide_;
    moveCount_--;

    if (lastMove.isPass()) {
        passCount_[curSide_]--;
    }
    else {
        const int x = lastMove.x();
        const int y = lastMove.y();
        flipBit(static_cast<Color>(curSide_), x, y);
        curHash_ ^= ZobristHash::zobristCell[curSide_][static_cast<int>(lastMove)];

        if (evaluator_)
            evaluator_->undo(static_cast<Color>(curSide_), x, y);
    }
}

int State::nonPassMoveCount() const
{
    return moveCount_ - passCount_[P_FIRST] - passCount_[P_SECOND];
}

int State::legalMoveCount(bool includePass) const
{
    int boardCells = boardSize_ * boardSize_;
    int stones     = nonPassMoveCount();
    return boardCells - stones + (includePass ? 1 : 0);
}

uint64_t State::hash() const
{
    return curHash_ ^ ZobristHash::zobristSide[curSide_];
}

uint64_t State::hashAfter(Move move) const
{
    assert(isLegal(move));
    return curHash_ ^ ZobristHash::zobristSide[curSide_]
           ^ (move.isPass() ? uint64_t {}
                            : ZobristHash::zobristCell[curSide_][static_cast<int>(move)]);
}

Eval State::terminalEval() const
{
    Player    lastSide = moveCount_ % 2 ? ~curSide_ : curSide_;
    Move      lastMove = getRecentMove();
    const int x        = lastMove.x();
    const int y        = lastMove.y();

    // Check if the last move forms a winning pattern
    if (check6(bitKey0_[y], lastSide)              // horizontal
        || check6(bitKey1_[x], lastSide)           // vertical
        || check6(bitKey2_[x + y], lastSide)       // anti-diagonal
        || check6(bitKey3_[31 - x + y], lastSide)  // diagonal
    )
        return curSide_ == lastSide ? MateIn(0) : MatedIn(0);

    // Check if the board is full
    if (nonPassMoveCount() == boardSize_ * boardSize_)
        return EVAL_DRAW;

    return EVAL_NONE;
}

bool State::isLegal(Move move) const
{
    return move && (move.isPass() || getColorAt(move.x(), move.y()) == C_EMPTY);
}

Move State::getHistoryMove(int index) const
{
    assert(0 <= index && index < moveCount_);
    return history_[index];
}

Move State::getRecentMove(int steps) const
{
    assert(0 <= steps);
    return steps >= moveCount_ ? Move::None : history_[moveCount_ - 1 - steps];
}

std::vector<Move> State::getLegalMoves(bool includePass) const
{
    std::vector<Move> moveList;
    moveList.reserve(legalMoveCount(includePass));

    for (int y = 0; y < boardSize_; y++) {
        uint32_t emptyBitKey = bitKey0_[y][C_BLACK] & bitKey0_[y][C_WHITE];
        while (emptyBitKey) {
            int x = PopLSB(emptyBitKey);
            moveList.push_back(Move(x, y));
        }
    }

    if (includePass)
        moveList.push_back(Move::Pass);

    return moveList;
}

Color State::getColorAt(int x, int y) const
{
    if (x < 0 || x >= boardSize_ || y < 0 || y >= boardSize_)
        return C_GAP;

    uint32_t blackBit = (bitKey0_[y][C_BLACK] >> x) & 1;
    uint32_t whiteBit = (bitKey0_[y][C_WHITE] >> x) & 1;
    switch (blackBit | (whiteBit << 1)) {
    default: return C_GAP;
    case 0b01: return C_WHITE;
    case 0b10: return C_BLACK;
    case 0b11: return C_EMPTY;
    }
}

void State::flipBit(Color c, int x, int y)
{
    bitKey0_[y][c] ^= 1 << x;
    bitKey1_[x][c] ^= 1 << y;
    bitKey2_[x + y][c] ^= 1 << x;
    bitKey3_[31 - x + y][c] ^= 1 << x;
}

std::string State::positionString() const
{
    std::string position;
    for (int i = 0; i < moveCount_; i++) {
        Move move = history_[i];
        std::format_to(std::back_inserter(position), "{}", move);
    }
    return position;
}

std::string State::traceString() const
{
    std::string result;
    result.reserve(2048);
    auto out = std::back_inserter(result);

    std::format_to(out, "Position [{}]\n", positionString());
    std::format_to(out, "Hash: {:x}\n", hash());
    std::format_to(out, "MoveCount: {}\n", moveCount());
    std::format_to(out, "NonPassMoveCount: {}\n", nonPassMoveCount());
    std::format_to(out,
                   "PassMoveCount: First={} Second={}\n",
                   passMoveCount(P_FIRST),
                   passMoveCount(P_SECOND));
    std::format_to(out, "CurrentSide: {} ({})\n", currentSide(), static_cast<Color>(currentSide()));
    std::format_to(out, "LastMove: {}\n", getRecentMove());
    std::format_to(out, "TerminalEval: {}\n", terminalEval());

    std::format_to(out, "Board:\n");
    for (int y = boardSize_ - 1; y >= 0; y--) {
        std::format_to(out, "{:2} ", y + 1);
        for (int x = 0; x < boardSize_; x++)
            std::format_to(out, " {}", getColorAt(x, y));
        std::format_to(out, "\n");
    }
    std::format_to(out, "   ");
    for (int x = 0; x < boardSize_; x++) {
        std::format_to(out, " {:c}", 'A' + x);
    }
    std::format_to(out, "\n");

    if (evaluator_) {
        PolicyBuffer policyBuf(boardSize_);
        for (Move move : getLegalMoves(false))
            policyBuf.setComputeFlag(move);

        // Calculate policy for the current side
        evaluator_->evaluatePolicy(currentSide(), policyBuf);

        // Output the raw policy logits
        std::format_to(out, "Policy (logits):  Pass={:.0f}\n", 100 * policyBuf(Move::Pass));
        for (int y = boardSize_ - 1; y >= 0; y--) {
            std::format_to(out, "{:2} ", y + 1);
            for (int x = 0; x < boardSize_; x++)
                std::format_to(out, "{:6.0f}", 100 * policyBuf(Move {x, y}));
            std::format_to(out, "\n");
        }
        std::format_to(out, "   ");
        for (int x = 0; x < boardSize_; x++) {
            std::format_to(out, "{:>6c}", 'A' + x);
        }
        std::format_to(out, "\n");

        // Output the policy after softmax
        policyBuf.applySoftmax();
        std::format_to(out, "Policy (softmaxed):  Pass={:.0f}\n", 100 * policyBuf(Move::Pass));
        for (int y = boardSize_ - 1; y >= 0; y--) {
            std::format_to(out, "{:2} ", y + 1);
            for (int x = 0; x < boardSize_; x++)
                std::format_to(out, "{:6.0f}", 100 * policyBuf(Move {x, y}));
            std::format_to(out, "\n");
        }
        std::format_to(out, "   ");
        for (int x = 0; x < boardSize_; x++) {
            std::format_to(out, "{:>6c}", 'A' + x);
        }
        std::format_to(out, "\n");

        // Output the value for the current side
        Value value = evaluator_->evaluateValue(currentSide());
        std::format_to(out, "Value:\n");
        std::format_to(out, "WinProb: {:.4f}\n", value.winProb());
        std::format_to(out, "LossProb: {:.4f}\n", value.lossProb());
        std::format_to(out, "DrawProb: {:.4f}\n", value.drawRate());
        std::format_to(out, "WinLossRate: {:.4f}\n", value.winLossRate());
        std::format_to(out, "WinningRate: {:.4f}\n", value.winningRate());
        std::format_to(out, "Eval: {}\n", value.eval());
    }

    return result;
}