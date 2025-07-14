#pragma once

#include <compare>
#include <cstdint>
#include <ostream>
#include <vector>

// --------------------------------------------------------

/// Player represents a side of the two-player game. (Index starts at zero)
/// Define SIDE_NB to be the total number of sides, which is two.
enum Player : uint8_t {
    P_FIRST  = 0,
    P_SECOND = 1,
    SIDE_NB  = 2,  // Two side of players
};

/// Get the next player after the full turn of the current player.
/// Since we only support two-player games now, the next player is simply the opposite side.
constexpr Player operator~(const Player p)
{
    return static_cast<Player>(1 - p);
}

/// Stringify the player to the stream.
std::ostream &operator<<(std::ostream &os, const Player p);

// --------------------------------------------------------

/// Eval represents the evaluation of the game position, stored as a signed 16-bit integer.
/// It can be used for accurately representing the result of terminal node.
/// It can also store a converted value from probabilistic evaluation.
/// The value is in the range [-30002, 30000], where:
///     - 0 means a draw,
///     - (30000 - x) means mate in the x steps,
///     - (-30000 + x) means mated in the x steps,
///     - (-30002) means no evaluation (not terminal).
enum Eval : int16_t {
    EVAL_DRAW             = 0,
    EVAL_MATE             = 30000,
    EVAL_INFINITE         = 30001,
    EVAL_NONE             = -30002,
    EVAL_MATE_IN_MAX_PLY  = EVAL_MATE - 1000,
    EVAL_MATED_IN_MAX_PLY = -EVAL_MATE + 1000,
    EVAL_MIN              = -8000,
    EVAL_MAX              = 8000,
};

constexpr Eval operator+(const Eval d1, const Eval d2)
{
    return static_cast<Eval>(static_cast<int>(d1) + static_cast<int>(d2));
}
constexpr Eval operator-(const Eval d1, const Eval d2)
{
    return static_cast<Eval>(static_cast<int>(d1) - static_cast<int>(d2));
}
constexpr Eval operator-(const Eval d)
{
    return static_cast<Eval>(-static_cast<int>(d));
}
inline Eval &operator+=(Eval &d1, const Eval d2)
{
    return d1 = d1 + d2;
}
inline Eval &operator-=(Eval &d1, const Eval d2)
{
    return d1 = d1 - d2;
}
constexpr Eval operator*(const int i, const Eval d)
{
    return static_cast<Eval>(i * static_cast<int>(d));
}
constexpr Eval operator*(const Eval d, const int i)
{
    return static_cast<Eval>(static_cast<int>(d) * i);
}
constexpr Eval operator/(const Eval d, const int i)
{
    return static_cast<Eval>(static_cast<int>(d) / i);
}
constexpr int operator/(const Eval d1, const Eval d2)
{
    return static_cast<int>(d1) / static_cast<int>(d2);
}
inline Eval &operator*=(Eval &d, const int i)
{
    return d = static_cast<Eval>(static_cast<int>(d) * i);
}
inline Eval &operator/=(Eval &d, const int i)
{
    return d = static_cast<Eval>(static_cast<int>(d) / i);
}
constexpr Eval operator+(const Eval v, const int i)
{
    return static_cast<Eval>(static_cast<int>(v) + i);
}
constexpr Eval operator-(const Eval v, const int i)
{
    return static_cast<Eval>(static_cast<int>(v) - i);
}
inline Eval &operator+=(Eval &v, const int i)
{
    return v = v + i;
}
inline Eval &operator-=(Eval &v, const int i)
{
    return v = v - i;
}

/// Construct a value for mate in N ply.
constexpr Eval MateIn(const int ply)
{
    return static_cast<Eval>(static_cast<int>(EVAL_MATE) - ply);
}

/// Construct a value for being mated in N ply
constexpr Eval MatedIn(const int ply)
{
    return static_cast<Eval>(static_cast<int>(-EVAL_MATE) + ply);
}

/// Get number of steps to mate from value and current ply
constexpr int MateStep(const Eval v, const int ply)
{
    return EVAL_MATE - ply - (v < 0 ? -v : v);
}

/// Stringify the display eval to the stream.
std::ostream &operator<<(std::ostream &os, const Eval ev);

// --------------------------------------------------------

/// Move represents a game action. It is stored as a 16-bit integer.
/// The action encoding space may not be consecutive, and the actual legality
/// of a move shoule be checked by the game state.
class Move
{
public:
    /// Max board size we can use. We use uint32_t bitboard and each cell takes 1 bit.
    static constexpr int MAX_BOARD_SIZE = 32;
    /// Max board array size.
    static constexpr int MAX_BOARD_CELLS = MAX_BOARD_SIZE * MAX_BOARD_SIZE;
    /// The maximum possible moves on an empty board (including a Pass).
    static constexpr int MAX_MOVES = MAX_BOARD_CELLS + 1;

    /// Uninitialized move
    Move() = default;
    /// Construct a move from an integer. The integer should be in the range [0, MAX_MOVES+1].
    constexpr explicit Move(int16_t move) : move_(move) {}
    /// Construct a move at the given coordinate on the board.
    constexpr Move(int x, int y) : move_(static_cast<int16_t>((y << 5) | x) + 1) {}
    /// Default copy constructor.
    constexpr Move(const Move &) = default;
    /// Get the x coordinate of the move.
    constexpr int x() const { return (move_ - 1) & 31; }
    /// Get the y coordinate of the move.
    constexpr int y() const { return (move_ - 1) >> 5; }
    /// Get the move as an integer.
    constexpr operator int() const { return move_; }
    /// Check if the move is in the action space
    constexpr operator bool() const { return move_ > 0 && move_ <= MAX_MOVES; }
    /// Check if the move is a pass move.
    constexpr bool isPass() const { return move_ == MAX_MOVES; }
    /// Spaceship operator for comparing moves.
    constexpr auto operator<=>(const Move &) const = default;

    static const Move None;
    static const Move Pass;

private:
    int16_t move_;
};

inline constexpr Move Move::None {0};
inline constexpr Move Move::Pass {Move::MAX_MOVES};

/// Stringify the display move to the stream.
std::ostream &operator<<(std::ostream &os, const Move p);

// --------------------------------------------------------

/// Direction represents one of the eight line directions on the board
enum Direction : int16_t {
    UP    = -Move::MAX_BOARD_SIZE,
    LEFT  = -1,
    DOWN  = -UP,
    RIGHT = -LEFT,

    UP_LEFT    = UP + LEFT,
    UP_RIGHT   = UP + RIGHT,
    DOWN_LEFT  = DOWN + LEFT,
    DOWN_RIGHT = DOWN + RIGHT
};

constexpr Direction Directions[] = {RIGHT, DOWN, UP_RIGHT, DOWN_RIGHT};

// Additional operators for adding a Direction to a Move

constexpr Move operator+(Move p, Direction i)
{
    return Move(int(p) + i);
}
constexpr Move operator-(Move p, Direction i)
{
    return Move(int(p) - i);
}
inline Move &operator+=(Move &p, Direction i)
{
    return p = Move(int(p) + i);
}
inline Move &operator-=(Move &p, Direction i)
{
    return p = Move(int(p) - i);
}
constexpr Direction operator+(Direction d1, Direction d2)
{
    return Direction(int(d1) + int(d2));
}
constexpr Direction operator*(int i, Direction d)
{
    return Direction(i * int(d));
}
constexpr Direction operator*(Direction d, int i)
{
    return Direction(int(d) * i);
}

// --------------------------------------------------------

/// Color represents a type on cell state on board.
enum Color {
    C_BLACK  = 0,
    C_WHITE  = 1,
    C_EMPTY  = 2,
    C_GAP    = 3,
    COLOR_NB = 4,
};

/// Get the opposite color of the given color.
/// Black <-> White, Empty <-> Gap
constexpr Color operator~(const Color c)
{
    constexpr Color opponents[4] = {C_WHITE, C_BLACK, C_GAP, C_EMPTY};
    return opponents[c];
}

/// Stringify the color to the stream.
std::ostream &operator<<(std::ostream &out, const Color color);

/// State class represents a board position state.
/// For simplicity of implementation, it also records the whole move history info, to ease the
/// move() and undo() update methods. Copy constructor is set as explicit to avoid unintended
/// expensive copy operation.
class State
{
public:
    /// Creates a new empty board state with the given size.
    /// @param boardSize Size of the board, in range [1, Move::MAX_BOARD_SIZE].
    /// @param evaluator The evaluator object to bind with this state.
    explicit State(int boardSize);
    /// Clone a board state object from another state.
    /// @param other State object to clone from.
    /// @param evaluator The evaluator object to bind with this state.
    explicit State(const State &other, class Evaluator *evaluator = nullptr);
    /// Destructor.
    ~State();
    /// Copy assignment operator is deleted to avoid unintended copy.
    State &operator=(const State &other) = delete;

    // --------------------------------------------------------
    // State modifier

    /// Reset the state to an empty board and discards all history infos.
    void reset();
    /// Make move and incremental update the board state.
    /// @param move A legal move to put at the next step. Pass move is allowed.
    void move(Move move);
    /// Undo the last move and rollback the board state.
    void undo();

    // --------------------------------------------------------
    // Global state query

    /// Get the board size of the current state.
    [[nodiscard]] int boardSize() const { return boardSize_; }
    /// Get the current player side to do the next move.
    [[nodiscard]] Player currentSide() const { return curSide_; }
    /// Get the total number of moves (including pass) played so far.
    [[nodiscard]] int moveCount() const { return moveCount_; }
    /// Get the number of passes for the given side.
    /// @param side The side to query.
    /// @return The number of passes for the given side.
    [[nodiscard]] int passMoveCount(const Player side) const { return passCount_[side]; }
    /// Get the number of non-pass moves played so far.
    [[nodiscard]] int nonPassMoveCount() const;
    /// Get the number of legal moves (empty cells) for the current state.
    [[nodiscard]] int legalMoveCount(bool includePass = false) const;
    /// Get the current hash key of the board state.
    [[nodiscard]] uint64_t hash() const;
    /// Look ahead the hash key after the applying the given move to current state.
    /// @param move The move to apply.
    [[nodiscard]] uint64_t hashAfter(Move move) const;
    /// Check if the game state has reached a terminal state.
    /// @return True if the game is over, otherwise false.
    [[nodiscard]] bool isTerminal() const { return terminalEval() != EVAL_NONE; }
    /// Get the terminal eval of the current state, from the current side of view.
    /// @return The theoretical eval of the current board state, which can be one of
    ///     1. Mate  (eval >= EVAL_MATE_IN_MAX_PLY),
    ///     2. Mated (eval <= EVAL_MATED_IN_MAX_PLY),
    ///     3. Draw  (eval == EVAL_DRAW),
    ///     4. Non terminal (eval == EVAL_NONE).
    [[nodiscard]] Eval terminalEval() const;
    /// Get the evaluator object bound to this state.
    /// @return The evaluator object bound to this state, or nullptr if not set.
    [[nodiscard]] class Evaluator *evaluator() const { return evaluator_; }

    // --------------------------------------------------------
    // Move and history query

    /// Check if the given move is legal at the current board state.
    /// A legal move should be on an empty cell or a pass move.
    /// @param move The move to check.
    /// @return True if the move is legal, otherwise false.
    [[nodiscard]] bool isLegal(Move move) const;
    /// Get the history move at the given move index.
    /// @param index The move index to query, in range [0, moveCount()).
    /// @return The move at the given index.
    [[nodiscard]] Move getHistoryMove(int index) const;
    /// Get the recent move played at the given steps before.
    /// @param steps The steps to look back, in range [0, moveCount()).
    ///     0 for the last move, 1 for the move before last move, and so on.
    /// @return The move at the given steps before.
    [[nodiscard]] Move getRecentMove(int steps = 0) const;

    // --------------------------------------------------------
    // Legal Move generator

    /// Get all legal moves at the current board state.
    /// @param includePass If true, include the pass move in the result.
    /// @return A (unsorted) list of all legal moves.
    [[nodiscard]] std::vector<Move> getLegalMoves(bool includePass = false) const;

private:
    // Bitkeys of 4 directions and black/white colors. (bit=1 for empty, bit=0 for occupied)
    uint32_t bitKey0_[Move::MAX_BOARD_SIZE][2] {};          // [RIGHT(MSB) - LEFT(LSB)]
    uint32_t bitKey1_[Move::MAX_BOARD_SIZE][2] {};          // [DOWN(MSB) - UP(LSB)]
    uint32_t bitKey2_[Move::MAX_BOARD_SIZE * 2 - 1][2] {};  // [UP_RIGHT(MSB) - DOWN_LEFT(LSB)]
    uint32_t bitKey3_[Move::MAX_BOARD_SIZE * 2 - 1][2] {};  // [DOWN_RIGHT(MSB) - UP_LEFT(LSB)]

    int              boardSize_;           // The actual board size
    int              moveCount_;           // The total number of moves played so far
    int              passCount_[SIDE_NB];  // Number of passes for both sides
    Player           curSide_;             // The current side to move
    uint64_t         curHash_;             // The current zobrist hash
    Move            *history_;             // The move history array
    class Evaluator *evaluator_;           // The evaluator object bound to this state

    /// Get the color at the given position.
    Color getColorAt(int x, int y) const;
    /// Flip the bit of the given color at the given position.
    void flipBit(Color c, int x, int y);
    /// Output the state for debugging
    friend std::ostream &operator<<(std::ostream &out, const State &state);
};
