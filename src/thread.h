#pragma once

#include "search.h"
#include "utils.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <thread>

class State;       // forward declaration
class Evaluator;   // forward declaration
class ThreadPool;  // forward declaration

/// SearchThread is the worker thread that performs the search. It contains all
/// per-thread state, including a game state, an evaluator, and all search variables.
class SearchThread
{
public:
    /// Instantiate a new search thread.
    /// @param sss The shared search state that is used by all threads.
    /// @param threadId The unique ID of this thread.
    SearchThread(SharedSearchState &sss, uint32_t threadId);
    /// Destory this search thread. Search must be stopped before entering.
    virtual ~SearchThread();
    /// Launch a custom task in this thread.
    void runTask(std::function<void(SearchThread &)> task);
    /// Wait until threadLoop() enters idle state.
    void waitForIdle();
    /// Get the ID of this thread.
    uint32_t id() const { return id_; }
    /// Return if this thread is the main thread (the first thread).
    bool isMainThread() const { return id_ == 0; }
    /// Clear the thread state between two search.
    virtual void clear(bool newGame);
    /// The search function that every thread will run.
    virtual void search();
    /// Setup the game state in this thread, and update the evaluator.
    virtual void setStateAndEvaluator(const State &st);

    /// The shared search state that is used by all threads.
    SharedSearchState &sss;
    /// Game state instance of this thread.
    std::unique_ptr<State> state;
    /// The evaluator instance of this thread.
    std::unique_ptr<Evaluator> evaluator;
    /// The root moves of this thread.
    std::vector<RootMove> rootMoves;

    // ----------------------------------------------------
    // Thread-related statistics

    /// The number of incremented visits in the current search.
    std::atomic<uint64_t> numNewVisits;
    /// The number of playouts in the current search.
    std::atomic<uint64_t> numPlayouts;
    /// The maximal search depth reached in the current thread.
    int selDepth;

private:
    /// The unique ID of this thread, starting from 0.
    uint32_t id_;
    /// Flags to control the thread loop.
    bool running_, exit_;
    /// The current task function to run in this thread.
    std::function<void(SearchThread &)> task_;
    /// Mutex to protect the thread state
    std::mutex mutex_;
    /// Condition variable to notify the thread state change
    std::condition_variable cv_;
    /// The thread instance to run the thread loop.
    std::thread thread_;
    /// The thread loop that runs the queued search task.
    void threadLoop();

    friend class ThreadPool;  // Allow ThreadPool to access private members
};

/// MainSearchThread class is the master thread in the multi-threaded search.
/// It controls the exit signal and is responsible for kicking off other threads.
class MainSearchThread : public SearchThread
{
public:
    MainSearchThread(std::unique_ptr<SharedSearchState> sss)
        : SearchThread(*sss, 0)
        , sharedSearchState(std::move(sss))
    {}

    /// Clear the main thread state between two search.
    void clear(bool newGame) override;
    /// The main search function entry point.
    void search() override;
    /// Check stop condition (time/nodes) and set shared search state's terminate flag.
    /// @return True if we should stop the search now.
    void checkStopInSearch();
    /// Start a custom task with all threads and wait for them to finish.
    /// @param task The custom task to run in each thread.
    /// @param includeSelf If true, the main thread will also run the task.
    void runCustomTaskAndWait(std::function<void(SearchThread &)> task, bool includeSelf);

    /// The shared search state that is used by all threads.
    std::unique_ptr<SharedSearchState> sharedSearchState;
    /// The best root move result.
    std::pair<Move, Move> bestMove;
    /// The start time point.
    Time startTime;
    /// The optimal and maximal elapsed time allowed for this turn.
    Time optimumTime, maximumTime;
    // The last number of newVisits and playouts that we printed search outputs.
    uint64_t lastOutputNewVisits, lastOutputPlayouts;
    // The last time that we printed search outputs.
    Time lastOutputTime;

    friend class ThreadPool;  // Allow ThreadPool to access private members
};

/// ThreadPool holds all the search threads and controls the init, launch and stop of all threads.
/// It also collects various accumulated statistics from all threads.
class ThreadPool : public std::vector<std::unique_ptr<SearchThread>>
{
public:
    ~ThreadPool();
    /// Wait for (other) search threads to finish their current works.
    /// @note When called inside the main thread, it will only wait for other
    ///     threads to finish their current works, excluding the main thread itself.
    void waitForIdle();
    /// Destroy all old threads and creates requested amount of threads.
    /// @param numThreads The number of threads to create.
    /// @note New threads will immediately go to sleep in threadLoop().
    ///     This must never be called in the worker threads.
    void setNumThreads(size_t numThreads);
    /// Clear all threads state, searcher state and thread pool state for a new game.
    void clear(bool newGame);
    /// Notify all threads to stop thinking immediately.
    void stopThinking();
    /// Start multi-threaded thinking for the given position.
    /// @param state The game state to start searching.
    /// @param options Options of this search.
    /// @param onStop Function to be called (in main thread) when search is finished or interrupted.
    /// @note This is a non-blocking function. It returns immediately after starting all threads.
    void startThinking(const State          &state,
                       const SearchOptions  &options,
                       std::function<void()> onStop = nullptr);
    /// Get the main search thread, which is the first thread in the pool.
    MainSearchThread *main() const { return static_cast<MainSearchThread *>(front().get()); }
    /// Get the number of total incremented visits searched.
    uint64_t newVisitsSearched() const { return accumulate(&SearchThread::numNewVisits); }
    /// Get the number of total playouts searched.
    uint64_t playoutsSearched() const { return accumulate(&SearchThread::numPlayouts); }

private:
    /// Accumulate any atomic member variable from all threads.
    template <typename T>
    T accumulate(std::atomic<T> SearchThread::*member, T init = T(0)) const
    {
        T sum = init;
        for (const auto &th : *this)
            sum += (th.get()->*member).load(std::memory_order_relaxed);
        return sum;
    }
};