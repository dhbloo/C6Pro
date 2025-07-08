#include "thread.h"

#include "eval.h"
#include "evaluator/factory.h"
#include "game.h"

#include <cassert>

SearchThread::SearchThread(SharedSearchState &sss, uint32_t threadId)
    : sss {sss}
    , state {nullptr}
    , evaluator {nullptr}
    , rootMoves {}
    , numVisits {0}
    , numPlayouts {0}
    , selDepth {0}
    , id_ {threadId}
    , running_ {false}
    , exit_ {false}
{
    // Launch the thread loop in a new thread.
    thread_ = std::thread(&SearchThread::threadLoop, this);
}

SearchThread::~SearchThread()
{
    exit_ = true;
    runTask(nullptr);
    thread_.join();
}

void SearchThread::runTask(std::function<void(SearchThread &)> task)
{
    if (std::this_thread::get_id() == thread_.get_id()) {
        // We *are* the worker ⇒ enqueue "tail task" without waiting.
        std::lock_guard<std::mutex> lock(mutex_);
        // at this point running is still true, we simply replace the functor
        task_ = std::move(task);
    }
    else {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [&] { return !running_; });
        task_    = std::move(task);
        running_ = true;
        lock.unlock();
        cv_.notify_one();
    }
}

void SearchThread::waitForIdle()
{
    // Check deadlock if we are already in the worker thread
    assert(std::this_thread::get_id() != thread_.get_id());
    if (!running_)
        return;
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [&] { return !running_; });
}

void SearchThread::setStateAndEvaluator(const State &state)
{
    // Reset board and evaluator instance in this thread to be null
    this->state.reset();
    this->evaluator.reset();
    // Create a new evaluator instance for this thread
    this->evaluator = CreateEvaluator(state.boardSize(), id());
    // Clone the board (this will also sync the evaluator to the board state)
    this->state = std::make_unique<State>(state, this->evaluator.get());
}

void SearchThread::threadLoop()
{
    while (true) {
        std::function<void(SearchThread &)> task;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (!task_) {
                running_ = false;
                cv_.notify_all();
                cv_.wait(lock, [&] { return running_ || exit_; });
            }
            if (exit_)
                return;
            std::swap(task, task_);
        }
        if (task)
            task(*this);
    }
}

void MainSearchThread::checkStopInSearch()
{
    auto &opts = sss.options;

    // Check if we have reached node/time limits
    if (opts.maxVisits && sss.pool.visitsSearched() >= opts.maxVisits
        || opts.useTimeLimit && Now() - startTime >= maximalTime) {
        sss.terminate.store(true, std::memory_order_relaxed);
    }
}

void MainSearchThread::runCustomTaskAndWait(std::function<void(SearchThread &)> task,
                                            bool                                includeSelf)
{
    if (!task)
        return;

    // Run task in non-main threads
    for (size_t i = 1; i < sharedSearchState->pool.size(); i++)
        sharedSearchState->pool[i]->runTask(task);

    // Run task in main thread
    if (includeSelf)
        task(*this);

    // Wait for all other threads to finish
    sharedSearchState->pool.waitForIdle();
}

void ThreadPool::waitForIdle()
{
    // Iterate all other threads and wait for them to finish
    for (auto &th : *this)
        if (th->thread_.get_id() != std::this_thread::get_id())
            th->waitForIdle();
}

void ThreadPool::setNumThreads(size_t numThreads)
{
    // Destroy all threads first (which will also wait for them to be idle)
    while (!empty())
        pop_back();  // std::unique_ptr will automatically destroy thread

    // Create requested amount of threads
    if (numThreads > 0) {
        auto sharedSearchState    = std::make_unique<SharedSearchState>(*this);
        auto sharedSearchStatePtr = sharedSearchState.get();

        // Make sure the first thread created is MainSearchThread
        push_back(std::make_unique<MainSearchThread>(std::move(sharedSearchState)));

        // Create the rest of the threads as SearchThread
        while (size() < numThreads) {
            push_back(std::make_unique<SearchThread>(*sharedSearchStatePtr, (uint32_t)size()));
        }
    }
}

void ThreadPool::clear(bool newGame)
{
    for (auto &th : *this)
        th->runTask([newGame](SearchThread &th) { th.clear(newGame); });
}

void ThreadPool::stopThinking()
{
    if (size() > 0)
        main()->sharedSearchState->terminate.store(true, std::memory_order_relaxed);
}

void ThreadPool::startThinking(const State          &state,
                               const SearchOptions  &options,
                               std::function<void()> onStop)
{
    assert(size() > 0);

    // If we are already thinking, wait for it first
    waitForIdle();

    // Clean up main thread state and copy options and game state
    main()->clear(false);
    main()->sharedSearchState->options = options;
    main()->setStateAndEvaluator(state);

    // TODO: Generate root moves for main search thread

    // Launch a small task to clear threads state and copy state from main thread
    main()->runCustomTaskAndWait(
        [mainTh = main()](SearchThread &th) {
            th.clear(false);
            th.setStateAndEvaluator(*mainTh->state);
            th.rootMoves = mainTh->rootMoves;
        },
        false);

    // Start the main search thread
    main()->runTask([this, onStop = std::move(onStop)](SearchThread &th) {
        th.search();  // Start the main search thread
        if (onStop)   // If onStop is set, queue a tail task to call it
            main()->runTask([onStop = std::move(onStop)](SearchThread &th) { onStop(); });
    });
}

ThreadPool::~ThreadPool()
{
    // Stop if there are still some threads thinking
    stopThinking();
    // Explicitly free all threads
    setNumThreads(0);
}