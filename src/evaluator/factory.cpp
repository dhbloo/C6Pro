#include "factory.h"

#include "dummy.h"

#include <functional>
#include <iostream>
#include <map>

std::string EvaluatorName      = "dummy";  // Default evaluator name
std::string EvaluatorModelPath = "";       // Default model path, can be set externally

/// EvaluatorArguments holds the arguments for creating an Evaluator instance.
struct EvaluatorArguments
{
    /// The board size of the game.
    int boardSize;
    /// The thread id that this evaluator will run on.
    int threadId;
    /// The path to the model file, if applicable.
    std::string modelPath;
};

using EvaluatorFactory = std::function<std::unique_ptr<Evaluator>(const EvaluatorArguments &)>;

auto EvaluatorFactories = []() {
    std::map<std::string, EvaluatorFactory> factories;

    factories["dummy"] = [](const EvaluatorArguments &args) {
        return std::make_unique<DummyEvaluator>(args.boardSize);
    };

    // Add other evaluators here as needed
    return factories;
}();

std::unique_ptr<Evaluator> CreateEvaluator(int boardSize, int threadId)
{
    auto it = EvaluatorFactories.find(EvaluatorName);
    if (it != EvaluatorFactories.end()) {
        return it->second(EvaluatorArguments {boardSize, threadId, EvaluatorModelPath});
    }
    else {
        std::cerr << "Evaluator type '" << EvaluatorName << "' not found." << std::endl;
        std::cerr << "Use dummy evaluator as fallback." << std::endl;
        return std::make_unique<DummyEvaluator>(boardSize);
    }
}