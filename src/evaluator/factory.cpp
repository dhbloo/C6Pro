#include "factory.h"

#include "dummy.h"
#include "mix9svqnnue.h"

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
    /// The numa node ID that this evaluator is bound to.
    int numaNodeId;
    /// The path to the model file, if applicable.
    std::string modelPath;
};

using EvaluatorFactory = std::function<std::unique_ptr<Evaluator>(const EvaluatorArguments &)>;

auto EvaluatorFactories = []() {
    std::map<std::string, EvaluatorFactory> factories;

    factories["dummy"] = [](const EvaluatorArguments &args) {
        return std::make_unique<DummyEvaluator>(args.boardSize);
    };
    factories["mix9svqnnue"] = [](const EvaluatorArguments &args) -> std::unique_ptr<Evaluator> {
        auto result = mix9svq::Evaluator::create(args.boardSize, args.numaNodeId, args.modelPath);
        if (result)
            return std::make_unique<mix9svq::Evaluator>(std::move(result.value()));
        else {
            std::cerr << "Failed to create mix9svqnnue evaluator: " << result.error() << std::endl;
            return std::make_unique<DummyEvaluator>(args.boardSize);
        }
    };

    // Add other evaluators here as needed
    return factories;
}();

std::unique_ptr<Evaluator> CreateEvaluator(int boardSize, int numaNodeId)
{
    auto it = EvaluatorFactories.find(EvaluatorName);
    if (it != EvaluatorFactories.end()) {
        return it->second(EvaluatorArguments {boardSize, numaNodeId, EvaluatorModelPath});
    }
    else {
        std::cerr << "Evaluator type '" << EvaluatorName << "' not found." << std::endl;
        std::cerr << "Use dummy evaluator as fallback." << std::endl;
        return std::make_unique<DummyEvaluator>(boardSize);
    }
}