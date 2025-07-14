#pragma once

#include "../eval.h"

#include <memory>
#include <string>

/// Global variable of the evaluator name to create.
extern std::string EvaluatorName;
/// Global variable of the evaluator model path to create.
extern std::string EvaluatorModelPath;

/// The factory function to create a new evaluator instance.
std::unique_ptr<Evaluator> CreateEvaluator(int boardSize, int numaNodeId);
