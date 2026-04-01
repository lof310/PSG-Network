#include <iostream>
#include "psgn.hpp"

using namespace psgn;

int main() {
    std::cout << "============================================================\n";
    std::cout << "       PSGN C++ Edition: Maze Solving Demonstration\n";
    std::cout << "============================================================\n\n";
    
    // Create maze solver
    MazeSolver solver(15, 25);
    
    std::cout << "[Step 1] Generated Random Maze:\n";
    solver.print_maze();
    
    std::cout << "\n[Step 2] Training PSGN on Maze Solution...\n";
    solver.train_on_solution();
    std::cout << "Training complete!\n";
    
    std::cout << "\n[Step 3] Using PSGN to Navigate Maze...\n";
    std::string path = solver.solve();
    solver.print_solution(path);
    
    // Also demonstrate basic PSGN text generation
    std::cout << "\n============================================================\n";
    std::cout << "       PSGN Text Generation Demo\n";
    std::cout << "============================================================\n\n";
    
    PSGN psgn(512, 0.05, 256, 0.1);
    
    // Train on simple sequences
    std::cout << "[Training] Learning patterns...\n";
    psgn.read_text("The cat sat on the mat .");
    psgn.read_text("The dog ran in the park .");
    psgn.read_text("A bird flew over the tree .");
    psgn.read_text("The fish swam in the pond .");
    
    // Generate completions
    std::cout << "\n[Generation] Completing 'The cat':\n";
    std::string gen1 = psgn.generate("The cat", 5);
    std::cout << "Result: " << gen1 << "\n";
    
    std::cout << "\n[Generation] Completing 'The dog':\n";
    std::string gen2 = psgn.generate("The dog", 5);
    std::cout << "Result: " << gen2 << "\n";
    
    std::cout << "\n============================================================\n";
    std::cout << "Demo Complete!\n";
    std::cout << "============================================================\n";
    
    return 0;
}
