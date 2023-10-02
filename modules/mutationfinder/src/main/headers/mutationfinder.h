#ifndef MUA_MUTATIONFINDER_H
#define MUA_MUTATIONFINDER_H

#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"

//------------------------------------------------------------------------------
// New PM interface
//------------------------------------------------------------------------------
struct MutatorPlugin : public llvm::PassInfoMixin<MutatorPlugin> {
  llvm::PreservedAnalyses run(llvm::Module &M,
                              llvm::ModuleAnalysisManager &);
  bool runOnModule(llvm::Module &M);

  // Without isRequired returning true, this pass will be skipped for functions
  // decorated with the optnone LLVM attribute. Note that clang -O0 decorates
  // all functions with optnone.
  static bool isRequired() { return true; }
};

// //------------------------------------------------------------------------------
// // Legacy PM interface
// //------------------------------------------------------------------------------
// struct LegacyMutatorPlugin : public llvm::ModulePass {
//   static char ID;
//   LegacyMutatorPlugin() : ModulePass(ID) {}
//   bool runOnModule(llvm::Module &M) override;

//   MutatorPlugin Impl;
// };

#endif